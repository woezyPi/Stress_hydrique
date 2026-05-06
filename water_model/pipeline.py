"""
Pipeline principale GreenDC — Water Stress Model.

Orchestre l'ensemble des étapes :
  1.  Téléchargement des données (Aqueduct, BRGM, ERA5)
  2.  Nettoyage des rasters (nodata, outliers)
  3.  Reprojection → EPSG:2154 (Lambert-93)
  4.  Harmonisation spatiale → grille 5 km commune
  5.  Calcul SPI-12 (indice de sécheresse)
  6a. Piézométrie Hub'Eau temps réel (anomalies de nappe)
  6.  Calcul GWS combiné (Aqueduct + BRGM + piézométrie Hub'Eau)
  7.  Calcul WSI = 0.6×BWS + 0.25×GWS + 0.15×DrR
  8.  Conversion WaterScore [0-100]
  9.  Projection sur grilles 5/15/30 km WGS84
  10. Validation scientifique
  11. Enrichissement scoring GreenDC

Usage :
    python -m water_model.pipeline --year 2022
    python -m water_model.pipeline --year 2022 --skip-download
    python -m water_model.pipeline --year 2022 --test       # test rapide (SPI statique)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from water_model.config import get_config, get_path


# ============================================================
# Étape 1 : Téléchargement des données
# ============================================================

def step_download(year: int, force: bool = False) -> dict[str, Path]:
    """
    Télécharge Aqueduct 4.0, BRGM BDLISA et ERA5.

    Returns:
        dict avec les chemins de tous les fichiers téléchargés
    """
    logger.info("=== ÉTAPE 1 : Téléchargement des données ===")
    from water_model.downloaders.download_aqueduct import run_aqueduct_download
    from water_model.downloaders.download_brgm import run_brgm_download
    from water_model.downloaders.download_meteofrance import run_era5_download

    raw_dir = get_path("raw_data")
    downloaded = {}

    # Aqueduct 4.0
    try:
        aq_tifs = run_aqueduct_download(raw_dir / "aqueduct", force=force)
        downloaded.update(aq_tifs)
        logger.success(f"Aqueduct : {list(aq_tifs.keys())}")
    except Exception as e:
        logger.error(f"Aqueduct téléchargement échoué : {e}")
        logger.info(
            "Solution manuelle : télécharger depuis https://www.wri.org/data/aqueduct-water-risk-atlas\n"
            f"  Placer bws_raw.tif, gws_raw.tif, drr_raw.tif dans : {raw_dir}/aqueduct/"
        )

    # BRGM BDLISA
    try:
        brgm_tif = run_brgm_download(raw_dir / "brgm", force=force)
        downloaded["brgm_gws"] = brgm_tif
        logger.success(f"BRGM : {brgm_tif}")
    except Exception as e:
        logger.warning(f"BRGM téléchargement échoué : {e}")

    # ERA5
    era5_dir = raw_dir / "era5"
    try:
        era5_paths = run_era5_download(
            era5_dir,
            year=year,
            download_baseline=True,
            force=force,
        )
        downloaded.update({f"era5_{k}": v for k, v in era5_paths.items()})
        logger.success(f"ERA5 : {list(era5_paths.keys())}")
    except ImportError:
        logger.warning(
            "cdsapi non installé → ERA5 indisponible\n"
            "  pip install cdsapi && configurer ~/.cdsapirc"
        )
    except Exception as e:
        logger.warning(f"ERA5 téléchargement échoué : {e}")

    return downloaded


# ============================================================
# Étape 2-3 : Nettoyage + Reprojection
# ============================================================

def step_preprocess(
    raw_tifs: dict[str, Path],
    test_mode: bool = False,
) -> dict[str, Path]:
    """
    Nettoie et reprojette tous les rasters vers EPSG:2154.

    Returns:
        dict des GeoTIFFs nettoyés et reprojetés
    """
    logger.info("=== ÉTAPE 2-3 : Nettoyage + Reprojection L93 ===")
    from water_model.preprocessing.raster_cleaning import clean_aqueduct_raster
    from water_model.preprocessing.reprojection import reproject_all

    processed_dir = get_path("processed_data")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Variables Aqueduct à traiter
    aq_vars = {k: v for k, v in raw_tifs.items()
               if k in ("bws_raw", "gws_raw", "drr_raw", "sev_raw")
               and v is not None and Path(v).exists()}

    # BRGM
    if "brgm_gws" in raw_tifs and raw_tifs["brgm_gws"] is not None:
        brgm_path = raw_tifs["brgm_gws"]
        if Path(brgm_path).exists():
            aq_vars["brgm_gws"] = brgm_path

    # Nettoyage nodata
    cleaned_tifs = {}
    for var_name, tif_path in aq_vars.items():
        clean_path = processed_dir / f"{var_name}_clean.tif"
        if not clean_path.exists():
            clean_aqueduct_raster(tif_path, var_name=var_name, output_path=clean_path)
        else:
            logger.info(f"Raster nettoyé en cache : {clean_path}")
        cleaned_tifs[var_name] = clean_path

    # Reprojection → L93 @ 5km
    l93_tifs = reproject_all(cleaned_tifs, processed_dir / "l93", resolution_m=5000.0)

    return l93_tifs


# ============================================================
# Étape 4 : Harmonisation spatiale
# ============================================================

def step_harmonize(
    l93_tifs: dict[str, Path],
    era5_ds: Optional[xr.Dataset],
) -> dict[str, xr.DataArray]:
    """
    Ré-échantillonne tous les rasters sur la grille commune 5km L93.

    Returns:
        dict de DataArrays harmonisés
    """
    logger.info("=== ÉTAPE 4 : Harmonisation spatiale → 5km L93 ===")
    from water_model.preprocessing.resampling import harmonize_all_rasters

    return harmonize_all_rasters(l93_tifs, era5_ds, get_path("processed_data") / "harmonized")


# ============================================================
# Étape 5 : Calcul SPI-12
# ============================================================

def step_compute_spi(
    year: int,
    test_mode: bool = False,
) -> xr.DataArray:
    """
    Calcule le SPI-12 depuis les données ERA5.

    Args:
        year: Année à évaluer
        test_mode: Si True, utilise des données synthétiques (pas d'ERA5)

    Returns:
        DataArray SPI-12 (time, y, x) ou (y, x) en mode test
    """
    logger.info("=== ÉTAPE 5 : Calcul SPI-12 ===")
    from water_model.modeling.drought_model import SPICalculator

    era5_dir = get_path("raw_data") / "era5"
    calc = SPICalculator(window=12)

    if test_mode:
        logger.warning("Mode test : SPI synthétique (bruit gaussien)")
        # Créer une grille France synthétique
        from water_model.preprocessing.resampling import build_reference_grid
        xs, ys = build_reference_grid(5000.0)
        np.random.seed(42)
        spi_data = np.random.normal(0, 1, size=(12, len(ys), len(xs))).astype(np.float32)
        spi = xr.DataArray(
            spi_data,
            dims=["time", "y", "x"],
            coords={
                "time": pd.date_range(f"{year}-01", periods=12, freq="MS"),
                "y": ys, "x": xs,
            },
            name="spi12",
            attrs={"long_name": "SPI-12 (synthetic test mode)", "units": "dimensionless"},
        )
        return spi

    # Charger ERA5
    from water_model.downloaders.download_meteofrance import ERA5Downloader
    dl = ERA5Downloader(dest_dir=era5_dir, year=year)

    # Mode série complète : rolling sum sur 1981-2022
    # → tous les mois de l'année cible ont un SPI valide
    try:
        full_precip = dl.load_full_precip_series()
        logger.info(
            f"Série complète chargée : {dict(full_precip.sizes)} — "
            f"mode série complète activé"
        )
        spi = calc.compute(full_precip, target_year=year)
    except (FileNotFoundError, KeyError) as e:
        logger.warning(
            f"Série complète non disponible ({e}) → fallback mode split"
        )
        try:
            era5_year = dl.load_monthly_data()
        except FileNotFoundError as e2:
            logger.warning(f"ERA5 non disponible ({e2}) → SPI synthétique")
            return step_compute_spi(year, test_mode=True)

        if "precip_mm" not in era5_year:
            logger.warning("precip_mm absent du dataset ERA5 → SPI synthétique")
            return step_compute_spi(year, test_mode=True)

        try:
            baseline_da = dl.load_baseline_series()
        except Exception as e3:
            logger.warning(f"Baseline non disponible ({e3}) → SPI sur série courte")
            baseline_da = None

        spi = calc.compute(era5_year["precip_mm"], baseline_da)

    # Reprojeter chaque mois WGS84 → L93 5km (conserver la dim time)
    from water_model.preprocessing.resampling import reproject_wgs84_to_l93

    if "time" in spi.dims and len(spi.time) > 1:
        slices_l93 = []
        for t in spi.time:
            spi_t = spi.sel(time=t)
            spi_t_l93 = reproject_wgs84_to_l93(spi_t, var_name="spi12")
            slices_l93.append(spi_t_l93)
        spi_l93 = xr.concat(slices_l93, dim=spi.time)
        spi_l93.name = "spi12"
        logger.info(f"SPI-12 mensuel reprojeté : {dict(spi_l93.sizes)}")
    else:
        spi_l93 = reproject_wgs84_to_l93(spi, var_name="spi12")
        spi_l93.name = "spi12"

    return spi_l93


# ============================================================
# Étape 6 : Calcul GWS combiné
# ============================================================

def step_fetch_piezo(
    year: int,
    reference_grid: Optional[xr.DataArray] = None,
    test_mode: bool = False,
) -> Optional[xr.DataArray]:
    """
    Télécharge les anomalies piézométriques Hub'Eau et les interpole sur la grille.

    Returns:
        DataArray piezo_stress normalisé [0, 1] sur la grille L93,
        ou None si données insuffisantes.
    """
    logger.info("=== ÉTAPE 6a : Piézométrie Hub'Eau (temps réel) ===")

    if test_mode:
        logger.warning("Mode test : piézométrie synthétique")
        if reference_grid is not None:
            piezo = xr.full_like(reference_grid, fill_value=0.4)
            piezo.name = "piezo_stress"
            return piezo
        return None

    try:
        from water_model.downloaders.download_hubeau_piezo import HubEauPiezoDownloader
        from scipy.interpolate import griddata

        dl = HubEauPiezoDownloader(dest_dir=get_path("raw_data") / "hubeau")
        stations = dl.fetch_stations(year=year)
        if stations.empty:
            logger.warning("Aucune station piézo → skip")
            return None

        anomalies = dl.compute_anomalies(stations_gdf=stations, year=year)
        if anomalies.empty or len(anomalies) < 50:
            logger.warning(f"Trop peu de stations piézo ({len(anomalies)}) → skip")
            return None

        # Reprojeter les stations en L93 pour interpoler sur la grille
        anomalies_l93 = anomalies.to_crs("EPSG:2154")
        station_x = anomalies_l93.geometry.x.values
        station_y = anomalies_l93.geometry.y.values
        station_vals = anomalies_l93["piezo_stress_norm"].values

        if reference_grid is None:
            logger.warning("Pas de grille de référence pour interpoler piézo → skip")
            return None

        # Grille cible
        grid_x, grid_y = np.meshgrid(
            reference_grid.coords["x"].values,
            reference_grid.coords["y"].values,
        )

        # Interpolation IDW (linear griddata + fallback nearest)
        piezo_vals = griddata(
            points=np.column_stack([station_x, station_y]),
            values=station_vals,
            xi=(grid_x, grid_y),
            method="linear",
        )
        # Remplir les NaN restants avec nearest
        mask_nan = np.isnan(piezo_vals)
        if mask_nan.any():
            nearest_vals = griddata(
                points=np.column_stack([station_x, station_y]),
                values=station_vals,
                xi=(grid_x[mask_nan], grid_y[mask_nan]),
                method="nearest",
            )
            piezo_vals[mask_nan] = nearest_vals

        piezo_da = xr.DataArray(
            piezo_vals.astype(np.float32).clip(0.0, 1.0),
            dims=reference_grid.dims,
            coords=reference_grid.coords,
            name="piezo_stress",
            attrs={
                "long_name": "Piezometric Stress (Hub'Eau anomaly)",
                "units": "dimensionless",
                "source": "Hub'Eau Piézométrie BRGM",
                "year": year,
                "n_stations": len(anomalies),
                "interpolation": "linear + nearest fallback",
            },
        )

        logger.success(
            f"Piézométrie interpolée : {len(anomalies)} stations → "
            f"grille {piezo_da.shape}, "
            f"stress moyen = {float(piezo_da.mean().values):.2f}"
        )
        return piezo_da

    except ImportError as e:
        logger.warning(f"Dépendance piézo manquante ({e}) → skip")
        return None
    except Exception as e:
        logger.warning(f"Piézométrie Hub'Eau échouée ({e}) → skip")
        return None


def step_compute_gws(
    harmonized: dict[str, xr.DataArray],
    piezo_stress: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Calcule le Groundwater Stress normalisé.

    v1.0 : Aqueduct + BRGM (statique)
    v1.1 : Aqueduct + BRGM + Hub'Eau piézométrie (dynamique)
    """
    logger.info("=== ÉTAPE 6 : Calcul Groundwater Stress ===")
    from water_model.modeling.groundwater_model import GroundwaterStressModel

    cfg = get_config()
    gws_w = cfg.get("hubeau_piezo", {}).get("gws_weights_v11", {})

    # Choisir les poids selon la disponibilité de la piézo
    if piezo_stress is not None:
        model = GroundwaterStressModel(
            weight_aqueduct=gws_w.get("aqueduct", 0.40),
            weight_brgm=gws_w.get("brgm", 0.30),
            weight_piezo=gws_w.get("piezo", 0.30),
        )
        logger.info("GWS v1.1 : Aqueduct + BRGM + Hub'Eau piézométrie")
    else:
        model = GroundwaterStressModel()
        logger.info("GWS v1.0 : Aqueduct + BRGM (pas de piézométrie)")

    # GWS Aqueduct (brut 0-5)
    gws_aq = harmonized.get("gws_raw")
    if gws_aq is None:
        logger.warning("gws_raw Aqueduct absent → GWS uniforme 0.3")
        from water_model.preprocessing.resampling import build_reference_grid
        xs, ys = build_reference_grid(5000.0)
        gws_aq = xr.DataArray(
            np.full((len(ys), len(xs)), 1.5, dtype=np.float32),
            dims=["y", "x"],
            coords={"y": ys, "x": xs},
            name="gws_raw",
        )

    # GWS BRGM (normalisé 0-1)
    gws_brgm = harmonized.get("brgm_gws")

    # Calcul combiné
    gws_combined = model.compute(gws_aq, gws_brgm, piezo_stress)

    return gws_combined


# ============================================================
# Étape 7-8 : WSI + WaterScore
# ============================================================

def step_compute_wsi(
    harmonized: dict[str, xr.DataArray],
    gws_norm: xr.DataArray,
    drought_norm: xr.DataArray,
    bnpe_stress: xr.DataArray | None = None,
) -> xr.Dataset:
    """
    Calcule WSI = 0.6×BWS + 0.25×GWS + 0.15×DrR puis WaterScore.

    Returns:
        xr.Dataset avec wsi_mean, wsi_p90, water_score, ...
    """
    logger.info("=== ÉTAPE 7-8 : Calcul WSI + WaterScore ===")
    from water_model.modeling.water_stress_model import (
        WaterStressModel,
        WSIWeights,
    )
    from water_model.config import get_config

    cfg = get_config()
    w = cfg["wsi"]["weights"]

    model = WaterStressModel(
        weights=WSIWeights(
            bws=w["bws"],
            gws=w["gws"],
            drought=w["drought"],
        ),
        normalization=cfg["wsi"].get("bws_normalization", "pfister"),
    )

    # BWS Aqueduct brut
    bws_raw = harmonized.get("bws_raw")
    if bws_raw is None:
        logger.warning("bws_raw absent → BWS uniforme 0.3 (stress modéré)")
        bws_raw = xr.full_like(gws_norm, fill_value=1.5)
        bws_raw.name = "bws_raw"

    # Normalisation BWS (Pfister)
    bws_norm = model.normalize_bws(bws_raw, bws_max=cfg["wsi"]["bws_score_max"])

    # Modulation BNPE : rehausser le BWS là où les prélèvements locaux sont élevés
    if bnpe_stress is not None:
        try:
            bnpe_aligned = bnpe_stress.interp_like(bws_norm, method="linear")
            # BWS_final = BWS_pfister + (1 - BWS_pfister) × bnpe_weight × BNPE_stress
            # → ne peut qu'augmenter le stress, jamais le réduire
            bnpe_weight = cfg["wsi"].get("bnpe_weight", 0.30)
            bws_norm = (bws_norm + (1.0 - bws_norm) * bnpe_weight * bnpe_aligned).clip(0.0, 1.0)
            logger.info(f"BWS modulé par BNPE (poids={bnpe_weight})")
        except Exception as e:
            logger.warning(f"Modulation BNPE échouée ({e})")

    # Alignement spatial des 3 couches sur la même grille
    try:
        gws_aligned = gws_norm.interp_like(bws_norm, method="linear")
    except Exception as e:
        logger.warning(f"Alignement GWS échoué ({e}) → utilisation brute")
        gws_aligned = gws_norm

    # Drought : interpoler seulement les dims spatiales (conserver time)
    try:
        if "time" in drought_norm.dims:
            drought_aligned = drought_norm.interp(
                y=bws_norm.y, x=bws_norm.x, method="linear",
            )
        else:
            drought_aligned = drought_norm.interp_like(bws_norm, method="linear")
    except Exception as e:
        logger.warning(f"Alignement drought échoué ({e}) → utilisation brute")
        drought_aligned = drought_norm

    logger.info(
        f"Alignement : bws={bws_norm.dims} gws={gws_aligned.dims} "
        f"drought={drought_aligned.dims} "
        f"(time={'time' in drought_aligned.dims})"
    )

    # Calcul WSI (propagation analytique premier ordre)
    wsi_ds = model.compute(bws_norm, gws_aligned, drought_aligned)

    # Conversion → WaterScore
    wsi_ds = model.to_water_score(wsi_ds)

    return wsi_ds


# ============================================================
# Étape 9 : Projection sur les 3 grilles
# ============================================================

def step_project_grids(
    wsi_ds: xr.Dataset,
    year: int,
) -> dict[str, xr.Dataset]:
    """
    Projette le WSI sur les grilles France 5/15/30 km WGS84.

    Returns:
        dict {'5km': Dataset, '15km': Dataset, '30km': Dataset}
    """
    logger.info("=== ÉTAPE 9 : Projection sur grilles GreenDC ===")
    from water_model.interpolation.grid_projection import project_wsi_to_grids

    output_dir = get_path("outputs")
    return project_wsi_to_grids(wsi_ds, year=year, output_dir=output_dir)


# ============================================================
# Étape 10 : Validation
# ============================================================

def step_validate(
    wsi_ds: xr.Dataset,
    spi: Optional[xr.DataArray],
    bws_norm: Optional[xr.DataArray],
) -> None:
    """Lance les tests de validation et sauvegarde le rapport."""
    logger.info("=== ÉTAPE 10 : Validation scientifique ===")
    from water_model.validation.validation import run_all_tests

    report = run_all_tests(wsi_ds, spi=spi, bws_norm=bws_norm)

    out_dir = get_path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "water_validation_report.txt"
    report_path.write_text(report.summary(), encoding="utf-8")
    logger.info(f"Rapport de validation : {report_path}")

    if not report.all_passed:
        logger.warning(
            "⚠ Certains tests de validation ont échoué.\n"
            "  Consulter water_validation_report.txt avant utilisation."
        )


# ============================================================
# Étape 11 : Enrichissement scoring GreenDC
# ============================================================

def step_enrich_scoring(year: int) -> None:
    """
    Enrichit les grilles scoring GreenDC existantes avec le WaterScore.
    """
    logger.info("=== ÉTAPE 11 : Enrichissement scoring GreenDC ===")
    from water_model.integration.greendc_water_score import enrich_scoring_grids

    poc_dir = Path("POC/data")
    scoring_grids = {
        "low":    poc_dir / "points_low.geojson",
        "medium": poc_dir / "points_medium.geojson",
        "high":   poc_dir / "points_high.geojson",
    }
    existing = {k: v for k, v in scoring_grids.items() if v.exists()}
    if not existing:
        logger.warning("Grilles scoring POC non trouvées — skip enrichissement")
        return

    output_dir = get_path("outputs")
    wsi_paths = {
        res: output_dir / f"water_stress_{res}_{year}.nc"
        for res in ["5km", "15km", "30km"]
    }

    enrich_scoring_grids(
        scoring_grid_paths=existing,
        wsi_grid_paths=wsi_paths,
        output_dir=output_dir / "greendc_scores_water",
    )


# ============================================================
# Pipeline complète
# ============================================================

def run_pipeline(
    year: int = 2022,
    test_mode: bool = False,
    skip_download: bool = False,
    force: bool = False,
) -> None:
    """
    Pipeline GreenDC Water Stress — complète.

    Args:
        year: Année de simulation (défaut 2022)
        test_mode: Mode rapide avec données synthétiques (test/CI)
        skip_download: Ne pas retélécharger les données
        force: Forcer le recalcul même si les caches existent
    """
    t_start = time.time()
    logger.info(
        f"GreenDC Water Stress Model — Année {year} "
        f"{'[TEST MODE]' if test_mode else ''}"
    )

    # 1. Téléchargements
    raw_tifs = {}
    if not skip_download:
        raw_tifs = step_download(year, force=force)
    else:
        # Reconstruire les chemins depuis les fichiers existants
        raw_dir = get_path("raw_data")
        for var in ("bws_raw", "gws_raw", "drr_raw"):
            p = raw_dir / "aqueduct" / f"{var}.tif"
            if p.exists():
                raw_tifs[var] = p
        brgm_p = raw_dir / "brgm" / "brgm_gws.tif"
        if brgm_p.exists():
            raw_tifs["brgm_gws"] = brgm_p

    # 2-3. Preprocessing
    l93_tifs = step_preprocess(raw_tifs, test_mode=test_mode)

    # ERA5 (optionnel)
    era5_ds = None
    if not test_mode:
        try:
            from water_model.downloaders.download_meteofrance import ERA5Downloader
            dl = ERA5Downloader(get_path("raw_data") / "era5", year=year)
            era5_ds = dl.load_monthly_data()
        except Exception as e:
            logger.warning(f"ERA5 non disponible ({e})")

    # 4. Harmonisation
    harmonized = step_harmonize(l93_tifs, era5_ds)

    # 5. SPI-12
    spi = step_compute_spi(year, test_mode=test_mode)

    # Drought Risk normalisé
    from water_model.modeling.drought_model import SPICalculator
    cfg = get_config()
    calc = SPICalculator(window=12)
    drought_norm = calc.drought_risk_normalized(spi)

    # 6a. Piézométrie Hub'Eau (temps réel)
    reference_grid = harmonized.get("bws_raw")
    if reference_grid is None:
        reference_grid = harmonized.get("gws_raw")
    piezo_stress = step_fetch_piezo(
        year=year,
        reference_grid=reference_grid,
        test_mode=test_mode,
    )

    # 6. GWS (avec piézo si disponible)
    gws_norm = step_compute_gws(harmonized, piezo_stress=piezo_stress)

    # 6b. Prélèvements BNPE (modulation locale du BWS)
    bnpe_stress = None
    if not test_mode:
        try:
            from water_model.downloaders.download_bnpe import BNPEDownloader
            bnpe_dl = BNPEDownloader(dest_dir=get_path("raw_data") / "bnpe", year=year)
            withdrawal = bnpe_dl.rasterize_to_grid()
            bnpe_stress = bnpe_dl.normalize_withdrawal_stress(withdrawal)
            logger.success(f"BNPE : stress moyen = {float(bnpe_stress.mean()):.3f}")
        except Exception as e:
            logger.warning(f"BNPE non disponible ({e}) → skip modulation locale")

    # 7-8. WSI + WaterScore
    wsi_ds = step_compute_wsi(harmonized, gws_norm, drought_norm, bnpe_stress=bnpe_stress)

    # 9. Grilles WGS84 + masque France métropolitaine
    # Le masque est appliqué APRÈS la projection (NN filling précède le masque)
    # via GridProjector._apply_france_mask() dans grid_projection.py
    grids = step_project_grids(wsi_ds, year)

    # 10. Validation
    # Utiliser le BWS harmonisé (même grille que le WSI)
    bws_norm_val = None
    if "bws_norm" in wsi_ds:
        bws_norm_val = wsi_ds["bws_norm"]
    step_validate(wsi_ds, spi=spi, bws_norm=bws_norm_val)

    # 11. Enrichissement scoring
    step_enrich_scoring(year)

    elapsed = time.time() - t_start
    output_dir = get_path("outputs")
    logger.success(
        f"\n{'='*60}\n"
        f"Pipeline Water Stress terminée en {elapsed/60:.1f} minutes\n"
        f"Sorties disponibles dans : {output_dir}\n"
        f"  water_stress_5km_{year}.nc   — WSI haute résolution\n"
        f"  water_stress_15km_{year}.nc  — WSI résolution moyenne\n"
        f"  water_stress_30km_{year}.nc  — WSI basse résolution\n"
        f"  water_validation_report.txt  — Rapport de validation\n"
        f"{'='*60}"
    )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GreenDC Water Stress Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python -m water_model.pipeline --year 2022
  python -m water_model.pipeline --year 2022 --skip-download
  python -m water_model.pipeline --test          # données synthétiques, rapide
  python -m water_model.pipeline --year 2022 --force  # recalcul complet
        """,
    )
    parser.add_argument("--year", type=int, default=2022, help="Année de simulation")
    parser.add_argument(
        "--test", action="store_true",
        help="Mode test : données synthétiques (pas besoin d'Aqueduct/ERA5)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Ne pas retélécharger les données (utiliser le cache)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Forcer le recalcul même si les caches existent",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            year=args.year,
            test_mode=args.test,
            skip_download=args.skip_download,
            force=args.force,
        )
    except KeyboardInterrupt:
        logger.info("Pipeline interrompue")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline échouée : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
