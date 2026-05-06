"""
Téléchargeur données météo — ERA5 via Copernicus CDS.

Source   : Copernicus Climate Data Store (CDS) — ECMWF
Licence  : Copernicus Licence to use Copernicus Products
API      : cdsapi (pip install cdsapi)
Compte   : https://cds.climate.copernicus.eu (gratuit)

Variables téléchargées :
    total_precipitation     (tp)   [m/h]   → précipitations mensuelles
    potential_evaporation   (pev)  [m/h]   → ETP (Penman-Monteith FAO56)
    2m_temperature          (t2m)  [K]     → température mensuelle (optionnel)

Résolution native ERA5 : 0.25° × 0.25° (~28 km)
Période recommandée     : 1979–présent (ERA5), 1950–présent (ERA5-Land)

Configuration CDS API :
    Créer ~/.cdsapirc avec :
        url: https://cds.climate.copernicus.eu/api/v2
        key: {uid}:{api_key}
    OU définir les variables d'environnement :
        CDS_UID=12345
        CDS_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

Usage :
    dl = ERA5Downloader(dest_dir=Path("data/raw/era5"), year=2022)
    dl.download_precipitation()
    dl.download_pet()
    ds = dl.load_monthly_data()
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Bbox France pour requêtes CDS
# CDS format : [Nord, Ouest, Sud, Est]  (lat décroissant)
# ──────────────────────────────────────────────────────────────────────────────
_CDS_BBOX_FRANCE = [52.0, -6.0, 41.0, 10.0]   # N, W, S, E

# Dataset ERA5 mensuel (plus léger que le hourly)
_CDS_DATASET_MONTHLY = "reanalysis-era5-single-levels-monthly-means"
_CDS_PRODUCT_TYPE = "monthly_averaged_reanalysis"


class ERA5Downloader:
    """
    Télécharge les données ERA5 mensuelles via Copernicus CDS.

    Args:
        dest_dir: Répertoire de destination
        year: Année à télécharger
        force: Forcer le re-téléchargement
        use_era5_land: Si True, utilise ERA5-Land (0.1° au lieu de 0.25°)
    """

    def __init__(
        self,
        dest_dir: Path,
        year: int = 2022,
        force: bool = False,
        use_era5_land: bool = False,
    ):
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.year = year
        self.force = force
        self.use_era5_land = use_era5_land
        self._client = None

    @staticmethod
    def _normalize_era5_ds(ds: xr.Dataset) -> xr.Dataset:
        """
        Normalise le format CDS post-2024 :
          - Renomme 'valid_time' → 'time' (nouveau format CDS migration 2024)
          - Supprime les coords superflues 'number', 'expver'
        """
        if "valid_time" in ds.dims and "time" not in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        for coord in ("number", "expver"):
            if coord in ds.coords:
                ds = ds.drop_vars(coord)
        return ds

    def _get_client(self):
        """Initialise le client CDS (singleton)."""
        if self._client is not None:
            return self._client

        try:
            import cdsapi
        except ImportError:
            raise ImportError(
                "cdsapi requis : pip install cdsapi\n"
                "Puis configurer ~/.cdsapirc avec vos identifiants CDS."
            )

        # Le client lit automatiquement ~/.cdsapirc ou les variables d'env
        try:
            self._client = cdsapi.Client()
        except Exception as e:
            raise RuntimeError(
                f"Client CDS non initialisé ({e}).\n"
                f"Vérifier ~/.cdsapirc ou les variables CDS_UID / CDS_API_KEY.\n"
                f"Compte gratuit : https://cds.climate.copernicus.eu"
            )
        return self._client

    # ──────────────────────────────────────────────────────────────────────────
    # Téléchargement des variables
    # ──────────────────────────────────────────────────────────────────────────

    def download_precipitation(self) -> Path:
        """
        Télécharge total_precipitation ERA5 mensuel sur la France.

        Returns:
            Path du fichier NetCDF
        """
        out_path = self.dest_dir / f"total_precipitation_{self.year}.nc"
        if out_path.exists() and not self.force:
            logger.info(f"ERA5 précipitations déjà présentes : {out_path}")
            return out_path

        logger.info(f"Téléchargement ERA5 précipitations {self.year}...")
        c = self._get_client()
        c.retrieve(
            _CDS_DATASET_MONTHLY,
            {
                "product_type": _CDS_PRODUCT_TYPE,
                "variable": "total_precipitation",
                "year": str(self.year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "time": "00:00",
                "area": _CDS_BBOX_FRANCE,
                "format": "netcdf",
                "grid": [0.25, 0.25],
            },
            str(out_path),
        )
        logger.success(f"ERA5 précipitations téléchargées : {out_path}")
        return out_path

    def download_pet(self) -> Path:
        """
        Télécharge potential_evaporation ERA5 mensuel.

        Note : ERA5 fournit 'potential_evaporation' (pev) basé sur
        Penman-Monteith FAO56. Valeurs en [m/h] → convertir en mm/mois.

        Returns:
            Path du fichier NetCDF
        """
        out_path = self.dest_dir / f"potential_evaporation_{self.year}.nc"
        if out_path.exists() and not self.force:
            logger.info(f"ERA5 ETP déjà présente : {out_path}")
            return out_path

        logger.info(f"Téléchargement ERA5 ETP {self.year}...")
        c = self._get_client()
        c.retrieve(
            _CDS_DATASET_MONTHLY,
            {
                "product_type": _CDS_PRODUCT_TYPE,
                "variable": "potential_evaporation",
                "year": str(self.year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "time": "00:00",
                "area": _CDS_BBOX_FRANCE,
                "format": "netcdf",
                "grid": [0.25, 0.25],
            },
            str(out_path),
        )
        logger.success(f"ERA5 ETP téléchargée : {out_path}")
        return out_path

    def download_temperature(self) -> Path:
        """Télécharge 2m_temperature (optionnel, pour SPEI)."""
        out_path = self.dest_dir / f"temperature_2m_{self.year}.nc"
        if out_path.exists() and not self.force:
            return out_path

        logger.info(f"Téléchargement ERA5 T2m {self.year}...")
        c = self._get_client()
        c.retrieve(
            _CDS_DATASET_MONTHLY,
            {
                "product_type": _CDS_PRODUCT_TYPE,
                "variable": "2m_temperature",
                "year": str(self.year),
                "month": [f"{m:02d}" for m in range(1, 13)],
                "time": "00:00",
                "area": _CDS_BBOX_FRANCE,
                "format": "netcdf",
                "grid": [0.25, 0.25],
            },
            str(out_path),
        )
        return out_path

    def download_baseline(
        self,
        start_year: int = 1981,
        end_year: int = 2010,
    ) -> Path:
        """
        Télécharge la série temporelle de référence pour le calcul SPI.

        La période 1981-2010 est la référence climatologique OMM standard.
        Télécharge total_precipitation et potential_evaporation pour tous
        les mois de start_year à end_year.

        Args:
            start_year: Début période de référence (défaut 1981)
            end_year: Fin période de référence (défaut 2010)

        Returns:
            Path du répertoire contenant les NetCDF
        """
        baseline_dir = self.dest_dir / "baseline"
        baseline_dir.mkdir(exist_ok=True)

        years = list(range(start_year, end_year + 1))
        logger.info(
            f"Téléchargement baseline ERA5 ({start_year}-{end_year}) "
            f"— {len(years)} années..."
        )

        # Téléchargement en un seul batch (CDS supporte les requêtes multi-années)
        out_path = baseline_dir / f"tp_baseline_{start_year}_{end_year}.nc"
        if out_path.exists() and not self.force:
            logger.info(f"Baseline déjà présente : {out_path}")
            return out_path

        c = self._get_client()
        c.retrieve(
            _CDS_DATASET_MONTHLY,
            {
                "product_type": _CDS_PRODUCT_TYPE,
                "variable": "total_precipitation",
                "year": [str(y) for y in years],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "time": "00:00",
                "area": _CDS_BBOX_FRANCE,
                "format": "netcdf",
                "grid": [0.25, 0.25],
            },
            str(out_path),
        )
        logger.success(
            f"Baseline précipitations téléchargée : {out_path} "
            f"({out_path.stat().st_size / 1e6:.1f} MB)"
        )
        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    # Chargement et post-traitement
    # ──────────────────────────────────────────────────────────────────────────

    def load_monthly_data(self) -> xr.Dataset:
        """
        Charge et fusionne les données ERA5 mensuelles.

        Conversions appliquées :
            tp  : m/h → mm/mois  (×1000 × heures_du_mois)
            pev : m   → mm/mois  (×1000, déjà cumulé sur le mois)
            t2m : K   → °C       (−273.15)

        Returns:
            xr.Dataset avec variables 'precip_mm', 'pet_mm', 'temp_c'
        """
        datasets = {}

        # Précipitations
        precip_path = self.dest_dir / f"total_precipitation_{self.year}.nc"
        if precip_path.exists():
            ds_tp = xr.open_dataset(str(precip_path))
            ds_tp = self._normalize_era5_ds(ds_tp)
            # Si le fichier couvre plusieurs années (ex: 1981-2026), filtrer l'année cible
            if len(ds_tp.time) > 12:
                ds_tp = ds_tp.sel(time=ds_tp.time.dt.year == self.year)
            # ERA5 monthly = taux moyen (m/h) × heures du mois
            if "tp" in ds_tp:
                times = pd.DatetimeIndex(ds_tp.time.values)
                hours_per_month = times.days_in_month * 24
                scale = xr.DataArray(hours_per_month.values, dims="time")
                # tp [m/h] × heures/mois × 1000 = mm/mois
                ds_tp["precip_mm"] = ds_tp["tp"] * scale * 1000.0
                ds_tp["precip_mm"].attrs = {
                    "units": "mm/month",
                    "long_name": "Total precipitation",
                    "source": "ERA5",
                }
                datasets["precip"] = ds_tp[["precip_mm"]]

        # ETP
        pet_path = self.dest_dir / f"potential_evaporation_{self.year}.nc"
        if pet_path.exists():
            ds_pev = xr.open_dataset(str(pet_path))
            ds_pev = self._normalize_era5_ds(ds_pev)
            if len(ds_pev.time) > 12:
                ds_pev = ds_pev.sel(time=ds_pev.time.dt.year == self.year)
            if "pev" in ds_pev:
                # pev ERA5 en m (négatif convention ECMWF → prendre valeur absolue)
                ds_pev["pet_mm"] = np.abs(ds_pev["pev"]) * 1000.0
                ds_pev["pet_mm"].attrs = {
                    "units": "mm/month",
                    "long_name": "Potential evapotranspiration",
                    "source": "ERA5",
                }
                datasets["pet"] = ds_pev[["pet_mm"]]

        # Température (optionnel)
        t2m_path = self.dest_dir / f"temperature_2m_{self.year}.nc"
        if t2m_path.exists():
            ds_t = xr.open_dataset(str(t2m_path))
            ds_t = self._normalize_era5_ds(ds_t)
            if len(ds_t.time) > 12:
                ds_t = ds_t.sel(time=ds_t.time.dt.year == self.year)
            if "t2m" in ds_t:
                ds_t["temp_c"] = ds_t["t2m"] - 273.15
                ds_t["temp_c"].attrs = {"units": "°C", "long_name": "2m temperature"}
                datasets["temp"] = ds_t[["temp_c"]]

        if not datasets:
            raise FileNotFoundError(
                f"Aucune donnée ERA5 trouvée dans {self.dest_dir}. "
                f"Lancer d'abord download_precipitation() et download_pet()."
            )

        # Fusionner sur les dimensions communes (time, latitude, longitude)
        ds_list = list(datasets.values())
        ds_merged = xr.merge(ds_list)

        # Renommer latitude/longitude si nécessaire (ERA5 peut varier)
        rename_map = {}
        if "latitude" in ds_merged.dims and "lat" not in ds_merged.dims:
            rename_map["latitude"] = "lat"
        if "longitude" in ds_merged.dims and "lon" not in ds_merged.dims:
            rename_map["longitude"] = "lon"
        if rename_map:
            ds_merged = ds_merged.rename(rename_map)

        logger.success(
            f"ERA5 chargé : {dict(ds_merged.dims)} — "
            f"variables : {list(ds_merged.data_vars)}"
        )
        return ds_merged

    def load_baseline_series(
        self,
        start_year: int = 1981,
        end_year: int = 2010,
    ) -> xr.DataArray:
        """
        Charge la série de précipitations mensuelles de référence pour SPI.

        Returns:
            DataArray (time, lat, lon) de précipitations mensuelles [mm]
        """
        baseline_path = (
            self.dest_dir / "baseline" / f"tp_baseline_{start_year}_{end_year}.nc"
        )

        # Fallback : si pas de fichier baseline dédié, chercher dans le fichier
        # principal (ex: total_precipitation_2024.nc couvrant 1981-2026)
        if not baseline_path.exists():
            main_path = self.dest_dir / f"total_precipitation_{self.year}.nc"
            if main_path.exists():
                logger.info(
                    f"Fichier baseline non trouvé, utilisation de {main_path.name} "
                    f"filtré sur {start_year}-{end_year}"
                )
                baseline_path = main_path
            else:
                baseline_path = self.download_baseline(start_year, end_year)

        ds = xr.open_dataset(str(baseline_path))
        ds = self._normalize_era5_ds(ds)

        if "tp" not in ds:
            raise KeyError(f"Variable 'tp' non trouvée dans {baseline_path}")

        # Filtrer sur la période de référence
        ds = ds.sel(time=(ds.time.dt.year >= start_year) & (ds.time.dt.year <= end_year))

        times = pd.DatetimeIndex(ds.time.values)
        hours_per_month = xr.DataArray(
            (times.days_in_month * 24).values, dims="time"
        )
        precip_mm = ds["tp"] * hours_per_month * 1000.0
        precip_mm.attrs = {"units": "mm/month", "source": "ERA5 baseline"}

        # Renommer coords si nécessaire
        rename_map = {}
        if "latitude" in precip_mm.dims:
            rename_map["latitude"] = "lat"
        if "longitude" in precip_mm.dims:
            rename_map["longitude"] = "lon"
        if rename_map:
            precip_mm = precip_mm.rename(rename_map)

        logger.success(
            f"Baseline ERA5 : {dict(precip_mm.sizes)} — "
            f"période {start_year}-{end_year}"
        )
        return precip_mm

    def load_full_precip_series(self) -> xr.DataArray:
        """
        Charge la série complète de précipitations (1981 → année cible incluse).

        Nécessaire pour le calcul SPI-12 : le rolling sum de 12 mois
        a besoin de 11 mois d'historique avant le premier mois cible.

        Cherche dans l'ordre :
            1. total_precipitation_baseline.nc (souvent 1981-2022+)
            2. baseline/tp_baseline_1981_2010.nc
            3. total_precipitation_{year}.nc

        Returns:
            DataArray (time, lat, lon) précipitations mensuelles [mm/mois]
        """
        candidates = [
            self.dest_dir / "total_precipitation_baseline.nc",
            self.dest_dir / "baseline" / f"tp_baseline_1981_2010.nc",
        ]
        source_path = None
        for p in candidates:
            if p.exists():
                source_path = p
                break

        if source_path is None:
            raise FileNotFoundError(
                f"Aucun fichier de série longue trouvé dans {self.dest_dir}. "
                "Lancer download_baseline() d'abord."
            )

        ds = xr.open_dataset(str(source_path))
        ds = self._normalize_era5_ds(ds)

        if "tp" not in ds:
            raise KeyError(f"Variable 'tp' absente de {source_path}")

        # Conversion tp [m/h] → mm/mois
        times = pd.DatetimeIndex(ds.time.values)
        hours_per_month = xr.DataArray(
            (times.days_in_month * 24).values, dims="time"
        )
        precip_mm = ds["tp"] * hours_per_month * 1000.0
        precip_mm.attrs = {"units": "mm/month", "source": "ERA5 full series"}

        # Renommer coords
        rename_map = {}
        if "latitude" in precip_mm.dims:
            rename_map["latitude"] = "lat"
        if "longitude" in precip_mm.dims:
            rename_map["longitude"] = "lon"
        if rename_map:
            precip_mm = precip_mm.rename(rename_map)

        logger.success(
            f"Série complète ERA5 : {dict(precip_mm.sizes)} — "
            f"{times[0].strftime('%Y-%m')} → {times[-1].strftime('%Y-%m')}"
        )
        return precip_mm


def run_era5_download(
    dest_dir: Path,
    year: int,
    download_baseline: bool = True,
    force: bool = False,
) -> dict[str, Path]:
    """
    Point d'entrée pipeline.

    Returns:
        dict {'precip': Path, 'pet': Path}
    """
    dl = ERA5Downloader(dest_dir=dest_dir, year=year, force=force)
    paths = {}
    paths["precip"] = dl.download_precipitation()
    paths["pet"] = dl.download_pet()

    if download_baseline:
        paths["baseline"] = dl.download_baseline()

    return paths


if __name__ == "__main__":
    import sys
    from water_model.config import get_path

    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
    dest = get_path("raw_data") / "era5"
    paths = run_era5_download(dest, year=year)
    print(f"\nERA5 {year} téléchargé :")
    for k, p in paths.items():
        print(f"  {k}: {p}")