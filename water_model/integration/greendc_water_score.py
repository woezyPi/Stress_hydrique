"""
Intégration du WaterScore dans le scoring GreenDC.

Ce module fournit l'interface principale entre le modèle hydrique et
la plateforme GreenDC :

    get_water_stress(lat, lon) → dict
    enrich_scoring_grids(...)  → dict[str, GeoDataFrame]

Il est le pendant exact de carbon_model/integration/greendc_scoring.py
pour la composante hydrique.

Score final GreenDC (pondérations v4) :
    score_final = (
        score_temperature * 0.25
      + score_water       * 0.20   ← ce module
      + score_carbon      * 0.20
      + score_environment * 0.10
      + score_technique   * 0.10
      + score_fiabilite   * 0.10
      + couverture_nappes * 0.05
    )

Usage :
    # Requête ponctuelle (API)
    result = get_water_stress(lat=48.86, lon=2.35)
    print(result["water_score"])   # 72.4

    # Enrichissement batch des grilles scoring
    provider = WaterStressProvider(wsi_grid_paths)
    scores = provider.get_batch(lats, lons)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from scipy.spatial import cKDTree

from water_model.modeling.water_stress_model import interpret_water_score


# ──────────────────────────────────────────────────────────────────────────────
# Provider d'accès au WaterScore (parallèle à CarbonIntensityProvider)
# ──────────────────────────────────────────────────────────────────────────────

class WaterStressProvider:
    """
    Interface d'accès au WaterScore pour n'importe quel point (lat, lon).

    Charge les grilles NetCDF en mémoire et utilise un KDTree pour
    les requêtes spatiales O(log n).

    Usage :
        provider = WaterStressProvider({'5km': path, '15km': path, '30km': path})
        score = provider.get_score(lat=48.86, lon=2.35)
        result = provider.get_full(lat=48.86, lon=2.35)
    """

    def __init__(self, wsi_grid_paths: dict[str, Path]):
        """
        Args:
            wsi_grid_paths: {'5km': Path, '15km': Path, '30km': Path}
        """
        self._datasets: dict[str, xr.Dataset] = {}
        self._kdtrees: dict[str, cKDTree] = {}
        self._coords: dict[str, np.ndarray] = {}

        for res, path in wsi_grid_paths.items():
            if path and Path(path).exists():
                try:
                    with xr.open_dataset(str(path)) as ds:
                        self._datasets[res] = ds.load()

                    lons = self._datasets[res].coords["lon"].values
                    lats = self._datasets[res].coords["lat"].values
                    lon_grid, lat_grid = np.meshgrid(lons, lats)
                    # Pseudo-métrique : convertir degrés → km
                    lat_rad = np.radians(lat_grid.ravel())
                    tree_x = lon_grid.ravel() * 111.0 * np.cos(lat_rad)
                    tree_y = lat_grid.ravel() * 111.0
                    coords_km = np.column_stack([tree_x, tree_y])
                    self._kdtrees[res] = cKDTree(coords_km)
                    self._coords[res] = coords_km

                    logger.info(
                        f"WaterStress provider ({res}) : "
                        f"{len(lats)}×{len(lons)} points chargés"
                    )
                except Exception as e:
                    logger.warning(f"Impossible de charger {path} ({e})")
            else:
                logger.warning(f"Grille WSI non trouvée ({res}) : {path}")

    def get_score(
        self,
        lat: float,
        lon: float,
        resolution: str = "15km",
    ) -> float:
        """
        Retourne le WaterScore [0-100] pour un point.

        Args:
            lat, lon: Coordonnées WGS84
            resolution: '5km', '15km' ou '30km'

        Returns:
            float WaterScore [0-100], ou 50.0 si aucune donnée
        """
        result = self.get_full(lat, lon, resolution)
        return result.get("water_score", 50.0)

    def get_full(
        self,
        lat: float,
        lon: float,
        resolution: str = "15km",
    ) -> dict:
        """
        Retourne le profil hydrique complet pour un point.

        Returns:
            dict avec water_score, wsi_mean, bws_norm, gws_norm,
            drought_norm, uncertainty, interpretation, resolution
        """
        # Fallback vers résolution disponible la plus fine
        res = self._resolve(resolution)
        if res is None:
            return self._empty_result(lat, lon, "Aucune donnée disponible")

        ds = self._datasets[res]
        tree = self._kdtrees[res]

        # KDTree query en km
        lat_rad = np.radians(lat)
        query_x = lon * 111.0 * np.cos(lat_rad)
        query_y = lat * 111.0
        dist, idx = tree.query([query_x, query_y])
        max_dist_km = 100.0  # >100 km = hors couverture

        if dist > max_dist_km:
            return self._empty_result(lat, lon, f"Hors couverture ({dist:.1f} km)")

        # Délinéariser l'index
        lons = ds.coords["lon"].values
        lats = ds.coords["lat"].values
        ny, nx = len(lats), len(lons)
        iy, ix = divmod(idx, nx)

        def _get(var: str, default: float = np.nan) -> float:
            try:
                return float(ds[var].isel(lat=iy, lon=ix).values)
            except Exception:
                return default

        water_score = _get("water_score", 50.0)

        return {
            "lat": lat,
            "lon": lon,
            "resolution": res,
            "water_score": round(water_score, 1),
            "wsi_mean": round(_get("wsi_mean", np.nan), 3),
            "wsi_p10": round(_get("wsi_p10", np.nan), 3),
            "wsi_p90": round(_get("wsi_p90", np.nan), 3),
            "bws_norm": round(_get("bws_norm", np.nan), 3),
            "gws_norm": round(_get("gws_norm", np.nan), 3),
            "drought_norm": round(_get("drought_norm_mean", np.nan), 3),
            "water_score_p10": round(_get("water_score_p10", np.nan), 1),
            "water_score_p90": round(_get("water_score_p90", np.nan), 1),
            "uncertainty_score_pts": round(_get("water_score_uncertainty", np.nan), 1),
            "nearest_point_km": round(dist, 1),  # déjà en km
            "interpretation": interpret_water_score(water_score),
        }

    def get_batch(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        resolution: str = "15km",
    ) -> pd.DataFrame:
        """
        Requête batch pour un ensemble de points.

        Optimisé pour les grilles scoring GreenDC (10k+ points).

        Args:
            lats, lons: Tableaux numpy des coordonnées
            resolution: Résolution des grilles

        Returns:
            DataFrame avec colonnes water_score, wsi_mean, bws_norm, ...
        """
        res = self._resolve(resolution)
        if res is None:
            n = len(lats)
            return pd.DataFrame({
                "water_score": [50.0] * n,
                "wsi_mean": [np.nan] * n,
            })

        ds = self._datasets[res]
        tree = self._kdtrees[res]

        lons_arr = ds.coords["lon"].values
        lats_arr = ds.coords["lat"].values
        nx = len(lons_arr)

        # KDTree batch query en km
        lats_np = np.asarray(lats)
        lons_np = np.asarray(lons)
        lat_rad = np.radians(lats_np)
        pts_x = lons_np * 111.0 * np.cos(lat_rad)
        pts_y = lats_np * 111.0
        pts = np.column_stack([pts_x, pts_y])
        dists, idxs = tree.query(pts)

        iys, ixs = np.divmod(idxs, nx)

        results = []
        for i, (iy, ix, dist) in enumerate(zip(iys, ixs, dists)):
            def _get(var: str, d: float = np.nan) -> float:
                try:
                    return float(ds[var].isel(lat=int(iy), lon=int(ix)).values)
                except Exception:
                    return d

            ws = _get("water_score", 50.0)
            results.append({
                "water_score": round(ws, 1),
                "wsi_mean": round(_get("wsi_mean", np.nan), 3),
                "bws_norm": round(_get("bws_norm", np.nan), 3),
                "gws_norm": round(_get("gws_norm", np.nan), 3),
                "drought_norm": round(_get("drought_norm_mean", np.nan), 3),
                "water_score_uncertainty": round(_get("water_score_uncertainty", np.nan), 1),
                "nearest_km": round(dist, 1),  # déjà en km
            })

        df = pd.DataFrame(results)
        logger.info(
            f"Batch water score ({resolution}) : "
            f"{len(df)} points, "
            f"moy={df['water_score'].mean():.1f}/100"
        )
        return df

    def _resolve(self, resolution: str) -> Optional[str]:
        """Retourne la résolution disponible la plus proche de la demandée."""
        if resolution in self._datasets:
            return resolution
        for fallback in ["15km", "5km", "30km"]:
            if fallback in self._datasets:
                logger.warning(
                    f"Résolution {resolution} non disponible → fallback {fallback}"
                )
                return fallback
        return None

    @staticmethod
    def _empty_result(lat: float, lon: float, reason: str) -> dict:
        return {
            "lat": lat, "lon": lon,
            "water_score": 50.0,
            "wsi_mean": np.nan,
            "error": reason,
        }

    def available_resolutions(self) -> list[str]:
        return list(self._datasets.keys())


# ──────────────────────────────────────────────────────────────────────────────
# API publique : get_water_stress()
# ──────────────────────────────────────────────────────────────────────────────

def get_water_stress(
    lat: float,
    lon: float,
    year: int = 2022,
    resolution: str = "15km",
    wsi_grid_dir: Optional[Path] = None,
) -> dict:
    """
    Point d'entrée principal — WaterScore pour un data center.

    Compatible avec l'API GreenDC (pendant de query_carbon_intensity()).

    Args:
        lat, lon: Coordonnées WGS84
        year: Année de référence des données
        resolution: Résolution de la grille ('5km', '15km', '30km')
        wsi_grid_dir: Répertoire des grilles NetCDF (None → auto-détection)

    Returns:
        dict avec water_score, WSI composantes, interprétation

    Exemple :
        >>> result = get_water_stress(lat=43.3, lon=5.4, year=2022)
        >>> print(f"Marseille — WaterScore: {result['water_score']}/100")
        Marseille — WaterScore: 35.2/100
    """
    if wsi_grid_dir is None:
        from water_model.config import get_path
        wsi_grid_dir = get_path("outputs")

    paths = {
        res: wsi_grid_dir / f"water_stress_{res}_{year}.nc"
        for res in ["5km", "15km", "30km"]
    }

    provider = WaterStressProvider(paths)
    return provider.get_full(lat, lon, resolution=resolution)


# ──────────────────────────────────────────────────────────────────────────────
# Enrichissement des grilles scoring GreenDC existantes
# ──────────────────────────────────────────────────────────────────────────────

def enrich_scoring_grids(
    scoring_grid_paths: dict[str, Path],
    wsi_grid_paths: dict[str, Path],
    output_dir: Optional[Path] = None,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Enrichit les grilles de scoring GreenDC avec le WaterScore hydrique.

    Remplace le proxy 'score_eau' existant (basé sur la pluviométrie annuelle)
    par un indicateur physiquement fondé issu du modèle WSI.

    Args:
        scoring_grid_paths: {'low': Path, 'medium': Path, 'high': Path}
        wsi_grid_paths: {'5km': Path, '15km': Path, '30km': Path}
        output_dir: Répertoire de sortie

    Returns:
        dict de GeoDataFrames enrichis
    """
    provider = WaterStressProvider(wsi_grid_paths)

    res_mapping = {
        "low":    "30km",
        "medium": "15km",
        "high":   "5km",
    }

    enriched = {}

    for grid_level, grid_path in scoring_grid_paths.items():
        if not Path(grid_path).exists():
            logger.warning(f"Grille scoring non trouvée : {grid_path}")
            continue

        logger.info(f"Enrichissement grille {grid_level}...")
        gdf = gpd.read_file(str(grid_path))

        wsi_res = res_mapping.get(grid_level, "15km")

        # Extraire coordonnées
        lats = gdf.geometry.y.values
        lons = gdf.geometry.x.values

        # Batch query
        water_df = provider.get_batch(lats, lons, resolution=wsi_res)

        gdf["water_score"]            = water_df["water_score"].values
        gdf["wsi_mean"]               = water_df["wsi_mean"].values
        gdf["bws_norm"]               = water_df["bws_norm"].values
        gdf["gws_norm"]               = water_df["gws_norm"].values
        gdf["drought_norm"]           = water_df["drought_norm"].values
        gdf["water_score_uncertainty"] = water_df["water_score_uncertainty"].values

        # Recalcul score final si possible
        if "score_final" in gdf.columns:
            gdf = _recompute_final_score_v4(gdf)

        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(output_dir) / f"points_{grid_level}_with_water.geojson"
            gdf.to_file(str(out_path), driver="GeoJSON")
            logger.success(
                f"Grille {grid_level} enrichie : {out_path}\n"
                f"  WaterScore moyen : {gdf['water_score'].mean():.1f}/100\n"
                f"  WSI moyen        : {gdf['wsi_mean'].mean():.3f}"
            )

        enriched[grid_level] = gdf

    return enriched


def _recompute_final_score_v4(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Recalcule le score final GreenDC v4 avec la composante hydrique.

    Pondérations v4 :
        score_temperature  × 0.25
        score_water        × 0.20  ← water_model (ce module)
        score_carbon       × 0.20  ← carbon_model
        score_environment  × 0.10
        score_technique    × 0.10
        score_fiabilite    × 0.10
        couverture_nappes  × 0.05
    """
    gdf = gdf.copy()

    weights = {
        "score_temperature":   0.25,
        "water_score":         0.20,   # ← remplace score_eau proxy
        "score_carbon":        0.20,   # ← carbon_model
        "score_environnement": 0.10,
        "score_technique":     0.10,
        "score_fiabilite":     0.10,
        "couverture_nappes_pct": 0.05,
    }

    score_v4 = pd.Series(0.0, index=gdf.index)
    total_w = 0.0
    for col, w in weights.items():
        if col in gdf.columns:
            score_v4 += gdf[col].fillna(0) * w
            total_w += w

    # Normaliser si certaines colonnes manquent
    if total_w > 0 and total_w < 1.0:
        score_v4 = score_v4 / total_w * 1.0

    gdf["score_final_v4"] = score_v4
    gdf["score_final_v4_rank"] = score_v4.rank(ascending=False)

    # Delta vs score original
    if "score_final" in gdf.columns:
        delta = score_v4 - gdf["score_final"]
        logger.info(
            f"Score V4 (avec water) :\n"
            f"  Moyen    : {score_v4.mean():.1f}/100\n"
            f"  Delta V3 : {delta.mean():+.1f} pts (moy)\n"
            f"  Top 10 gagnants : {delta.nlargest(10).index.tolist()}"
        )
    return gdf