"""
Projection du WSI sur les grilles GreenDC (5 km / 15 km / 30 km).

Ce module produit les 3 fichiers NetCDF finaux du modèle hydrique :
    water_stress_5km_{year}.nc
    water_stress_15km_{year}.nc
    water_stress_30km_{year}.nc

Format de sortie (xr.Dataset) :
    Dimensions : lat, lon  (coordonnées WGS84 EPSG:4326)
    Variables  :
        water_score          [0-100]
        wsi_mean             [0-1]
        wsi_p10, wsi_p90     [0-1]
        bws_norm             [0-1]
        gws_norm             [0-1]
        drought_norm_mean    [0-1]
        wsi_uncertainty      [0-1]
        water_score_uncertainty [score points]

Choix de projection :
    Les grilles sont générées en EPSG:4326 (WGS84) pour compatibilité
    avec les autres outputs GreenDC (cartes Leaflet, scoring API).
    La grille 5km est la grille de référence.
    Les grilles 15km et 30km sont obtenues par coarsening.

Coordonnées lat/lon :
    Bien que le modèle tourne en EPSG:2154 (Lambert-93), les fichiers
    de sortie sont en EPSG:4326 pour :
    - Cohérence avec le modèle carbone (carbon_model/)
    - Compatibilité directe avec l'API GreenDC (get_water_stress(lat, lon))
    - Lecture directe dans QGIS, Folium, etc.

Usage :
    proj = GridProjector(year=2022)
    grids = proj.project_all(wsi_dataset_l93)
    # Retourne {'5km': Dataset, '15km': Dataset, '30km': Dataset}
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from loguru import logger
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator


# ──────────────────────────────────────────────────────────────────────────────
# Définition des grilles de sortie (EPSG:4326)
# ──────────────────────────────────────────────────────────────────────────────

# Résolutions en degrés approximatives (1° ≈ 111 km)
_RESOLUTIONS_DEG = {
    "5km":  5.0 / 111.0,    # ≈ 0.04505°
    "15km": 15.0 / 111.0,   # ≈ 0.13514°
    "30km": 30.0 / 111.0,   # ≈ 0.27027°
}

# Bbox France EPSG:4326
_FRANCE_BBOX = (-5.2, 41.2, 9.8, 51.2)  # (minlon, minlat, maxlon, maxlat)


def make_france_grid_wgs84(
    resolution_key: str = "5km",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Génère la grille régulière France en WGS84.

    Returns:
        (lons, lats) — tableaux 1D des coordonnées WGS84
    """
    step = _RESOLUTIONS_DEG[resolution_key]
    minlon, minlat, maxlon, maxlat = _FRANCE_BBOX

    lons = np.arange(minlon + step / 2, maxlon, step)
    lats = np.arange(minlat + step / 2, maxlat, step)

    return lons, lats


class GridProjector:
    """
    Projette le Dataset WSI (EPSG:2154) vers les grilles GreenDC (WGS84).

    Args:
        year: Année de simulation
        output_dir: Répertoire de sortie
        france_mask: Masque France pour clip final (optionnel)
    """

    def __init__(
        self,
        year: int,
        output_dir: Optional[Path] = None,
        france_mask: bool = True,
    ):
        self.year = year
        self.output_dir = output_dir
        self.france_mask = france_mask

        # Transformateur L93 → WGS84
        self.transformer_l93_to_wgs84 = Transformer.from_crs(
            "EPSG:2154", "EPSG:4326", always_xy=True
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Projection principale
    # ──────────────────────────────────────────────────────────────────────────

    def project_all(
        self,
        wsi_ds: xr.Dataset,
    ) -> dict[str, xr.Dataset]:
        """
        Projette le Dataset WSI sur les 3 grilles GreenDC.

        Args:
            wsi_ds: Dataset WSI en EPSG:2154 (dimensions y, x en mètres L93)

        Returns:
            dict {'5km': Dataset, '15km': Dataset, '30km': Dataset}
            Chaque Dataset en EPSG:4326 (dimensions lat, lon)
        """
        logger.info("Projection WSI sur les grilles GreenDC...")

        # Extraire les coordonnées L93 du dataset source
        x_l93 = wsi_ds.coords["x"].values
        y_l93 = wsi_ds.coords["y"].values

        # Convertir en WGS84 pour construire les interpolateurs
        # (meshgrid 2D → transformer → grid régulière approx.)
        xx, yy = np.meshgrid(x_l93, y_l93)
        lons_src, lats_src = self.transformer_l93_to_wgs84.transform(xx, yy)

        grids = {}
        for res_key in ["5km", "15km", "30km"]:
            logger.info(f"Projection → grille {res_key}...")
            ds_wgs84 = self._project_to_grid(
                wsi_ds,
                lons_src, lats_src,
                x_l93, y_l93,
                resolution_key=res_key,
            )

            if self.france_mask:
                ds_wgs84 = self._apply_france_mask(ds_wgs84)

            if self.output_dir is not None:
                self._save(ds_wgs84, res_key)

            grids[res_key] = ds_wgs84

        return grids

    def _project_to_grid(
        self,
        wsi_ds: xr.Dataset,
        lons_src: np.ndarray,
        lats_src: np.ndarray,
        x_l93: np.ndarray,
        y_l93: np.ndarray,
        resolution_key: str,
    ) -> xr.Dataset:
        """
        Interpole chaque variable WSI sur la grille WGS84 cible.

        Méthode :
            1. Construire un interpolateur bilinéaire sur la grille L93 source
            2. Évaluer sur les points WGS84 cibles (après reprojection inverse)
        """
        lons_tgt, lats_tgt = make_france_grid_wgs84(resolution_key)

        # Transformer les coordonnées cibles WGS84 → L93 pour interpolation
        transformer_wgs84_to_l93 = Transformer.from_crs(
            "EPSG:4326", "EPSG:2154", always_xy=True
        )
        lon_grid, lat_grid = np.meshgrid(lons_tgt, lats_tgt)
        x_tgt, y_tgt = transformer_wgs84_to_l93.transform(lon_grid, lat_grid)

        # Préparer y_l93 trié (croissant, requis par RegularGridInterpolator)
        if y_l93[0] > y_l93[-1]:
            y_l93_sorted = y_l93[::-1]
            flip_y = True
        else:
            y_l93_sorted = y_l93.copy()
            flip_y = False

        pts = np.column_stack([y_tgt.ravel(), x_tgt.ravel()])

        data_vars = {}
        for var_name in wsi_ds.data_vars:
            da = wsi_ds[var_name]
            has_time = "time" in da.dims

            if has_time:
                # Normaliser l'ordre des dims → (time, y, x)
                spatial_dims = [d for d in da.dims if d != "time"]
                da = da.transpose("time", *spatial_dims)
                time_coords = da.coords["time"].values
                slices = []
                for t_idx in range(len(time_coords)):
                    arr = da.values[t_idx].astype(np.float64)
                    if flip_y:
                        arr = arr[::-1, :]
                    result_t = self._interp_2d(
                        arr, y_l93_sorted, x_l93, pts,
                        len(lats_tgt), len(lons_tgt),
                        y_tgt, x_tgt,
                    )
                    slices.append(result_t)

                result_3d = np.stack(slices, axis=0).astype(np.float32)
                data_vars[var_name] = xr.DataArray(
                    result_3d,
                    dims=["time", "lat", "lon"],
                    coords={
                        "time": time_coords,
                        "lat": lats_tgt,
                        "lon": lons_tgt,
                    },
                    name=var_name,
                    attrs=da.attrs,
                )
            else:
                # Variable 2D (y, x)
                arr = da.values.astype(np.float64)
                if flip_y:
                    arr = arr[::-1, :]
                result = self._interp_2d(
                    arr, y_l93_sorted, x_l93, pts,
                    len(lats_tgt), len(lons_tgt),
                    y_tgt, x_tgt,
                )
                data_vars[var_name] = xr.DataArray(
                    result.astype(np.float32),
                    dims=["lat", "lon"],
                    coords={"lat": lats_tgt, "lon": lons_tgt},
                    name=var_name,
                    attrs=da.attrs,
                )

        ds_out = xr.Dataset(
            data_vars,
            coords={"lat": lats_tgt, "lon": lons_tgt},
            attrs={
                **wsi_ds.attrs,
                "year": self.year,
                "resolution": resolution_key,
                "crs": "EPSG:4326 (WGS84)",
                "projection_source": "EPSG:2154 (Lambert-93)",
                "grid_lats": f"{lats_tgt[0]:.2f}° à {lats_tgt[-1]:.2f}°",
                "grid_lons": f"{lons_tgt[0]:.2f}° à {lons_tgt[-1]:.2f}°",
                "n_points": int(len(lats_tgt) * len(lons_tgt)),
            },
        )

        logger.info(
            f"Grille {resolution_key} : "
            f"{len(lats_tgt)}×{len(lons_tgt)} = {len(lats_tgt)*len(lons_tgt)} points"
        )
        return ds_out

    @staticmethod
    def _interp_2d(
        arr: np.ndarray,
        y_sorted: np.ndarray,
        x_sorted: np.ndarray,
        pts: np.ndarray,
        n_lat: int,
        n_lon: int,
        y_tgt: np.ndarray,
        x_tgt: np.ndarray,
    ) -> np.ndarray:
        """Interpole un array 2D (y, x) sur les points cibles + NN fallback."""
        interp = RegularGridInterpolator(
            (y_sorted, x_sorted),
            arr,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        result = interp(pts).reshape(n_lat, n_lon)

        nan_mask = np.isnan(result)
        if nan_mask.any() and (~nan_mask).any():
            valid = ~nan_mask
            nn = NearestNDInterpolator(
                np.column_stack([y_tgt[valid], x_tgt[valid]]),
                result[valid],
            )
            result[nan_mask] = nn(
                np.column_stack([y_tgt[nan_mask], x_tgt[nan_mask]])
            )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Masque France
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_france_mask(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Applique un masque pour exclure les points hors France métropolitaine.

        Utilise un polygone simplifié de la France (~50 points, précision ~10 km)
        défini dans preprocessing.resampling. Suffisant pour la résolution 5 km.
        """
        from water_model.preprocessing.resampling import apply_france_mask
        return apply_france_mask(ds, crs="WGS84", buffer_km=10.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Sauvegarde
    # ──────────────────────────────────────────────────────────────────────────

    def _save(self, ds: xr.Dataset, resolution_key: str) -> Path:
        """Sauvegarde le Dataset en NetCDF avec compression Zstd."""
        if self.output_dir is None:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"water_stress_{resolution_key}_{self.year}.nc"
        out_path = self.output_dir / fname

        # Encodage NetCDF avec compression
        encoding = {}
        for var in ds.data_vars:
            encoding[var] = {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "_FillValue": -9999.0,
            }

        ds.to_netcdf(str(out_path), encoding=encoding, mode="w")

        logger.success(
            f"Sauvegardé : {out_path} "
            f"({out_path.stat().st_size / 1e6:.1f} MB)"
        )
        return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Fonction utilitaire pipeline
# ──────────────────────────────────────────────────────────────────────────────

def project_wsi_to_grids(
    wsi_ds: xr.Dataset,
    year: int,
    output_dir: Path,
) -> dict[str, xr.Dataset]:
    """
    Point d'entrée principal — projette et sauvegarde les 3 grilles.

    Args:
        wsi_ds: Dataset WSI en EPSG:2154
        year: Année de simulation
        output_dir: Répertoire de sortie

    Returns:
        dict {'5km': Dataset, '15km': Dataset, '30km': Dataset}
    """
    projector = GridProjector(year=year, output_dir=output_dir, france_mask=True)
    return projector.project_all(wsi_ds)
