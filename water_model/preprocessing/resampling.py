"""
Harmonisation spatiale — tous les rasters sur une grille commune 5 km.

Problème : les sources de données ont des résolutions différentes :
    - Aqueduct 4.0  : ~10 km (polygones rasterisés, 5 arc-min)
    - BRGM BDLISA   : variable (polygones rasterisés à 10 km)
    - ERA5          : ~28 km (0.25° × 0.25°)

Solution : régression vers une grille cible commune en EPSG:2154 :
    - Résolution de référence : 5 km (haute résolution GreenDC)
    - Les grilles 15 km et 30 km sont dérivées par moyennage

Méthode :
    1. Raster fin (5-10 km) → bilinear interpolation sur la grille 5km
    2. Raster grossier (ERA5 28 km) → bilinear sur grille 5km
       (puis lissage par bloc pour éviter les artefacts)
    3. Grille 15km et 30km → coarsening xarray

Usage :
    resampler = RasterResampler(resolution_m=5000)
    da_5km = resampler.resample(da_source, "precip_mm")
    da_15km = resampler.coarsen_to(da_5km, resolution_m=15000)
    da_30km = resampler.coarsen_to(da_5km, resolution_m=30000)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from loguru import logger
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator


# ──────────────────────────────────────────────────────────────────────────────
# Grille de référence France EPSG:2154 (Lambert-93)
# ──────────────────────────────────────────────────────────────────────────────
_FRANCE_GRID_PARAMS = {
    5_000:  {"left": -700_000, "bottom": 6_000_000, "right": 1_300_000, "top": 7_200_000},
    15_000: {"left": -700_000, "bottom": 6_000_000, "right": 1_300_000, "top": 7_200_000},
    30_000: {"left": -700_000, "bottom": 6_000_000, "right": 1_300_000, "top": 7_200_000},
}


def build_reference_grid(resolution_m: float = 5000.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Construit la grille de référence France en EPSG:2154.

    Args:
        resolution_m: Résolution en mètres

    Returns:
        (xs, ys) — tableaux 1D des coordonnées Lambert-93
    """
    p = _FRANCE_GRID_PARAMS.get(int(resolution_m), _FRANCE_GRID_PARAMS[5_000])
    xs = np.arange(p["left"] + resolution_m / 2, p["right"], resolution_m)
    ys = np.arange(p["top"] - resolution_m / 2, p["bottom"], -resolution_m)  # Nord→Sud
    return xs, ys


class RasterResampler:
    """
    Ré-échantillonne n'importe quel raster sur la grille France 5km (L93).

    Args:
        resolution_m: Résolution cible de la grille de base (défaut 5000 m)
        method: Méthode d'interpolation ('linear', 'nearest', 'cubic')
    """

    def __init__(
        self,
        resolution_m: float = 5000.0,
        method: str = "linear",
    ):
        self.resolution_m = resolution_m
        self.method = method
        self._xs_ref, self._ys_ref = build_reference_grid(resolution_m)

    # ──────────────────────────────────────────────────────────────────────────
    # Rééchantillonnage depuis GeoTIFF L93
    # ──────────────────────────────────────────────────────────────────────────

    def resample_tif(
        self,
        tif_path: Path,
        var_name: str = "value",
    ) -> xr.DataArray:
        """
        Lit un GeoTIFF L93 et le ré-échantillonne sur la grille de référence.

        Le GeoTIFF source doit déjà être en EPSG:2154 (après reprojection.py).

        Args:
            tif_path: GeoTIFF source en EPSG:2154
            var_name: Nom de la variable dans le DataArray résultant

        Returns:
            xr.DataArray (y, x) sur la grille de référence France 5km
        """
        import rasterio

        with rasterio.open(str(tif_path)) as src:
            arr = src.read(1).astype(np.float64)
            transform = src.transform
            height, width = arr.shape

            # Coordonnées du centre des pixels source
            xs_src = transform.c + np.arange(width) * transform.a + transform.a / 2
            ys_src = transform.f + np.arange(height) * transform.e + transform.e / 2

            # Ordonnées croissantes requises pour RegularGridInterpolator
            if ys_src[0] > ys_src[-1]:
                ys_src = ys_src[::-1]
                arr = arr[::-1, :]

            nodata = src.nodata
            if nodata is not None:
                arr = np.where(np.isclose(arr, nodata, atol=1.0), np.nan, arr)

        return self._interpolate_to_grid(arr, xs_src, ys_src, var_name)

    # ──────────────────────────────────────────────────────────────────────────
    # Rééchantillonnage depuis xarray (ERA5)
    # ──────────────────────────────────────────────────────────────────────────

    def resample_xarray(
        self,
        da: xr.DataArray,
        x_coord: str = "x",
        y_coord: str = "y",
        var_name: Optional[str] = None,
    ) -> xr.DataArray:
        """
        Ré-échantillonne un DataArray L93 sur la grille de référence.

        Pour ERA5 (coordonnées en degrés), utiliser d'abord
        reprojection.RasterReprojector.reproject_to_xarray() avant ce step.

        Args:
            da: DataArray 2D (y, x) ou (lat, lon)
            x_coord, y_coord: Noms des dimensions
            var_name: Nom de la variable résultante

        Returns:
            DataArray (y, x) sur la grille de référence France 5km
        """
        if var_name is None:
            var_name = da.name or "value"

        arr = da.values.astype(np.float64)

        # Auto-détection des noms de coordonnées (ERA5 = lat/lon, L93 = x/y)
        if x_coord not in da.coords:
            for candidate in ("lon", "longitude", "x"):
                if candidate in da.coords:
                    x_coord = candidate
                    break
        if y_coord not in da.coords:
            for candidate in ("lat", "latitude", "y"):
                if candidate in da.coords:
                    y_coord = candidate
                    break

        # Extraire coordonnées 1D
        xs_src = da[x_coord].values
        ys_src = da[y_coord].values

        # Assurer ordre croissant Y
        if ys_src[0] > ys_src[-1]:
            ys_src = ys_src[::-1]
            arr = arr[::-1, :]

        return self._interpolate_to_grid(arr, xs_src, ys_src, var_name)

    # ──────────────────────────────────────────────────────────────────────────
    # Coarsening vers 15km et 30km
    # ──────────────────────────────────────────────────────────────────────────

    def coarsen_to(
        self,
        da_5km: xr.DataArray,
        resolution_m: float,
        aggregation: str = "mean",
    ) -> xr.DataArray:
        """
        Dégrade la grille 5km vers 15km ou 30km par agrégation spatiale.

        Utilise xarray.coarsen() avec padding si nécessaire.

        Args:
            da_5km: DataArray sur la grille 5km
            resolution_m: Résolution cible (15000 ou 30000)
            aggregation: 'mean', 'median', 'max'

        Returns:
            DataArray ré-agrégé
        """
        factor = int(resolution_m / self.resolution_m)
        if factor <= 1:
            return da_5km

        logger.info(
            f"Coarsening {da_5km.name or ''} : "
            f"{int(self.resolution_m/1000)} km → {int(resolution_m/1000)} km "
            f"(facteur ×{factor})"
        )

        # Padding pour que les dimensions soient divisibles
        ny, nx = da_5km.shape[-2], da_5km.shape[-1]
        pad_y = (-ny) % factor
        pad_x = (-nx) % factor

        if pad_y > 0 or pad_x > 0:
            da_5km = da_5km.pad(
                {"y": (0, pad_y), "x": (0, pad_x)},
                mode="edge",
                constant_values=np.nan,
            )

        coarsen_kwargs = {"y": factor, "x": factor, "boundary": "trim"}

        if aggregation == "mean":
            da_coarse = da_5km.coarsen(**coarsen_kwargs).mean()
        elif aggregation == "median":
            da_coarse = da_5km.coarsen(**coarsen_kwargs).median()
        elif aggregation == "max":
            da_coarse = da_5km.coarsen(**coarsen_kwargs).max()
        else:
            raise ValueError(f"Agrégation inconnue : {aggregation}")

        da_coarse.attrs.update({
            **da_5km.attrs,
            "resolution_m": resolution_m,
        })

        return da_coarse

    # ──────────────────────────────────────────────────────────────────────────
    # Cœur de l'interpolation
    # ──────────────────────────────────────────────────────────────────────────

    def _interpolate_to_grid(
        self,
        arr: np.ndarray,
        xs_src: np.ndarray,
        ys_src: np.ndarray,
        var_name: str,
    ) -> xr.DataArray:
        """
        Interpole arr(ys_src, xs_src) sur la grille de référence (self._ys_ref, self._xs_ref).

        Gestion des NaN :
            1. Interpolation bilinéaire sur les pixels valides
            2. Les NaN résiduels (bords, zones hors bbox) sont remplis
               par le plus proche voisin valide

        Returns:
            DataArray (y, x)
        """
        # Masquer les NaN pour l'interpolateur
        nan_mask = np.isnan(arr)
        arr_filled = np.where(nan_mask, 0.0, arr)

        # Points cibles (meshgrid de la grille de référence)
        yy, xx = np.meshgrid(self._ys_ref, self._xs_ref, indexing="ij")
        pts = np.column_stack([yy.ravel(), xx.ravel()])

        # Interpolateur bilinéaire (axes ordonnés croissants)
        interp = RegularGridInterpolator(
            (ys_src, xs_src),
            arr_filled,
            method=self.method,
            bounds_error=False,
            fill_value=np.nan,
        )
        result = interp(pts).reshape(len(self._ys_ref), len(self._xs_ref))

        # Interpoler le masque NaN pour détecter les pixels pollués par les 0
        # Un pixel cible est invalidé si le poids des NaN sources > 0
        if nan_mask.any():
            nan_weight = nan_mask.astype(np.float64)
            mask_interp = RegularGridInterpolator(
                (ys_src, xs_src),
                nan_weight,
                method=self.method,
                bounds_error=False,
                fill_value=1.0,
            )
            nan_influence = mask_interp(pts).reshape(result.shape)
            result[nan_influence > 0.01] = np.nan

            # Remplir les NaN résiduels par NearestND (plus proche voisin valide)
            ys_flat = np.repeat(ys_src, len(xs_src))
            xs_flat = np.tile(xs_src, len(ys_src))
            valid = ~nan_mask.ravel()
            if valid.any():
                nn_interp = NearestNDInterpolator(
                    np.column_stack([ys_flat[valid], xs_flat[valid]]),
                    arr.ravel()[valid],
                )
                nan_result = np.isnan(result)
                if nan_result.any():
                    result[nan_result] = nn_interp(
                        np.column_stack([yy[nan_result], xx[nan_result]])
                    )

        da = xr.DataArray(
            result.astype(np.float32),
            dims=["y", "x"],
            coords={"y": self._ys_ref, "x": self._xs_ref},
            name=var_name,
            attrs={"resolution_m": self.resolution_m, "crs": "EPSG:2154"},
        )
        logger.info(
            f"Rééchantillonné {var_name} : "
            f"{len(self._ys_ref)}×{len(self._xs_ref)} px "
            f"@ {self.resolution_m/1000:.0f} km"
        )
        return da


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline complet
# ──────────────────────────────────────────────────────────────────────────────

def harmonize_all_rasters(
    reprojected_tifs: dict[str, Path],
    era5_dataset: Optional[xr.Dataset],
    output_dir: Path,
    resolution_m: float = 5000.0,
) -> dict[str, xr.DataArray]:
    """
    Harmonise tous les rasters sur la grille France 5km.

    Args:
        reprojected_tifs: {'bws_raw': Path L93, 'gws_raw': Path L93, ...}
        era5_dataset: xr.Dataset ERA5 reprojeté (time, y, x)
        output_dir: Répertoire de sortie pour les DataArrays

    Returns:
        dict de DataArrays harmonisés en EPSG:2154
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    resampler = RasterResampler(resolution_m=resolution_m)
    harmonized = {}

    # Rasters Aqueduct / BRGM (GeoTIFF L93)
    for var_name, tif_path in reprojected_tifs.items():
        if not tif_path.exists():
            logger.warning(f"GeoTIFF manquant : {tif_path}")
            continue
        da = resampler.resample_tif(tif_path, var_name=var_name)
        harmonized[var_name] = da

    # ERA5 (xarray — déjà en L93 après reprojection)
    if era5_dataset is not None:
        for var in era5_dataset.data_vars:
            da_era5 = era5_dataset[var]
            # ERA5 peut avoir une dimension time → traiter par pas de temps
            if "time" in da_era5.dims:
                slices = []
                for t in da_era5.time:
                    da_t = da_era5.sel(time=t)
                    da_r = resampler.resample_xarray(da_t, var_name=var)
                    slices.append(da_r)
                da_stacked = xr.concat(slices, dim=da_era5.time)
                da_stacked.name = var
                harmonized[var] = da_stacked
            else:
                da_r = resampler.resample_xarray(da_era5, var_name=var)
                harmonized[var] = da_r

    logger.success(f"Harmonisation terminée : {list(harmonized.keys())}")
    return harmonized


def reproject_wgs84_to_l93(
    da: xr.DataArray,
    var_name: Optional[str] = None,
    resolution_m: float = 5000.0,
) -> xr.DataArray:
    """
    Reprojette un DataArray 2D (lat, lon) WGS84 sur la grille L93 de référence.

    Algorithme :
        1. Transformer les points de la grille L93 cible en WGS84 (pyproj)
        2. Interpoler depuis la grille ERA5 régulière lat/lon avec RegularGridInterpolator
        3. Remplir les NaN de bordure avec NearestND

    Args:
        da: DataArray 2D avec dims (lat, lon) ou (latitude, longitude) en WGS84
        var_name: Nom de la variable de sortie
        resolution_m: Résolution de la grille cible (défaut 5000 m)

    Returns:
        DataArray (y, x) en EPSG:2154
    """
    from pyproj import Transformer

    if var_name is None:
        var_name = da.name or "value"

    # Détecter les noms de coords lat/lon
    lat_coord = next((c for c in ("lat", "latitude") if c in da.coords), None)
    lon_coord = next((c for c in ("lon", "longitude") if c in da.coords), None)
    if lat_coord is None or lon_coord is None:
        raise ValueError(f"Coords lat/lon introuvables dans {list(da.coords)}")

    lats = da[lat_coord].values.copy()
    lons = da[lon_coord].values.copy()
    arr = da.values.astype(np.float64)

    # Assurer ordre croissant des latitudes
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        arr = arr[::-1, :]

    # Assurer ordre croissant des longitudes
    if len(lons) > 1 and lons[0] > lons[-1]:
        lons = lons[::-1]
        arr = arr[:, ::-1]

    # Interpolateur ERA5 sur grille WGS84 régulière
    interp = RegularGridInterpolator(
        (lats, lons), arr,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Grille cible L93
    xs_l93, ys_l93 = build_reference_grid(resolution_m)
    yy, xx = np.meshgrid(ys_l93, xs_l93, indexing="ij")

    # Transformer les coordonnées L93 → WGS84
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    lon_tgt, lat_tgt = transformer.transform(xx.ravel(), yy.ravel())

    # Interpoler
    result = interp(np.column_stack([lat_tgt, lon_tgt]))
    result = result.reshape(len(ys_l93), len(xs_l93)).astype(np.float32)

    # Remplir les NaN de bordure par NearestND
    nan_mask = np.isnan(result)
    if nan_mask.any():
        valid_arr = arr.ravel()
        valid_pts = ~np.isnan(valid_arr)
        if valid_pts.any():
            lons_g, lats_g = np.meshgrid(lons, lats)
            # Convertir les points source ERA5 en L93 pour NearestND
            lon_src_flat = lons_g.ravel()[valid_pts]
            lat_src_flat = lats_g.ravel()[valid_pts]
            x_src, y_src = Transformer.from_crs(
                "EPSG:4326", "EPSG:2154", always_xy=True
            ).transform(lon_src_flat, lat_src_flat)
            nn = NearestNDInterpolator(
                np.column_stack([y_src, x_src]),
                valid_arr[valid_pts],
            )
            result[nan_mask] = nn(
                np.column_stack([yy.ravel()[nan_mask.ravel()], xx.ravel()[nan_mask.ravel()]])
            ).astype(np.float32)

    logger.info(
        f"Reprojeté WGS84→L93 {var_name} : "
        f"{len(ys_l93)}×{len(xs_l93)} px @ {resolution_m/1000:.0f} km"
    )
    return xr.DataArray(
        result,
        dims=["y", "x"],
        coords={"y": ys_l93, "x": xs_l93},
        name=var_name,
        attrs={"resolution_m": resolution_m, "crs": "EPSG:2154"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Masque France métropolitaine
# ──────────────────────────────────────────────────────────────────────────────

# Polygone simplifié de la France métropolitaine (WGS84).
# ~50 points, précision ~10 km — largement suffisant pour grille 5 km.
_FRANCE_METRO_COORDS_WGS84 = [
    (-1.86, 48.63),   # Saint-Malo
    (-1.27, 49.65),   # Cherbourg
    (0.08, 49.48),    # Le Havre
    (1.59, 50.96),    # Calais
    (2.54, 51.09),    # Dunkerque
    (3.20, 50.36),    # Valenciennes
    (4.22, 49.96),    # Charleville-Mézières
    (5.89, 49.50),    # Longwy
    (7.36, 49.15),    # Wissembourg (nord Alsace)
    (8.10, 48.98),    # Est Strasbourg (frontière Rhin)
    (7.80, 48.50),    # Sélestat
    (7.59, 47.59),    # Bâle
    (7.02, 47.32),    # Belfort
    (6.86, 46.45),    # Genève
    (6.80, 45.83),    # Chamonix
    (7.07, 44.69),    # Col de Tende
    (7.53, 43.79),    # Menton
    (6.87, 43.43),    # Fréjus
    (5.93, 43.14),    # Toulon
    (5.37, 43.30),    # Marseille
    (4.84, 43.39),    # Étang de Berre
    (4.63, 43.40),    # Arles
    (3.88, 43.56),    # Montpellier
    (3.03, 42.53),    # Perpignan
    (1.72, 42.43),    # Andorre
    (0.73, 42.80),    # Saint-Girons
    (-0.55, 42.80),   # Col du Somport
    (-1.79, 43.36),   # Bayonne
    (-1.63, 43.95),    # Dax
    (-1.25, 44.63),   # Arcachon
    (-1.18, 45.60),   # Royan
    (-1.17, 46.18),   # La Rochelle
    (-2.21, 46.70),   # Noirmoutier
    (-2.34, 47.27),   # Belle-Île
    (-3.43, 47.64),   # Lorient
    (-4.50, 48.02),   # Pointe du Raz
    (-4.77, 48.39),   # Brest
    (-3.99, 48.73),   # Roscoff
    (-3.12, 48.83),   # Lannion
    (-1.86, 48.63),   # fermeture
]


def create_france_mask_wgs84(
    lats: np.ndarray,
    lons: np.ndarray,
    buffer_km: float = 10.0,
) -> np.ndarray:
    """
    Crée un masque booléen France métropolitaine sur une grille WGS84.

    Le polygone est bufferisé de buffer_km pour éviter de couper
    des pixels de bordure qui pourraient contenir des données valides.

    Args:
        lats: Coordonnées latitude 1D (WGS84)
        lons: Coordonnées longitude 1D (WGS84)
        buffer_km: Marge autour du polygone en km (défaut 10 km)

    Returns:
        np.ndarray booléen 2D (len(lats), len(lons)), True = France
    """
    from shapely.geometry import Polygon, Point
    from shapely.prepared import prep

    poly = Polygon(_FRANCE_METRO_COORDS_WGS84)
    # Buffer en degrés (~1° ≈ 111 km à cette latitude)
    buffer_deg = buffer_km / 111.0
    poly_buffered = poly.buffer(buffer_deg)
    prepared = prep(poly_buffered)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    mask = np.zeros(lon_grid.shape, dtype=bool)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i, j] = prepared.contains(Point(lon_grid[i, j], lat_grid[i, j]))

    n_in = int(mask.sum())
    n_total = mask.size
    logger.info(
        f"Masque France : {n_in}/{n_total} pixels dans le domaine "
        f"({n_in / n_total * 100:.1f}%)"
    )
    return mask


def create_france_mask_l93(
    xs: np.ndarray,
    ys: np.ndarray,
    buffer_km: float = 10.0,
) -> np.ndarray:
    """
    Crée un masque booléen France métropolitaine sur une grille Lambert-93.

    Projette le polygone WGS84 en L93 puis fait un test point-in-polygon.

    Args:
        xs: Coordonnées x L93 1D (mètres)
        ys: Coordonnées y L93 1D (mètres)
        buffer_km: Marge en km

    Returns:
        np.ndarray booléen 2D (len(ys), len(xs)), True = France
    """
    from pyproj import Transformer
    from shapely.geometry import Polygon, Point
    from shapely.prepared import prep

    # Projeter le polygone WGS84 → L93
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:2154", always_xy=True
    )
    coords_l93 = [
        transformer.transform(lon, lat)
        for lon, lat in _FRANCE_METRO_COORDS_WGS84
    ]
    poly = Polygon(coords_l93)
    poly_buffered = poly.buffer(buffer_km * 1000.0)  # mètres
    prepared = prep(poly_buffered)

    xx, yy = np.meshgrid(xs, ys)
    mask = np.zeros(xx.shape, dtype=bool)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i, j] = prepared.contains(Point(xx[i, j], yy[i, j]))

    n_in = int(mask.sum())
    logger.info(
        f"Masque France L93 : {n_in}/{mask.size} pixels "
        f"({n_in / mask.size * 100:.1f}%)"
    )
    return mask


def apply_france_mask(
    ds: xr.Dataset,
    crs: str = "WGS84",
    buffer_km: float = 10.0,
) -> xr.Dataset:
    """
    Applique le masque France métropolitaine à un Dataset.

    Met à NaN tous les pixels hors France (mer, étranger).

    Args:
        ds: Dataset avec coordonnées (lat, lon) ou (y, x)
        crs: "WGS84" ou "L93"
        buffer_km: Marge autour du polygone

    Returns:
        Dataset masqué (NaN hors France)
    """
    if crs == "L93" and "x" in ds.coords and "y" in ds.coords:
        mask = create_france_mask_l93(
            ds.x.values, ds.y.values, buffer_km=buffer_km
        )
    elif "lat" in ds.coords and "lon" in ds.coords:
        mask = create_france_mask_wgs84(
            ds.lat.values, ds.lon.values, buffer_km=buffer_km
        )
    else:
        logger.warning(
            f"Coordonnées non reconnues pour le masque France : "
            f"{list(ds.coords)}. Masque non appliqué."
        )
        return ds

    mask_da = xr.DataArray(mask, dims=ds[list(ds.data_vars)[0]].dims[-2:])
    ds_masked = ds.where(mask_da)

    n_masked = int((~mask).sum())
    logger.info(f"Masque France appliqué : {n_masked} pixels mis à NaN")
    return ds_masked
