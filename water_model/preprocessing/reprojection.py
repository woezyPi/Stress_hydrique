"""
Reprojection géographique — tous les rasters vers EPSG:2154 (Lambert-93).

Justification du choix EPSG:2154 :
    - Projection officielle pour la France métropolitaine (IGN)
    - Préserve les distances et surfaces localement (conforme Lambert)
    - Cohérente avec le modèle carbone (carbon_model/config.yaml: crs_metric)
    - Unités en mètres → calculs de résolution précis

Pipeline de reprojection :
    EPSG:4326 (GeoTIFF source) → EPSG:2154 (5 km grille cible)

Méthode :
    rasterio.warp.reproject avec resampling bilinéaire (données continues)

Usage :
    proj = RasterReprojector(target_crs="EPSG:2154", resolution_m=5000)
    out_path = proj.reproject(in_tif, out_tif)
    xda = proj.reproject_to_xarray(in_tif)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr
from loguru import logger

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
except ImportError:
    raise ImportError("rasterio requis : pip install rasterio")


# ──────────────────────────────────────────────────────────────────────────────
# Bbox France en EPSG:2154 (Lambert-93) — coins SW et NE
# Valeurs obtenues via : pyproj.Transformer.from_crs(4326, 2154)
# ──────────────────────────────────────────────────────────────────────────────
_FRANCE_BBOX_L93 = {
    "left":   -700_000,   # m (Bretagne)
    "bottom": 6_000_000,  # m (Pyrénées)
    "right":  1_300_000,  # m (Alsace)
    "top":    7_200_000,  # m (Nord)
}


class RasterReprojector:
    """
    Reprojette des rasters GeoTIFF vers EPSG:2154 (Lambert-93).

    Args:
        target_crs: CRS cible (défaut EPSG:2154)
        resolution_m: Résolution de la grille cible en mètres (défaut 5000 = 5 km)
        resampling: Méthode de rééchantillonnage rasterio
    """

    def __init__(
        self,
        target_crs: str = "EPSG:2154",
        resolution_m: float = 5000.0,
        resampling: Resampling = Resampling.bilinear,
    ):
        self.target_crs = CRS.from_string(target_crs)
        self.resolution_m = resolution_m
        self.resampling = resampling

    # ──────────────────────────────────────────────────────────────────────────
    # Reprojection vers GeoTIFF
    # ──────────────────────────────────────────────────────────────────────────

    def reproject(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        bbox_l93: Optional[dict] = None,
        nodata_in: float = -9999.0,
        nodata_out: float = np.nan,
    ) -> Path:
        """
        Reprojette un GeoTIFF vers le CRS cible et le clip sur la France.

        Args:
            input_path: GeoTIFF source (n'importe quel CRS)
            output_path: GeoTIFF de sortie (EPSG:2154)
            bbox_l93: Dictionnaire {left, bottom, right, top} en coordonnées L93
                       None → utilise _FRANCE_BBOX_L93
            nodata_in: Valeur nodata source
            nodata_out: Valeur nodata destination (nan recommandé pour float32)

        Returns:
            Path du GeoTIFF reprojeté

        Algorithme :
            1. Lire le raster source avec rasterio
            2. Calculer la transformation cible via calculate_default_transform
            3. Clipper sur la bbox France L93
            4. Appliquer reproject() avec interpolation bilinéaire
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if bbox_l93 is None:
            bbox_l93 = _FRANCE_BBOX_L93

        with rasterio.open(str(input_path)) as src:
            src_crs = src.crs
            src_transform = src.transform
            src_data = src.read()
            src_nodata = src.nodata or nodata_in
            src_dtype = src.dtypes[0]

            # Calculer la transformation optimale vers le CRS cible
            transform, width, height = calculate_default_transform(
                src_crs,
                self.target_crs,
                src.width,
                src.height,
                left=src.bounds.left,
                bottom=src.bounds.bottom,
                right=src.bounds.right,
                top=src.bounds.top,
                resolution=self.resolution_m,
            )

            # Clipper sur la bbox France L93 si plus petite
            transform, width, height = self._clip_transform(
                transform, width, height, bbox_l93
            )

            # Nodata source → NaN avant reprojection
            src_float = src_data.astype(np.float64)
            if src_nodata is not None:
                src_float = np.where(
                    np.isclose(src_float, src_nodata, atol=1.0),
                    np.nan,
                    src_float,
                )

            dst_data = np.full(
                (src.count, height, width),
                np.nan,
                dtype=np.float64,
            )

            reproject(
                source=src_float,
                destination=dst_data,
                src_transform=src_transform,
                src_crs=src_crs,
                src_nodata=np.nan,
                dst_transform=transform,
                dst_crs=self.target_crs,
                dst_nodata=np.nan,
                resampling=self.resampling,
            )

            profile = {
                "driver":    "GTiff",
                "dtype":     "float32",
                "width":     width,
                "height":    height,
                "count":     src.count,
                "crs":       self.target_crs,
                "transform": transform,
                "nodata":    nodata_out if np.isfinite(nodata_out) else None,
                "compress":  "lzw",
                "tiled":     True,
                "blockxsize": 256,
                "blockysize": 256,
            }

            with rasterio.open(str(output_path), "w", **profile) as dst:
                dst.write(dst_data.astype(np.float32))

        logger.success(
            f"Reprojeté : {input_path.name} → {output_path.name} "
            f"({width}×{height} px @ {self.resolution_m/1000:.1f} km)"
        )
        return output_path

    def reproject_to_xarray(
        self,
        input_path: Union[str, Path],
        var_name: str = "value",
    ) -> xr.DataArray:
        """
        Reprojette et charge directement en xarray.DataArray (EPSG:2154).

        Les coordonnées x, y sont en mètres Lambert-93.
        Un index lat/lon WGS84 est ajouté comme coordonnées auxiliaires.

        Returns:
            DataArray (y, x) avec attrs CRS et coordonnées lat/lon auxiliaires
        """
        import tempfile
        from pyproj import Transformer

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            self.reproject(input_path, tmp_path)

            with rasterio.open(str(tmp_path)) as src:
                arr = src.read(1).astype(np.float32)
                transform = src.transform
                height, width = arr.shape

                # Coordonnées X, Y du centre des pixels (L93, en mètres)
                xs = transform.c + np.arange(width) * transform.a + transform.a / 2
                ys = transform.f + np.arange(height) * transform.e + transform.e / 2

            # Convertir X, Y L93 → lat/lon WGS84 pour coordonnées auxiliaires
            transformer = Transformer.from_crs(
                "EPSG:2154", "EPSG:4326", always_xy=True
            )
            xx, yy = np.meshgrid(xs, ys)
            lons, lats = transformer.transform(xx, yy)

            da = xr.DataArray(
                arr,
                dims=["y", "x"],
                coords={"y": ys, "x": xs},
                name=var_name,
                attrs={
                    "crs": "EPSG:2154",
                    "resolution_m": self.resolution_m,
                    "source": str(input_path),
                },
            )
            # Ajouter lat/lon comme coordonnées 2D
            da = da.assign_coords(
                lat=(["y", "x"], lats),
                lon=(["y", "x"], lons),
            )

        finally:
            tmp_path.unlink(missing_ok=True)

        logger.info(
            f"xarray créé : {var_name} shape={arr.shape}, "
            f"résolution={self.resolution_m/1000:.1f} km"
        )
        return da

    # ──────────────────────────────────────────────────────────────────────────
    # Utilitaires
    # ──────────────────────────────────────────────────────────────────────────

    def _clip_transform(
        self,
        transform,
        width: int,
        height: int,
        bbox: dict,
    ) -> tuple:
        """
        Ajuste la transformation pour clipper sur une bbox L93.

        Redimensionne width/height si la grille calculée dépasse la bbox France.
        """
        from rasterio.transform import from_origin

        # Coin supérieur gauche de la grille calculée
        grid_left = transform.c
        grid_top = transform.f

        # Utiliser la grille calculée si elle est plus petite que la bbox France
        new_left = max(grid_left, bbox["left"])
        new_top = min(grid_top, bbox["top"])
        new_right = min(grid_left + width * transform.a, bbox["right"])
        new_bottom = max(grid_top + height * transform.e, bbox["bottom"])

        res = self.resolution_m
        new_width = max(1, int((new_right - new_left) / res))
        new_height = max(1, int((new_top - new_bottom) / res))

        new_transform = from_origin(new_left, new_top, res, res)
        return new_transform, new_width, new_height


# ──────────────────────────────────────────────────────────────────────────────
# Fonction utilitaire pour le pipeline
# ──────────────────────────────────────────────────────────────────────────────

def reproject_all(
    input_tifs: dict[str, Path],
    output_dir: Path,
    resolution_m: float = 5000.0,
) -> dict[str, Path]:
    """
    Reprojette tous les rasters d'un dictionnaire vers EPSG:2154.

    Args:
        input_tifs: {'bws_raw': Path, 'gws_raw': Path, ...}
        output_dir: Répertoire de sortie
        resolution_m: Résolution cible en mètres

    Returns:
        dict des chemins reprojetés
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    proj = RasterReprojector(target_crs="EPSG:2154", resolution_m=resolution_m)
    output_paths = {}

    for var_name, in_path in input_tifs.items():
        if not in_path.exists():
            logger.warning(f"Fichier source manquant : {in_path}")
            continue
        out_path = output_dir / f"{var_name}_l93_{int(resolution_m/1000)}km.tif"
        proj.reproject(in_path, out_path)
        output_paths[var_name] = out_path

    return output_paths
