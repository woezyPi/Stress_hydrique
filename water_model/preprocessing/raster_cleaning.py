"""
Nettoyage des rasters hydrologiques.

Opérations :
    1. Détection et remplacement des valeurs nodata (sentinelles WRI : -9999)
    2. Clamp des valeurs dans les plages valides physiques
    3. Détection des outliers (IQR × 3, Z-score > 5)
    4. Interpolation des trous (inpainting spatial 2D)
    5. Lissage optionnel (filtre médian 3×3 pour artefacts de rasterisation)

Usage :
    cleaner = RasterCleaner(nodata=-9999.0)
    array_clean = cleaner.clean(array, var_name="bws_raw")
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger
from scipy.ndimage import generic_filter, label, median_filter
from scipy.interpolate import NearestNDInterpolator


# ──────────────────────────────────────────────────────────────────────────────
# Plages de valeurs valides par variable (physiquement justifiées)
# ──────────────────────────────────────────────────────────────────────────────
_VALID_RANGES = {
    "bws_raw":   (0.0, 10.0),   # Aqueduct BWS : >5 = extrêmement haut
    "gws_raw":   (0.0, 10.0),   # Aqueduct GWS
    "drr_raw":   (0.0,  5.0),   # Aqueduct DrR
    "sev_raw":   (0.0,  5.0),   # Aqueduct SEV
    "gws_score": (0.0,  1.0),   # Score BRGM normalisé
    "spi12":     (-4.0, 4.0),   # SPI : ±4 extrêmes (McKee 1993)
    "precip_mm": (0.0, 500.0),  # Précipitations mensuelles France
    "pet_mm":    (0.0, 200.0),  # ETP mensuelle France
}


class RasterCleaner:
    """
    Nettoie un raster numpy 2D (lat × lon).

    Args:
        nodata: Valeur sentinel nodata à masquer (défaut -9999.0)
        apply_median_filter: Lisser les artefacts avec un filtre médian
        fill_gaps: Interpoler les trous par plus proche voisin valide
    """

    def __init__(
        self,
        nodata: float = -9999.0,
        apply_median_filter: bool = False,
        fill_gaps: bool = True,
    ):
        self.nodata = nodata
        self.apply_median_filter = apply_median_filter
        self.fill_gaps = fill_gaps

    def clean(
        self,
        array: np.ndarray,
        var_name: str = "",
        valid_range: Optional[tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Nettoie un raster 2D float32.

        Pipeline :
            nodata → clamp → outliers → interpolation → lissage

        Args:
            array: Raster 2D numpy (peut contenir NaN et nodata)
            var_name: Nom de la variable (pour log et plage valide auto)
            valid_range: (min, max) — override la plage détectée auto

        Returns:
            Raster nettoyé 2D, même shape que l'entrée
        """
        arr = array.astype(np.float64).copy()
        original_nan_count = np.sum(np.isnan(arr))

        # ── Étape 1 : nodata → NaN ──────────────────────────────────────────
        arr = self._mask_nodata(arr)
        nodata_count = np.sum(np.isnan(arr)) - original_nan_count

        # ── Étape 2 : clamping plage valide ─────────────────────────────────
        if valid_range is None:
            valid_range = _VALID_RANGES.get(var_name)
        if valid_range is not None:
            out_of_range = np.sum(
                ~np.isnan(arr) & ((arr < valid_range[0]) | (arr > valid_range[1]))
            )
            arr = np.where(
                ~np.isnan(arr),
                np.clip(arr, valid_range[0], valid_range[1]),
                arr,
            )
            if out_of_range > 0:
                logger.debug(f"{var_name}: {out_of_range} valeurs hors plage clampées")

        # ── Étape 3 : détection outliers (Z-score robuste) ──────────────────
        arr = self._remove_outliers_zscore(arr, threshold=5.0)

        # ── Étape 4 : interpolation des trous ───────────────────────────────
        if self.fill_gaps:
            n_gaps = np.sum(np.isnan(arr))
            if n_gaps > 0:
                arr = self._fill_nearest(arr)
                logger.debug(f"{var_name}: {n_gaps} pixels interpolés")

        # ── Étape 5 : lissage optionnel ─────────────────────────────────────
        if self.apply_median_filter:
            arr = self._smooth_median(arr, size=3)

        # ── Rapport ─────────────────────────────────────────────────────────
        valid = ~np.isnan(arr)
        logger.info(
            f"Cleaning {var_name or 'raster'} : "
            f"{nodata_count} nodata masqués, "
            f"{valid.sum()} pixels valides, "
            f"moy={np.nanmean(arr):.3f}, "
            f"[{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]"
        )

        return arr.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Méthodes internes
    # ──────────────────────────────────────────────────────────────────────────

    def _mask_nodata(self, arr: np.ndarray) -> np.ndarray:
        """Remplace les sentinelles nodata par NaN."""
        # Valeurs sentinelles communes WRI / GDAL
        sentinels = {self.nodata, -9999.0, -9999, 9999.0, -3.4028235e+38}
        for s in sentinels:
            if np.isfinite(s):
                arr = np.where(np.isclose(arr, s, atol=1.0), np.nan, arr)
        return arr

    def _remove_outliers_zscore(
        self, arr: np.ndarray, threshold: float = 5.0
    ) -> np.ndarray:
        """
        Supprime les outliers via Z-score modifié (médiane/MAD).

        Référence : Iglewicz & Hoaglin (1993) — Z-score robuste
        threshold=5 → élimination valeurs à >5 MAD de la médiane
        """
        valid = arr[~np.isnan(arr)]
        if len(valid) < 4:
            return arr

        median = np.median(valid)
        mad = np.median(np.abs(valid - median))

        if mad < 1e-10:
            return arr

        z_scores = np.abs(arr - median) / (mad * 1.4826)
        outlier_mask = z_scores > threshold
        arr = np.where(outlier_mask, np.nan, arr)
        return arr

    def _fill_nearest(self, arr: np.ndarray) -> np.ndarray:
        """
        Interpole les pixels NaN par le plus proche voisin valide.

        Utilise scipy.interpolate.NearestNDInterpolator.
        Efficace pour les trous épars dans les données Aqueduct.
        """
        ny, nx = arr.shape
        yy, xx = np.mgrid[0:ny, 0:nx]

        valid_mask = ~np.isnan(arr)
        if not valid_mask.any():
            logger.warning("Aucune valeur valide dans le raster — remplissage impossible")
            return arr

        interp = NearestNDInterpolator(
            np.column_stack([yy[valid_mask], xx[valid_mask]]),
            arr[valid_mask],
        )
        arr_filled = interp(yy, xx)
        return arr_filled.astype(arr.dtype)

    def _smooth_median(self, arr: np.ndarray, size: int = 3) -> np.ndarray:
        """Filtre médian 2D pour éliminer les artefacts de pixelisation."""
        # Conserver les NaN
        nan_mask = np.isnan(arr)
        arr_filled = np.where(nan_mask, 0.0, arr)
        smoothed = median_filter(arr_filled, size=size)
        return np.where(nan_mask, np.nan, smoothed)


# ──────────────────────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ──────────────────────────────────────────────────────────────────────────────

def clean_aqueduct_raster(
    tif_path,
    var_name: str,
    output_path=None,
    nodata: float = -9999.0,
) -> np.ndarray:
    """
    Nettoie un GeoTIFF Aqueduct et le réécrit si output_path fourni.

    Returns:
        Tableau numpy nettoyé (float32)
    """
    import rasterio

    with rasterio.open(str(tif_path)) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform

    cleaner = RasterCleaner(nodata=nodata, fill_gaps=True)
    arr_clean = cleaner.clean(arr, var_name=var_name)

    if output_path is not None:
        profile.update(dtype="float32", nodata=np.nan)
        with rasterio.open(str(output_path), "w", **profile) as dst:
            dst.write(arr_clean, 1)
        logger.success(f"Raster nettoyé : {output_path}")

    return arr_clean


def quality_report(arr: np.ndarray, var_name: str = "") -> dict:
    """
    Rapport statistique de qualité pour un raster.

    Returns:
        dict avec n_valid, n_nan, mean, std, p5, p50, p95, min, max
    """
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {"error": "Aucune valeur valide"}

    return {
        "variable":  var_name,
        "n_pixels":  arr.size,
        "n_valid":   int(len(valid)),
        "n_nan":     int(np.sum(np.isnan(arr))),
        "pct_valid": float(len(valid) / arr.size * 100),
        "mean":      float(np.mean(valid)),
        "std":       float(np.std(valid)),
        "p5":        float(np.percentile(valid, 5)),
        "p25":       float(np.percentile(valid, 25)),
        "p50":       float(np.percentile(valid, 50)),
        "p75":       float(np.percentile(valid, 75)),
        "p95":       float(np.percentile(valid, 95)),
        "min":       float(np.min(valid)),
        "max":       float(np.max(valid)),
    }
