"""
Calcul du Standardized Precipitation Index (SPI) et du Drought Risk.

Référence scientifique :
    McKee, T.B., Doesken, N.J. & Kleist, J. (1993). The relationship of drought
    frequency and duration to time scales. Preprints, 8th Conf. Applied Climatology,
    17–22 January 1993, Anaheim, CA. AMS, Boston, pp. 179–184.

Méthode SPI :
    1. Accumuler les précipitations sur N mois (fenêtre glissante)
    2. Ajuster une distribution gamma (α, β) sur la période de référence (1981-2010)
       → utiliser scipy.stats.gamma.fit() avec floc=0 (contrainte physique : P ≥ 0)
    3. Calculer la probabilité cumulée : p = Γcdf(P_acc | α, β)
    4. Convertir en unités normales : SPI = Φ⁻¹(p) où Φ est la CDF normale standard

Interprétation SPI (OMM, 2012) :
    SPI ≥  2.0   : Extrêmement humide
    SPI  [1.5, 2) : Très humide
    SPI  [1.0, 1.5): Modérément humide
    SPI  [-1, 1]  : Normal
    SPI  (-1.5,-1]: Modérément sec
    SPI  (-2,-1.5]: Très sec
    SPI ≤ -2.0   : Extrêmement sec

Drought Risk normalisé :
    DR_norm = clip((−SPI + 1.5) / 3.5, 0, 1)
    → SPI = -2.0 → DR_norm = 1.0 (stress maximal)
    → SPI =  1.5 → DR_norm = 0.0 (pas de stress)

Usage :
    calc = SPICalculator(window=12)
    spi = calc.compute(precip_da, baseline_da)
    dr_norm = calc.drought_risk_normalized(spi)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr
from loguru import logger
from scipy import stats


class SPICalculator:
    """
    Calcule le Standardized Precipitation Index (SPI-N).

    Args:
        window: Fenêtre temporelle en mois (SPI-12 recommandé pour hydrologie)
        baseline_start: Début période de référence (défaut 1981)
        baseline_end: Fin période de référence (défaut 2010)
    """

    def __init__(
        self,
        window: int = 12,
        baseline_start: int = 1981,
        baseline_end: int = 2010,
    ):
        self.window = window
        self.baseline_start = baseline_start
        self.baseline_end = baseline_end

    # ──────────────────────────────────────────────────────────────────────────
    # Calcul SPI principal
    # ──────────────────────────────────────────────────────────────────────────

    def compute(
        self,
        precip_da: xr.DataArray,
        baseline_da: Optional[xr.DataArray] = None,
        target_year: Optional[int] = None,
    ) -> xr.DataArray:
        """
        Calcule le SPI pour chaque point de la grille.

        Deux modes de fonctionnement :

        1. Mode série complète (target_year fourni) — RECOMMANDÉ :
           precip_da contient la série longue contiguë (ex: 1981-2022).
           Le rolling sum est calculé sur la série entière (tous les mois
           ont assez d'historique). Le fit gamma est fait sur la période
           baseline_start–baseline_end. Seuls les SPI de target_year
           sont retournés.

        2. Mode split (target_year=None) — LEGACY :
           precip_da = année à évaluer (12 mois).
           baseline_da = période de référence (ex: 360 mois).
           ATTENTION : le rolling sum sur 12 mois ne produit qu'une seule
           valeur valide (le dernier mois).

        Args:
            precip_da: Précipitations mensuelles [mm] (time, y, x)
            baseline_da: Précipitations de référence [mm] (time, y, x)
                         Ignoré en mode série complète.
            target_year: Année cible pour le SPI (active le mode série complète)

        Returns:
            DataArray SPI (time, y, x) — adimensionnel
        """
        if target_year is not None:
            return self._compute_full_series(precip_da, target_year)

        # ── Mode split legacy ──────────────────────────────────────────
        if baseline_da is None:
            baseline_da = precip_da

        # Accumulation glissante N mois
        precip_acc = self._rolling_sum(precip_da)
        baseline_acc = self._rolling_sum(baseline_da)

        # Vérifier combien de mois ont des valeurs valides
        n_valid_months = int((~np.isnan(precip_acc.values)).any(axis=tuple(
            range(1, precip_acc.ndim)
        )).sum())
        if n_valid_months <= 1:
            logger.warning(
                f"SPI rolling sum : seulement {n_valid_months} mois valide(s) "
                f"sur {len(precip_da.time)}. Le rolling sum de {self.window} mois "
                f"nécessite au moins {self.window} mois de données. "
                f"Utiliser target_year= avec la série complète."
            )

        logger.info(
            f"Calcul SPI-{self.window} (mode split) : "
            f"grille {dict(precip_acc.sizes)}, "
            f"baseline {dict(baseline_acc.sizes)}"
        )

        spi = self._apply_spi_ufunc(precip_acc, baseline_acc)
        _log_spi_stats(spi)
        return spi

    def _compute_full_series(
        self,
        full_precip: xr.DataArray,
        target_year: int,
    ) -> xr.DataArray:
        """
        Calcule le SPI en mode série complète.

        Le rolling sum est calculé sur toute la série contiguë,
        ce qui garantit que tous les mois de target_year ont une
        valeur valide (à condition que la série commence au moins
        window mois avant target_year).
        """
        # Rolling sum sur la série entière
        full_acc = self._rolling_sum(full_precip)

        # Séparer baseline (pour fit gamma) et target (pour évaluation)
        baseline_mask = (
            (full_precip.time.dt.year >= self.baseline_start)
            & (full_precip.time.dt.year <= self.baseline_end)
        )
        target_mask = full_precip.time.dt.year == target_year

        baseline_acc = full_acc.sel(time=baseline_mask)
        target_acc = full_acc.sel(time=target_mask)

        n_target = int(target_mask.sum())
        n_target_valid = int((~np.isnan(target_acc.values)).any(axis=tuple(
            range(1, target_acc.ndim)
        )).sum())

        logger.info(
            f"Calcul SPI-{self.window} (série complète) : "
            f"série {dict(full_precip.sizes)}, "
            f"baseline {int(baseline_mask.sum())} mois, "
            f"target {target_year} = {n_target_valid}/{n_target} mois valides"
        )

        if n_target_valid < n_target:
            logger.warning(
                f"Seulement {n_target_valid}/{n_target} mois avec rolling sum "
                f"complet. Vérifier que la série commence au moins {self.window} "
                f"mois avant {target_year}."
            )

        spi = self._apply_spi_ufunc(target_acc, baseline_acc)
        _log_spi_stats(spi)
        return spi

    def _apply_spi_ufunc(
        self,
        precip_acc: xr.DataArray,
        baseline_acc: xr.DataArray,
    ) -> xr.DataArray:
        """Applique le calcul SPI pixel par pixel via apply_ufunc."""
        # Renommer la dim time de la baseline pour éviter le conflit
        baseline_for_ufunc = baseline_acc.rename({"time": "time_baseline"})

        spi = xr.apply_ufunc(
            self._compute_spi_pixel,
            precip_acc,
            baseline_for_ufunc,
            input_core_dims=[["time"], ["time_baseline"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float32],
        )

        spi = spi.transpose(*precip_acc.dims)
        spi.name = f"spi{self.window}"
        spi.attrs = {
            "long_name": f"Standardized Precipitation Index (SPI-{self.window})",
            "units": "dimensionless",
            "reference": "McKee et al. (1993)",
            "baseline": f"{self.baseline_start}-{self.baseline_end}",
            "window_months": self.window,
        }
        return spi

    def _rolling_sum(self, da: xr.DataArray) -> xr.DataArray:
        """
        Calcule la somme glissante sur window mois.

        Pour les N premiers timesteps où la fenêtre est incomplète,
        retourne NaN (comportement correct pour SPI).
        """
        return da.rolling(time=self.window, min_periods=self.window).sum()

    def _compute_spi_pixel(
        self,
        precip_acc: np.ndarray,    # (time,) — période à évaluer
        baseline_acc: np.ndarray,  # (time_baseline,) — période de référence
    ) -> np.ndarray:
        """
        Calcule le SPI pour un seul pixel.

        Fit gamma sur baseline, évalue sur precip_acc.

        Note sur le fit gamma :
            - floc=0 : la distribution gamma est contrainte à démarrer à 0
              (physiquement correct : précipitations ≥ 0)
            - Les pixels secs (Σ = 0) sont traités séparément
            - Une proportion q0 de valeurs nulles est intégrée comme
              masse de probabilité discrète (cf. Thom, 1966)
        """
        # Supprimer les NaN
        baseline_valid = baseline_acc[~np.isnan(baseline_acc)]

        if len(baseline_valid) < 10:
            return np.full_like(precip_acc, np.nan, dtype=np.float32)

        # Proportion de valeurs nulles (zones semi-arides)
        q0 = np.sum(baseline_valid == 0.0) / len(baseline_valid)
        baseline_pos = baseline_valid[baseline_valid > 0.0]

        if len(baseline_pos) < 4:
            # Pas assez de valeurs positives → SPI=0 partout
            return np.zeros_like(precip_acc, dtype=np.float32)

        # Fit gamma sur les valeurs positives (floc=0)
        try:
            alpha, loc, beta = stats.gamma.fit(baseline_pos, floc=0.0)
        except Exception:
            return np.full_like(precip_acc, np.nan, dtype=np.float32)

        # Probabilité cumulative pour chaque valeur de la période cible
        spi_out = np.full_like(precip_acc, np.nan, dtype=np.float64)

        for i, p in enumerate(precip_acc):
            if np.isnan(p):
                continue
            if p == 0.0:
                # Masse de probabilité discrète pour les zéros
                prob = q0
            else:
                # Probabilité composite : P(X ≤ p) = q0 + (1-q0) × Gamma_cdf(p)
                prob = q0 + (1.0 - q0) * stats.gamma.cdf(p, alpha, loc=loc, scale=beta)

            # Clamper pour éviter les infinis dans ppf
            prob = np.clip(prob, 1e-6, 1.0 - 1e-6)

            # SPI = quantile normal standard
            spi_out[i] = stats.norm.ppf(prob)

        return spi_out.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Conversion SPI → Drought Risk normalisé
    # ──────────────────────────────────────────────────────────────────────────

    def drought_risk_normalized(
        self,
        spi: xr.DataArray,
    ) -> xr.DataArray:
        """
        Convertit le SPI en score de risque sécheresse normalisé [0, 1].

        Formule :
            DR_norm = clip(-SPI / 3, 0, 1)

            SPI ≥  0.0 → DR_norm = 0.0   (normal ou humide)
            SPI = -1.0 → DR_norm = 0.33   (sécheresse modérée)
            SPI = -2.0 → DR_norm = 0.67   (sécheresse sévère)
            SPI ≤ -3.0 → DR_norm = 1.0    (sécheresse extrême)

        Args:
            spi: DataArray SPI

        Returns:
            DataArray DR_norm [0, 1] — même shape que spi
        """
        spi_extreme = 3.0  # SPI = -3 → stress maximal
        dr_norm = (-spi / spi_extreme).clip(0.0, 1.0)
        dr_norm.name = "drought_risk_norm"
        dr_norm.attrs = {
            "long_name": "Normalized Drought Risk",
            "units": "dimensionless",
            "range": "[0, 1]",
            "formula": "clip(-SPI / 3, 0, 1)",
            "source_spi": f"SPI-{self.window}",
        }
        return dr_norm

    def compute_temporal_mean(
        self,
        spi: xr.DataArray,
        months: Optional[list[int]] = None,
    ) -> xr.DataArray:
        """
        Calcule la moyenne temporelle du SPI (ou sur mois sélectionnés).

        Args:
            spi: DataArray SPI (time, y, x)
            months: Liste de mois [1-12] — None = tous les mois

        Returns:
            DataArray (y, x) — SPI moyen annuel ou saisonnier
        """
        if months is not None:
            spi = spi.sel(time=spi.time.dt.month.isin(months))
        return spi.mean(dim="time", skipna=True)


# ──────────────────────────────────────────────────────────────────────────────
# SPEI — Standardized Precipitation-Evapotranspiration Index (optionnel)
# ──────────────────────────────────────────────────────────────────────────────

def compute_spei(
    precip_mm: xr.DataArray,
    pet_mm: xr.DataArray,
    window: int = 12,
) -> xr.DataArray:
    """
    Calcule le SPEI (Vicente-Serrano et al., 2010).

    SPEI = SPI appliqué sur le bilan hydrique D = P − PET.
    Utilise la distribution log-logistique (3 paramètres) à la place de gamma.

    Référence :
        Vicente-Serrano, S.M. et al. (2010). A Multiscalar Drought Index Sensitive
        to Global Warming: The Standardized Precipitation Evapotranspiration Index.
        Journal of Climate, 23(7), 1696–1718.

    Args:
        precip_mm: Précipitations mensuelles [mm]
        pet_mm: Évapotranspiration potentielle [mm]
        window: Fenêtre temporelle (mois)

    Returns:
        DataArray SPEI — même dimensions que les entrées
    """
    # Bilan hydrique
    D = precip_mm - pet_mm  # Déficit/excédent [mm]

    # Accumulation glissante
    D_acc = D.rolling(time=window, min_periods=window).sum()

    def _spei_pixel(d_acc: np.ndarray) -> np.ndarray:
        valid = d_acc[~np.isnan(d_acc)]
        if len(valid) < 10:
            return np.full_like(d_acc, np.nan, dtype=np.float32)

        # Distribution log-logistique (3 paramètres)
        # Décaler pour valeurs positives : d_shifted = D + min(D) + 1
        shift = -np.min(valid) + 1.0
        d_shifted = valid + shift

        try:
            c, loc, scale = stats.fisk.fit(d_shifted, floc=0)
        except Exception:
            return np.full_like(d_acc, np.nan, dtype=np.float32)

        spei_out = np.full_like(d_acc, np.nan, dtype=np.float64)
        for i, d in enumerate(d_acc):
            if np.isnan(d):
                continue
            prob = stats.fisk.cdf(d + shift, c, loc=loc, scale=scale)
            prob = np.clip(prob, 1e-6, 1.0 - 1e-6)
            spei_out[i] = stats.norm.ppf(prob)

        return spei_out.astype(np.float32)

    spei = xr.apply_ufunc(
        _spei_pixel,
        D_acc,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    )
    spei = spei.transpose(*D_acc.dims)
    spei.name = f"spei{window}"
    spei.attrs = {
        "long_name": f"Standardized Precipitation-Evapotranspiration Index (SPEI-{window})",
        "units": "dimensionless",
        "reference": "Vicente-Serrano et al. (2010)",
    }
    return spei


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────────

def _log_spi_stats(spi: xr.DataArray) -> None:
    """Log les statistiques descriptives du SPI calculé."""
    vals = spi.values[~np.isnan(spi.values)]
    if len(vals) == 0:
        logger.warning("SPI : aucune valeur valide")
        return
    dry = np.sum(vals < -1.0) / len(vals) * 100
    very_dry = np.sum(vals < -2.0) / len(vals) * 100
    logger.success(
        f"SPI-{spi.attrs.get('window_months', '?')} calculé :\n"
        f"  Pixels valides   : {len(vals)}\n"
        f"  Moyenne          : {np.mean(vals):.3f} (doit être ≈ 0)\n"
        f"  Écart-type       : {np.std(vals):.3f} (doit être ≈ 1)\n"
        f"  % secs (< -1.0)  : {dry:.1f}%\n"
        f"  % très secs(<-2) : {very_dry:.1f}%\n"
        f"  [p5, p95]        : [{np.percentile(vals, 5):.2f}, {np.percentile(vals, 95):.2f}]"
    )