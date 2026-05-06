"""
Calcul du Water Stress Index (WSI) — modèle principal.

Formule GreenDC :
    WSI = 0.60 × BWS_norm
        + 0.25 × GWS_norm
        + 0.15 × DroughtRisk_norm

    WaterScore = 100 × (1 − clip(WSI, 0, 1))

Justification des pondérations :
    ┌─────────────────────────────────────────────────────────────────┐
    │ Composante          │ Poids │ Justification                     │
    ├─────────────────────────────────────────────────────────────────┤
    │ BWS (Aqueduct)      │ 0.60  │ Signal le plus direct de pression │
    │                     │       │ sur la ressource en eau de surface│
    │                     │       │ (ratio prélèvements/débit moyen)  │
    ├─────────────────────────────────────────────────────────────────┤
    │ GWS (Aquif.+BRGM)  │ 0.25  │ Ressource souterraine, recharge   │
    │                     │       │ lente → vulnérabilité long terme  │
    ├─────────────────────────────────────────────────────────────────┤
    │ Drought Risk (SPI)  │ 0.15  │ Variabilité climatique, signal    │
    │                     │       │ de court terme (sécheresse 2022)  │
    └─────────────────────────────────────────────────────────────────┘

Normalisation BWS :
    Aqueduct 4.0 fournit BWS sur une échelle 0-5 (catégories de risque).
    On applique une transformation sigmoïde pour accentuer la sensibilité
    dans la zone intermédiaire (BWS 1-3) et saturer les extrêmes :

        BWS_norm = sigmoid_transform(bws_raw, midpoint=2.0, steepness=0.8)

        `
    Alternativement : normalisation linéaire simple BWS_norm = bws_raw / 5.0
    → choix exposé dans config.yaml (normalization_method: 'linear' | 'sigmoid')

Propagation d'incertitude :
    Deux méthodes disponibles :

    1) Propagation analytique (premier ordre, par défaut) :
       Hypothèse d'erreurs indépendantes, incertitudes RELATIVES (fraction, 1σ) :
           ε_BWS ≈ 12% → σ_BWS_abs = ε_BWS × BWS_norm
           ε_GWS ≈ 18% → σ_GWS_abs = ε_GWS × GWS_norm
           ε_DR  ≈ 22% → σ_DR_abs  = ε_DR  × DroughtRisk_norm

       L'incertitude absolue propagée sur WSI (quadrature) :
           σ_WSI² = (w_BWS × σ_BWS_abs)² + (w_GWS × σ_GWS_abs)² + (w_DR × σ_DR_abs)²

    2) Monte Carlo (N=5000 tirages) :
       Chaque composante est perturbée par un bruit gaussien
       centré sur sa valeur normalisée, avec σ = ε × valeur (relatif).
       WSI recalculé pour chaque tirage → extraction de p10, p90, std.

Limitations :
    - Hypothèse d'indépendance entre composantes (BWS/GWS/DR)
    - Pondérations expertes (pas de calibration statistique formelle)
    - Normalisation non universelle (calée sur la France métropolitaine)
    - Pas de saisonnalité explicite dans BWS et GWS (snapshot annuel)
    - Pas de projection climatique future (état actuel uniquement)
    - Incertitudes relatives estimées d'après la littérature, non calibrées localement

Références :
    - Hofste, R. et al. (2019). Aqueduct 4.0 WRI Technical Note.
    - Gosling, S.N. & Arnell, N.W. (2016). A global assessment of the impact of
      climate change on water scarcity. Climatic Change, 134(3), 371-385.
    - McKee, T.B. et al. (1993). SPI definition.
    - van Vliet, M. et al. (2021). Global water scarcity including surface water
      quality and expansions of clean water technologies. Environmental Research
      Letters, 16(2), 024020.

Usage :
    model = WaterStressModel()                          # v1.0, analytique
    wsi_ds = model.compute(bws_norm, gws_norm, drought_risk_norm)
    score_ds = model.to_water_score(wsi_ds)

    # Avec Monte Carlo :
    wsi_mc = model.compute_monte_carlo(bws_norm, gws_norm, drought_risk_norm, n_samples=5000)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xarray as xr
from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Configuration des pondérations
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WSIWeights:
    """Pondérations du Water Stress Index."""
    bws:     float = 0.60
    gws:     float = 0.25
    drought: float = 0.15

    def __post_init__(self):
        total = self.bws + self.gws + self.drought
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Les pondérations doivent sommer à 1.0 (somme = {total:.4f})"
            )


@dataclass
class UncertaintyParams:
    """
    Paramètres d'incertitude RELATIVE (fraction, 1σ).

    Chaque ε représente l'incertitude relative sur la composante normalisée :
        σ_abs = ε × valeur_normalisée
ca
    Exemple : ε_BWS = 0.12 signifie que pour BWS_norm = 0.60,
              σ_BWS_abs = 0.12 × 0.60 = 0.072 (en unités normalisées).
    """
    sigma_bws:     float = 0.12   # 12% relatif (Gosling & Arnell 2016)
    sigma_gws:     float = 0.18   # 18% relatif (données GRACE + modèles hydro)
    sigma_drought: float = 0.22   # 22% relatif (variabilité SPI, longueur série)


MODEL_VERSION = "water_model_v1.0"


class WaterStressModel:
    """
    Calcule le Water Stress Index (WSI) et le WaterScore GreenDC.

    Version : water_model_v1.0
        - 1 normalisation officielle (linear par défaut, sigmoid optionnel)
        - 1 set de poids officiel (0.60 / 0.25 / 0.15)
        - Incertitudes relatives (propagation analytique + Monte Carlo optionnel)

    Args:
        weights: Pondérations BWS/GWS/DroughtRisk
        normalization: 'linear' | 'sigmoid' pour la normalisation BWS
        uncertainty_params: Paramètres d'incertitude (propagation d'erreur)
    """

    def __init__(
        self,
        weights: Optional[WSIWeights] = None,
        normalization: str = "linear",
        uncertainty_params: Optional[UncertaintyParams] = None,
    ):
        self.weights = weights or WSIWeights()
        self.normalization = normalization
        self.uncertainty = uncertainty_params or UncertaintyParams()
        self.version = MODEL_VERSION

    # ──────────────────────────────────────────────────────────────────────────
    # Normalisation BWS
    # ──────────────────────────────────────────────────────────────────────────

    def normalize_bws(
        self,
        bws_raw: xr.DataArray,
        bws_max: float = 5.0,
    ) -> xr.DataArray:
        """
        Normalise le Baseline Water Stress (0-5) vers [0, 1].

        Trois méthodes :
            'linear'  : BWS_norm = bws_raw / bws_max
            'sigmoid' : BWS_norm = 1 / (1 + exp(-k × (bws_raw - mid)))
            'pfister' : Logistique de Pfister et al. (2009)
                        WTA = bws_raw / bws_max
                        WSI = 1 / (1 + exp(-6.4 × WTA / (1 - WTA)))
                        Référence peer-reviewed pour LCA water scarcity.

        Args:
            bws_raw: DataArray BWS brut (0-5)
            bws_max: Valeur maximale de l'échelle (défaut 5.0)

        Returns:
            DataArray BWS normalisé [0, 1]
        """
        if self.normalization == "pfister":
            wta = (bws_raw / bws_max).clip(0.001, 0.999)
            pfister_raw = 1.0 / (1.0 + np.exp(-6.4 * wta / (1.0 - wta)))
            bws_norm = ((pfister_raw - 0.5) / 0.5).clip(0.0, 1.0)
        elif self.normalization == "sigmoid":
            mid = bws_max / 2.5
            k = 0.8
            bws_norm = 1.0 / (1.0 + np.exp(-k * (bws_raw - mid)))
        else:  # linear (default, used in production)
            bws_norm = (bws_raw / bws_max).clip(0.0, 1.0)

        bws_norm = bws_norm.clip(0.0, 1.0)
        bws_norm.name = "bws_norm"
        bws_norm.attrs = {
            "long_name": "Baseline Water Stress (normalized)",
            "units": "dimensionless",
            "source": "WRI Aqueduct 4.0",
            "normalization": self.normalization,
            "reference": "Pfister et al. (2009)" if self.normalization == "pfister" else "",
        }
        return bws_norm

    # ──────────────────────────────────────────────────────────────────────────
    # Calcul WSI
    # ──────────────────────────────────────────────────────────────────────────

    def compute(
        self,
        bws_norm: xr.DataArray,
        gws_norm: xr.DataArray,
        drought_norm: xr.DataArray,
        time_dim: Optional[str] = "time",
    ) -> xr.Dataset:
        """
        Calcule le WSI à partir des trois composantes normalisées.

        Si drought_norm a une dimension temporelle, le WSI est calculé
        pour chaque timestep (puis moyenné et agrégé en percentiles).

        Args:
            bws_norm:     DataArray BWS normalisé [0, 1] (y, x)
            gws_norm:     DataArray GWS normalisé [0, 1] (y, x)
            drought_norm: DataArray Drought Risk normalisé [0, 1] (time?, y, x)
            time_dim:     Nom de la dimension temporelle (None si statique)

        Returns:
            xr.Dataset avec variables :
                - wsi_mean   : WSI moyen [0, 1]
                - wsi_p90    : WSI 90e percentile temporel du WSI
                - bws_norm   : BWS normalisé
                - gws_norm   : GWS normalisé
                - drought_norm_mean : Drought Risk moyen
                - wsi_uncertainty   : Incertitude [0, 1]
        """
        w = self.weights

        # ── Gestion temporelle du drought ────────────────────────────────────
        if time_dim and time_dim in drought_norm.dims:
            # Garde-fou STRICT : vérifier que la dimension temporelle porte
            # une vraie variabilité. Sans ça les percentiles sont inutiles
            # et le pipeline produit des sorties trompeuses.
            dr_temporal_std = float(drought_norm.std(dim=time_dim).mean().values)
            if dr_temporal_std < 1e-6:
                raise ValueError(
                    f"PIPELINE HALT: drought_norm a une dimension '{time_dim}' "
                    f"mais ZÉRO variabilité temporelle (std={dr_temporal_std:.2e}). "
                    f"Le SPI rolling sum n'a probablement produit qu'un seul mois "
                    f"valide. Utiliser SPICalculator.compute(target_year=) avec la "
                    f"série complète 1981-2022."
                )

            # WSI dynamique (time, y, x)
            wsi_dynamic = (
                w.bws * bws_norm
                + w.gws * gws_norm
                + w.drought * drought_norm
            ).clip(0.0, 1.0)
            wsi_dynamic.name = "wsi"

            wsi_mean = wsi_dynamic.mean(dim=time_dim)
            wsi_p10 = wsi_dynamic.quantile(0.10, dim=time_dim)
            wsi_p90 = wsi_dynamic.quantile(0.90, dim=time_dim)
            drought_mean = drought_norm.mean(dim=time_dim)
        else:
            # WSI statique (y, x)
            wsi_mean = (
                w.bws * bws_norm
                + w.gws * gws_norm
                + w.drought * drought_norm
            ).clip(0.0, 1.0)
            wsi_p10 = wsi_mean.copy()
            wsi_p90 = wsi_mean.copy()
            drought_mean = drought_norm

        # ── Incertitude propagée ─────────────────────────────────────────────
        u = self.uncertainty
        wsi_uncertainty_val = np.sqrt(
            (w.bws * u.sigma_bws * bws_norm) ** 2
            + (w.gws * u.sigma_gws * gws_norm) ** 2
            + (w.drought * u.sigma_drought * drought_mean) ** 2
        )
        wsi_uncertainty = wsi_uncertainty_val.clip(0.0, 1.0)

        # ── Assemblage Dataset ───────────────────────────────────────────────
        data_vars = {
            "wsi_mean": wsi_mean.astype(np.float32),
            "wsi_p10":  wsi_p10.drop_vars("quantile", errors="ignore").astype(np.float32),
            "wsi_p90":  wsi_p90.drop_vars("quantile", errors="ignore").astype(np.float32),
            "bws_norm":          bws_norm.astype(np.float32),
            "gws_norm":          gws_norm.astype(np.float32),
            "drought_norm_mean": drought_mean.astype(np.float32),
            "wsi_uncertainty":   wsi_uncertainty.astype(np.float32),
        }

        # Séries mensuelles (si temporel) — pour l'API et le dashboard
        if time_dim and time_dim in drought_norm.dims:
            data_vars["wsi_monthly"] = wsi_dynamic.astype(np.float32)
            data_vars["drought_norm_monthly"] = drought_norm.astype(np.float32)

        ds = xr.Dataset(
            data_vars,
            attrs={
                "title":        "GreenDC Water Stress Index",
                "version":      self.version,
                "formula":      (
                    f"WSI = {w.bws}×BWS + {w.gws}×GWS + {w.drought}×DrR"
                ),
                "crs":          "EPSG:2154 (Lambert-93)",
                "bws_weight":   w.bws,
                "gws_weight":   w.gws,
                "drought_weight": w.drought,
                "uncertainty_method": (
                    "Propagation analytique premier ordre (quadrature, "
                    "incertitudes relatives)"
                ),
                "references":   (
                    "Hofste et al. (2019); McKee et al. (1993); "
                    "Gosling & Arnell (2016)"
                ),
                "limitations": (
                    "Indépendance composantes; poids experts; "
                    "normalisation France; pas de saisonnalité BWS/GWS; "
                    "pas de projection future"
                ),
            },
        )

        # ── Assertions post-compute ──────────────────────────────────────────
        self._assert_output_integrity(ds, has_time=(time_dim and time_dim in drought_norm.dims))
        self._log_wsi_stats(ds)
        return ds

    # ──────────────────────────────────────────────────────────────────────────
    # Monte Carlo — vraie propagation stochastique
    # ──────────────────────────────────────────────────────────────────────────

    def compute_monte_carlo(
        self,
        bws_norm: xr.DataArray,
        gws_norm: xr.DataArray,
        drought_norm: xr.DataArray,
        n_samples: int = 5000,
        seed: int = 42,
    ) -> xr.Dataset:
        """
        Propagation d'incertitude par Monte Carlo (N tirages).

        Chaque composante normalisée est perturbée par un bruit gaussien
        avec σ_abs = ε_relatif × valeur, tronqué à [0, 1].
        Le WSI est recalculé pour chaque tirage.

        Args:
            bws_norm:     DataArray BWS normalisé [0, 1]
            gws_norm:     DataArray GWS normalisé [0, 1]
            drought_norm: DataArray Drought Risk normalisé [0, 1]
                          (si temporel, on utilise la moyenne)
            n_samples:    Nombre de tirages Monte Carlo (défaut 5000)
            seed:         Graine aléatoire pour reproductibilité

        Returns:
            xr.Dataset avec :
                - wsi_mc_mean : WSI moyen Monte Carlo
                - wsi_mc_std  : Écart-type Monte Carlo
                - wsi_mc_p10  : 10e percentile
                - wsi_mc_p90  : 90e percentile
        """
        rng = np.random.default_rng(seed)
        w = self.weights
        u = self.uncertainty

        # Aplatir la dimension temporelle du drought si présente
        if "time" in drought_norm.dims:
            drought_val = drought_norm.mean(dim="time").values
        else:
            drought_val = drought_norm.values

        bws_val = bws_norm.values
        gws_val = gws_norm.values

        shape = bws_val.shape
        wsi_samples = np.empty((n_samples, *shape), dtype=np.float32)

        for i in range(n_samples):
            bws_p = np.clip(
                bws_val + rng.normal(0, u.sigma_bws * np.maximum(bws_val, 1e-6), shape),
                0.0, 1.0,
            )
            gws_p = np.clip(
                gws_val + rng.normal(0, u.sigma_gws * np.maximum(gws_val, 1e-6), shape),
                0.0, 1.0,
            )
            dr_p = np.clip(
                drought_val + rng.normal(0, u.sigma_drought * np.maximum(drought_val, 1e-6), shape),
                0.0, 1.0,
            )
            wsi_samples[i] = np.clip(
                w.bws * bws_p + w.gws * gws_p + w.drought * dr_p,
                0.0, 1.0,
            )

        wsi_mc_mean = np.mean(wsi_samples, axis=0)
        wsi_mc_std = np.std(wsi_samples, axis=0)
        wsi_mc_p10 = np.percentile(wsi_samples, 10, axis=0)
        wsi_mc_p90 = np.percentile(wsi_samples, 90, axis=0)

        coords = bws_norm.coords
        ds = xr.Dataset(
            {
                "wsi_mc_mean": xr.DataArray(wsi_mc_mean, coords=coords, dims=bws_norm.dims),
                "wsi_mc_std":  xr.DataArray(wsi_mc_std, coords=coords, dims=bws_norm.dims),
                "wsi_mc_p10":  xr.DataArray(wsi_mc_p10, coords=coords, dims=bws_norm.dims),
                "wsi_mc_p90":  xr.DataArray(wsi_mc_p90, coords=coords, dims=bws_norm.dims),
            },
            attrs={
                "title":      "GreenDC WSI — Monte Carlo Uncertainty",
                "version":    self.version,
                "n_samples":  n_samples,
                "seed":       seed,
                "method":     "Monte Carlo (perturbation gaussienne, σ relatif)",
            },
        )

        logger.info(
            f"Monte Carlo ({n_samples} tirages) :\n"
            f"  WSI moyen  : {np.nanmean(wsi_mc_mean):.3f}\n"
            f"  σ moyen    : ±{np.nanmean(wsi_mc_std):.3f}\n"
            f"  [p10, p90] : [{np.nanmean(wsi_mc_p10):.3f}, {np.nanmean(wsi_mc_p90):.3f}]"
        )
        return ds

    # ──────────────────────────────────────────────────────────────────────────
    # Conversion en WaterScore GreenDC [0-100]
    # ──────────────────────────────────────────────────────────────────────────

    def to_water_score(self, wsi_ds: xr.Dataset) -> xr.Dataset:
        """
        Convertit le WSI en score GreenDC [0, 100].

        Formule :
            WaterScore = 100 × (1 − WSI_mean)
            WaterScore est borné dans [0, 100].

        Interprétation (seuils alignés avec interpret_water_score) :
            80-100 = Excellent (nord et ouest de la France typiquement)
            65-80  = Bon
            50-65  = Modéré (Île-de-France, bassin parisien)
            35-50  = Défavorable (Hérault, Var en été)
            0-35   = Critique (méditerranéen en stress hydrique)

        Args:
            wsi_ds: Dataset WSI (output de compute())

        Returns:
            Dataset enrichi avec colonnes 'water_score', 'water_score_uncertainty'
        """
        # Score nominal
        water_score = (100.0 * (1.0 - wsi_ds["wsi_mean"])).clip(0.0, 100.0)
        water_score.name = "water_score"
        water_score.attrs = {
            "long_name": "GreenDC Water Availability Score",
            "units": "score [0-100]",
            "range": "[0=crisis, 100=excellent]",
            "formula": "100 × (1 − WSI_mean)",
        }

        # Incertitude sur le score (propagée linéairement)
        score_uncertainty = (100.0 * wsi_ds["wsi_uncertainty"]).clip(0.0, 20.0)
        score_uncertainty.name = "water_score_uncertainty"
        score_uncertainty.attrs = {
            "long_name": "Water Score Uncertainty (1σ)",
            "units": "score points",
        }

        # Score p10 et p90 (plage de variabilité temporelle)
        score_p10 = (100.0 * (1.0 - wsi_ds["wsi_p90"])).clip(0.0, 100.0)
        score_p90 = (100.0 * (1.0 - wsi_ds["wsi_p10"])).clip(0.0, 100.0)
        score_p10.name = "water_score_p10"
        score_p90.name = "water_score_p90"

        ds = wsi_ds.copy()
        ds["water_score"] = water_score
        ds["water_score_uncertainty"] = score_uncertainty
        ds["water_score_p10"] = score_p10
        ds["water_score_p90"] = score_p90

        # Score mensuel (si la série temporelle est disponible)
        if "wsi_monthly" in wsi_ds:
            water_score_monthly = (
                100.0 * (1.0 - wsi_ds["wsi_monthly"])
            ).clip(0.0, 100.0)
            water_score_monthly.name = "water_score_monthly"
            water_score_monthly.attrs = {
                "long_name": "GreenDC Water Score (monthly)",
                "units": "score [0-100]",
                "formula": "100 × (1 − WSI_monthly)",
            }
            ds["water_score_monthly"] = water_score_monthly

            # ── Scores métier saisonniers ──────────────────────────────────
            # worst_month : score du pire mois — dimensionnant pour le
            # refroidissement DC (le mois qui contraint le design)
            worst_month_score = water_score_monthly.min(dim="time")
            worst_month_score.name = "worst_month_score"
            worst_month_score.attrs = {
                "long_name": "Water Score — Worst Month",
                "units": "score [0-100]",
                "description": (
                    "Score du mois le plus défavorable. "
                    "Dimensionnant pour le design refroidissement DC."
                ),
            }
            ds["worst_month_score"] = worst_month_score

            # summer_mean : moyenne juin-septembre — période de stress
            # hydrique typique en France métropolitaine
            summer_months = water_score_monthly.time.dt.month
            summer_mask = (summer_months >= 6) & (summer_months <= 9)
            if summer_mask.any():
                summer_score = water_score_monthly.sel(
                    time=summer_mask
                ).mean(dim="time")
                summer_score.name = "summer_mean_score"
                summer_score.attrs = {
                    "long_name": "Water Score — Summer Mean (Jun-Sep)",
                    "units": "score [0-100]",
                    "description": (
                        "Moyenne estivale (juin-septembre). "
                        "Période typique de stress hydrique en France."
                    ),
                }
                ds["summer_mean_score"] = summer_score

        logger.success(
            f"WaterScore calculé :\n"
            f"  Moyenne nationale : {float(water_score.mean().values):.1f}/100\n"
            f"  Médiane           : {float(water_score.median().values):.1f}/100\n"
            f"  Min / Max         : "
            f"{float(water_score.min().values):.1f} / "
            f"{float(water_score.max().values):.1f}"
        )
        return ds

    # ──────────────────────────────────────────────────────────────────────────
    # Assertions (non-régression)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _assert_output_integrity(ds: xr.Dataset, has_time: bool) -> None:
        """
        Vérifications strictes sur le Dataset WSI.

        Ces checks sont des ERREURS, pas des warnings :
        si l'un échoue, les outputs sont invalides et ne doivent pas être publiés.
        """
        wsi_mean = ds["wsi_mean"].values
        wsi_p10 = ds["wsi_p10"].values
        wsi_p90 = ds["wsi_p90"].values
        valid = ~np.isnan(wsi_mean)

        if valid.sum() == 0:
            raise ValueError("PIPELINE HALT: aucun pixel WSI valide.")

        # 1. Si temporel, p10 et p90 ne doivent PAS être identiques partout
        if has_time:
            diff = np.abs(wsi_p90[valid] - wsi_p10[valid])
            if np.max(diff) < 1e-6:
                raise ValueError(
                    "PIPELINE HALT: wsi_p10 == wsi_p90 partout. "
                    "La variabilité temporelle n'a pas été propagée. "
                    "Le drought_norm a probablement une variance nulle."
                )
            pct_identical = np.sum(diff < 1e-6) / len(diff) * 100
            if pct_identical > 10:
                logger.warning(
                    f"Attention : {pct_identical:.1f}% des pixels ont p10==p90. "
                    f"Possible perte de variabilité temporelle sur certaines zones."
                )

        # 2. WSI dans [0, 1]
        if np.any(wsi_mean[valid] < -0.01) or np.any(wsi_mean[valid] > 1.01):
            raise ValueError(
                f"PIPELINE HALT: WSI hors limites [0,1]. "
                f"min={np.nanmin(wsi_mean):.4f}, max={np.nanmax(wsi_mean):.4f}"
            )

        # 3. Série mensuelle cohérente (si présente)
        if "wsi_monthly" in ds:
            wsi_m = ds["wsi_monthly"].values
            if np.all(np.isnan(wsi_m)):
                raise ValueError(
                    "PIPELINE HALT: wsi_monthly est entièrement NaN."
                )

        logger.info(
            f"Integrity check OK : {int(valid.sum())} pixels valides, "
            f"WSI [{np.nanmin(wsi_mean):.3f}, {np.nanmax(wsi_mean):.3f}]"
            + (f", p10!=p90 sur {100 - pct_identical:.1f}% des pixels" if has_time else "")
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────────────

    def _log_wsi_stats(self, ds: xr.Dataset) -> None:
        wsi = ds["wsi_mean"].values
        valid = wsi[~np.isnan(wsi)]
        if len(valid) == 0:
            return
        logger.success(
            f"WSI calculé ({len(valid)} pixels) :\n"
            f"  Formule : WSI = "
            f"{self.weights.bws}×BWS + {self.weights.gws}×GWS + {self.weights.drought}×DrR\n"
            f"  Moyenne           : {np.mean(valid):.3f}\n"
            f"  Médiane           : {np.median(valid):.3f}\n"
            f"  [p5, p95]         : [{np.percentile(valid, 5):.3f}, "
            f"{np.percentile(valid, 95):.3f}]\n"
            f"  % stress élevé (>0.7) : "
            f"{np.sum(valid > 0.7) / len(valid) * 100:.1f}%\n"
            f"  Incertitude moy.  : "
            f"±{float(ds['wsi_uncertainty'].mean().values):.3f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ──────────────────────────────────────────────────────────────────────────────

def compute_wsi_from_components(
    bws_raw: xr.DataArray,
    gws_norm: xr.DataArray,
    drought_norm: xr.DataArray,
    bws_max: float = 5.0,
    weights: Optional[WSIWeights] = None,
) -> xr.Dataset:
    """
    Fonction tout-en-un pour le pipeline.

    Args:
        bws_raw: BWS brut Aqueduct (0-5)
        gws_norm: GWS normalisé (0-1)
        drought_norm: Drought Risk normalisé (0-1)
        bws_max: Maximum de l'échelle BWS
        weights: Pondérations (None → valeurs par défaut)

    Returns:
        xr.Dataset avec WSI et WaterScore
    """
    model = WaterStressModel(weights=weights)
    bws_norm = model.normalize_bws(bws_raw, bws_max=bws_max)
    wsi_ds = model.compute(bws_norm, gws_norm, drought_norm)
    return model.to_water_score(wsi_ds)


def interpret_water_score(score: float) -> dict:
    """
    Interprétation qualitative du WaterScore GreenDC.

    Returns:
        dict avec niveau, couleur, description
    """
    if score >= 80:
        return {
            "level": "Excellent",
            "color": "#1a9641",
            "description": (
                "Ressource en eau abondante, stress hydrique très faible. "
                "Idéal pour data center avec refroidissement par eau."
            ),
        }
    elif score >= 65:
        return {
            "level": "Bon",
            "color": "#a6d96a",
            "description": (
                "Bonne disponibilité, pression modérée. "
                "Refroidissement par eau viable avec gestion modérée."
            ),
        }
    elif score >= 50:
        return {
            "level": "Modéré",
            "color": "#fdae61",
            "description": (
                "Stress hydrique notable, surtout en été. "
                "Recommander free-cooling ou refroidissement adiabatique."
            ),
        }
    elif score >= 35:
        return {
            "level": "Défavorable",
            "color": "#f46d43",
            "description": (
                "Pression hydrique significative. "
                "Refroidissement par eau déconseillé en période sèche."
            ),
        }
    else:
        return {
            "level": "Critique",
            "color": "#d73027",
            "description": (
                "Zone en stress hydrique sévère. "
                "Refroidissement à air ou tours à circuit fermé uniquement."
            ),
        }
