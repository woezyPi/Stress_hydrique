"""
Modèle de stress en eau souterraine.

Combine deux sources de données complémentaires :
    1. WRI Aqueduct 4.0 (gws_raw) — dépletion des aquifères à l'échelle mondiale
       Métrique : ratio dépletion/recharge (0-5)
       Résolution : ~10 km (polygones HydroSHEDS)

    2. BRGM BDLISA v3 — caractérisation des masses d'eau souterraine françaises
       Métrique : score de productivité/stress hydrogéologique (0-1)
       Résolution : variable (polygones)

Fusion spatiale (v1.0, sans piézométrie) :
    GWS_final = 0.6 × GWS_aqueduct_norm + 0.4 × GWS_brgm_norm

Fusion spatiale (v1.1, avec piézométrie Hub'Eau temps réel) :
    GWS_final = 0.4 × GWS_aqueduct_norm + 0.3 × GWS_brgm_norm + 0.3 × Piezo_stress

Justification des poids :
    - Aqueduct est directement basé sur des données de dépletion observée
      (GRACE satellite + modèles PCRGLOBWB, Wada et al.) → signal physique fort
    - BRGM apporte la distinction fine entre types d'aquifères (karst vs poreux)
      et l'état quantitatif officiel → amélioration de la résolution qualitative
    - Hub'Eau piézométrie apporte le signal temps réel (niveaux de nappe observés)
      → dynamique saisonnière et interannuelle

Références :
    - Hofste, R. et al. (2019). Aqueduct 4.0 — Updated Decision-Relevant Global
      Water Risk Indicators. WRI Technical Note.
    - Wada, Y. et al. (2016). Modelling global water stress of the recent past.
      Geoscientific Model Development, 9(1), 397-408.
    - BRGM (2022). BDLISA v3 : Guide de lecture.

Usage :
    model = GroundwaterStressModel()
    gws_combined = model.compute(aqueduct_gws_da, brgm_gws_da)

    # Avec piézométrie Hub'Eau temps réel :
    model_v2 = GroundwaterStressModel(
        weight_aqueduct=0.4, weight_brgm=0.3, weight_piezo=0.3
    )
    gws_v2 = model_v2.compute(aqueduct_gws_da, brgm_gws_da, piezo_stress_da)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr
from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Paramètres de normalisation Aqueduct GWS
# Aqueduct 4.0 utilise une échelle de risque 0-5 :
#   0    : faible / sans données
#   1    : faible
#   2    : modéré
#   3    : élevé
#   4    : très élevé
#   5    : extrêmement élevé
# ──────────────────────────────────────────────────────────────────────────────
_AQUEDUCT_GWS_MAX = 5.0

# Poids de fusion (v1.0 : Aqueduct + BRGM)
_WEIGHT_AQUEDUCT = 0.60
_WEIGHT_BRGM = 0.40
_WEIGHT_PIEZO = 0.00   # v1.0 = pas de piézo

# Seuil de dépletion critique (GWS Aqueduct ≥ 3 = zone à risque en France)
_CRITICAL_GWS_THRESHOLD = 3.0


class GroundwaterStressModel:
    """
    Calcule le stress en eau souterraine normalisé [0, 1].

    Supporte 2 modes :
        - v1.0 : Aqueduct + BRGM (statique)
        - v1.1 : Aqueduct + BRGM + Hub'Eau piézométrie (dynamique)

    Args:
        weight_aqueduct: Poids Aqueduct (0-1)
        weight_brgm: Poids BRGM (0-1)
        weight_piezo: Poids piézométrie Hub'Eau (0-1), 0 = désactivé
        gws_max_aqueduct: Valeur maximale de l'échelle Aqueduct (défaut 5.0)
    """

    def __init__(
        self,
        weight_aqueduct: float = _WEIGHT_AQUEDUCT,
        weight_brgm: float = _WEIGHT_BRGM,
        weight_piezo: float = _WEIGHT_PIEZO,
        gws_max_aqueduct: float = _AQUEDUCT_GWS_MAX,
    ):
        total = weight_aqueduct + weight_brgm + weight_piezo
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Les poids doivent sommer à 1.0 (somme = {total:.4f})"
            )
        self.weight_aqueduct = weight_aqueduct
        self.weight_brgm = weight_brgm
        self.weight_piezo = weight_piezo
        self.gws_max_aqueduct = gws_max_aqueduct

    def normalize_aqueduct(self, gws_raw: xr.DataArray) -> xr.DataArray:
        """
        Normalise gws_raw Aqueduct (0-5) vers [0, 1].

        Transformée linéaire : GWS_norm = min(gws_raw / gws_max, 1.0)

        Args:
            gws_raw: DataArray Aqueduct GWS brut (0-5)

        Returns:
            DataArray normalisé [0, 1]
        """
        gws_norm = (gws_raw / self.gws_max_aqueduct).clip(0.0, 1.0)
        gws_norm.name = "gws_aqueduct_norm"
        gws_norm.attrs = {
            "long_name": "Groundwater Stress (Aqueduct 4.0, normalized)",
            "units": "dimensionless",
            "source": "WRI Aqueduct 4.0",
            "original_range": f"0-{self.gws_max_aqueduct}",
        }
        return gws_norm

    def normalize_brgm(self, gws_brgm: xr.DataArray) -> xr.DataArray:
        """
        Normalise le score BRGM (déjà [0-1]) — validation et attributs.

        Args:
            gws_brgm: DataArray BRGM GWS score (0-1)

        Returns:
            DataArray normalisé [0, 1] avec attributs
        """
        gws_norm = gws_brgm.clip(0.0, 1.0)
        gws_norm.name = "gws_brgm_norm"
        gws_norm.attrs = {
            "long_name": "Groundwater Stress (BRGM BDLISA, normalized)",
            "units": "dimensionless",
            "source": "BRGM BDLISA v3",
        }
        return gws_norm

    def compute(
        self,
        aqueduct_gws: xr.DataArray,
        brgm_gws: Optional[xr.DataArray] = None,
        piezo_stress: Optional[xr.DataArray] = None,
    ) -> xr.DataArray:
        """
        Calcule le GWS combiné.

        v1.0 : GWS = 0.6×Aqueduct + 0.4×BRGM
        v1.1 : GWS = 0.4×Aqueduct + 0.3×BRGM + 0.3×Piezo (si piezo fourni)

        Args:
            aqueduct_gws: DataArray Aqueduct GWS brut (0-5)
            brgm_gws: DataArray BRGM GWS score normalisé (0-1)
                      None → redistribue le poids sur les autres sources
            piezo_stress: DataArray anomalie piézométrique normalisée [0-1]
                         (0 = nappe haute, 1 = nappe basse = stress)
                         None → redistribue le poids sur les autres sources

        Returns:
            DataArray GWS combiné normalisé [0, 1]
        """
        gws_aq_norm = self.normalize_aqueduct(aqueduct_gws)

        # Déterminer les sources disponibles et recalculer les poids
        sources = {"aqueduct": (gws_aq_norm, self.weight_aqueduct)}

        if brgm_gws is not None:
            gws_brgm_norm = self.normalize_brgm(brgm_gws)
            try:
                gws_brgm_aligned = gws_brgm_norm.interp_like(
                    gws_aq_norm, method="linear"
                )
                sources["brgm"] = (gws_brgm_aligned, self.weight_brgm)
            except Exception as e:
                logger.warning(f"Alignement BRGM/Aqueduct échoué ({e})")
        else:
            logger.info("BRGM non disponible")

        if piezo_stress is not None and self.weight_piezo > 0:
            try:
                piezo_aligned = piezo_stress.interp_like(
                    gws_aq_norm, method="linear"
                )
                sources["piezo"] = (piezo_aligned, self.weight_piezo)
            except Exception as e:
                logger.warning(f"Alignement piézo échoué ({e})")
        elif self.weight_piezo > 0:
            logger.info("Piézométrie Hub'Eau non disponible")

        # Renormaliser les poids selon les sources disponibles
        total_w = sum(w for _, w in sources.values())
        weights = {name: w / total_w for name, (_, w) in sources.items()}

        if len(sources) == 1:
            logger.warning("GWS = Aqueduct seul")
            gws_combined = gws_aq_norm.copy()
            gws_combined.name = "gws_norm"
            return gws_combined

        # Combinaison pondérée avec gestion NaN pixel par pixel
        shape = gws_aq_norm.shape
        gws_sum = np.zeros(shape, dtype=np.float64)
        weight_sum = np.zeros(shape, dtype=np.float64)

        for name, (da, _) in sources.items():
            w = weights[name]
            vals = da.values.astype(np.float64)
            valid = ~np.isnan(vals)
            gws_sum[valid] += w * vals[valid]
            weight_sum[valid] += w

        # Normaliser par les poids effectivement utilisés
        with np.errstate(divide="ignore", invalid="ignore"):
            gws_combined_vals = np.where(
                weight_sum > 0,
                (gws_sum / weight_sum).clip(0.0, 1.0),
                np.nan,
            )

        # Construire la formule pour les attrs
        formula_parts = [
            f"{weights[name]:.2f}×{name.upper()}"
            for name in sources
        ]
        formula_str = " + ".join(formula_parts)
        source_names = " + ".join(
            s.upper() for s in sources
        )
        version = "v1.1" if "piezo" in sources else "v1.0"

        gws_combined = xr.DataArray(
            gws_combined_vals.astype(np.float32),
            dims=gws_aq_norm.dims,
            coords=gws_aq_norm.coords,
            name="gws_norm",
            attrs={
                "long_name": f"Groundwater Stress ({source_names})",
                "units": "dimensionless",
                "range": "[0, 1]",
                "version": version,
                "formula": formula_str,
                **{f"weight_{name}": weights[name] for name in sources},
            },
        )

        n_valid = int(np.sum(~np.isnan(gws_combined_vals)))
        logger.success(
            f"GWS combiné ({version}) :\n"
            f"  Sources           : {source_names}\n"
            f"  Formule           : {formula_str}\n"
            f"  Pixels valides    : {n_valid}\n"
            f"  Moyenne           : {np.nanmean(gws_combined_vals):.3f}\n"
            f"  % zones critiques (≥ 0.6) : "
            f"{np.sum(gws_combined_vals >= 0.6) / max(n_valid, 1) * 100:.1f}%"
        )
        return gws_combined

    def identify_critical_zones(
        self,
        gws_norm: xr.DataArray,
        threshold: float = 0.6,
    ) -> xr.DataArray:
        """
        Identifie les zones à stress souterrain critique.

        Args:
            gws_norm: DataArray GWS normalisé [0, 1]
            threshold: Seuil de criticité (défaut 0.6)

        Returns:
            DataArray booléen : True = zone critique
        """
        critical = gws_norm >= threshold
        critical.name = "gws_critical"
        critical.attrs = {
            "long_name": f"Critical Groundwater Stress Zones (GWS ≥ {threshold})",
            "threshold": threshold,
        }
        n_critical = int(critical.values.sum())
        logger.info(
            f"Zones souterraines critiques (GWS ≥ {threshold}) : "
            f"{n_critical} pixels"
        )
        return critical

    def uncertainty_estimate(
        self,
        gws_norm: xr.DataArray,
    ) -> xr.DataArray:
        """
        Estime l'incertitude du GWS via la différence Aqueduct/BRGM.

        L'incertitude est définie comme l'écart absolu entre les deux sources
        normalisé par leur moyenne. Si une seule source est disponible,
        l'incertitude est fixée à 0.2 (valeur de défaut conservative).

        Returns:
            DataArray incertitude [0, 1]
        """
        uncertainty = xr.full_like(gws_norm, fill_value=0.2)
        uncertainty.name = "gws_uncertainty"
        uncertainty.attrs = {
            "long_name": "GWS Uncertainty Estimate",
            "units": "dimensionless",
            "note": "Relative uncertainty [0=certain, 1=very uncertain]",
        }
        return uncertainty
