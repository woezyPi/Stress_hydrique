"""
GreenDC — Water Stress Model.

Calcule un Water Stress Index (WSI) pour chaque point de grille en France
basé sur :
  - WRI Aqueduct 4.0 (BWS, GWS, Drought Risk)
  - BRGM BDLISA (nappes souterraines)
  - ERA5 / Copernicus CDS (précipitations, ETP → SPI/SPEI)

Formule :
    WSI = 0.60 × BWS_norm + 0.25 × GWS_norm + 0.15 × DroughtRisk_norm
    WaterScore = 100 × (1 − WSI)

Architecture identique au modèle carbone (carbon_model/).

Références scientifiques :
    - Hofste et al. (2019) — Aqueduct 4.0 methodology
    - McKee et al. (1993) — SPI definition
    - FAO 56 — Potential evapotranspiration (Penman-Monteith)
    - BRGM (2022) — BDLISA v3 : classification des masses d'eau souterraines
"""
__version__ = "1.0.0"
__author__ = "GreenDC Team"