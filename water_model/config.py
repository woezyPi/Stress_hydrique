"""
Chargement et accès centralisé à la configuration Water Stress Model.
Miroir exact du pattern carbon_model/config.py.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_MODULE_ROOT = Path(__file__).parent
PROJECT_ROOT = _MODULE_ROOT.parent


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = _MODULE_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_cfg: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def get_path(key: str, **format_kwargs) -> Path:
    """
    Retourne un Path absolu pour une clé définie dans config.paths.
    Supporte les templates : get_path('wsi_5km', year=2022)
    """
    cfg = get_config()
    rel = cfg["paths"][key]
    if format_kwargs:
        rel = rel.format(**format_kwargs)
    return _MODULE_ROOT / rel


def get_api_keys() -> dict[str, str]:
    """
    Charge les clés API depuis l'environnement ou le fichier .env.

    Fichier .env attendu dans water_model/ :
        CDS_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

    Nouveau format CDS 2024+ : clé unique (sans UID séparé).
    Le ~/.cdsapirc est aussi lu automatiquement par cdsapi.
    """
    from dotenv import load_dotenv
    load_dotenv(_MODULE_ROOT / ".env")

    return {
        "cds_api_key": os.getenv("CDS_API_KEY", ""),
    }