"""
API Water Stress — GreenDC.

Endpoints :
    GET  /                           → page HTML interactive
    GET  /api/query?lat=&lon=&month= → score hydrique pour un point (optionnel: mois)
    GET  /api/monthly?lat=&lon=      → série mensuelle complète pour un point
    GET  /api/stats                  → statistiques globales du dataset
    GET  /api/heatmap?step=&month=   → grille pour overlay carte (optionnel: mois)
    GET  /api/validation             → rapport de validation

Usage :
    cd water-model   (racine du repo, parent de water_model/)
    uvicorn water_model.api:app --reload --port 8001
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from scipy.spatial import cKDTree

from water_model.modeling.water_stress_model import interpret_water_score

# ──────────────────────────────────────────────────────────────────────────────
# Chargement du dataset (une seule fois au démarrage)
# ──────────────────────────────────────────────────────────────────────────────

_MODULE_ROOT = Path(__file__).parent
_NC_PATH = _MODULE_ROOT / "data" / "outputs" / "water_stress_5km_2022.nc"

_ds: xr.Dataset | None = None
_tree: cKDTree | None = None
_lat2d: np.ndarray | None = None
_lon2d: np.ndarray | None = None
_has_monthly: bool = False
_months: list[int] = []
_year: int = 2022


def _load():
    global _ds, _tree, _lat2d, _lon2d, _has_monthly, _months, _year
    if _ds is not None:
        return
    _ds = xr.open_dataset(str(_NC_PATH))
    lats = _ds.lat.values
    lons = _ds.lon.values
    _lon2d, _lat2d = np.meshgrid(lons, lats)
    # Convertir en pseudo-métrique pour un KDTree correct
    # 1° lat ≈ 111 km, 1° lon ≈ 111 km × cos(lat)
    lat_rad = np.radians(_lat2d.ravel())
    tree_x = _lon2d.ravel() * 111.0 * np.cos(lat_rad)
    tree_y = _lat2d.ravel() * 111.0
    _tree = cKDTree(np.column_stack([tree_y, tree_x]))

    # Détecter les données mensuelles
    _has_monthly = "water_score_monthly" in _ds
    if _has_monthly and "time" in _ds.coords:
        import pandas as pd
        times = pd.DatetimeIndex(_ds.time.values)
        _months = times.month.tolist()
        _year = int(times[0].year)


def _find_pixel(lat: float, lon: float) -> tuple[int, int, float] | None:
    """Retourne (row, col, distance_km) ou None si hors zone."""
    _load()
    lat_rad = np.radians(lat)
    query_x = lon * 111.0 * np.cos(lat_rad)
    query_y = lat * 111.0
    dist, idx = _tree.query([query_y, query_x])
    distance_km = round(dist, 1)  # déjà en km
    if distance_km > 50:
        return None
    row = idx // _lon2d.shape[1]
    col = idx % _lon2d.shape[1]
    return row, col, distance_km


# ──────────────────────────────────────────────────────────────────────────────
# App FastAPI
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="GreenDC Water Stress API", version="2.0.0")


@app.on_event("startup")
def startup():
    _load()


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = _MODULE_ROOT / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/query")
def query_point(
    lat: float = Query(..., description="Latitude (WGS84)"),
    lon: float = Query(..., description="Longitude (WGS84)"),
    month: Optional[int] = Query(
        None, ge=1, le=12,
        description="Mois (1-12). Si omis, retourne la moyenne annuelle.",
    ),
):
    """
    Retourne le détail du stress hydrique pour un point (lat, lon).

    Si month est fourni ET que les données mensuelles existent,
    retourne le score du mois spécifique au lieu de la moyenne annuelle.
    """
    _load()

    pixel = _find_pixel(lat, lon)
    if pixel is None:
        return {
            "error": "Hors zone de couverture (France métropolitaine uniquement)",
            "query": {"lat": lat, "lon": lon},
        }
    row, col, distance_km = pixel

    def _val(var: str) -> float | None:
        if var not in _ds:
            return None
        v = float(_ds[var].values[row, col])
        return None if np.isnan(v) else round(v, 4)

    def _val_monthly(var: str, month_idx: int) -> float | None:
        if var not in _ds:
            return None
        v = float(_ds[var].values[month_idx, row, col])
        return None if np.isnan(v) else round(v, 4)

    # Score pour un mois spécifique
    use_monthly = month is not None and _has_monthly and month in _months
    if use_monthly:
        month_idx = _months.index(month)
        water_score = _val_monthly("water_score_monthly", month_idx)
        wsi = _val_monthly("wsi_monthly", month_idx)
        drought = _val_monthly("drought_norm_monthly", month_idx)
    else:
        water_score = _val("water_score")
        wsi = _val("wsi_mean")
        drought = _val("drought_norm_mean")

    interpretation = interpret_water_score(water_score) if water_score is not None else None

    result = {
        "query": {"lat": lat, "lon": lon, "month": month, "year": _year},
        "mode": f"month_{month}" if use_monthly else "annual_mean",
        "nearest_grid": {
            "lat": round(float(_lat2d.ravel()[row * _lon2d.shape[1] + col]), 4),
            "lon": round(float(_lon2d.ravel()[row * _lon2d.shape[1] + col]), 4),
            "distance_km": distance_km,
        },
        "water_score": water_score,
        "interpretation": interpretation,
        "components": {
            "wsi_mean": _val("wsi_mean"),
            "wsi": wsi,
            "wsi_p10": _val("wsi_p10"),
            "wsi_p90": _val("wsi_p90"),
            "bws_norm": _val("bws_norm"),
            "gws_norm": _val("gws_norm"),
            "drought_norm": drought,
            "drought_norm_mean": _val("drought_norm_mean"),
            "wsi_uncertainty": _val("wsi_uncertainty"),
        },
    }

    # Scores saisonniers (toujours inclus)
    result["seasonal"] = {
        "worst_month_score": _val("worst_month_score"),
        "summer_mean_score": _val("summer_mean_score"),
    }

    # Ajouter le contexte annuel si on est en mode mensuel
    if use_monthly:
        result["annual"] = {
            "water_score": _val("water_score"),
            "wsi_mean": _val("wsi_mean"),
            "wsi_p10": _val("wsi_p10"),
            "wsi_p90": _val("wsi_p90"),
        }

    if not use_monthly and month is not None and not _has_monthly:
        result["warning"] = (
            "Données mensuelles non disponibles dans le NetCDF. "
            "Relancer le pipeline pour générer les séries mensuelles."
        )

    return result


@app.get("/api/monthly")
def monthly_series(
    lat: float = Query(..., description="Latitude (WGS84)"),
    lon: float = Query(..., description="Longitude (WGS84)"),
):
    """
    Retourne la série mensuelle complète (12 mois) pour un point.

    Permet de visualiser la saisonnalité du stress hydrique.
    """
    _load()

    pixel = _find_pixel(lat, lon)
    if pixel is None:
        return {
            "error": "Hors zone de couverture (France métropolitaine uniquement)",
            "query": {"lat": lat, "lon": lon},
        }
    row, col, distance_km = pixel

    if not _has_monthly:
        # Fallback : retourner la moyenne annuelle répétée
        ws = float(_ds["water_score"].values[row, col])
        if np.isnan(ws):
            return {"error": "Pixel masqué (hors France métropolitaine)"}
        return {
            "query": {"lat": lat, "lon": lon, "year": _year},
            "warning": "Données mensuelles non disponibles, moyenne annuelle utilisée",
            "months": [
                {"month": m, "water_score": round(ws, 1), "drought_norm": None}
                for m in range(1, 13)
            ],
        }

    months_data = []
    for i, m in enumerate(_months):
        ws_m = float(_ds["water_score_monthly"].values[i, row, col])
        dr_m = float(_ds["drought_norm_monthly"].values[i, row, col])
        wsi_m = float(_ds["wsi_monthly"].values[i, row, col])
        months_data.append({
            "month": m,
            "water_score": None if np.isnan(ws_m) else round(ws_m, 1),
            "wsi": None if np.isnan(wsi_m) else round(wsi_m, 4),
            "drought_norm": None if np.isnan(dr_m) else round(dr_m, 4),
        })

    # Annuel + saisonnier pour contexte
    ws_annual = float(_ds["water_score"].values[row, col])

    def _safe(var):
        if var not in _ds:
            return None
        v = float(_ds[var].values[row, col])
        return None if np.isnan(v) else round(v, 1)

    return {
        "query": {"lat": lat, "lon": lon, "year": _year},
        "annual": {
            "water_score": round(ws_annual, 1) if not np.isnan(ws_annual) else None,
            "water_score_p10": _safe("water_score_p10"),
            "water_score_p90": _safe("water_score_p90"),
            "worst_month_score": _safe("worst_month_score"),
            "summer_mean_score": _safe("summer_mean_score"),
        },
        "months": months_data,
    }


@app.get("/api/stats")
def dataset_stats():
    """Statistiques globales du dataset."""
    _load()

    def _stats(var: str) -> dict | None:
        if var not in _ds:
            return None
        arr = _ds[var].values
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return None
        return {
            "mean": round(float(np.mean(valid)), 4),
            "median": round(float(np.median(valid)), 4),
            "min": round(float(np.min(valid)), 4),
            "max": round(float(np.max(valid)), 4),
            "p10": round(float(np.percentile(valid, 10)), 4),
            "p90": round(float(np.percentile(valid, 90)), 4),
            "std": round(float(np.std(valid)), 4),
            "n_valid": int(len(valid)),
        }

    ws_arr = _ds["water_score"].values
    ws_valid = ws_arr[~np.isnan(ws_arr)]

    result = {
        "dataset": str(_NC_PATH.name),
        "year": _year,
        "has_monthly": _has_monthly,
        "months_available": _months if _has_monthly else [],
        "grid": {
            "n_lat": len(_ds.lat),
            "n_lon": len(_ds.lon),
            "bbox": {
                "lat_min": round(float(_ds.lat.min()), 2),
                "lat_max": round(float(_ds.lat.max()), 2),
                "lon_min": round(float(_ds.lon.min()), 2),
                "lon_max": round(float(_ds.lon.max()), 2),
            },
        },
        "water_score": {
            "mean": round(float(np.mean(ws_valid)), 1),
            "median": round(float(np.median(ws_valid)), 1),
            "min": round(float(np.min(ws_valid)), 1),
            "max": round(float(np.max(ws_valid)), 1),
            "p10": round(float(np.percentile(ws_valid, 10)), 1),
            "p90": round(float(np.percentile(ws_valid, 90)), 1),
        },
        "components": {
            "bws_norm": _stats("bws_norm"),
            "gws_norm": _stats("gws_norm"),
            "drought_norm_mean": _stats("drought_norm_mean"),
            "wsi_uncertainty": _stats("wsi_uncertainty"),
        },
    }
    return result


@app.get("/api/heatmap")
def heatmap_grid(
    step: int = Query(3, description="Sous-échantillonnage (1=full, 3=every 3rd pixel)"),
    month: Optional[int] = Query(None, ge=1, le=12, description="Mois (1-12). Omis = moyenne annuelle."),
):
    """Retourne la grille WaterScore pour affichage heatmap sur la carte."""
    _load()
    lats = _ds.lat.values[::step]
    lons = _ds.lon.values[::step]

    # Score annuel ou mensuel
    use_monthly = month is not None and _has_monthly and month in _months
    if use_monthly:
        month_idx = _months.index(month)
        ws = _ds["water_score_monthly"].values[month_idx, ::step, ::step]
    else:
        ws = _ds["water_score"].values[::step, ::step]

    bws = _ds["bws_norm"].values[::step, ::step] if "bws_norm" in _ds else None
    gws = _ds["gws_norm"].values[::step, ::step] if "gws_norm" in _ds else None

    if use_monthly and "drought_norm_monthly" in _ds:
        dr = _ds["drought_norm_monthly"].values[month_idx, ::step, ::step]
    elif "drought_norm_mean" in _ds:
        dr = _ds["drought_norm_mean"].values[::step, ::step]
    else:
        dr = None

    dlat = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.05
    dlon = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.05

    points = []
    for i, lat_v in enumerate(lats):
        for j, lon_v in enumerate(lons):
            v = float(ws[i, j])
            if np.isnan(v):
                continue
            pt = {"lat": round(float(lat_v), 4), "lon": round(float(lon_v), 4), "ws": round(v, 1)}
            if bws is not None and not np.isnan(bws[i, j]):
                pt["bws"] = round(float(bws[i, j]), 3)
            if gws is not None and not np.isnan(gws[i, j]):
                pt["gws"] = round(float(gws[i, j]), 3)
            if dr is not None and not np.isnan(dr[i, j]):
                pt["dr"] = round(float(dr[i, j]), 3)
            points.append(pt)

    return {
        "points": points,
        "dlat": round(dlat, 5),
        "dlon": round(dlon, 5),
        "step": step,
        "month": month if use_monthly else None,
        "mode": f"month_{month}" if use_monthly else "annual",
    }


@app.get("/api/validation")
def validation_report():
    """Retourne le rapport de validation si disponible."""
    report_path = _MODULE_ROOT / "data" / "outputs" / "water_validation_report.txt"
    if not report_path.exists():
        return {"error": "Rapport de validation non trouvé"}
    return {"report": report_path.read_text(encoding="utf-8")}
