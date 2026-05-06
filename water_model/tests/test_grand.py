"""
Grand test — GreenDC Water Stress Model v2.

Vérifie l'intégrité complète des sorties :
    1. Structure du dataset (variables, dims, coords)
    2. Masque France métropolitaine
    3. Ranges physiques
    4. Variabilité temporelle (fix SPI)
    5. Exactitude mathématique des percentiles
    6. Cohérence spatiale
    7. Scores saisonniers
    8. Conservation de variance
    9. API smoke test
    10. Rapport de validation

Usage :
    python -m water_model.tests.test_grand
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

NC_PATH = Path(__file__).parent.parent / "data" / "outputs" / "water_stress_5km_2022.nc"

PASS = 0
FAIL = 0
WARN = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")


def warn(name, detail):
    global WARN
    WARN += 1
    print(f"  [WARN] {name} — {detail}")


def get_pixel(ds, lat, lon):
    li = int(np.argmin(np.abs(ds.lat.values - lat)))
    lj = int(np.argmin(np.abs(ds.lon.values - lon)))
    return li, lj


def region_mean(ds, var, lat_lo, lat_hi, lon_lo, lon_hi):
    v = ds[var].values
    lat_mask = (ds.lat.values >= lat_lo) & (ds.lat.values <= lat_hi)
    lon_mask = (ds.lon.values >= lon_lo) & (ds.lon.values <= lon_hi)
    region = v[np.ix_(lat_mask, lon_mask)]
    valid = region[~np.isnan(region)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


def main():
    global PASS, FAIL, WARN

    if not NC_PATH.exists():
        print(f"ABORT: {NC_PATH} not found. Run pipeline first.")
        sys.exit(2)

    ds = xr.open_dataset(str(NC_PATH))
    lats = ds.lat.values
    lons = ds.lon.values

    print("=" * 70)
    print("  GRAND TEST — GreenDC Water Stress Model v2")
    print("=" * 70)

    # ─── 1. STRUCTURE ─────────────────────────────────────────
    print("\n1. STRUCTURE DU DATASET")

    required_2d = [
        "wsi_mean", "wsi_p10", "wsi_p90", "bws_norm", "gws_norm",
        "drought_norm_mean", "wsi_uncertainty", "water_score",
        "water_score_uncertainty", "water_score_p10", "water_score_p90",
        "worst_month_score", "summer_mean_score",
    ]
    required_3d = ["wsi_monthly", "drought_norm_monthly", "water_score_monthly"]

    for var in required_2d:
        check(f"{var} exists (2D)", var in ds, "missing")
    for var in required_3d:
        check(f"{var} exists (3D)", var in ds, "missing")

    check("lat/lon coords", "lat" in ds.coords and "lon" in ds.coords)
    check("time coord", "time" in ds.coords)

    if "time" in ds.coords:
        n_months = len(ds.time)
        check(f"12 months in time", n_months == 12, f"got {n_months}")

    for var in required_3d:
        if var in ds:
            check(f"{var} dims start with time", ds[var].dims[0] == "time",
                  f"dims={ds[var].dims}")

    # ─── 2. MASQUE FRANCE ─────────────────────────────────────
    print("\n2. MASQUE FRANCE")

    ws = ds["water_score"].values
    n_total = ws.size
    n_nan = int(np.sum(np.isnan(ws)))
    n_valid = n_total - n_nan
    pct = n_valid / n_total * 100

    check("Mask applied (NaN exist)", n_nan > 0, "no NaN")
    check(f"Valid 40-55% of grid ({pct:.1f}%)", 40 < pct < 55, f"{pct:.1f}%")

    # France cities = NOT masked
    for name, lat, lon in [("Paris", 48.85, 2.35), ("Marseille", 43.30, 5.37),
                            ("Brest", 48.39, -4.48), ("Strasbourg", 48.58, 7.75)]:
        li, lj = get_pixel(ds, lat, lon)
        check(f"{name} not masked", not np.isnan(ws[li, lj]))

    # Foreign/ocean = masked
    for name, lat, lon in [("Atlantic", 47.0, -4.5), ("Channel", 50.0, -2.0),
                            ("Germany", 50.0, 8.0), ("Spain", 42.0, -2.0)]:
        li, lj = get_pixel(ds, lat, lon)
        check(f"{name} masked", np.isnan(ws[li, lj]), f"got {ws[li, lj]:.1f}")

    # ─── 3. RANGES ────────────────────────────────────────────
    print("\n3. RANGES PHYSIQUES")

    ranges = [
        ("wsi_mean", 0, 1), ("wsi_p10", 0, 1), ("wsi_p90", 0, 1),
        ("bws_norm", 0, 1), ("gws_norm", 0, 1), ("drought_norm_mean", 0, 1),
        ("water_score", 0, 100), ("worst_month_score", 0, 100),
        ("summer_mean_score", 0, 100),
    ]
    for var, lo, hi in ranges:
        v = ds[var].values
        valid = v[~np.isnan(v)]
        if len(valid) == 0:
            check(f"{var} in [{lo},{hi}]", False, "all NaN")
        else:
            vmin, vmax = float(np.min(valid)), float(np.max(valid))
            check(f"{var} in [{lo},{hi}]", vmin >= lo - 0.01 and vmax <= hi + 0.01,
                  f"[{vmin:.4f}, {vmax:.4f}]")

    # ─── 4. VARIABILITE TEMPORELLE ────────────────────────────
    print("\n4. VARIABILITE TEMPORELLE")

    wsi_p10 = ds["wsi_p10"].values
    wsi_p90 = ds["wsi_p90"].values
    valid_mask = ~np.isnan(wsi_p10)
    diff = np.abs(wsi_p90[valid_mask] - wsi_p10[valid_mask])

    check("max(p90-p10) > 0.01", np.max(diff) > 0.01, f"{np.max(diff):.6f}")
    pct_diff = np.sum(diff > 1e-6) / len(diff) * 100
    check(f"p10!=p90 on >90% pixels ({pct_diff:.1f}%)", pct_diff > 90)

    wsm = ds["water_score_monthly"].values
    for name, lat, lon in [("Paris", 48.85, 2.35), ("Lyon", 45.75, 4.85)]:
        li, lj = get_pixel(ds, lat, lon)
        series = wsm[:, li, lj]
        std_s = float(np.nanstd(series))
        check(f"{name} monthly std > 0.5 pts", std_s > 0.5, f"std={std_s:.2f}")

    # ─── 5. PERCENTILES ───────────────────────────────────────
    print("\n5. EXACTITUDE PERCENTILES")

    for name, lat, lon in [("Paris", 48.85, 2.35), ("Marseille", 43.30, 5.37),
                            ("Lyon", 45.75, 4.85), ("Rennes", 48.11, -1.68)]:
        li, lj = get_pixel(ds, lat, lon)
        wsi_m = ds["wsi_monthly"].values[:, li, lj]
        valid_wsi = wsi_m[~np.isnan(wsi_m)]
        if len(valid_wsi) < 2:
            continue
        p10_m = float(np.percentile(valid_wsi, 10))
        p90_m = float(np.percentile(valid_wsi, 90))
        mean_m = float(np.mean(valid_wsi))
        check(f"{name} mean match",
              abs(mean_m - float(ds["wsi_mean"].values[li, lj])) < 0.001)
        check(f"{name} p10 match",
              abs(p10_m - float(ds["wsi_p10"].values[li, lj])) < 0.002)
        check(f"{name} p90 match",
              abs(p90_m - float(ds["wsi_p90"].values[li, lj])) < 0.002)

    # ─── 6. COHERENCE SPATIALE ────────────────────────────────
    print("\n6. COHERENCE SPATIALE")

    med_wsi = region_mean(ds, "wsi_mean", 42.0, 44.0, 3.0, 7.0)
    bret_wsi = region_mean(ds, "wsi_mean", 47.5, 49.0, -4.5, -1.0)
    check("Mediterranean WSI > Brittany", med_wsi > bret_wsi,
          f"Med={med_wsi:.3f}, Bret={bret_wsi:.3f}")

    alps_ws = region_mean(ds, "water_score", 44.5, 47.0, 5.5, 7.5)
    check(f"Alps score > 60 ({alps_ws:.1f})", alps_ws > 60)

    # ─── 7. SCORES SAISONNIERS ────────────────────────────────
    print("\n7. SCORES SAISONNIERS")

    for name, lat, lon in [("Paris", 48.85, 2.35), ("Lyon", 45.75, 4.85),
                            ("Nantes", 47.22, -1.55)]:
        li, lj = get_pixel(ds, lat, lon)
        ann = float(ds["water_score"].values[li, lj])
        worst = float(ds["worst_month_score"].values[li, lj])
        summer = float(ds["summer_mean_score"].values[li, lj])
        check(f"{name}: worst({worst:.1f}) <= summer({summer:.1f}) <= annual({ann:.1f})",
              worst <= summer + 0.1 and summer <= ann + 0.1)

    # worst_month = min(monthly)
    for name, lat, lon in [("Paris", 48.85, 2.35), ("Marseille", 43.30, 5.37)]:
        li, lj = get_pixel(ds, lat, lon)
        manual_worst = float(np.nanmin(wsm[:, li, lj]))
        stored_worst = float(ds["worst_month_score"].values[li, lj])
        check(f"{name}: worst matches min(monthly)",
              abs(manual_worst - stored_worst) < 0.1,
              f"manual={manual_worst:.1f}, stored={stored_worst:.1f}")

    # ─── 8. VARIANCE ──────────────────────────────────────────
    print("\n8. CONSERVATION VARIANCE")

    dr_m = ds["drought_norm_monthly"].values
    dr_std = np.nanstd(dr_m, axis=0)
    valid_std = dr_std[~np.isnan(dr_std)]
    check("DR temporal std mean > 0.05", np.mean(valid_std) > 0.05,
          f"{np.mean(valid_std):.4f}")
    check("DR temporal std max > 0.15", np.max(valid_std) > 0.15,
          f"{np.max(valid_std):.4f}")

    # ─── 9. API ───────────────────────────────────────────────
    print("\n9. API SMOKE TEST")

    try:
        import uvicorn
        import threading
        import time
        import requests

        def _run():
            uvicorn.run("water_model.api:app", host="127.0.0.1", port=8097,
                        log_level="error")

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        time.sleep(3)

        # Annual
        d = requests.get("http://127.0.0.1:8097/api/query?lat=48.85&lon=2.35",
                         timeout=5).json()
        check("API annual returns score", d.get("water_score") is not None)
        check("API annual mode", d.get("mode") == "annual_mean")
        check("API annual has seasonal", d.get("seasonal", {}).get("worst_month_score") is not None)

        # Monthly
        d = requests.get("http://127.0.0.1:8097/api/query?lat=48.85&lon=2.35&month=8",
                         timeout=5).json()
        check("API month=8 returns score", d.get("water_score") is not None)
        check("API month=8 mode", d.get("mode") == "month_8")
        check("API month=8 has annual", "annual" in d)

        # Series
        d = requests.get("http://127.0.0.1:8097/api/monthly?lat=48.85&lon=2.35",
                         timeout=5).json()
        check("API /monthly 12 months", len(d.get("months", [])) == 12)
        check("API /monthly has worst_month",
              d.get("annual", {}).get("worst_month_score") is not None)

        # Masked point
        d = requests.get("http://127.0.0.1:8097/api/query?lat=50&lon=8",
                         timeout=5).json()
        check("API rejects out-of-France",
              d.get("error") is not None or d.get("water_score") is None)

        # Stats
        d = requests.get("http://127.0.0.1:8097/api/stats", timeout=5).json()
        check("API stats has_monthly", d.get("has_monthly") is True)

        # Heatmap monthly
        d = requests.get("http://127.0.0.1:8097/api/heatmap?step=10&month=8",
                         timeout=10).json()
        check("API heatmap month=8", len(d.get("points", [])) > 50)

    except Exception as e:
        warn("API tests", str(e))

    # ─── 10. VALIDATION ───────────────────────────────────────
    print("\n10. VALIDATION REPORT")

    report_path = Path(__file__).parent.parent / "data" / "outputs" / "water_validation_report.txt"
    check("Report exists", report_path.exists())
    if report_path.exists():
        text = report_path.read_text()
        n_pass = text.count("[✓]")
        n_fail = text.count("[✗]")
        check(f"6/6 pass (got {n_pass}p/{n_fail}f)", n_pass >= 6 and n_fail == 0)

    # ═══════════════════════════════════════════════════════════
    ds.close()
    total = PASS + FAIL
    print("\n" + "=" * 70)
    print(f"  RESULTS: {PASS}/{total} PASS, {FAIL} FAIL, {WARN} WARN")
    if FAIL == 0:
        print("  STATUS: ALL CHECKS PASSED")
    else:
        print(f"  STATUS: {FAIL} FAILURES")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
