"""
Vague 5 — debug du calcul SPEI.

Diagnostic prealable a toute correction du modele.
On observe que >50% des SPEI sont satures a -4.75 = inv_normal(1e-6).

Etapes :
  1. Plotter D_acc (deficit cumule P-PET) pour 5 zones representatives
  2. Comparer 3 methodes :
       (a) Fisk (log-logistique) — methode actuelle
       (b) ECDF (rank-based, sans fit parametrique)
       (c) Pearson III (recommande Vicente-Serrano 2010)
  3. Quantifier la saturation pour chaque methode
  4. Sanity check : mediane SPEI ~ 0, std ~ 1, symetrique
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
ERA5_NC = ROOT / "water_model" / "data" / "raw" / "era5" / "era5_monthly_1981_2022.nc"
OUT_DIR = ROOT / "data" / "wave5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPEI_BASELINE = (1981, 2010)
TARGET_YEAR, TARGET_MONTH = 2022, 7

# 5 zones representatives, gradient climat
ZONES = {
    "Brest":      (48.39, -4.49),   # oceanique tres humide
    "Paris":      (48.85, 2.35),    # tempere
    "Strasbourg": (48.58, 7.75),    # semi-continental
    "Lyon":       (45.76, 4.83),    # continental
    "Marseille":  (43.30, 5.37),    # mediterraneen sec
}


# ----------------------------------------------------------------------
def load_era5_series(lat, lon):
    ds = xr.open_dataset(ERA5_NC)
    td = "valid_time" if "valid_time" in ds.dims else "time"
    sub = ds.sel(latitude=lat, longitude=lon, method="nearest")
    times = pd.DatetimeIndex(sub[td].values)
    days = times.days_in_month.values.astype(float)
    P = np.maximum(sub["tp"].values * days * 1000.0, 0.0)        # mm/mois
    PET = np.maximum(-sub["pev"].values * days * 1000.0, 0.0)
    return P, PET, times


def D_acc(P, PET, window):
    D = P - PET
    n = len(D)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = np.sum(D[i - window + 1:i + 1])
    return out


def baseline_mask(times):
    return ((times.year >= SPEI_BASELINE[0]) &
            (times.year <= SPEI_BASELINE[1]))


# --- Methode A : Fisk (actuel)
def spei_fisk(D_acc_arr, times):
    months = times.month
    mb = baseline_mask(times)
    monthly = {}
    for m in range(1, 13):
        mask = (months == m) & mb & ~np.isnan(D_acc_arr)
        base = D_acc_arr[mask]
        if len(base) < 10:
            monthly[m] = None; continue
        shift = -np.min(base) + 1.0
        try:
            c, _, scale = stats.fisk.fit(base + shift, floc=0)
            monthly[m] = (c, scale, shift)
        except Exception:
            monthly[m] = None
    spei = np.full(len(D_acc_arr), np.nan)
    for i in range(len(D_acc_arr)):
        if np.isnan(D_acc_arr[i]): continue
        p = monthly.get(months[i])
        if p is None: continue
        c, scale, shift = p
        cdf = stats.fisk.cdf(D_acc_arr[i] + shift, c, scale=scale)
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        spei[i] = stats.norm.ppf(cdf)
    return spei


# --- Methode B : ECDF (rank-based, non-parametrique)
def spei_ecdf(D_acc_arr, times):
    months = times.month
    mb = baseline_mask(times)
    spei = np.full(len(D_acc_arr), np.nan)
    for m in range(1, 13):
        mask_base = (months == m) & mb & ~np.isnan(D_acc_arr)
        base = D_acc_arr[mask_base]
        if len(base) < 10: continue
        sorted_base = np.sort(base)
        n_b = len(sorted_base)
        mask_target = (months == m) & ~np.isnan(D_acc_arr)
        for i in np.where(mask_target)[0]:
            # rank avec position medianne en cas d'egalite
            rank = np.searchsorted(sorted_base, D_acc_arr[i], side="right")
            cdf = (rank + 0.5) / (n_b + 1)
            cdf = np.clip(cdf, 1e-4, 1 - 1e-4)   # plancher relache
            spei[i] = stats.norm.ppf(cdf)
    return spei


# --- Methode C : Pearson III (standard scientifique SPEI)
def spei_pearson3(D_acc_arr, times):
    months = times.month
    mb = baseline_mask(times)
    monthly = {}
    for m in range(1, 13):
        mask = (months == m) & mb & ~np.isnan(D_acc_arr)
        base = D_acc_arr[mask]
        if len(base) < 10:
            monthly[m] = None; continue
        try:
            skew, loc, scale = stats.pearson3.fit(base)
            monthly[m] = (skew, loc, scale)
        except Exception:
            monthly[m] = None
    spei = np.full(len(D_acc_arr), np.nan)
    for i in range(len(D_acc_arr)):
        if np.isnan(D_acc_arr[i]): continue
        p = monthly.get(months[i])
        if p is None: continue
        skew, loc, scale = p
        cdf = stats.pearson3.cdf(D_acc_arr[i], skew, loc=loc, scale=scale)
        cdf = np.clip(cdf, 1e-4, 1 - 1e-4)
        spei[i] = stats.norm.ppf(cdf)
    return spei


# ----------------------------------------------------------------------
def sanity(spei, label):
    base = spei[~np.isnan(spei)]
    sat_low = np.mean(base <= -3.5)
    sat_hi = np.mean(base >= 3.5)
    return {
        "method": label,
        "mean": float(np.nanmean(spei)),
        "median": float(np.nanmedian(spei)),
        "std": float(np.nanstd(spei)),
        "min": float(np.nanmin(spei)),
        "max": float(np.nanmax(spei)),
        "sat_low_pct": 100 * sat_low,
        "sat_high_pct": 100 * sat_hi,
    }


def main():
    print("Chargement ERA5 ...")
    rows = []
    fig, axes = plt.subplots(len(ZONES), 4, figsize=(18, 3.2 * len(ZONES)))

    for irow, (name, (lat, lon)) in enumerate(ZONES.items()):
        P, PET, times = load_era5_series(lat, lon)
        D12 = D_acc(P, PET, 12)
        D3 = D_acc(P, PET, 3)
        # On debug sur SPEI-12 (le plus stable)
        sp_fisk = spei_fisk(D12, times)
        sp_ecdf = spei_ecdf(D12, times)
        sp_p3 = spei_pearson3(D12, times)

        for label, sp in [("Fisk", sp_fisk), ("ECDF", sp_ecdf), ("Pearson3", sp_p3)]:
            s = sanity(sp, f"{name} | {label} SPEI-12")
            rows.append(s)

        # Plot
        ax = axes[irow, 0]
        ax.plot(times, D12, color="#0277BD", lw=0.8)
        ax.set_title(f"{name} : D_acc 12 mois")
        ax.grid(alpha=0.3)
        ax.set_ylabel("mm")

        ax = axes[irow, 1]
        base = D12[baseline_mask(times) & ~np.isnan(D12)]
        ax.hist(base, bins=40, color="#0277BD", alpha=0.7)
        ax.set_title(f"{name} : distribution baseline 1981-2010")
        ax.grid(alpha=0.3)

        ax = axes[irow, 2]
        for label, sp, c in [("Fisk", sp_fisk, "#E65100"),
                             ("ECDF", sp_ecdf, "#0277BD"),
                             ("Pearson3", sp_p3, "#388E3C")]:
            ax.hist(sp[~np.isnan(sp)], bins=40, alpha=0.45, label=label, color=c)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{name} : distribution SPEI-12 selon methode")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[irow, 3]
        sp_target = {
            "Fisk":      sp_fisk[(times.year == TARGET_YEAR) & (times.month == TARGET_MONTH)],
            "ECDF":      sp_ecdf[(times.year == TARGET_YEAR) & (times.month == TARGET_MONTH)],
            "Pearson3":  sp_p3[(times.year == TARGET_YEAR) & (times.month == TARGET_MONTH)],
        }
        ax.bar(list(sp_target.keys()),
               [float(v[0]) if len(v) else np.nan for v in sp_target.values()],
               color=["#E65100", "#0277BD", "#388E3C"])
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{name} : SPEI-12 a juillet 2022")
        ax.set_ylabel("SPEI")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig_path = OUT_DIR / "spei_debug.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df_path = OUT_DIR / "spei_sanity.csv"
    df.to_csv(df_path, index=False)

    print("\n=== SANITY CHECK PAR METHODE (5 zones x 3 methodes) ===\n")
    print(df.round(3).to_string(index=False))
    print(f"\nFigure : {fig_path}")
    print(f"CSV    : {df_path}")

    # Diagnostic global agrege
    print("\n=== AGREGATION PAR METHODE ===\n")
    print(df.groupby("method").agg(
        median=("median", "mean"),
        mean=("mean", "mean"),
        std=("std", "mean"),
        sat_low_pct=("sat_low_pct", "mean"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
