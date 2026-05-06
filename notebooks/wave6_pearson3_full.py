"""
Vague 6 — pipeline complet H_local avec SPEI Pearson III + tests demandes.

1. Re-genere le SPEI (Pearson III) pour les 453 zones, a juillet 2022
   et aux lags t-1 (juin), t-2 (mai).
2. Recalcule H_v2 et H_v3 avec ce SPEI.
3. Tests :
   (a) sanity sur 30 zones aleatoires (pas seulement urbaines)
   (b) rho individuel par variable (BWS, GWS, SV, aridite, SPEI3, SPEI12) vs Propluvia
   (c) rho avec lag t-1, t-2 sur SPEI
   (d) v2 vs v3 recalcule
   (e) comparaison aridite vs SPEI

Outputs :
  data/wave6/zones_h_local_pearson3.csv
  data/wave6/wave6_report.md
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import geopandas as gpd
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Paths
ROOT = Path(__file__).resolve().parents[1]
ZONES_CSV = ROOT / "data" / "propluvia" / "zones_2022_07_15.csv"
ZONES_SHP = ROOT / "data" / "propluvia" / "all_zones.shp"
WM = ROOT / "water_model" / "data"
BWS_TIF = WM / "raw" / "aqueduct" / "bws_raw.tif"
GWS_TIF = WM / "raw" / "aqueduct" / "gws_raw.tif"
ERA5_NC = WM / "raw" / "era5" / "era5_monthly_1981_2022.nc"
AQ_GDB = WM / "Aqueduct40_waterrisk_download_Y2023M07D05" / "GDB" / "Aq40_Y2023D07M05.gdb"

OUT_DIR = ROOT / "data" / "wave6"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "zones_h_local_pearson3.csv"
OUT_REPORT = OUT_DIR / "wave6_report.md"
OUT_FIG = OUT_DIR / "wave6_summary.png"

# Constantes (verbatim HlocalV3)
BWS_MAX_FR, GWS_MAX_FR, SV_MAX_FR = 0.60, 0.15, 3.00
EPSILON_GEOM, C_ASYM, TAU_BOLTZ, BETA_0 = 0.05, 2.0, 5.0, 0.70
SPEI_BASELINE = (1981, 2010)
TARGET = (2022, 7)
LAG1 = (2022, 6)   # juin
LAG2 = (2022, 5)   # mai
W_V2 = {"bws": 0.50, "aridity": 0.36, "gws": 0.14}
W_V3 = {"bws": 0.532, "aridity": 0.185, "gws": 0.185, "sv": 0.097}

RNG = np.random.default_rng(42)
N_BOOT = 5000


# ----------------------------------------------------------------------
# Modele
# ----------------------------------------------------------------------
def f_asym(spei, c=C_ASYM):
    if np.isnan(spei): return np.nan
    if spei >= 0: return 0.0
    return float(-spei / (c + abs(spei)))

def s_struct_v2(x_bws, x_a, x_gws):
    comps = [(W_V2["bws"], x_bws), (W_V2["aridity"], x_a), (W_V2["gws"], x_gws)]
    valid = [(w, v) for w, v in comps if not np.isnan(v)]
    if not valid: return np.nan
    return sum(w*v for w, v in valid) / sum(w for w, _ in valid)

def s_struct_v3(x_bws, x_a, x_gws, x_sv):
    comps = [(W_V3["bws"], x_bws), (W_V3["aridity"], x_a),
             (W_V3["gws"], x_gws), (W_V3["sv"], x_sv)]
    valid = [(w, v) for w, v in comps if v is not None and not np.isnan(v)]
    if not valid: return np.nan
    w_total = sum(w for w, _ in valid)
    log_sum = sum(w * np.log(v + EPSILON_GEOM) for w, v in valid) / w_total
    return float(np.clip(np.exp(log_sum) - EPSILON_GEOM, 0, 1))

def s_conj_v2(d3, d12, tau=TAU_BOLTZ):
    if np.isnan(d3) and np.isnan(d12): return np.nan
    if np.isnan(d3): return d12
    if np.isnan(d12): return d3
    e3, e12 = np.exp(tau*d3), np.exp(tau*d12)
    return float((d3*e3 + d12*e12)/(e3 + e12))

def s_conj_v3(d3, d12):
    if np.isnan(d3) and np.isnan(d12): return np.nan
    if np.isnan(d3): return d12
    if np.isnan(d12): return d3
    return float(max(d3, d12))

def assemble(s_struct, s_conj, beta_0=BETA_0):
    if np.isnan(s_struct) and np.isnan(s_conj): return np.nan
    if np.isnan(s_struct): return s_conj
    if np.isnan(s_conj): return s_struct
    beta_t = beta_0*(1 - s_conj)
    return float(np.clip(beta_t*s_struct + (1-beta_t)*s_conj, 0, 1))


# ----------------------------------------------------------------------
# SPEI Pearson III (avec clip relache 1e-4)
# ----------------------------------------------------------------------
def D_acc_arr(D, w):
    n = len(D)
    out = np.full(n, np.nan)
    for i in range(w-1, n):
        out[i] = np.sum(D[i-w+1:i+1])
    return out

def fit_pearson3_monthly(D_acc, times):
    months = times.month
    mb = (times.year >= SPEI_BASELINE[0]) & (times.year <= SPEI_BASELINE[1])
    monthly = {}
    for m in range(1, 13):
        mask = (months == m) & mb & ~np.isnan(D_acc)
        base = D_acc[mask]
        if len(base) < 10:
            monthly[m] = None; continue
        try:
            sk, lo, sc = stats.pearson3.fit(base)
            monthly[m] = (sk, lo, sc)
        except Exception:
            monthly[m] = None
    return monthly

def spei_pearson3_at(D_acc, times, params, year, month):
    months = times.month
    idx = np.where((times.year == year) & (months == month))[0]
    if len(idx) == 0: return np.nan
    i = idx[0]
    if np.isnan(D_acc[i]): return np.nan
    p = params.get(months[i])
    if p is None: return np.nan
    sk, lo, sc = p
    cdf = stats.pearson3.cdf(D_acc[i], sk, loc=lo, scale=sc)
    cdf = float(np.clip(cdf, 1e-4, 1 - 1e-4))
    return float(stats.norm.ppf(cdf))


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def find_land_gridpoint(lat, lon, ds, td, threshold=0.60):
    lats, lons = ds.latitude.values, ds.longitude.values
    i_lat = int(np.argmin(np.abs(lats - lat)))
    i_lon = int(np.argmin(np.abs(lons - lon)))
    times = pd.DatetimeIndex(ds[td].values)
    month_idx = next(k for k in range(len(times)-1, -1, -1) if times[k].month == 7)
    lat_sl = slice(max(0, i_lat-1), min(len(lats), i_lat+2))
    lon_sl = slice(max(0, i_lon-1), min(len(lons), i_lon+2))
    pev_3x3 = ds["pev"].isel({td: month_idx, "latitude": lat_sl, "longitude": lon_sl})
    pet_3x3 = np.maximum(-pev_3x3.values.astype(float), 0.0)
    pet_center = pet_3x3[min(1, i_lat), min(1, i_lon)]
    pet_max = np.max(pet_3x3)
    ratio = pet_center/pet_max if pet_max > 0 else 1.0
    if ratio >= threshold:
        return float(lats[i_lat]), float(lons[i_lon])
    local_lats, local_lons = lats[lat_sl], lons[lon_sl]
    pos = np.unravel_index(np.argmax(pet_3x3), pet_3x3.shape)
    return float(local_lats[pos[0]]), float(local_lons[pos[1]])


def build_zones() -> pd.DataFrame:
    z = pd.read_csv(ZONES_CSV, dtype={"code_dept": str, "zone_id": str})
    g = gpd.read_file(ZONES_SHP)
    g["zone_id"] = g["id_zone"].astype(str)
    g = g[g["est_max_v"] == 1].drop_duplicates("zone_id")
    g["centroid"] = g.geometry.representative_point()
    g["lat"] = g["centroid"].y
    g["lon"] = g["centroid"].x
    merged = z.merge(g[["zone_id", "lat", "lon"]], on="zone_id", how="inner")
    return merged.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def extract_aqueduct(sites: pd.DataFrame) -> pd.DataFrame:
    coords = list(zip(sites["lon"], sites["lat"]))
    with rasterio.open(BWS_TIF) as src:
        bws = np.array([float(v[0]) for v in src.sample(coords)])
        if src.nodata is not None: bws[bws == src.nodata] = np.nan
    with rasterio.open(GWS_TIF) as src:
        gws = np.array([float(v[0]) for v in src.sample(coords)])
        if src.nodata is not None: gws[gws == src.nodata] = np.nan
    aq = gpd.read_file(AQ_GDB, layer="baseline_annual", columns=["sev_raw"])
    if aq.crs is None: aq = aq.set_crs(epsg=4326)
    elif aq.crs.to_epsg() != 4326: aq = aq.to_crs(epsg=4326)
    pts = gpd.GeoDataFrame(sites[["zone_id"]].copy(),
                            geometry=gpd.points_from_xy(sites["lon"], sites["lat"]),
                            crs="EPSG:4326")
    j = gpd.sjoin(pts, aq, how="left", predicate="within")
    miss = j["sev_raw"].isna()
    if miss.any():
        n = gpd.sjoin_nearest(pts.loc[miss, ["geometry"]],
                              aq[["geometry", "sev_raw"]], how="left",
                              max_distance=1.0)
        j.loc[miss, "sev_raw"] = n["sev_raw"].values
    sev = j.sort_index()["sev_raw"].to_numpy(dtype=float)
    out = sites.copy()
    out["bws_raw"] = bws; out["gws_raw"] = gws; out["sev_raw"] = sev
    return out


def build_era5_with_lags(sites: pd.DataFrame, ds, td) -> tuple[dict, pd.Series]:
    """
    Pour chaque cellule ERA5 unique : x_a + spei3 et spei12 a juillet, juin, mai 2022.
    """
    times = pd.DatetimeIndex(ds[td].values)
    days = times.days_in_month.values.astype(float)
    n_years = SPEI_BASELINE[1] - SPEI_BASELINE[0] + 1
    mb = (times.year >= SPEI_BASELINE[0]) & (times.year <= SPEI_BASELINE[1])

    print("  Mapping zones -> cellules ERA5 ...")
    cells = [find_land_gridpoint(lat, lon, ds, td)
             for lat, lon in zip(sites["lat"], sites["lon"])]
    sites_cells = pd.Series(cells, index=sites.index)
    unique_cells = list(set(cells))
    print(f"  {len(sites)} zones → {len(unique_cells)} cellules ERA5 uniques")

    cache = {}
    for k, (lat_g, lon_g) in enumerate(unique_cells, 1):
        sub = ds.sel(latitude=lat_g, longitude=lon_g, method="nearest")
        P = np.maximum(sub["tp"].values * days * 1000.0, 0.0)
        PET = np.maximum(-sub["pev"].values * days * 1000.0, 0.0)
        # aridite
        p_ann = np.sum(P[mb])/n_years
        pet_ann = np.sum(PET[mb])/n_years
        x_a = 0.0 if pet_ann <= 0 else float(1.0/(1.0 + (p_ann/pet_ann/0.65)**3))
        # SPEI Pearson III
        D = P - PET
        D3 = D_acc_arr(D, 3)
        D12 = D_acc_arr(D, 12)
        params3 = fit_pearson3_monthly(D3, times)
        params12 = fit_pearson3_monthly(D12, times)
        info = {"x_a": x_a}
        for tag, (yr, mo) in [("t", TARGET), ("lag1", LAG1), ("lag2", LAG2)]:
            info[f"spei3_{tag}"]  = spei_pearson3_at(D3,  times, params3,  yr, mo)
            info[f"spei12_{tag}"] = spei_pearson3_at(D12, times, params12, yr, mo)
        cache[(lat_g, lon_g)] = info
        if k % 30 == 0:
            print(f"  {k}/{len(unique_cells)} ...", flush=True)
    return cache, sites_cells


def assemble_table(sites, cache, sites_cells) -> pd.DataFrame:
    rows = []
    for i, r in sites.iterrows():
        cinfo = cache.get(sites_cells.iloc[i])
        if cinfo is None: continue
        x_bws = np.clip(r["bws_raw"]/BWS_MAX_FR, 0, 1) if not np.isnan(r["bws_raw"]) else np.nan
        x_gws = np.clip(r["gws_raw"]/GWS_MAX_FR, 0, 1) if not np.isnan(r["gws_raw"]) else np.nan
        x_sv  = np.clip(r["sev_raw"]/SV_MAX_FR,  0, 1) if not np.isnan(r["sev_raw"]) else np.nan
        x_a   = cinfo["x_a"]
        # H_local au temps t (juillet 2022) = ce qu'on garde pour comparer a Propluvia
        s3, s12 = cinfo["spei3_t"], cinfo["spei12_t"]
        d3, d12 = f_asym(s3), f_asym(s12)
        S2 = s_struct_v2(x_bws, x_a, x_gws)
        S3 = s_struct_v3(x_bws, x_a, x_gws, x_sv)
        C2 = s_conj_v2(d3, d12); C3 = s_conj_v3(d3, d12)
        rows.append({
            **r.to_dict(),
            "x_bws": x_bws, "x_gws": x_gws, "x_sv": x_sv, "x_a": x_a,
            "spei3_t":   s3, "spei12_t":   s12,
            "spei3_lag1": cinfo["spei3_lag1"], "spei12_lag1": cinfo["spei12_lag1"],
            "spei3_lag2": cinfo["spei3_lag2"], "spei12_lag2": cinfo["spei12_lag2"],
            "S_struct_v2": S2, "S_conj_v2": C2, "H_v2": assemble(S2, C2),
            "S_struct_v3": S3, "S_conj_v3": C3, "H_v3": assemble(S3, C3),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Analyses
# ----------------------------------------------------------------------
def boot_rho(H, P, n_boot=N_BOOT):
    n = len(H)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, _ = stats.spearmanr(H[idx], P[idx])
        out[i] = r
    return out[~np.isnan(out)]


def paired_delta(H1, H2, P, n_boot=N_BOOT):
    n = len(P)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1, _ = stats.spearmanr(H1[idx], P[idx])
            r2, _ = stats.spearmanr(H2[idx], P[idx])
        deltas[i] = r1 - r2
    return deltas[~np.isnan(deltas)]


def rho_with_ic(x, P, label):
    rho_pt, p = stats.spearmanr(x, P)
    rhos = boot_rho(np.asarray(x), np.asarray(P))
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return {"feature": label, "rho": rho_pt, "p": p,
            "ic_lo": lo, "ic_hi": hi}


def stability_test(df: pd.DataFrame, n=30, seed=42):
    """Sanity sur n zones aleatoires : SPEI doit avoir mediane~0, std~1."""
    rng = np.random.default_rng(seed)
    sample = df.sample(min(n, len(df)), random_state=seed)
    sp3 = sample["spei3_t"].dropna()
    sp12 = sample["spei12_t"].dropna()
    return {
        "n": len(sample),
        "spei3_med": float(np.median(sp3)),
        "spei3_std": float(np.std(sp3)),
        "spei3_sat_pct": 100*np.mean(sp3 <= -3.5),
        "spei12_med": float(np.median(sp12)),
        "spei12_std": float(np.std(sp12)),
        "spei12_sat_pct": 100*np.mean(sp12 <= -3.5),
    }


def main():
    if OUT_CSV.exists():
        print(f"Reuse {OUT_CSV} (delete pour recomputer)")
        df = pd.read_csv(OUT_CSV, dtype={"code_dept": str, "zone_id": str})
    else:
        print("Build zones ...")
        sites = build_zones()
        sites = extract_aqueduct(sites)
        print("Open ERA5 ...")
        ds = xr.open_dataset(ERA5_NC)
        td = "valid_time" if "valid_time" in ds.dims else "time"
        cache, sites_cells = build_era5_with_lags(sites, ds, td)
        df = assemble_table(sites, cache, sites_cells)
        df.to_csv(OUT_CSV, index=False)
        print(f"Ecrit : {OUT_CSV}  ({len(df)} lignes)")

    sub = df.dropna(subset=["H_v2", "H_v3", "niveau_int", "spei3_t", "spei12_t"]).copy()
    n = len(sub)
    P = sub["niveau_int"].to_numpy()

    # 1. Sanity SPEI
    sanity = stability_test(sub, n=30)
    print(f"\n=== SANITY 30 zones aleatoires ===")
    for k, v in sanity.items():
        print(f"  {k:<22} {v}")

    # 2. rho individuel par feature (juillet 2022)
    print("\n=== rho individuel par feature (Pearson III) ===")
    feats = []
    for col in ["x_bws", "x_a", "x_gws", "x_sv",
                "spei3_t",  "spei12_t",
                "spei3_lag1", "spei12_lag1",
                "spei3_lag2", "spei12_lag2"]:
        x = sub[col].to_numpy()
        valid = ~np.isnan(x)
        if valid.sum() < 10: continue
        # SPEI : on veut + de stress -> + de niveau, donc on inverse le signe
        sign = -1 if col.startswith("spei") else 1
        info = rho_with_ic(sign*x[valid], P[valid], col + (" (inverted)" if sign==-1 else ""))
        feats.append(info)
    feats_df = pd.DataFrame(feats)
    print(feats_df.round(3).to_string(index=False))

    # 3. v2 vs v3 recalcule
    print("\n=== v2 vs v3 (SPEI Pearson III) ===")
    H2, H3 = sub["H_v2"].to_numpy(), sub["H_v3"].to_numpy()
    rho2_pt, p2 = stats.spearmanr(H2, P); rho3_pt, p3 = stats.spearmanr(H3, P)
    rhos2 = boot_rho(H2, P); rhos3 = boot_rho(H3, P)
    deltas = paired_delta(H2, H3, P)
    lo2, hi2 = np.percentile(rhos2, [2.5, 97.5])
    lo3, hi3 = np.percentile(rhos3, [2.5, 97.5])
    d_med = float(np.median(deltas))
    d_lo, d_hi = np.percentile(deltas, [2.5, 97.5])
    print(f"v2 : rho={rho2_pt:+.3f} (p={p2:.2e})  IC95 [{lo2:+.3f}, {hi2:+.3f}]")
    print(f"v3 : rho={rho3_pt:+.3f} (p={p3:.2e})  IC95 [{lo3:+.3f}, {hi3:+.3f}]")
    print(f"Delta v2-v3 = {d_med:+.3f}  IC95 [{d_lo:+.3f}, {d_hi:+.3f}]  zero in: {d_lo<=0<=d_hi}")

    # 4. Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0,0]
    ax.hist(sub["spei12_t"].dropna(), bins=40, color="#0277BD", alpha=0.7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"SPEI-12 juillet 2022 (Pearson III) — n={n}")
    ax.grid(alpha=0.3)

    ax = axes[0,1]
    feats_df_plot = feats_df.set_index("feature")
    y = np.arange(len(feats_df_plot))
    ax.errorbar(feats_df_plot["rho"], y,
                xerr=[feats_df_plot["rho"]-feats_df_plot["ic_lo"],
                      feats_df_plot["ic_hi"]-feats_df_plot["rho"]],
                fmt="o", capsize=3, color="#0277BD")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(feats_df_plot.index, fontsize=8)
    ax.set_xlabel("rho Spearman vs Propluvia (IC95)")
    ax.set_title("Feature individuelles (lag t, t-1, t-2)")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1,0]
    ax.hist(rhos2, bins=40, alpha=0.55, label=f"v2 (med={np.median(rhos2):+.3f})", color="#4FC3F7")
    ax.hist(rhos3, bins=40, alpha=0.55, label=f"v3 (med={np.median(rhos3):+.3f})", color="#AB47BC")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Bootstrap rho v2 vs v3"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1,1]
    ax.hist(deltas, bins=40, color="#FFB74D", alpha=0.85)
    ax.axvline(0, color="k", lw=1.0)
    ax.axvline(d_med, color="#E65100", lw=1.2, ls="--", label=f"med={d_med:+.3f}")
    ax.axvspan(d_lo, d_hi, alpha=0.15, color="#FFB74D",
               label=f"IC95 [{d_lo:+.3f}, {d_hi:+.3f}]")
    ax.set_title("Delta rho v2-v3"); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130)
    plt.close(fig)

    # 5. Rapport
    lines = [
        f"# Vague 6 — H_local avec SPEI Pearson III (clip 1e-4)\n",
        f"- n zones valides : **{n}**",
        f"- bootstrap : {N_BOOT} replicats, seed=42",
        "",
        "## Sanity SPEI (30 zones aleatoires)\n",
        f"- spei3 mediane = {sanity['spei3_med']:+.3f}, std = {sanity['spei3_std']:.3f}, saturation_bas = {sanity['spei3_sat_pct']:.1f}%",
        f"- spei12 mediane = {sanity['spei12_med']:+.3f}, std = {sanity['spei12_std']:.3f}, saturation_bas = {sanity['spei12_sat_pct']:.1f}%",
        "",
        "## rho individuel par feature\n",
        "| feature | rho | p | IC95 lo | IC95 hi |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in feats:
        lines.append(f"| {r['feature']} | {r['rho']:+.3f} | {r['p']:.2e} | {r['ic_lo']:+.3f} | {r['ic_hi']:+.3f} |")
    lines += [
        "",
        "## v2 vs v3 (SPEI Pearson III)\n",
        f"- v2 : rho_pt = {rho2_pt:+.3f} (p={p2:.2e}), IC95 = [{lo2:+.3f}, {hi2:+.3f}]",
        f"- v3 : rho_pt = {rho3_pt:+.3f} (p={p3:.2e}), IC95 = [{lo3:+.3f}, {hi3:+.3f}]",
        f"- Delta v2-v3 = {d_med:+.3f}, IC95 = [{d_lo:+.3f}, {d_hi:+.3f}]",
        f"- Zero dans IC95 : {'OUI' if d_lo<=0<=d_hi else 'NON'}",
    ]
    OUT_REPORT.write_text("\n".join(lines))
    print(f"\nRapport : {OUT_REPORT}")
    print(f"Figure  : {OUT_FIG}")


if __name__ == "__main__":
    main()
