"""
Vague 3 — H_local sur N=1110 zones d'alerte (vs N=86 en vague 2).

Resolution beaucoup plus fine que le departement : on prend le centroide
de chaque zone d'alerte Propluvia comme point d'analyse.

Optimisations :
  - SPEI/aridite caches par cellule ERA5 (28 km) : <100 cellules pour 1110 zones
  - Aqueduct GDB charge une fois, sjoin global en une passe

Inputs :
  data/propluvia/zones_2022_07_15.csv   (1110 zones + niveau)
  data/propluvia/all_zones.shp          (geometries 8535 zones)
  water_model/data/raw/aqueduct/{bws,gws}_raw.tif
  water_model/data/Aqueduct40_*/GDB/    (sev_raw)
  water_model/data/raw/era5/era5_monthly_1981_2022.nc

Outputs :
  data/wave3/zones_h_local_2022.csv
  data/wave3/wave3_report.md
  data/wave3/wave3_bootstrap.png
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

# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ZONES_CSV = ROOT / "data" / "propluvia" / "zones_2022_07_15.csv"
ZONES_SHP = ROOT / "data" / "propluvia" / "all_zones.shp"
WM = ROOT / "water_model" / "data"
BWS_TIF = WM / "raw" / "aqueduct" / "bws_raw.tif"
GWS_TIF = WM / "raw" / "aqueduct" / "gws_raw.tif"
ERA5_NC = WM / "raw" / "era5" / "era5_monthly_1981_2022.nc"
AQ_GDB = WM / "Aqueduct40_waterrisk_download_Y2023M07D05" / "GDB" / "Aq40_Y2023D07M05.gdb"

OUT_DIR = ROOT / "data" / "wave3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "zones_h_local_2022.csv"
OUT_REPORT = OUT_DIR / "wave3_report.md"
OUT_FIG = OUT_DIR / "wave3_bootstrap.png"

# Constantes (verbatim HlocalV3)
BWS_MAX_FR, GWS_MAX_FR, SV_MAX_FR = 0.60, 0.15, 3.00
EPSILON_GEOM = 0.05
C_ASYM = 2.0
TAU_BOLTZ = 5.0
BETA_0 = 0.70
SPEI_BASELINE = (1981, 2010)
TARGET_YEAR, TARGET_MONTH = 2022, 7
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
    return sum(w * v for w, v in valid) / sum(w for w, _ in valid)


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
    e3, e12 = np.exp(tau * d3), np.exp(tau * d12)
    return float((d3 * e3 + d12 * e12) / (e3 + e12))


def s_conj_v3(d3, d12):
    if np.isnan(d3) and np.isnan(d12): return np.nan
    if np.isnan(d3): return d12
    if np.isnan(d12): return d3
    return float(max(d3, d12))


def assemble_H(s_struct, s_conj, beta_0=BETA_0):
    if np.isnan(s_struct) and np.isnan(s_conj): return np.nan
    if np.isnan(s_struct): return s_conj
    if np.isnan(s_conj): return s_struct
    beta_t = beta_0 * (1 - s_conj)
    return float(np.clip(beta_t * s_struct + (1 - beta_t) * s_conj, 0, 1))


# ----------------------------------------------------------------------
# Step 1 : charger zones + geometries → centroides
# ----------------------------------------------------------------------
def build_zone_points() -> pd.DataFrame:
    z = pd.read_csv(ZONES_CSV, dtype={"code_dept": str, "zone_id": str})
    print(f"Zones cibles : {len(z)}")
    g = gpd.read_file(ZONES_SHP)
    g["zone_id"] = g["id_zone"].astype(str)
    g = g[g["est_max_v"] == 1]                   # garde la version max
    g = g.drop_duplicates("zone_id")
    g["centroid"] = g.geometry.representative_point()
    g["lat"] = g["centroid"].y
    g["lon"] = g["centroid"].x

    merged = z.merge(g[["zone_id", "lat", "lon"]], on="zone_id", how="left")
    miss = merged["lat"].isna().sum()
    print(f"Centroides obtenus : {(~merged['lat'].isna()).sum()} / {len(merged)} (drop {miss})")
    merged = merged.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return merged


# ----------------------------------------------------------------------
# Step 2 : extractions Aqueduct
# ----------------------------------------------------------------------
def extract_aqueduct(sites: pd.DataFrame) -> pd.DataFrame:
    print("Extraction BWS/GWS (rasters TIF) ...")
    coords = list(zip(sites["lon"], sites["lat"]))
    with rasterio.open(BWS_TIF) as src:
        bws = np.array([float(v[0]) for v in src.sample(coords)])
        if src.nodata is not None:
            bws[bws == src.nodata] = np.nan
    with rasterio.open(GWS_TIF) as src:
        gws = np.array([float(v[0]) for v in src.sample(coords)])
        if src.nodata is not None:
            gws[gws == src.nodata] = np.nan

    print("Extraction SV (Aqueduct GDB) ...")
    aq = gpd.read_file(AQ_GDB, layer="baseline_annual", columns=["sev_raw"])
    if aq.crs is None:
        aq = aq.set_crs(epsg=4326)
    elif aq.crs.to_epsg() != 4326:
        aq = aq.to_crs(epsg=4326)
    pts = gpd.GeoDataFrame(
        sites[["zone_id"]].copy(),
        geometry=gpd.points_from_xy(sites["lon"], sites["lat"]),
        crs="EPSG:4326",
    )
    j = gpd.sjoin(pts, aq, how="left", predicate="within")
    miss = j["sev_raw"].isna()
    if miss.any():
        n = gpd.sjoin_nearest(pts.loc[miss, ["geometry"]],
                              aq[["geometry", "sev_raw"]], how="left",
                              max_distance=1.0)
        j.loc[miss, "sev_raw"] = n["sev_raw"].values
    sev = j.sort_index()["sev_raw"].to_numpy(dtype=float)

    sites = sites.copy()
    sites["bws_raw"] = bws
    sites["gws_raw"] = gws
    sites["sev_raw"] = sev
    return sites


# ----------------------------------------------------------------------
# Step 3 : ERA5 cache par cellule
# ----------------------------------------------------------------------
def find_land_gridpoint(lat, lon, ds, td, threshold=0.60):
    lats, lons = ds.latitude.values, ds.longitude.values
    i_lat = int(np.argmin(np.abs(lats - lat)))
    i_lon = int(np.argmin(np.abs(lons - lon)))
    times = pd.DatetimeIndex(ds[td].values)
    month_idx = next(k for k in range(len(times) - 1, -1, -1) if times[k].month == 7)
    lat_sl = slice(max(0, i_lat - 1), min(len(lats), i_lat + 2))
    lon_sl = slice(max(0, i_lon - 1), min(len(lons), i_lon + 2))
    pev_3x3 = ds["pev"].isel({td: month_idx, "latitude": lat_sl, "longitude": lon_sl})
    pet_3x3 = np.maximum(-pev_3x3.values.astype(float), 0.0)
    pet_center = pet_3x3[min(1, i_lat), min(1, i_lon)]
    pet_max = np.max(pet_3x3)
    ratio = pet_center / pet_max if pet_max > 0 else 1.0
    if ratio >= threshold:
        return float(lats[i_lat]), float(lons[i_lon])
    local_lats, local_lons = lats[lat_sl], lons[lon_sl]
    max_pos = np.unravel_index(np.argmax(pet_3x3), pet_3x3.shape)
    return float(local_lats[max_pos[0]]), float(local_lons[max_pos[1]])


def compute_spei_series(D_acc: np.ndarray, dates: pd.DatetimeIndex, monthly_params: dict):
    n = len(D_acc)
    spei = np.full(n, np.nan)
    months = dates.month
    for i in range(n):
        if np.isnan(D_acc[i]):
            continue
        params = monthly_params.get(months[i])
        if params is None:
            continue
        c, scale, shift = params
        cdf = stats.fisk.cdf(D_acc[i] + shift, c, scale=scale)
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        spei[i] = stats.norm.ppf(cdf)
    return spei


def fit_monthly(D_acc: np.ndarray, dates: pd.DatetimeIndex):
    months = dates.month
    mask_base = (dates.year >= SPEI_BASELINE[0]) & (dates.year <= SPEI_BASELINE[1])
    monthly = {}
    for m in range(1, 13):
        mask = (months == m) & mask_base & ~np.isnan(D_acc)
        base = D_acc[mask]
        if len(base) < 10:
            monthly[m] = None
            continue
        shift = -np.min(base) + 1.0
        try:
            c, _, scale = stats.fisk.fit(base + shift, floc=0)
            monthly[m] = (c, scale, shift)
        except Exception:
            monthly[m] = None
    return monthly


def build_era5_cache(sites: pd.DataFrame, ds, td) -> dict:
    """
    Pour chaque cellule ERA5 land utilisee, calcule x_a, spei3_jul2022, spei12_jul2022.
    Renvoie {(lat_g, lon_g): {x_a, spei3, spei12}}.
    """
    times = pd.DatetimeIndex(ds[td].values)
    days = times.days_in_month.values.astype(float)
    n_years = SPEI_BASELINE[1] - SPEI_BASELINE[0] + 1
    mask_base = (times.year >= SPEI_BASELINE[0]) & (times.year <= SPEI_BASELINE[1])

    # 1. Mapper chaque site -> cellule terrestre
    print("  Mapping zones -> cellules ERA5 ...")
    cells = []
    for lat, lon in zip(sites["lat"], sites["lon"]):
        cells.append(find_land_gridpoint(lat, lon, ds, td))
    sites_cells = pd.Series(cells, index=sites.index)
    unique_cells = list(set(cells))
    print(f"  {len(sites)} sites → {len(unique_cells)} cellules ERA5 uniques")

    cache = {}
    for k, (lat_g, lon_g) in enumerate(unique_cells, 1):
        sub = ds.sel(latitude=lat_g, longitude=lon_g, method="nearest")
        P = np.maximum(sub["tp"].values * days * 1000.0, 0.0)
        PET = np.maximum(-sub["pev"].values * days * 1000.0, 0.0)
        # Aridite
        p_ann = np.sum(P[mask_base]) / n_years
        pet_ann = np.sum(PET[mask_base]) / n_years
        if pet_ann <= 0:
            x_a = 0.0
        else:
            ai = p_ann / pet_ann
            x_a = float(1.0 / (1.0 + (ai / 0.65) ** 3.0))
        # SPEI
        D = P - PET
        n = len(D)
        D3 = np.full(n, np.nan); D12 = np.full(n, np.nan)
        for i in range(2, n):  D3[i]  = np.sum(D[i-2:i+1])
        for i in range(11, n): D12[i] = np.sum(D[i-11:i+1])
        params3 = fit_monthly(D3, times)
        params12 = fit_monthly(D12, times)
        spei3 = compute_spei_series(D3, times, params3)
        spei12 = compute_spei_series(D12, times, params12)
        idx = np.where((times.year == TARGET_YEAR) & (times.month == TARGET_MONTH))[0]
        if len(idx) == 0:
            cache[(lat_g, lon_g)] = None
        else:
            i = idx[0]
            cache[(lat_g, lon_g)] = {"x_a": x_a, "spei3": spei3[i], "spei12": spei12[i]}
        if k % 10 == 0:
            print(f"  {k}/{len(unique_cells)} cellules ...", flush=True)
    return cache, sites_cells


# ----------------------------------------------------------------------
# Step 4 : assemble H par zone
# ----------------------------------------------------------------------
def compute_H_table(sites: pd.DataFrame, era5_cache: dict, sites_cells: pd.Series):
    rows = []
    for i, r in sites.iterrows():
        cell = sites_cells.iloc[i]
        cinfo = era5_cache.get(cell)
        if cinfo is None:
            continue
        x_bws = np.clip(r["bws_raw"] / BWS_MAX_FR, 0, 1) if not np.isnan(r["bws_raw"]) else np.nan
        x_gws = np.clip(r["gws_raw"] / GWS_MAX_FR, 0, 1) if not np.isnan(r["gws_raw"]) else np.nan
        x_sv = np.clip(r["sev_raw"] / SV_MAX_FR, 0, 1) if not np.isnan(r["sev_raw"]) else np.nan
        x_a = cinfo["x_a"]
        s3, s12 = cinfo["spei3"], cinfo["spei12"]
        d3, d12 = f_asym(s3), f_asym(s12)
        S2 = s_struct_v2(x_bws, x_a, x_gws)
        S3 = s_struct_v3(x_bws, x_a, x_gws, x_sv)
        C2 = s_conj_v2(d3, d12)
        C3 = s_conj_v3(d3, d12)
        rows.append({
            **r.to_dict(),
            "x_bws": x_bws, "x_gws": x_gws, "x_sv": x_sv, "x_a": x_a,
            "spei3": s3, "spei12": s12,
            "S_struct_v2": S2, "S_conj_v2": C2, "H_v2": assemble_H(S2, C2),
            "S_struct_v3": S3, "S_conj_v3": C3, "H_v3": assemble_H(S3, C3),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Bootstrap analyse
# ----------------------------------------------------------------------
def bootstrap_rho(H, P, n_boot=N_BOOT):
    n = len(H)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, _ = stats.spearmanr(H[idx], P[idx])
        rhos[i] = r
    return rhos[~np.isnan(rhos)]


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


def analyze(df: pd.DataFrame):
    sub = df.dropna(subset=["H_v2", "H_v3", "niveau_int"]).copy()
    n = len(sub)
    H2 = sub["H_v2"].to_numpy()
    H3 = sub["H_v3"].to_numpy()
    P = sub["niveau_int"].to_numpy()

    rho2_pt, p2 = stats.spearmanr(H2, P)
    rho3_pt, p3 = stats.spearmanr(H3, P)
    rhos2 = bootstrap_rho(H2, P)
    rhos3 = bootstrap_rho(H3, P)
    deltas = paired_delta(H2, H3, P)

    lo2, hi2 = np.percentile(rhos2, [2.5, 97.5])
    lo3, hi3 = np.percentile(rhos3, [2.5, 97.5])
    d_med = float(np.median(deltas))
    d_lo, d_hi = np.percentile(deltas, [2.5, 97.5])
    zero_in = (d_lo <= 0 <= d_hi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.hist(rhos2, bins=50, alpha=0.55, label=f"v2 (med={np.median(rhos2):+.3f})", color="#4FC3F7")
    ax.hist(rhos3, bins=50, alpha=0.55, label=f"v3 (med={np.median(rhos3):+.3f})", color="#AB47BC")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"rho_Spearman vs Propluvia, n={n}")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.hist(deltas, bins=50, color="#FFB74D", alpha=0.85)
    ax.axvline(0, color="k", lw=1.0)
    ax.axvline(d_med, color="#E65100", lw=1.2, ls="--", label=f"med={d_med:+.3f}")
    ax.axvspan(d_lo, d_hi, alpha=0.15, color="#FFB74D",
               label=f"IC95 [{d_lo:+.3f}, {d_hi:+.3f}]")
    ax.set_title("Delta rho v2-v3 (apparie)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130)
    plt.close(fig)

    lines = [
        f"# Vague 3 — H_local sur N={n} zones d'alerte\n",
        f"- bootstrap : {N_BOOT} replicats, seed=42",
        f"- date : 15 juillet 2022",
        "",
        "## Concordance Propluvia (zone par zone)\n",
        "| Modele | rho ponctuel | mediane | IC95 | p-value |",
        "|---|---:|---:|:---:|---:|",
        f"| v2 | {rho2_pt:+.3f} | {np.median(rhos2):+.3f} | [{lo2:+.3f}, {hi2:+.3f}] | {p2:.4g} |",
        f"| v3 | {rho3_pt:+.3f} | {np.median(rhos3):+.3f} | [{lo3:+.3f}, {hi3:+.3f}] | {p3:.4g} |",
        "",
        "## Difference appariee\n",
        f"- Delta rho mediane : **{d_med:+.3f}**",
        f"- IC95 : [{d_lo:+.3f}, {d_hi:+.3f}]",
        f"- Zero dans IC95 : **{'OUI' if zero_in else 'NON'}**",
    ]
    OUT_REPORT.write_text("\n".join(lines))

    print("\n=== RESULTATS VAGUE 3 ===")
    print(f"n = {n}")
    print(f"rho v2 = {rho2_pt:+.3f} (p={p2:.4g}) | IC95 [{lo2:+.3f}, {hi2:+.3f}]")
    print(f"rho v3 = {rho3_pt:+.3f} (p={p3:.4g}) | IC95 [{lo3:+.3f}, {hi3:+.3f}]")
    print(f"Delta = {d_med:+.3f} | IC95 [{d_lo:+.3f}, {d_hi:+.3f}] | zero in: {zero_in}")
    print(f"Rapport : {OUT_REPORT}")
    print(f"Figure  : {OUT_FIG}")


def main():
    if OUT_CSV.exists():
        print(f"Reuse {OUT_CSV} (delete pour recomputer)")
        df = pd.read_csv(OUT_CSV, dtype={"code_dept": str, "zone_id": str})
        analyze(df)
        return
    sites = build_zone_points()
    sites = extract_aqueduct(sites)
    print("Chargement ERA5 ...")
    ds = xr.open_dataset(ERA5_NC)
    td = "valid_time" if "valid_time" in ds.dims else "time"
    cache, sites_cells = build_era5_cache(sites, ds, td)
    df = compute_H_table(sites, cache, sites_cells)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nEcrit : {OUT_CSV}  ({len(df)} lignes)")
    analyze(df)


if __name__ == "__main__":
    main()
