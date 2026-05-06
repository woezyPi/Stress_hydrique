"""
Vague 2 — etape 3 : calcul H_v2 et H_v3 sur les 86 prefectures
+ bootstrap rho_Spearman vs Propluvia 2022-07-15.

Reutilise les formules de HlocalV3.ipynb (re-codees ici pour
isolation et reproductibilite).

Inputs :
  data/propluvia/sites_dept_2022.csv       (86 prefectures + niveau)
  water_model/data/raw/aqueduct/bws_raw.tif
  water_model/data/raw/aqueduct/gws_raw.tif
  water_model/data/Aqueduct40_*/GDB/Aq40*.gdb (pour sev_raw)
  water_model/data/raw/era5/era5_monthly_1981_2022.nc

Outputs :
  data/wave2/sites_h_local_2022.csv
  data/wave2/wave2_report.md
  data/wave2/wave2_bootstrap.png
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SITES_CSV = ROOT / "data" / "propluvia" / "sites_dept_2022.csv"
WM = ROOT / "water_model" / "data"
BWS_TIF = WM / "raw" / "aqueduct" / "bws_raw.tif"
GWS_TIF = WM / "raw" / "aqueduct" / "gws_raw.tif"
ERA5_NC = WM / "raw" / "era5" / "era5_monthly_1981_2022.nc"
AQ_GDB = WM / "Aqueduct40_waterrisk_download_Y2023M07D05" / "GDB" / "Aq40_Y2023D07M05.gdb"

OUT_DIR = ROOT / "data" / "wave2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "sites_h_local_2022.csv"
OUT_REPORT = OUT_DIR / "wave2_report.md"
OUT_FIG = OUT_DIR / "wave2_bootstrap.png"

# ----------------------------------------------------------------------
# Constantes du modele (identiques a HlocalV3)
# ----------------------------------------------------------------------
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
# Modele H_local (reprise verbatim de HlocalV3)
# ----------------------------------------------------------------------
def f_asym(spei, c=C_ASYM):
    if np.isnan(spei): return np.nan
    if spei >= 0: return 0.0
    return float(-spei / (c + abs(spei)))


def s_struct_v2_arith(x_bws, x_a, x_gws):
    comps = [(W_V2["bws"], x_bws), (W_V2["aridity"], x_a), (W_V2["gws"], x_gws)]
    valid = [(w, v) for w, v in comps if not np.isnan(v)]
    if not valid: return np.nan
    num = sum(w * v for w, v in valid)
    den = sum(w for w, _ in valid)
    return num / den


def s_struct_v3_geom(x_bws, x_a, x_gws, x_sv):
    comps = [(W_V3["bws"], x_bws), (W_V3["aridity"], x_a),
             (W_V3["gws"], x_gws), (W_V3["sv"], x_sv)]
    valid = [(w, v) for w, v in comps if v is not None and not np.isnan(v)]
    if not valid: return np.nan
    w_total = sum(w for w, _ in valid)
    log_sum = sum(w * np.log(v + EPSILON_GEOM) for w, v in valid) / w_total
    return float(np.clip(np.exp(log_sum) - EPSILON_GEOM, 0, 1))


def s_conj_boltzmann(d3, d12, tau=TAU_BOLTZ):
    if np.isnan(d3) and np.isnan(d12): return np.nan
    if np.isnan(d3): return d12
    if np.isnan(d12): return d3
    e3, e12 = np.exp(tau * d3), np.exp(tau * d12)
    return float((d3 * e3 + d12 * e12) / (e3 + e12))


def s_conj_max(d3, d12):
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
# Extraction donnees
# ----------------------------------------------------------------------
def extract_raster_point(path: Path, lat: float, lon: float) -> float:
    if not path.exists():
        return np.nan
    with rasterio.open(path) as src:
        vals = list(src.sample([(lon, lat)]))
        val = float(vals[0][0])
        if src.nodata is not None and val == src.nodata:
            return np.nan
        return val


def load_sev_lookup() -> gpd.GeoDataFrame:
    """Charge la table Aqueduct (GDB) pour faire des sjoin nearest."""
    print(f"  Lecture GDB Aqueduct ...", flush=True)
    gdf = gpd.read_file(AQ_GDB, columns=["sev_raw"])
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def sev_for_points(gdf_aq: gpd.GeoDataFrame, sites: pd.DataFrame) -> np.ndarray:
    """Spatial join : pour chaque site, on prend le polygone qui le contient
    (sinon le plus proche). Retourne sev_raw."""
    pts = gpd.GeoDataFrame(
        sites[["code_dept"]].copy(),
        geometry=[Point(lon, lat) for lat, lon in zip(sites["lat"], sites["lon"])],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, gdf_aq, how="left", predicate="within")
    # remplir manquants par nearest (cotier qui sort des polygones)
    miss = joined["sev_raw"].isna()
    if miss.any():
        joined_near = gpd.sjoin_nearest(
            pts.loc[miss, ["geometry"]], gdf_aq[["geometry", "sev_raw"]],
            how="left", max_distance=1.0,
        )
        joined.loc[miss, "sev_raw"] = joined_near["sev_raw"].values
    return joined["sev_raw"].to_numpy(dtype=float)


def find_land_gridpoint(lat, lon, ds, threshold=0.60):
    td = "valid_time" if "valid_time" in ds.dims else "time"
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
        return lats[i_lat], lons[i_lon]
    local_lats, local_lons = lats[lat_sl], lons[lon_sl]
    max_pos = np.unravel_index(np.argmax(pet_3x3), pet_3x3.shape)
    return float(local_lats[max_pos[0]]), float(local_lons[max_pos[1]])


def compute_spei_series(P, PET, dates, window):
    D = P - PET
    n = len(D)
    D_acc = np.full(n, np.nan)
    for i in range(window - 1, n):
        D_acc[i] = np.sum(D[i - window + 1:i + 1])
    months = dates.month
    mask_base = (dates.year >= SPEI_BASELINE[0]) & (dates.year <= SPEI_BASELINE[1])
    monthly_params = {}
    for m in range(1, 13):
        mask_m = (months == m) & mask_base & ~np.isnan(D_acc)
        base = D_acc[mask_m]
        if len(base) < 10:
            monthly_params[m] = None
            continue
        shift = -np.min(base) + 1.0
        try:
            c, _, scale = stats.fisk.fit(base + shift, floc=0)
            monthly_params[m] = (c, scale, shift)
        except Exception:
            monthly_params[m] = None
    spei = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(D_acc[i]):
            continue
        params = monthly_params[months[i]]
        if params is None:
            continue
        c, scale, shift = params
        cdf = stats.fisk.cdf(D_acc[i] + shift, c, scale=scale)
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        spei[i] = stats.norm.ppf(cdf)
    return spei


def compute_aridity_value(P, PET, dates, ai_mid=0.65, k=3.0):
    mask = (dates.year >= SPEI_BASELINE[0]) & (dates.year <= SPEI_BASELINE[1])
    n_years = SPEI_BASELINE[1] - SPEI_BASELINE[0] + 1
    p_ann = np.sum(P[mask]) / n_years
    pet_ann = np.sum(PET[mask]) / n_years
    if pet_ann <= 0:
        return 0.0
    ai = p_ann / pet_ann
    return float(1.0 / (1.0 + (ai / ai_mid) ** k))


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def compute_for_site(lat, lon, sev_raw_val, ds, td):
    bws_raw = extract_raster_point(BWS_TIF, lat, lon)
    gws_raw = extract_raster_point(GWS_TIF, lat, lon)

    x_bws = np.clip(bws_raw / BWS_MAX_FR, 0, 1) if not np.isnan(bws_raw) else np.nan
    x_gws = np.clip(gws_raw / GWS_MAX_FR, 0, 1) if not np.isnan(gws_raw) else np.nan
    x_sv = np.clip(sev_raw_val / SV_MAX_FR, 0, 1) if not np.isnan(sev_raw_val) else np.nan

    # ERA5 - point terrestre
    lat_g, lon_g = find_land_gridpoint(lat, lon, ds)
    sub = ds.sel(latitude=lat_g, longitude=lon_g, method="nearest")
    times = pd.DatetimeIndex(sub[td].values)
    P = sub["tp"].values.astype(float) * 1000.0   # m -> mm
    days_in_month = times.days_in_month.values
    P = P * days_in_month                          # tp ERA5 = m/jour mensuel
    PET = np.maximum(-sub["pev"].values.astype(float) * 1000.0 * days_in_month, 0.0)

    x_a = compute_aridity_value(P, PET, times)

    spei3 = compute_spei_series(P, PET, times, 3)
    spei12 = compute_spei_series(P, PET, times, 12)
    idx = np.where((times.year == TARGET_YEAR) & (times.month == TARGET_MONTH))[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    s3, s12 = spei3[i], spei12[i]
    d3, d12 = f_asym(s3), f_asym(s12)

    s_struct_v2 = s_struct_v2_arith(x_bws, x_a, x_gws)
    s_struct_v3 = s_struct_v3_geom(x_bws, x_a, x_gws, x_sv)
    s_conj_v2 = s_conj_boltzmann(d3, d12)
    s_conj_v3 = s_conj_max(d3, d12)
    H_v2 = assemble_H(s_struct_v2, s_conj_v2)
    H_v3 = assemble_H(s_struct_v3, s_conj_v3)

    return {
        "x_bws": x_bws, "x_gws": x_gws, "x_sv": x_sv, "x_aridity": x_a,
        "spei3": s3, "spei12": s12,
        "S_struct_v2": s_struct_v2, "S_conj_v2": s_conj_v2, "H_v2": H_v2,
        "S_struct_v3": s_struct_v3, "S_conj_v3": s_conj_v3, "H_v3": H_v3,
    }


def run_pipeline():
    sites = pd.read_csv(SITES_CSV, dtype={"code_dept": str})
    print(f"Sites a traiter : {len(sites)}")

    print("Chargement Aqueduct GDB ...")
    gdf_aq = load_sev_lookup()
    sev_vals = sev_for_points(gdf_aq, sites)

    print("Chargement ERA5 ...")
    ds = xr.open_dataset(ERA5_NC)
    td = "valid_time" if "valid_time" in ds.dims else "time"

    out = []
    for i, row in sites.iterrows():
        sev = sev_vals[i]
        try:
            res = compute_for_site(row["lat"], row["lon"], sev, ds, td)
        except Exception as e:
            print(f"  [err] {row['code_dept']} {row['prefecture']}: {e}")
            res = None
        rec = row.to_dict()
        if res:
            rec.update(res)
        out.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sites)} ...", flush=True)

    df = pd.DataFrame(out)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nEcrit : {OUT_CSV}")
    return df


# ----------------------------------------------------------------------
# Bootstrap rho_Spearman avec n=86
# ----------------------------------------------------------------------
def bootstrap_rho(H, P, n_boot=N_BOOT):
    n = len(H)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        if len(np.unique(idx)) < 5:
            rhos[i] = np.nan
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, _ = stats.spearmanr(H[idx], P[idx])
        rhos[i] = r
    return rhos[~np.isnan(rhos)]


def paired_delta_rho(H1, H2, P, n_boot=N_BOOT):
    n = len(P)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        if len(np.unique(idx)) < 5:
            deltas[i] = np.nan
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1, _ = stats.spearmanr(H1[idx], P[idx])
            r2, _ = stats.spearmanr(H2[idx], P[idx])
        deltas[i] = r1 - r2
    return deltas[~np.isnan(deltas)]


def analyze(df: pd.DataFrame):
    sub = df.dropna(subset=["H_v2", "H_v3", "niveau_int"]).copy()
    n = len(sub)
    H_v2 = sub["H_v2"].to_numpy()
    H_v3 = sub["H_v3"].to_numpy()
    P = sub["niveau_int"].to_numpy()

    rho_v2_pt, p_v2 = stats.spearmanr(H_v2, P)
    rho_v3_pt, p_v3 = stats.spearmanr(H_v3, P)
    rhos_v2 = bootstrap_rho(H_v2, P)
    rhos_v3 = bootstrap_rho(H_v3, P)
    deltas = paired_delta_rho(H_v2, H_v3, P)

    lo_v2, hi_v2 = np.percentile(rhos_v2, [2.5, 97.5])
    lo_v3, hi_v3 = np.percentile(rhos_v3, [2.5, 97.5])
    d_med = float(np.median(deltas))
    d_lo, d_hi = np.percentile(deltas, [2.5, 97.5])
    zero_in = (d_lo <= 0 <= d_hi)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.hist(rhos_v2, bins=40, alpha=0.55, label=f"v2 (med={np.median(rhos_v2):+.3f})", color="#4FC3F7")
    ax.hist(rhos_v3, bins=40, alpha=0.55, label=f"v3 (med={np.median(rhos_v3):+.3f})", color="#AB47BC")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("rho_Spearman(H, niveau Propluvia)")
    ax.set_title(f"Distribution rho — bootstrap n={n} sites, {N_BOOT} replicats")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(deltas, bins=40, color="#FFB74D", alpha=0.85)
    ax.axvline(0, color="k", lw=1.0)
    ax.axvline(d_med, color="#E65100", lw=1.2, ls="--", label=f"med = {d_med:+.3f}")
    ax.axvspan(d_lo, d_hi, alpha=0.15, color="#FFB74D",
               label=f"IC95 = [{d_lo:+.3f}, {d_hi:+.3f}]")
    ax.set_xlabel("Delta rho = rho_v2 - rho_v3 (apparie)")
    ax.set_title("Difference v2 vs v3 — significative ?")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130)
    plt.close(fig)

    # Report
    lines = [
        "# Vague 2 — H_local v2 vs v3 sur 86 prefectures\n",
        f"- n sites valides : **{n}**",
        f"- bootstrap : **{N_BOOT}** replicats, seed=42",
        f"- date validation : 15 juillet 2022\n",
        "## Concordance Propluvia 2022-07-15 — IC95 bootstrap\n",
        "| Modele | rho ponctuel | rho mediane | IC95 |",
        "|---|---:|---:|:---:|",
        f"| v2 | {rho_v2_pt:+.3f} | {np.median(rhos_v2):+.3f} | [{lo_v2:+.3f}, {hi_v2:+.3f}] |",
        f"| v3 | {rho_v3_pt:+.3f} | {np.median(rhos_v3):+.3f} | [{lo_v3:+.3f}, {hi_v3:+.3f}] |",
        "",
        "## Difference appariee\n",
        f"- Delta rho mediane : **{d_med:+.3f}**",
        f"- IC95 : [{d_lo:+.3f}, {d_hi:+.3f}]",
        f"- Zero dans IC95 : **{'OUI' if zero_in else 'NON'}**",
        "",
    ]
    if zero_in:
        lines.append("> Pas de difference statistiquement significative v2 vs v3.")
    else:
        sign = "v2 meilleur" if d_med > 0 else "v3 meilleur"
        lines.append(f"> Difference significative — {sign} (IC95 ne contient pas zero).")
    OUT_REPORT.write_text("\n".join(lines))

    print("\n=== RESULTATS ===")
    print(f"n = {n}")
    print(f"rho v2 = {rho_v2_pt:+.3f} (p={p_v2:.4f}) | IC95 [{lo_v2:+.3f}, {hi_v2:+.3f}]")
    print(f"rho v3 = {rho_v3_pt:+.3f} (p={p_v3:.4f}) | IC95 [{lo_v3:+.3f}, {hi_v3:+.3f}]")
    print(f"Delta rho mediane = {d_med:+.3f} | IC95 [{d_lo:+.3f}, {d_hi:+.3f}]")
    print(f"Zero dans IC95 : {'OUI' if zero_in else 'NON'}")
    print(f"\nRapport : {OUT_REPORT}")
    print(f"Figure  : {OUT_FIG}")


def main():
    if OUT_CSV.exists():
        print(f"Reuse {OUT_CSV} (delete pour recomputer)")
        df = pd.read_csv(OUT_CSV, dtype={"code_dept": str})
    else:
        df = run_pipeline()
    analyze(df)


if __name__ == "__main__":
    main()
