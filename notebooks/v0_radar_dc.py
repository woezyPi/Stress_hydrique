"""
Olara V0 produit — radar de risque hydrique DC, 4 metriques x 4 sites.

Sites :
  - Paris-Saint-Denis (urbain Seine)
  - Marseille L2 (mediterraneen sec)
  - Tours (cluster Val-de-Loire, gestion dominante)
  - Lyon-Venissieux (continental Rhone regule)

Metriques :
  f1 : nb annees 1991-2022 ou min(z_Q mensuel) < -2
  f2 : run-length moyenne de mois consecutifs avec z_Q < -1
  f3 : pente OLS de z_Q annuel (par decennie)
  f4 : nb d'occurrences (mois jjja) avec z_Q < -1 ET >=3 jours consecutifs T_max > 35 C

Sources :
  - Hub'Eau hydrometrie API : QmM par station Banque Hydro
  - Open-Meteo archive API  : T_max journaliere par coordonnee (ERA5 reanalyse)

Outputs :
  data/v0/radars_4sites.csv
  data/v0/radar_v0.png
  Doc/v0_produit.md
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "v0"
DOC_DIR = ROOT / "Doc"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOC_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "radars_4sites.csv"
OUT_FIG = OUT_DIR / "radar_v0.png"
OUT_DOC = DOC_DIR / "v0_produit.md"

BASELINE = (1991, 2020)
RANGE = (1991, 2022)
T_MAX_THRESHOLD = 35.0
T_RUN_LEN = 3
Z_Q_DRY = -1.0
Z_Q_CRIT = -2.0

SITES = {
    "Paris-Saint-Denis": (48.93, 2.36),
    "Marseille-L2":      (43.30, 5.40),
    "Tours":             (47.39, 0.69),
    "Lyon-Venissieux":   (45.70, 4.89),
}

# ----------------------------------------------------------------------
# Hub'Eau API
# ----------------------------------------------------------------------
HUBEAU_STATIONS = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel/stations"
HUBEAU_QMM     = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"


def find_station(lat: float, lon: float, radius_km: float = 30.0) -> dict | None:
    """Trouve la station Banque Hydro la plus proche avec serie longue."""
    params = {
        "latitude": lat, "longitude": lon, "distance": radius_km,
        "en_service": "true", "fields": "code_station,libelle_station,latitude_station,"
                                          "longitude_station,date_ouverture_station,en_service",
        "size": 50,
    }
    r = requests.get(HUBEAU_STATIONS, params=params, timeout=30)
    r.raise_for_status()
    items = r.json().get("data", [])
    if not items:
        return None

    cutoff_year = BASELINE[0]
    candidates = []
    for s in items:
        d = s.get("date_ouverture_station")
        if not d:
            continue
        if int(d[:4]) > cutoff_year:
            continue
        # distance
        dlat = s["latitude_station"] - lat
        dlon = (s["longitude_station"] - lon) * np.cos(np.radians(lat))
        dist_km = np.sqrt(dlat**2 + dlon**2) * 111
        candidates.append((dist_km, s))
    if not candidates:
        # fallback : sans filtre date
        for s in items:
            dlat = s["latitude_station"] - lat
            dlon = (s["longitude_station"] - lon) * np.cos(np.radians(lat))
            dist_km = np.sqrt(dlat**2 + dlon**2) * 111
            candidates.append((dist_km, s))
    candidates.sort()
    return candidates[0][1] if candidates else None


def fetch_qmm(code_station: str) -> pd.Series:
    """QmM mensuel pour la station, 1991-2022."""
    out = []
    for year in range(RANGE[0], RANGE[1] + 1):
        params = {
            "code_entite": code_station,
            "grandeur_hydro_elab": "QmM",
            "date_debut_obs_elab": f"{year}-01-01",
            "date_fin_obs_elab":   f"{year}-12-31",
            "size": 200,
        }
        r = requests.get(HUBEAU_QMM, params=params, timeout=30)
        if r.status_code != 200:
            continue
        for it in r.json().get("data", []):
            d = it.get("date_obs_elab")
            v = it.get("resultat_obs_elab")
            if d and v is not None:
                out.append((pd.Timestamp(d), float(v)))
    if not out:
        return pd.Series(dtype=float)
    df = pd.DataFrame(out, columns=["date", "Q"]).set_index("date").sort_index()
    return df["Q"]


# ----------------------------------------------------------------------
# Open-Meteo Archive (T_max journaliere)
# ----------------------------------------------------------------------
OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def fetch_tmax_daily(lat: float, lon: float) -> pd.Series:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{RANGE[0]}-01-01",
        "end_date":   f"{RANGE[1]}-12-31",
        "daily": "temperature_2m_max",
        "timezone": "Europe/Paris",
    }
    r = requests.get(OM_ARCHIVE, params=params, timeout=120)
    r.raise_for_status()
    j = r.json()
    dates = pd.to_datetime(j["daily"]["time"])
    tmax = j["daily"]["temperature_2m_max"]
    return pd.Series(tmax, index=dates, name="tmax")


# ----------------------------------------------------------------------
# Calculs metriques
# ----------------------------------------------------------------------
def z_q_monthly(q: pd.Series) -> pd.Series:
    df = q.to_frame("Q").copy()
    df["year"]  = df.index.year
    df["month"] = df.index.month
    base = df[(df["year"] >= BASELINE[0]) & (df["year"] <= BASELINE[1])]
    mu  = base.groupby("month")["Q"].mean()
    sig = base.groupby("month")["Q"].std()
    df["z"] = df.apply(lambda r: (r["Q"] - mu[r["month"]]) / sig[r["month"]]
                       if sig[r["month"]] > 0 else np.nan, axis=1)
    return df["z"]


def metric_f1(z: pd.Series) -> int:
    """Annees ou le min mensuel < -2."""
    df = z.to_frame("z"); df["year"] = df.index.year
    return int((df.groupby("year")["z"].min() < Z_Q_CRIT).sum())


def metric_f2(z: pd.Series) -> float:
    """Run-length moyenne des mois consecutifs avec z < -1."""
    s = (z < Z_Q_DRY).astype(int)
    runs = []
    cur = 0
    for v in s:
        if v == 1:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return float(np.mean(runs)) if runs else 0.0


def metric_f3(z: pd.Series) -> float:
    """Pente OLS de z_Q moyen annuel par decennie."""
    df = z.to_frame("z"); df["year"] = df.index.year
    annual = df.groupby("year")["z"].mean().dropna()
    if len(annual) < 5:
        return 0.0
    yrs = annual.index.values.astype(float)
    a = np.polyfit(yrs, annual.values, 1)[0]
    return float(a * 10.0)   # par decennie


def metric_f4(z: pd.Series, tmax: pd.Series) -> int:
    """Mois jjja avec z<-1 ET >=3 jours consecutifs T_max>35."""
    z_dry = z[(z.index.month.isin([6, 7, 8])) & (z < Z_Q_DRY)]
    count = 0
    for d in z_dry.index:
        m_start = pd.Timestamp(d.year, d.month, 1)
        m_end = (m_start + pd.offsets.MonthEnd(0))
        t = tmax.loc[m_start:m_end].dropna()
        if t.empty:
            continue
        hot = (t > T_MAX_THRESHOLD).astype(int).values
        # plus longue run de 1
        max_run = cur = 0
        for v in hot:
            cur = cur + 1 if v else 0
            max_run = max(max_run, cur)
        if max_run >= T_RUN_LEN:
            count += 1
    return int(count)


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def run_site(name: str, lat: float, lon: float) -> dict:
    print(f"\n=== {name} (lat={lat}, lon={lon}) ===")
    st = find_station(lat, lon)
    if st is None:
        print("  Pas de station hydro trouvee.")
        return {"site": name, "lat": lat, "lon": lon, "station": None,
                "f1": np.nan, "f2": np.nan, "f3": np.nan, "f4": np.nan}
    print(f"  Station Hub'Eau : {st['code_station']} — {st['libelle_station']}")

    q = fetch_qmm(st["code_station"])
    print(f"  QmM points : {len(q)} (annees: {q.index.min()} → {q.index.max() if len(q) else '-'})")
    if len(q) < 24:
        print("  Serie trop courte.")
        return {"site": name, "lat": lat, "lon": lon, "station": st["code_station"],
                "f1": np.nan, "f2": np.nan, "f3": np.nan, "f4": np.nan}

    z = z_q_monthly(q)
    tmax = fetch_tmax_daily(lat, lon)
    print(f"  T_max points : {len(tmax)}")

    f1 = metric_f1(z); f2 = metric_f2(z); f3 = metric_f3(z); f4 = metric_f4(z, tmax)
    print(f"  f1 = {f1} annees etiage critique")
    print(f"  f2 = {f2:.2f} mois moyen run-length z<-1")
    print(f"  f3 = {f3:+.3f} z par decennie")
    print(f"  f4 = {f4} mois jjja avec etiage + canicule >=3j")
    return {"site": name, "lat": lat, "lon": lon,
            "station": st["code_station"], "station_nom": st["libelle_station"],
            "f1": f1, "f2": f2, "f3": f3, "f4": f4}


def normalize_for_radar(values: list[float], higher_is_worse: bool) -> list[float]:
    """Min-max normalisation sur les 4 sites (0 = best, 1 = worst)."""
    arr = np.array(values, dtype=float)
    if higher_is_worse:
        lo, hi = np.nanmin(arr), np.nanmax(arr)
    else:
        lo, hi = np.nanmax(arr), np.nanmin(arr)   # inverse
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in arr]


def plot_radar(rows: list[dict]):
    labels = ["f1 freq etiage", "f2 duree etiage", "f3 tendance", "f4 canicule+etiage"]
    norms = {
        "f1": normalize_for_radar([r["f1"] for r in rows], higher_is_worse=True),
        "f2": normalize_for_radar([r["f2"] for r in rows], higher_is_worse=True),
        "f3": normalize_for_radar([r["f3"] for r in rows], higher_is_worse=False),  # plus negatif = pire
        "f4": normalize_for_radar([r["f4"] for r in rows], higher_is_worse=True),
    }
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += [angles[0]]
    colors = ["#0277BD", "#E65100", "#388E3C", "#AB47BC"]
    fig, ax = plt.subplots(figsize=(8.5, 8), subplot_kw={"polar": True})
    for i, r in enumerate(rows):
        vals = [norms["f1"][i], norms["f2"][i], norms["f3"][i], norms["f4"][i]]
        vals += [vals[0]]
        ax.plot(angles, vals, "-o", linewidth=2, color=colors[i], label=r["site"])
        ax.fill(angles, vals, alpha=0.12, color=colors[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0 (worst)"])
    ax.set_title("Olara V0 — radar de risque hydrique DC (relativise inter-sites)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10))
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_doc(rows: list[dict]):
    lines = [
        "# Olara V0 — radar de risque hydrique DC\n",
        "**Périmètre** : 4 sites représentatifs, 4 métriques physiques.",
        "**Fenêtre temporelle** : 1991–2022 (baseline climato 1991–2020).",
        "",
        "## Métriques",
        "- **f1** : nombre d'années (sur 32) où le débit mensuel a chuté en dessous de z = −2 (étiage critique).",
        "- **f2** : durée moyenne (en mois consécutifs) des épisodes z < −1.",
        "- **f3** : tendance d'aggravation = pente du z_Q annuel par décennie. Négatif = aggravation.",
        f"- **f4** : nombre de mois (juin/juillet/août) sur 32 ans avec étiage z<−1 ET ≥{T_RUN_LEN} jours consécutifs T_max > {T_MAX_THRESHOLD:.0f} °C.",
        "",
        "## Résultats — 4 sites",
        "| Site | Station Banque Hydro | f1 | f2 | f3 | f4 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['site']} | {r['station']} ({r.get('station_nom','')}) "
                     f"| {r['f1']} | {r['f2']:.2f} | {r['f3']:+.3f} | {r['f4']} |")
    lines += [
        "",
        "## Lecture",
        "Chaque axe du radar (cf. `data/v0/radar_v0.png`) est normalisé entre les 4 sites :",
        "0 = meilleur des 4, 1 = pire des 4. C'est une **comparaison relative**, pas un score absolu.",
        "Pour v1, prévoir un benchmark national (percentiles France) afin de produire une grille absolue.",
        "",
        "## Limites assumées V0",
        "- Une seule station Banque Hydro par site (la plus proche, ≤ 30 km).",
        "- Open-Meteo archive utilise ERA5 → résolution 0,25° (~28 km).",
        "- Pas de prise en compte régulation barrages, réseau, mitigation cooling.",
        "- Tendance f3 sensible aux extrêmes (ex. 2022).",
        "",
        "## Validation visuelle attendue",
        "- Marseille en tête sur f1, f3, f4 (méditerranéen sec et chaud).",
        "- Tours différencié par f2 (cluster où la gestion humaine domine — durée d'étiage cohérente avec le climat).",
        "- Paris/Lyon similaires structurellement, différenciés par f4 (canicule urbaine + débits régulés).",
    ]
    OUT_DOC.write_text("\n".join(lines))


def main():
    rows = []
    for name, (lat, lon) in SITES.items():
        rows.append(run_site(name, lat, lon))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    plot_radar(rows)
    write_doc(rows)
    print(f"\n=== ECRIT ===")
    print(f"  CSV : {OUT_CSV}")
    print(f"  PNG : {OUT_FIG}")
    print(f"  DOC : {OUT_DOC}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
