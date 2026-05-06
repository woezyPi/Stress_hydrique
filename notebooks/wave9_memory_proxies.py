"""
Vague 9 — test rapide : effet memoire avec proxies simples.

On ne touche pas a Hub'Eau pour l'instant. On teste si l'ajout de variables
agregees temporellement (rolling mean, min, trend) sur les SPEI deja calcules
ameliore specifiquement le cluster 0 (Val-de-Loire) qui faisait planter le RF.

Proxies testes (a partir des spei3/spei12 a t, t-1, t-2 existants) :
  - rolling_mean_spei3 / spei12 = mean(t, t-1, t-2)        (lissage memoire)
  - rolling_min_spei3 / spei12  = min  (pire mois recent)  (worst-case)
  - trend_spei3 / spei12        = lag1 - lag2              (assechement)

KPI principal : reduction de la std entre clusters spatiaux + perf cluster 0.
"""

from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "wave6" / "zones_h_local_pearson3.csv"
OUT_DIR = ROOT / "data" / "wave9"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_REPORT = OUT_DIR / "wave9_report.md"
OUT_FIG = OUT_DIR / "wave9_summary.png"

RNG = np.random.default_rng(42)


def f_asym(spei, c=2.0):
    s = np.asarray(spei, dtype=float)
    out = np.where(s >= 0, 0.0, -s / (c + np.abs(s)))
    return np.clip(out, 0, 1)


def fit_rf(X, y, seed=42):
    return RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=4,
        random_state=seed, n_jobs=-1).fit(X, y)


def cv_spatial(sub, features, target_col="niveau_int", n_clusters=5):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    sub = sub.copy()
    sub["cluster"] = km.fit_predict(sub[["lat", "lon"]].to_numpy())
    gkf = GroupKFold(n_splits=n_clusters)
    y = sub[target_col].to_numpy().astype(float)
    rhos = {}
    for fold, (tr, te) in enumerate(gkf.split(sub, y, sub["cluster"].to_numpy())):
        Xtr, Xte = sub.iloc[tr][features].to_numpy(), sub.iloc[te][features].to_numpy()
        rf = fit_rf(Xtr, y[tr])
        r, _ = stats.spearmanr(rf.predict(Xte), y[te])
        rhos[int(sub.iloc[te]["cluster"].iloc[0])] = r
    return rhos


def main():
    df = pd.read_csv(CSV, dtype={"code_dept": str, "zone_id": str})
    needed = ["x_bws", "x_a", "x_gws", "x_sv",
              "spei3_t", "spei3_lag1", "spei3_lag2",
              "spei12_t", "spei12_lag1", "spei12_lag2",
              "niveau_int", "lat", "lon"]
    sub = df.dropna(subset=needed).copy()
    print(f"n = {len(sub)}")

    # Proxies memoire
    sub["spei3_roll_mean"]  = sub[["spei3_t", "spei3_lag1", "spei3_lag2"]].mean(axis=1)
    sub["spei3_roll_min"]   = sub[["spei3_t", "spei3_lag1", "spei3_lag2"]].min(axis=1)
    sub["spei3_trend"]      = sub["spei3_lag1"] - sub["spei3_lag2"]
    sub["spei12_roll_mean"] = sub[["spei12_t", "spei12_lag1", "spei12_lag2"]].mean(axis=1)
    sub["spei12_roll_min"]  = sub[["spei12_t", "spei12_lag1", "spei12_lag2"]].min(axis=1)
    sub["spei12_trend"]     = sub["spei12_lag1"] - sub["spei12_lag2"]
    sub["d3_lag1"] = f_asym(sub["spei3_lag1"])
    sub["d12_lag1"] = f_asym(sub["spei12_lag1"])

    # 3 jeux de features pour comparer
    SET_BASE = ["x_bws", "x_gws", "x_sv", "x_a",
                "spei3_lag1", "spei12_lag1", "d3_lag1", "d12_lag1"]
    SET_MEM = SET_BASE + [
        "spei3_roll_mean", "spei3_roll_min", "spei3_trend",
        "spei12_roll_mean", "spei12_roll_min", "spei12_trend",
    ]
    SET_MEM_NO_BWS = [f for f in SET_MEM if f != "x_bws"]

    all_results = {}
    for name, feats in [("base (vague 8)", SET_BASE),
                        ("base + memoire", SET_MEM),
                        ("memoire sans BWS", SET_MEM_NO_BWS)]:
        rhos = cv_spatial(sub, feats)
        all_results[name] = rhos
        vals = list(rhos.values())
        print(f"\n{name} ({len(feats)} features) :")
        for c in sorted(rhos):
            print(f"  cluster {c}: rho = {rhos[c]:+.3f}")
        print(f"  mediane = {np.median(vals):+.3f}, std = {np.std(vals):.3f}, "
              f"min = {min(vals):+.3f}, max = {max(vals):+.3f}")

    # Plot comparaison
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.27
    cluster_ids = sorted(all_results["base (vague 8)"].keys())
    for i, (name, rhos) in enumerate(all_results.items()):
        vals = [rhos[c] for c in cluster_ids]
        ax.bar([c + (i - 1) * width for c in cluster_ids], vals, width, label=name)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(cluster_ids)
    ax.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax.set_xlabel("Cluster spatial laisse out")
    ax.set_ylabel("rho Spearman test")
    ax.set_title("CV spatiale : effet des proxies memoire")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130)
    plt.close(fig)

    # Rapport
    base_vals = list(all_results["base (vague 8)"].values())
    mem_vals = list(all_results["base + memoire"].values())
    mem_no_bws_vals = list(all_results["memoire sans BWS"].values())

    lines = [
        "# Vague 9 — proxies memoire (rolling/min/trend SPEI)\n",
        f"- n = {len(sub)}",
        "- 5 clusters spatiaux KMeans",
        "- proxies ajoutes : spei{3,12}_roll_mean, _roll_min, _trend",
        "",
        "## Comparaison CV spatiale\n",
        "| variante | mediane | std (inter-cluster) | min | max | C0 (echec vague 8) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, vals_dict in all_results.items():
        vals = list(vals_dict.values())
        lines.append(f"| {name} | {np.median(vals):+.3f} | {np.std(vals):.3f} | {min(vals):+.3f} | {max(vals):+.3f} | {vals_dict[0]:+.3f} |")
    lines += [
        "",
        "## Lecture",
        "Le KPI prioritaire est la **std inter-cluster** (homogeneite) et le rho sur **C0** (cluster Val-de-Loire qui s'effondrait).",
        "Si memoire >> base sur C0 et std memoire < std base : effet memoire confirme, Hub'Eau prometteur.",
        "Si pas de gain significatif : le probleme C0 n'est pas la memoire courte mais l'hydrologie reelle (Hub'Eau / Banque Hydro indispensable).",
    ]
    OUT_REPORT.write_text("\n".join(lines))
    print(f"\nRapport : {OUT_REPORT}")
    print(f"Fig     : {OUT_FIG}")


if __name__ == "__main__":
    main()
