"""
Vague 7 — trois tests demandes par le reviewer :

1. Baseline : H = SPEI-3(t-1) + aridite (sans BWS)
2. Interaction : H = SPEI-3(t-1) * (1 - BWS) — BWS comme modulateur
3. Random Forest sur les inputs valides → feature importance

Donnees : data/wave6/zones_h_local_pearson3.csv (n=453)
"""

from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "wave6" / "zones_h_local_pearson3.csv"
OUT_DIR = ROOT / "data" / "wave7"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_REPORT = OUT_DIR / "wave7_report.md"
OUT_FIG = OUT_DIR / "wave7_summary.png"

RNG = np.random.default_rng(42)
N_BOOT = 5000

# d_asym pour transformer SPEI -> stress conjoncturel [0,1]
def f_asym(spei, c=2.0):
    s = np.asarray(spei, dtype=float)
    out = np.where(s >= 0, 0.0, -s / (c + np.abs(s)))
    return np.clip(out, 0, 1)


def rho_with_ic(x, P, label, n_boot=N_BOOT):
    valid = ~np.isnan(x)
    x, P = np.asarray(x[valid]), np.asarray(P[valid])
    rho_pt, p = stats.spearmanr(x, P)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, len(x), size=len(x))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, _ = stats.spearmanr(x[idx], P[idx])
        rhos[i] = r
    rhos = rhos[~np.isnan(rhos)]
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return {"model": label, "rho": rho_pt, "p": p,
            "ic_lo": lo, "ic_hi": hi, "n": len(x)}


def main():
    df = pd.read_csv(CSV, dtype={"code_dept": str, "zone_id": str})
    sub = df.dropna(subset=["x_bws", "x_a", "x_gws", "x_sv",
                            "spei3_lag1", "spei12_lag1",
                            "niveau_int"]).copy()
    print(f"n = {len(sub)}")

    # Composantes utiles
    sub["d3_lag1"]  = f_asym(sub["spei3_lag1"])
    sub["d12_lag1"] = f_asym(sub["spei12_lag1"])
    P = sub["niveau_int"].to_numpy()

    # ----------------------------------------------------------------
    # 1. Modele MINIMAL : SPEI-3(t-1) + aridite
    # ----------------------------------------------------------------
    print("\n=== 1. Baseline minimaliste ===")
    rows = []
    # 1a. SPEI-3(t-1) seul
    rows.append(rho_with_ic(sub["d3_lag1"], P, "d3_lag1 seul"))
    # 1b. aridite seule
    rows.append(rho_with_ic(sub["x_a"], P, "aridite seule"))
    # 1c. moyenne 50/50
    H_avg = 0.5 * sub["d3_lag1"] + 0.5 * sub["x_a"]
    rows.append(rho_with_ic(H_avg, P, "0.5*d3_lag1 + 0.5*aridite"))
    # 1d. moyenne 70/30
    H_70 = 0.7 * sub["d3_lag1"] + 0.3 * sub["x_a"]
    rows.append(rho_with_ic(H_70, P, "0.7*d3_lag1 + 0.3*aridite"))
    H_30 = 0.3 * sub["d3_lag1"] + 0.7 * sub["x_a"]
    rows.append(rho_with_ic(H_30, P, "0.3*d3_lag1 + 0.7*aridite"))

    # ----------------------------------------------------------------
    # 2. INTERACTION : BWS comme modulateur
    # ----------------------------------------------------------------
    print("\n=== 2. Interaction BWS comme modulateur ===")
    # Hypothese : BWS eleve = forte demande = forte adaptation -> reduit le stress percu
    H_int_a = sub["d3_lag1"] * (1 - sub["x_bws"])
    rows.append(rho_with_ic(H_int_a, P, "d3_lag1 * (1 - BWS)"))
    H_int_b = sub["x_a"] * (1 - sub["x_bws"])
    rows.append(rho_with_ic(H_int_b, P, "aridite * (1 - BWS)"))
    H_int_c = (0.5*sub["d3_lag1"] + 0.5*sub["x_a"]) * (1 - sub["x_bws"])
    rows.append(rho_with_ic(H_int_c, P, "(0.5 d3 + 0.5 a) * (1 - BWS)"))
    # Inverse : BWS comme amplificateur
    H_amp = (0.5*sub["d3_lag1"] + 0.5*sub["x_a"]) * (1 + sub["x_bws"])
    rows.append(rho_with_ic(H_amp, P, "(0.5 d3 + 0.5 a) * (1 + BWS)"))

    # ----------------------------------------------------------------
    # 3. Random Forest
    # ----------------------------------------------------------------
    print("\n=== 3. Random Forest (importance par permutation) ===")
    X_cols = ["x_bws", "x_gws", "x_sv", "x_a",
              "d3_lag1", "d12_lag1", "spei3_lag1", "spei12_lag1"]
    X = sub[X_cols].to_numpy()
    y = sub["niveau_int"].to_numpy().astype(float)

    # train/test stratifie 70/30 (sur niveau_int)
    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
        X, y, sub.index, test_size=0.30, random_state=42,
        stratify=sub["niveau_int"]
    )
    rf = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=8,
                               min_samples_leaf=4, n_jobs=-1)
    rf.fit(Xtr, ytr)
    yhat_te = rf.predict(Xte)
    rho_rf, p_rf = stats.spearmanr(yhat_te, yte)
    rows.append({"model": "RandomForest (test set)", "rho": rho_rf, "p": p_rf,
                 "ic_lo": np.nan, "ic_hi": np.nan, "n": len(yte)})

    # Permutation importance sur le test set
    perm = permutation_importance(rf, Xte, yte, n_repeats=30, random_state=42, n_jobs=-1)
    fi = pd.DataFrame({"feature": X_cols,
                       "importance": perm.importances_mean,
                       "std": perm.importances_std}).sort_values("importance", ascending=False)
    print(fi.round(4).to_string(index=False))

    # ----------------------------------------------------------------
    # Resume
    # ----------------------------------------------------------------
    res = pd.DataFrame(rows)
    print("\n=== TABLEAU GENERAL ===")
    print(res.round(3).to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    yy = np.arange(len(res))
    err = np.array([
        [r - lo if not np.isnan(lo) else 0
         for r, lo in zip(res["rho"], res["ic_lo"])],
        [hi - r if not np.isnan(hi) else 0
         for r, hi in zip(res["rho"], res["ic_hi"])],
    ])
    ax.errorbar(res["rho"], yy, xerr=err, fmt="o", capsize=3, color="#0277BD")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels(res["model"], fontsize=9)
    ax.set_xlabel("rho Spearman vs Propluvia (IC95)")
    ax.set_title("Modeles testes")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    yyy = np.arange(len(fi))
    ax.barh(yyy, fi["importance"], xerr=fi["std"], color="#388E3C")
    ax.set_yticks(yyy); ax.set_yticklabels(fi["feature"])
    ax.set_xlabel("Importance par permutation (RF, test set)")
    ax.set_title("Feature importance — Random Forest")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130)
    plt.close(fig)

    # Rapport markdown
    lines = [
        "# Vague 7 — Baseline minimaliste, interaction, Random Forest\n",
        f"- n zones : {len(sub)}",
        f"- bootstrap : {N_BOOT} replicats, seed=42",
        f"- RF : 300 arbres, depth=8, leaf>=4, train/test stratifie 70/30",
        "",
        "## Modeles testes (rho_Spearman vs Propluvia)\n",
        "| modele | rho | p | IC95 lo | IC95 hi |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in res.iterrows():
        ic = f"[{r['ic_lo']:+.3f}, {r['ic_hi']:+.3f}]" if not np.isnan(r["ic_lo"]) else "(N/A)"
        lines.append(f"| {r['model']} | {r['rho']:+.3f} | {r['p']:.2e} | {r['ic_lo']:+.3f} | {r['ic_hi']:+.3f} |")
    lines += ["", "## Random Forest — importance par permutation\n",
              "| feature | importance | std |",
              "|---|---:|---:|"]
    for _, r in fi.iterrows():
        lines.append(f"| {r['feature']} | {r['importance']:+.4f} | {r['std']:.4f} |")
    OUT_REPORT.write_text("\n".join(lines))
    print(f"\nRapport : {OUT_REPORT}")
    print(f"Fig     : {OUT_FIG}")


if __name__ == "__main__":
    main()
