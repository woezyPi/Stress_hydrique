"""
Vague 8 — robustesse du Random Forest (n=362, ρ=0.507 a etablir).

Tests demandes :
  1. CV spatiale : 5 clusters (lat, lon) → leave-one-cluster-out
  2. RF avec vs sans BWS sur les memes folds
  3. Permutation test : shuffle y → distribution de rho_null
  4. SHAP : interaction summary, top features

Limites assumees :
  - on a un seul snapshot temporel (juillet 2022). Pas de CV temporelle possible.
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
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "wave6" / "zones_h_local_pearson3.csv"
OUT_DIR = ROOT / "data" / "wave8"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_REPORT = OUT_DIR / "wave8_report.md"
OUT_FIG = OUT_DIR / "wave8_summary.png"
OUT_SHAP = OUT_DIR / "wave8_shap.png"

RNG = np.random.default_rng(42)
N_PERM = 1000
N_SPATIAL_CLUSTERS = 5

FEATURES_FULL = ["x_bws", "x_gws", "x_sv", "x_a",
                 "spei3_lag1", "spei12_lag1", "d3_lag1", "d12_lag1"]
FEATURES_NO_BWS = [f for f in FEATURES_FULL if f != "x_bws"]

def f_asym(spei, c=2.0):
    s = np.asarray(spei, dtype=float)
    out = np.where(s >= 0, 0.0, -s / (c + np.abs(s)))
    return np.clip(out, 0, 1)


def fit_rf(X, y, seed=42):
    return RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=4,
        random_state=seed, n_jobs=-1).fit(X, y)


def main():
    df = pd.read_csv(CSV, dtype={"code_dept": str, "zone_id": str})
    df["d3_lag1"]  = f_asym(df["spei3_lag1"])
    df["d12_lag1"] = f_asym(df["spei12_lag1"])
    sub = df.dropna(subset=FEATURES_FULL + ["niveau_int", "lat", "lon"]).copy()
    print(f"n = {len(sub)}")

    # --- Cluster spatial
    km = KMeans(n_clusters=N_SPATIAL_CLUSTERS, random_state=42, n_init=10)
    sub["cluster"] = km.fit_predict(sub[["lat", "lon"]].to_numpy())
    print(f"Clusters spatiaux ({N_SPATIAL_CLUSTERS}) :")
    print(sub.groupby("cluster").size().to_string())

    # ----------------------------------------------------------------
    # 1. CV spatiale : leave-one-cluster-out
    # ----------------------------------------------------------------
    gkf = GroupKFold(n_splits=N_SPATIAL_CLUSTERS)
    groups = sub["cluster"].to_numpy()
    y = sub["niveau_int"].to_numpy().astype(float)

    rho_full, rho_no_bws, rho_null = [], [], []
    for fold, (tr, te) in enumerate(gkf.split(sub, y, groups)):
        Xtr_full, Xte_full = sub.iloc[tr][FEATURES_FULL], sub.iloc[te][FEATURES_FULL]
        Xtr_nob,  Xte_nob  = sub.iloc[tr][FEATURES_NO_BWS], sub.iloc[te][FEATURES_NO_BWS]
        ytr, yte = y[tr], y[te]

        rf_full = fit_rf(Xtr_full.values, ytr)
        rf_nob  = fit_rf(Xtr_nob.values, ytr)
        rho_f, _ = stats.spearmanr(rf_full.predict(Xte_full.values), yte)
        rho_n, _ = stats.spearmanr(rf_nob.predict(Xte_nob.values), yte)
        rho_full.append(rho_f); rho_no_bws.append(rho_n)
        clu_left = int(sub.iloc[te]["cluster"].iloc[0])
        n_te = len(te)
        print(f"  fold {fold}: cluster {clu_left} (n_test={n_te}) | rho full={rho_f:+.3f} | rho no_bws={rho_n:+.3f} | delta={rho_f-rho_n:+.3f}")

    rho_full = np.array(rho_full); rho_no_bws = np.array(rho_no_bws)
    print(f"\n=== CV spatiale (5 folds) ===")
    print(f"rho RF complet   : mediane={np.median(rho_full):+.3f}, mean={np.mean(rho_full):+.3f}, std={np.std(rho_full):.3f}, min={rho_full.min():+.3f}, max={rho_full.max():+.3f}")
    print(f"rho RF sans BWS  : mediane={np.median(rho_no_bws):+.3f}, mean={np.mean(rho_no_bws):+.3f}, std={np.std(rho_no_bws):.3f}, min={rho_no_bws.min():+.3f}, max={rho_no_bws.max():+.3f}")
    print(f"Delta moyen      : {np.mean(rho_full - rho_no_bws):+.3f}")

    # ----------------------------------------------------------------
    # 2. Permutation test (sur le random split 70/30 standard)
    # ----------------------------------------------------------------
    print(f"\n=== Permutation test (shuffle y, n={N_PERM}) ===")
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(
        sub[FEATURES_FULL].to_numpy(), y, test_size=0.30, random_state=42,
        stratify=sub["niveau_int"]
    )
    rf = fit_rf(Xtr, ytr)
    rho_real, _ = stats.spearmanr(rf.predict(Xte), yte)

    rho_null = []
    for k in range(N_PERM):
        ytr_shuf = ytr.copy()
        RNG.shuffle(ytr_shuf)
        rf_n = fit_rf(Xtr, ytr_shuf, seed=k)
        r, _ = stats.spearmanr(rf_n.predict(Xte), yte)
        if not np.isnan(r):
            rho_null.append(r)
    rho_null = np.array(rho_null)
    p_perm = np.mean(rho_null >= rho_real)
    print(f"rho reel (random split)   = {rho_real:+.3f}")
    print(f"rho null mediane          = {np.median(rho_null):+.3f}")
    print(f"rho null IC95             = [{np.percentile(rho_null, 2.5):+.3f}, {np.percentile(rho_null, 97.5):+.3f}]")
    print(f"p (perm) : {p_perm:.4f}")

    # ----------------------------------------------------------------
    # 3. SHAP (sur le random split, RF complet)
    # ----------------------------------------------------------------
    print("\n=== SHAP ===")
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(Xte)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    fi = pd.DataFrame({"feature": FEATURES_FULL,
                       "shap_mean_abs": mean_abs}).sort_values("shap_mean_abs", ascending=False)
    print(fi.round(4).to_string(index=False))

    fig_shap = plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_vals, pd.DataFrame(Xte, columns=FEATURES_FULL),
                      show=False, max_display=8)
    plt.tight_layout()
    plt.savefig(OUT_SHAP, dpi=130, bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------------------
    # 4. Plot recap
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    x = np.arange(N_SPATIAL_CLUSTERS)
    ax.bar(x - 0.2, rho_full, 0.4, label="RF complet", color="#0277BD")
    ax.bar(x + 0.2, rho_no_bws, 0.4, label="RF sans BWS", color="#FFB74D")
    ax.axhline(np.median(rho_full), color="#0277BD", lw=0.7, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([f"C{i}" for i in x])
    ax.set_xlabel("Cluster spatial laisse out")
    ax.set_ylabel("rho Spearman test")
    ax.set_title(f"CV spatiale (5 folds)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.hist(rho_null, bins=40, color="#bbb", alpha=0.85, label="null (y shuffle)")
    ax.axvline(rho_real, color="#E65100", lw=2, label=f"rho reel = {rho_real:+.3f}")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("rho")
    ax.set_title(f"Permutation test (n={len(rho_null)}), p={p_perm:.4f}")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    yy = np.arange(len(fi))[::-1]
    ax.barh(yy, fi["shap_mean_abs"], color="#388E3C")
    ax.set_yticks(yy); ax.set_yticklabels(fi["feature"])
    ax.set_xlabel("|SHAP| mean")
    ax.set_title("Importance SHAP (RF complet)")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130)
    plt.close(fig)

    # Rapport
    lines = [
        "# Vague 8 — robustesse Random Forest\n",
        f"- n = {len(sub)}",
        f"- {N_SPATIAL_CLUSTERS} clusters spatiaux KMeans(lat, lon)",
        f"- {N_PERM} permutations pour le test null",
        "",
        "## CV spatiale (leave-one-cluster-out, 5 folds)\n",
        "| fold | RF complet | RF sans BWS | delta |",
        "|---:|---:|---:|---:|",
    ]
    for i, (rf_, rn_) in enumerate(zip(rho_full, rho_no_bws)):
        lines.append(f"| {i} | {rf_:+.3f} | {rn_:+.3f} | {rf_-rn_:+.3f} |")
    lines += [
        f"| **mediane** | **{np.median(rho_full):+.3f}** | **{np.median(rho_no_bws):+.3f}** | **{np.median(rho_full-rho_no_bws):+.3f}** |",
        f"| std         | {np.std(rho_full):.3f} | {np.std(rho_no_bws):.3f} | {np.std(rho_full-rho_no_bws):.3f} |",
        "",
        "## Permutation test\n",
        f"- rho reel (random split 70/30) = {rho_real:+.3f}",
        f"- rho null mediane = {np.median(rho_null):+.3f}",
        f"- rho null IC95 = [{np.percentile(rho_null, 2.5):+.3f}, {np.percentile(rho_null, 97.5):+.3f}]",
        f"- p-value permutation = **{p_perm:.4f}**",
        "",
        "## SHAP — top features (mean |shap|)\n",
        "| feature | shap_mean_abs |",
        "|---|---:|",
    ]
    for _, r in fi.iterrows():
        lines.append(f"| {r['feature']} | {r['shap_mean_abs']:.4f} |")
    lines += [
        "",
        "## Limites assumees",
        "- Pas de CV temporelle : un seul snapshot juillet 2022.",
        "- n=362 reste modeste pour un modele non-lineaire (fragilite de la mediane).",
        "- L'absence de gain quand on retire BWS, si elle est observee, contredirait l'importance RF.",
    ]
    OUT_REPORT.write_text("\n".join(lines))
    print(f"\nRapport : {OUT_REPORT}")
    print(f"Fig     : {OUT_FIG}")
    print(f"SHAP    : {OUT_SHAP}")


if __name__ == "__main__":
    main()
