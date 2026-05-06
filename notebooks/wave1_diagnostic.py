"""
Vague 1 — diagnostic H_local v2 vs v3.

Aucun changement de modele. On instrumente l'existant pour separer
le signal du bruit avant toute correction.

Inputs  : notebooks/comparaison_v2_v3.csv (genere par HlocalV3.ipynb)
Outputs : notebooks/wave1_report.md
          notebooks/wave1_bootstrap_rho.png
          notebooks/wave1_ablation.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

RNG = np.random.default_rng(42)
N_BOOT = 5000

PROPLUVIA_2022 = {
    "Marseille": 4, "Montpellier": 3, "Lyon": 4, "Bordeaux": 3,
    "Paris-Saclay": 2, "Strasbourg": 2, "Rennes": 4, "Brest": 1,
}

BETA_0 = 0.70  # meme valeur que HlocalV3

HERE = Path(__file__).parent
CSV_IN = HERE / "comparaison_v2_v3.csv"
REPORT = HERE / "wave1_report.md"
FIG_BOOT = HERE / "wave1_bootstrap_rho.png"
FIG_ABL = HERE / "wave1_ablation.png"


# ----------------------------------------------------------------------
# 1. Chargement et filtrage 2022 (la seule annee avec Propluvia connu)
# ----------------------------------------------------------------------
def load_2022() -> pd.DataFrame:
    df = pd.read_csv(CSV_IN)
    df = df[df["Year"] == 2022].copy()
    df["Propluvia"] = df["Site"].map(PROPLUVIA_2022)
    df = df.dropna(subset=["H_v2", "H_v3", "Propluvia"]).reset_index(drop=True)
    if len(df) < 5:
        raise RuntimeError(f"Trop peu de sites 2022 valides : {len(df)}")
    return df


# ----------------------------------------------------------------------
# 2. Bootstrap IC95 sur rho_Spearman(H, Propluvia)
# ----------------------------------------------------------------------
def bootstrap_rho(H: np.ndarray, P: np.ndarray, n_boot: int = N_BOOT):
    n = len(H)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        # cas degenere : tous les memes indices => correlation NaN
        if len(np.unique(idx)) < 3:
            rhos[i] = np.nan
            continue
        r, _ = stats.spearmanr(H[idx], P[idx])
        rhos[i] = r
    rhos = rhos[~np.isnan(rhos)]
    return rhos


def paired_delta_rho(H1, H2, P, n_boot: int = N_BOOT):
    """Bootstrap apparie : meme resampling pour les deux modeles."""
    n = len(P)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        if len(np.unique(idx)) < 3:
            deltas[i] = np.nan
            continue
        r1, _ = stats.spearmanr(H1[idx], P[idx])
        r2, _ = stats.spearmanr(H2[idx], P[idx])
        deltas[i] = r1 - r2
    return deltas[~np.isnan(deltas)]


# ----------------------------------------------------------------------
# 3. Ablation decomposee :
#    on connait S_struct_v2/v3 et S_conj_v2/v3, on recombine en 4 variantes.
# ----------------------------------------------------------------------
def H_from(s_struct: np.ndarray, s_conj: np.ndarray, beta_0: float = BETA_0):
    beta_t = beta_0 * (1.0 - s_conj)
    return np.clip(beta_t * s_struct + (1 - beta_t) * s_conj, 0, 1)


def ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Cree 4 variantes et calcule rho_Spearman(H, Propluvia) pour chacune."""
    variants = {
        "v2 (struct=v2, conj=v2)": (df["S_struct_v2"], df["S_conj_v2"]),
        "v2+conj_v3 (struct=v2, conj=v3)": (df["S_struct_v2"], df["S_conj_v3"]),
        "v2+struct_v3 (struct=v3, conj=v2)": (df["S_struct_v3"], df["S_conj_v2"]),
        "v3 (struct=v3, conj=v3)": (df["S_struct_v3"], df["S_conj_v3"]),
    }
    P = df["Propluvia"].to_numpy()
    rows = []
    for name, (sstr, sco) in variants.items():
        H = H_from(sstr.to_numpy(), sco.to_numpy())
        rho, p = stats.spearmanr(H, P)
        rhos_boot = bootstrap_rho(H, P)
        lo, hi = np.percentile(rhos_boot, [2.5, 97.5])
        rows.append({
            "variante": name,
            "rho_point": rho,
            "p_value": p,
            "rho_lo95": lo,
            "rho_hi95": hi,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 4. Plots
# ----------------------------------------------------------------------
def plot_bootstrap(rhos_v2, rhos_v3, deltas, fig_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(rhos_v2, bins=40, alpha=0.55, label="v2", color="#4FC3F7")
    ax.hist(rhos_v3, bins=40, alpha=0.55, label="v3", color="#AB47BC")
    ax.axvline(np.median(rhos_v2), color="#0277BD", lw=1.2, ls="--")
    ax.axvline(np.median(rhos_v3), color="#6A1B9A", lw=1.2, ls="--")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("rho_Spearman(H, Propluvia)")
    ax.set_ylabel("frequence bootstrap")
    ax.set_title(f"Distribution rho — bootstrap n={len(rhos_v2)}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(deltas, bins=40, color="#FFB74D", alpha=0.8)
    ax.axvline(0, color="k", lw=1.0)
    ax.axvline(np.median(deltas), color="#E65100", lw=1.2, ls="--",
               label=f"median = {np.median(deltas):+.3f}")
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    ax.axvspan(lo, hi, alpha=0.15, color="#FFB74D",
               label=f"IC95 = [{lo:+.3f}, {hi:+.3f}]")
    ax.set_xlabel("Delta rho = rho_v2 - rho_v3 (apparie)")
    ax.set_title("Difference v2 vs v3 — significative ?")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


def plot_ablation(table: pd.DataFrame, fig_path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(table))
    ax.errorbar(
        table["rho_point"], y,
        xerr=[table["rho_point"] - table["rho_lo95"],
              table["rho_hi95"] - table["rho_point"]],
        fmt="o", color="#0277BD", capsize=4, lw=1.4, markersize=8,
    )
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(table["variante"])
    ax.set_xlabel("rho_Spearman(H, Propluvia 2022) — IC95 bootstrap")
    ax.set_title("Ablation decomposee : qui fait chuter rho ?")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# 5. Rapport
# ----------------------------------------------------------------------
def write_report(df, rhos_v2, rhos_v3, deltas, ablation):
    n = len(df)
    rho_v2 = float(np.median(rhos_v2))
    rho_v3 = float(np.median(rhos_v3))
    lo_v2, hi_v2 = np.percentile(rhos_v2, [2.5, 97.5])
    lo_v3, hi_v3 = np.percentile(rhos_v3, [2.5, 97.5])
    d_med = float(np.median(deltas))
    d_lo, d_hi = np.percentile(deltas, [2.5, 97.5])
    p_zero_in = (d_lo <= 0 <= d_hi)

    lines = []
    lines.append("# Vague 1 — diagnostic H_local v2 vs v3\n")
    lines.append(f"- n sites 2022 : **{n}**")
    lines.append(f"- bootstrap : **{N_BOOT}** replicats, seed=42")
    lines.append("")
    lines.append("## 1. Concordance Propluvia 2022 — IC95 bootstrap\n")
    lines.append("| Modele | rho mediane | IC95 bootstrap |")
    lines.append("|---|---:|:---:|")
    lines.append(f"| v2 | {rho_v2:+.3f} | [{lo_v2:+.3f}, {hi_v2:+.3f}] |")
    lines.append(f"| v3 | {rho_v3:+.3f} | [{lo_v3:+.3f}, {hi_v3:+.3f}] |")
    lines.append("")
    lines.append("## 2. Difference v2 - v3 (bootstrap apparie)\n")
    lines.append(f"- Delta rho mediane : **{d_med:+.3f}**")
    lines.append(f"- IC95 : [{d_lo:+.3f}, {d_hi:+.3f}]")
    lines.append(f"- Zero dans IC95 : **{'OUI' if p_zero_in else 'NON'}**")
    lines.append("")
    if p_zero_in:
        lines.append("> **Conclusion** : la 'degradation v2 -> v3' n'est pas")
        lines.append("> statistiquement distinguable de zero a n=8.")
        lines.append("> Le rho ponctuel a 0.111 vs 0.457 etait du bruit.")
    else:
        lines.append("> **Conclusion** : v3 degrade significativement la concordance.")
        lines.append("> L'ablation ci-dessous indique le coupable.")
    lines.append("")
    lines.append("## 3. Ablation decomposee\n")
    lines.append("Recombinaison des composantes pour isoler le coupable :\n")
    lines.append("| variante | rho | p | IC95 lo | IC95 hi |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in ablation.iterrows():
        lines.append(
            f"| {r['variante']} | {r['rho_point']:+.3f} | {r['p_value']:.3f} "
            f"| {r['rho_lo95']:+.3f} | {r['rho_hi95']:+.3f} |"
        )
    lines.append("")
    lines.append("Lecture : si une variante hybride conserve le rho de v2,")
    lines.append("c'est l'autre composante qui fait chuter v3.")
    lines.append("")
    lines.append("## 4. Limites de cette vague")
    lines.append("- n=8 reste un plancher statistique. Tout IC est large.")
    lines.append("- Propluvia est un benchmark administratif, pas physique.")
    lines.append("- Pas de Monte-Carlo sur les inputs ici (vague 1bis).")
    lines.append("- Pas de matrice corr BWS/GWS ici (vague 1bis, besoin rasters).")
    lines.append("")
    REPORT.write_text("\n".join(lines))


# ----------------------------------------------------------------------
def main():
    df = load_2022()
    print(f"Sites 2022 : {len(df)}")
    print(df[["Site", "H_v2", "H_v3", "Propluvia"]].to_string(index=False))

    H_v2 = df["H_v2"].to_numpy()
    H_v3 = df["H_v3"].to_numpy()
    P = df["Propluvia"].to_numpy()

    rhos_v2 = bootstrap_rho(H_v2, P)
    rhos_v3 = bootstrap_rho(H_v3, P)
    deltas = paired_delta_rho(H_v2, H_v3, P)

    ablation = ablation_table(df)
    print("\nAblation :")
    print(ablation.round(3).to_string(index=False))

    plot_bootstrap(rhos_v2, rhos_v3, deltas, FIG_BOOT)
    plot_ablation(ablation, FIG_ABL)
    write_report(df, rhos_v2, rhos_v3, deltas, ablation)

    print(f"\nRapport     : {REPORT}")
    print(f"Fig boot    : {FIG_BOOT}")
    print(f"Fig ablation: {FIG_ABL}")


if __name__ == "__main__":
    main()
