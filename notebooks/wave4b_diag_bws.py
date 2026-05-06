"""
Vague 4b — diagnostic : pourquoi w_bws -> 0 a la calibration ?

On regarde la correlation INDIVIDUELLE de chaque input avec Propluvia.
Si BWS tout seul a rho ~ 0, on sait pourquoi l'optim le rejette.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "wave3" / "zones_h_local_2022.csv",
                 dtype={"code_dept": str, "zone_id": str})
df = df.dropna(subset=["x_bws", "x_a", "x_gws", "x_sv", "spei3", "spei12", "niveau_int"])

P = df["niveau_int"].to_numpy()

print(f"n = {len(df)}")
print()
print("Correlation Spearman INDIVIDUELLE de chaque input avec Propluvia :\n")
print(f"{'Variable':<15} {'rho':>8} {'p':>10} {'min':>7} {'mean':>7} {'max':>7} {'std':>7}")
print("-" * 70)
for col in ["x_bws", "x_a", "x_gws", "x_sv", "spei3", "spei12"]:
    x = df[col].to_numpy()
    rho, p = stats.spearmanr(x, P)
    # Pour spei3/spei12 on inverse le signe (plus negatif = plus sec)
    sign = "(inverted)" if col.startswith("spei") else ""
    print(f"{col:<15} {rho:+.3f}   {p:.2e}  {x.min():+.2f}   {x.mean():+.2f}   {x.max():+.2f}   {x.std():.2f}  {sign}")

print()
print("Distribution x_bws :")
print(df["x_bws"].describe().round(3).to_string())

print()
print("=== CONCLUSION DIAGNOSTIQUE ===")
rho_bws, p_bws = stats.spearmanr(df["x_bws"], P)
rho_a, p_a = stats.spearmanr(df["x_a"], P)
rho_gws, p_gws = stats.spearmanr(df["x_gws"], P)
if abs(rho_bws) < 0.10:
    print(f"BWS seul rho={rho_bws:+.3f} : trop faible pour porter du signal.")
    print("  -> BWS Aqueduct ne discrimine pas les zones Propluvia.")
elif abs(rho_a) > abs(rho_bws) * 2:
    print(f"Aridite domine (rho={rho_a:+.3f} vs BWS rho={rho_bws:+.3f}).")
print(f"\nrho_aridite seule = {rho_a:+.3f} (p={p_a:.2e})")
print(f"rho_BWS seule     = {rho_bws:+.3f} (p={p_bws:.2e})")
print(f"rho_GWS seule     = {rho_gws:+.3f} (p={p_gws:.2e})")
