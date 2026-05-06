import json

with open("notebooks/pipeline_v2.ipynb") as f:
    nb = json.load(f)

source = '''# ── 5.2 Analyse des flux et saturation ────────────────────────────────
import matplotlib.pyplot as plt

flows = net.lines_t.p0
flow_abs = flows.abs()
s_nom = net.lines["s_nom"].replace(0, np.nan)

# Stats
print("── Statistiques flux (MW) ──")
print(f"  Flux moyen abs  : {flow_abs.mean().mean():.1f} MW  (médiane {flow_abs.mean().median():.1f})")
print(f"  Flux max abs    : {flow_abs.max().max():.0f} MW")
print(f"  Écart-type moyen: {flow_abs.std().mean():.1f} MW")

# Utilisation
util_mean = flow_abs.mean() / s_nom * 100
util_max = flow_abs.max() / s_nom * 100
print(f"\\n── Utilisation lignes (% s_nom) ──")
print(f"  Utilisation moy. : {util_mean.mean():.1f}%")
print(f"  Utilisation max  : {util_max.max():.0f}%")
print(f"  Lignes >90% cap  : {(util_max > 90).sum()}")
print(f"  Lignes >100% cap : {(util_max > 100).sum()}")

# Angles
theta = net.buses_t.v_ang
print(f"\\n── Angles θ ──")
print(f"  θ max  : {theta.max().max():.4f} rad ({np.degrees(theta.max().max()):.2f}°)")
print(f"  θ min  : {theta.min().min():.4f} rad ({np.degrees(theta.min().min()):.2f}°)")
print(f"  θ mean : {theta.mean().mean():.6f} rad")

# Conservation p0 + p1
if hasattr(net.lines_t, "p1") and len(net.lines_t.p1) > 0:
    resid = (net.lines_t.p0 + net.lines_t.p1).abs()
    print(f"\\n── Conservation (p0 + p1) ──")
    print(f"  Résidu max : {resid.max().max():.6f} MW")
    print(f"  Résidu moy : {resid.mean().mean():.6f} MW")

# --- Graphiques ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Distribution flux moyens
ax = axes[0]
ax.hist(flow_abs.mean(), bins=100, color='steelblue', edgecolor='none')
med = flow_abs.mean().median()
ax.axvline(med, color='red', ls='--', label=f'médiane={med:.0f}')
ax.set_xlabel('Flux moyen absolu (MW)')
ax.set_ylabel('Nombre de lignes')
ax.set_title('Distribution des flux moyens')
ax.legend()

# 2. Taux utilisation moyen
ax = axes[1]
ax.hist(util_mean.dropna(), bins=100, color='darkorange', edgecolor='none')
ax.set_xlabel('Utilisation moyenne (% s_nom)')
ax.set_ylabel('Nombre de lignes')
ax.set_title("Taux d'utilisation moyen")

# 3. Flux total réseau journalier
ax = axes[2]
daily_total = flow_abs.sum(axis=1).resample('D').mean() / 1e3
ax.plot(daily_total.index, daily_total.values, color='green', lw=0.8)
ax.set_xlabel('Date')
ax.set_ylabel('Flux total (GW)')
ax.set_title('Flux total réseau (moy. journalière)')

plt.tight_layout()
plt.show()'''

cell_52 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in source.split("\n")]
}

# Insert after cell 35 (5.1) and before current cell 36 (5.3 bilan)
nb["cells"].insert(36, cell_52)

with open("notebooks/pipeline_v2.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Cell 5.2 inserted at index 36")
print(f"Total cells: {len(nb['cells'])}")