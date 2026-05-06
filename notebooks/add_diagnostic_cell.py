import json

with open("notebooks/pipeline_v2.ipynb") as f:
    nb = json.load(f)

diag_source = '''# ── 6.4 DIAGNOSTIC — Où disparaît le carbone ? ────────────────────────
from carbon_model.plants.emission_factors import EMISSION_FACTORS, normalize_technology

# ═══════════════════════════════════════════════════════════
# TEST 1 : CI production directe (sans tracing)
# CI_prod(t) = Σ EF_g * P_g(t) / Σ P_g(t)
# ═══════════════════════════════════════════════════════════
carrier_ef = {}
for gen_carrier in net.generators['carrier'].unique():
    carrier_ef[gen_carrier] = EMISSION_FACTORS.get(gen_carrier, 300.0)

print('── EF par carrier (réseau) ──')
for c, ef in sorted(carrier_ef.items(), key=lambda x: -x[1]):
    n_gen = (net.generators['carrier'] == c).sum()
    print(f'  {c:25s} : {ef:6.0f} gCO2/kWh  ({n_gen} générateurs)')

# Production horaire par carrier (depuis generators_t.p_set)
gen_p = net.generators_t.p_set  # [time x generators]
gen_carriers = net.generators['carrier']

co2_total = pd.Series(0.0, index=gen_p.index)
mw_total = pd.Series(0.0, index=gen_p.index)

for carrier in gen_carriers.unique():
    mask = gen_carriers == carrier
    carrier_mw = gen_p.loc[:, mask].sum(axis=1).clip(lower=0)
    ef = carrier_ef[carrier]
    co2_total += carrier_mw * ef
    mw_total += carrier_mw

ci_prod_direct = co2_total / mw_total.clip(lower=1.0)

print(f'\\n── CI production directe (sans tracing) ──')
print(f'  Mean : {ci_prod_direct.mean():.1f} gCO2/kWh')
print(f'  RTE  : {ci_rte.mean():.1f} gCO2/kWh')
print(f'  Ratio: {ci_prod_direct.mean()/ci_rte.mean():.2f}')

# ═══════════════════════════════════════════════════════════
# TEST 2 : Carbone total injecté vs consommé (conservation)
# C_prod = Σ_{g,t} EF_g * P_g(t)
# C_load = Σ_{b,t} CI_b(t) * L_b(t)
# ═══════════════════════════════════════════════════════════
C_prod_total = co2_total.sum()  # en gCO2 * MW (= gCO2/kWh * MW*h)

# Carbone consommé : CI nodale * charge
ci_arr_np = ci_nodal_ds['carbon_intensity'].values  # [time, bus]
bus_names = ci_nodal_ds.coords['bus'].values

# Loads : si loads_t est vide, utiliser loads.p_set
if len(net.loads_t.p_set) > 0:
    load_p = net.loads_t.p_set
else:
    # Reconstruire depuis loads statiques
    load_p = pd.DataFrame(
        np.outer(np.ones(len(net.snapshots)), net.loads['p_set'].values),
        index=net.snapshots,
        columns=net.loads.index
    )

# Map loads to buses
load_bus_map = net.loads['bus']
load_per_bus = pd.DataFrame(0.0, index=net.snapshots, columns=bus_names)
for load_id in load_p.columns:
    bus = load_bus_map[load_id]
    if str(bus) in load_per_bus.columns:
        load_per_bus[str(bus)] += load_p[load_id].values

C_load_total = (ci_arr_np * load_per_bus.values).sum()

print(f'\\n── Conservation carbone ──')
print(f'  C_prod (injecté)  : {C_prod_total/1e9:.1f} TgCO2·MW')
print(f'  C_load (consommé) : {C_load_total/1e9:.1f} TgCO2·MW')
if C_prod_total > 0:
    ratio = C_load_total / C_prod_total
    print(f'  Ratio C_load/C_prod : {ratio:.3f}')
    if ratio < 0.8:
        print(f'  ⚠ Le tracing perd {(1-ratio)*100:.0f}% du carbone !')
    elif ratio > 1.2:
        print(f'  ⚠ Le tracing crée du carbone (+{(ratio-1)*100:.0f}%) !')
    else:
        print(f'  ✓ Conservation raisonnable')

# ═══════════════════════════════════════════════════════════
# TEST 3 : Comparaison CI_prod vs CI_tracing vs CI_RTE
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Série temporelle
common_idx = ci_prod_direct.index.intersection(ci_rte.index).intersection(ci_national_load.index)
daily_prod = ci_prod_direct[common_idx].resample('D').mean()
daily_model = ci_national_load[common_idx].resample('D').mean()
daily_rte = ci_rte[common_idx].resample('D').mean()

axes[0].plot(daily_prod, label=f'CI prod directe ({ci_prod_direct.mean():.0f})', alpha=0.8)
axes[0].plot(daily_model, label=f'CI tracing ({ci_national_load.mean():.0f})', alpha=0.8)
axes[0].plot(daily_rte, label=f'CI RTE ({ci_rte.mean():.0f})', alpha=0.8)
axes[0].set_ylabel('gCO2/kWh')
axes[0].set_title('CI nationale : prod directe vs tracing vs RTE')
axes[0].legend()

# Mix production : part de chaque carrier
carrier_shares = {}
for carrier in gen_carriers.unique():
    mask = gen_carriers == carrier
    total = gen_p.loc[:, mask].sum().sum()
    carrier_shares[carrier] = total

total_all = sum(carrier_shares.values())
carriers_sorted = sorted(carrier_shares.items(), key=lambda x: -x[1])

names = [c for c, _ in carriers_sorted[:10]]
shares = [v/total_all*100 for _, v in carriers_sorted[:10]]
colors_map = {
    'nuclear': '#f4d44d', 'wind': '#74c476', 'solar': '#fdae6b',
    'hydro': '#6baed6', 'gas': '#d62728', 'coal': '#2c2c2c',
    'oil': '#8c564b', 'biomass': '#2ca02c', 'storage': '#9467bd',
    'other': '#7f7f7f', 'wind_offshore': '#31a354'
}
bar_colors = [colors_map.get(n, '#999') for n in names]
axes[1].barh(names, shares, color=bar_colors)
axes[1].set_xlabel('Part production (%)')
axes[1].set_title('Mix production (dispatch)')

plt.tight_layout()
plt.show()'''

cell_diag = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in diag_source.split("\n")]
}

# Insert at index 43 (after 6.3 bilan, before step 7 markdown)
nb["cells"].insert(43, cell_diag)

with open("notebooks/pipeline_v2.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Diagnostic cell 6.4 inserted at index 43")
print(f"Total cells: {len(nb['cells'])}")
