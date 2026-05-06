# Vague 9 — proxies memoire (rolling/min/trend SPEI)

- n = 362
- 5 clusters spatiaux KMeans
- proxies ajoutes : spei{3,12}_roll_mean, _roll_min, _trend

## Comparaison CV spatiale

| variante | mediane | std (inter-cluster) | min | max | C0 (echec vague 8) |
|---|---:|---:|---:|---:|---:|
| base (vague 8) | +0.281 | 0.158 | -0.088 | +0.365 | -0.088 |
| base + memoire | +0.050 | 0.146 | +0.008 | +0.375 | +0.050 |
| memoire sans BWS | +0.072 | 0.064 | -0.010 | +0.165 | +0.165 |

## Lecture
Le KPI prioritaire est la **std inter-cluster** (homogeneite) et le rho sur **C0** (cluster Val-de-Loire qui s'effondrait).
Si memoire >> base sur C0 et std memoire < std base : effet memoire confirme, Hub'Eau prometteur.
Si pas de gain significatif : le probleme C0 n'est pas la memoire courte mais l'hydrologie reelle (Hub'Eau / Banque Hydro indispensable).