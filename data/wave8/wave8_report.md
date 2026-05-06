# Vague 8 — robustesse Random Forest

- n = 362
- 5 clusters spatiaux KMeans(lat, lon)
- 1000 permutations pour le test null

## CV spatiale (leave-one-cluster-out, 5 folds)

| fold | RF complet | RF sans BWS | delta |
|---:|---:|---:|---:|
| 0 | +0.365 | +0.429 | -0.064 |
| 1 | -0.088 | +0.142 | -0.230 |
| 2 | +0.240 | +0.087 | +0.153 |
| 3 | +0.281 | +0.154 | +0.127 |
| 4 | +0.292 | +0.083 | +0.209 |
| **mediane** | **+0.281** | **+0.142** | **+0.127** |
| std         | 0.158 | 0.128 | 0.163 |

## Permutation test

- rho reel (random split 70/30) = +0.505
- rho null mediane = +0.003
- rho null IC95 = [-0.252, +0.263]
- p-value permutation = **0.0000**

## SHAP — top features (mean |shap|)

| feature | shap_mean_abs |
|---|---:|
| x_bws | 0.1996 |
| x_a | 0.1880 |
| spei12_lag1 | 0.1625 |
| d12_lag1 | 0.1395 |
| x_sv | 0.0760 |
| spei3_lag1 | 0.0484 |
| d3_lag1 | 0.0409 |
| x_gws | 0.0019 |

## Limites assumees
- Pas de CV temporelle : un seul snapshot juillet 2022.
- n=362 reste modeste pour un modele non-lineaire (fragilite de la mediane).
- L'absence de gain quand on retire BWS, si elle est observee, contredirait l'importance RF.