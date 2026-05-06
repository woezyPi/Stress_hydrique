# Vague 7 — Baseline minimaliste, interaction, Random Forest

- n zones : 362
- bootstrap : 5000 replicats, seed=42
- RF : 300 arbres, depth=8, leaf>=4, train/test stratifie 70/30

## Modeles testes (rho_Spearman vs Propluvia)

| modele | rho | p | IC95 lo | IC95 hi |
|---|---:|---:|---:|---:|
| d3_lag1 seul | +0.282 | 4.81e-08 | +0.194 | +0.367 |
| aridite seule | +0.321 | 4.19e-10 | +0.234 | +0.403 |
| 0.5*d3_lag1 + 0.5*aridite | +0.345 | 1.51e-11 | +0.253 | +0.429 |
| 0.7*d3_lag1 + 0.3*aridite | +0.326 | 2.01e-10 | +0.233 | +0.414 |
| 0.3*d3_lag1 + 0.7*aridite | +0.347 | 1.10e-11 | +0.258 | +0.433 |
| d3_lag1 * (1 - BWS) | +0.280 | 6.08e-08 | +0.183 | +0.376 |
| aridite * (1 - BWS) | +0.298 | 7.12e-09 | +0.203 | +0.391 |
| (0.5 d3 + 0.5 a) * (1 - BWS) | +0.307 | 2.37e-09 | +0.207 | +0.403 |
| (0.5 d3 + 0.5 a) * (1 + BWS) | +0.183 | 4.54e-04 | +0.085 | +0.276 |
| RandomForest (test set) | +0.507 | 1.87e-08 | +nan | +nan |

## Random Forest — importance par permutation

| feature | importance | std |
|---|---:|---:|
| x_bws | +0.1418 | 0.0440 |
| x_a | +0.1132 | 0.0414 |
| d12_lag1 | +0.0953 | 0.0229 |
| spei12_lag1 | +0.0781 | 0.0200 |
| x_sv | +0.0176 | 0.0167 |
| spei3_lag1 | +0.0033 | 0.0100 |
| d3_lag1 | +0.0022 | 0.0116 |
| x_gws | -0.0015 | 0.0006 |