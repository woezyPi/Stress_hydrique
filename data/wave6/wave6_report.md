# Vague 6 — H_local avec SPEI Pearson III (clip 1e-4)

- n zones valides : **453**
- bootstrap : 5000 replicats, seed=42

## Sanity SPEI (30 zones aleatoires)

- spei3 mediane = -2.135, std = 0.651, saturation_bas = 3.3%
- spei12 mediane = -3.719, std = 1.003, saturation_bas = 53.3%

## rho individuel par feature

| feature | rho | p | IC95 lo | IC95 hi |
|---|---:|---:|---:|---:|
| x_bws | -0.293 | 2.06e-10 | -0.381 | -0.200 |
| x_a | +0.376 | 1.18e-16 | +0.300 | +0.449 |
| x_gws | +0.063 | 2.31e-01 | -0.019 | +0.141 |
| x_sv | -0.044 | 3.52e-01 | -0.128 | +0.041 |
| spei3_t (inverted) | +0.263 | 1.32e-08 | +0.171 | +0.347 |
| spei12_t (inverted) | +0.212 | 5.22e-06 | +0.123 | +0.297 |
| spei3_lag1 (inverted) | +0.327 | 9.30e-13 | +0.248 | +0.404 |
| spei12_lag1 (inverted) | +0.293 | 1.91e-10 | +0.205 | +0.379 |
| spei3_lag2 (inverted) | +0.218 | 2.87e-06 | +0.132 | +0.304 |
| spei12_lag2 (inverted) | +0.202 | 1.54e-05 | +0.112 | +0.294 |

## v2 vs v3 (SPEI Pearson III)

- v2 : rho_pt = +0.192 (p=3.83e-05), IC95 = [+0.102, +0.283]
- v3 : rho_pt = +0.126 (p=7.16e-03), IC95 = [+0.036, +0.217]
- Delta v2-v3 = +0.066, IC95 = [+0.037, +0.095]
- Zero dans IC95 : NON