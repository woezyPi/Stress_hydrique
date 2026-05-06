# Vague 1 — diagnostic H_local v2 vs v3

- n sites 2022 : **8**
- bootstrap : **5000** replicats, seed=42

## 1. Concordance Propluvia 2022 — IC95 bootstrap

| Modele | rho mediane | IC95 bootstrap |
|---|---:|:---:|
| v2 | +0.475 | [-0.378, +0.943] |
| v3 | +0.113 | [-0.877, +0.798] |

## 2. Difference v2 - v3 (bootstrap apparie)

- Delta rho mediane : **+0.292**
- IC95 : [-0.270, +1.105]
- Zero dans IC95 : **OUI**

> **Conclusion** : la 'degradation v2 -> v3' n'est pas
> statistiquement distinguable de zero a n=8.
> Le rho ponctuel a 0.111 vs 0.457 etait du bruit.

## 3. Ablation decomposee

Recombinaison des composantes pour isoler le coupable :

| variante | rho | p | IC95 lo | IC95 hi |
|---|---:|---:|---:|---:|
| v2 (struct=v2, conj=v2) | +0.457 | 0.255 | -0.336 | +0.949 |
| v2+conj_v3 (struct=v2, conj=v3) | +0.296 | 0.476 | -0.578 | +0.955 |
| v2+struct_v3 (struct=v3, conj=v2) | +0.432 | 0.285 | -0.309 | +0.961 |
| v3 (struct=v3, conj=v3) | +0.111 | 0.793 | -0.879 | +0.796 |

Lecture : si une variante hybride conserve le rho de v2,
c'est l'autre composante qui fait chuter v3.

## 4. Limites de cette vague
- n=8 reste un plancher statistique. Tout IC est large.
- Propluvia est un benchmark administratif, pas physique.
- Pas de Monte-Carlo sur les inputs ici (vague 1bis).
- Pas de matrice corr BWS/GWS ici (vague 1bis, besoin rasters).
