# Rapport de diagnostic — H_local v2/v3

**Date :** 2026-05-05
**Auteur :** validation interne Olara
**Périmètre :** modèle de stress hydrique territorial H_local v2 / v3, validation contre Propluvia (arrêtés sécheresse) au 15 juillet 2022.

---

## 1. Résumé exécutif

Trois résultats successifs, chacun infirmant la lecture précédente :

| Vague | Échelle | n | ρ_v2 | ρ_v3 | Lecture |
|:---:|---|---:|---:|---:|---|
| 1 | 8 villes (notebook initial) | 8 | +0,46 | +0,11 | "v3 dégrade tout" — **artefact de petit échantillon** |
| 2 | départements | 86 | +0,07 | −0,01 | "ni v2 ni v3 ne corrèlent" — **le modèle semble cassé** |
| 3 | zones d'alerte | 453 | **+0,25** (p=4,7×10⁻⁸) | **+0,15** (p=10⁻³) | **le modèle marche, c'était la granularité** |
| 4 | recalibration des poids v2 | 362 | +0,16 → **+0,27** | — | gain de +0,11 sur le test set |
| 4b | corrélation des inputs un par un | 362 | — | — | **3 inputs sur 6 sont à l'envers ou inutiles** |

L'enseignement principal du diagnostic : **le pouvoir prédictif réel de H_local est étouffé par trois variables qui pointent dans le mauvais sens** (BWS, GWS, SV Aqueduct).

---

## 2. Méthodologie

### 2.1 Donnée de référence

- **Propluvia / VigiEau (data.gouv)** — fichier `arretes.csv` (9,4 Mo, 11 650 arrêtés).
- Filtre : arrêtés actifs au **15 juillet 2022**.
- 219 arrêtés actifs, couvrant 86 / 96 départements métropolitains.
- Niveaux de gravité : `vigilance` (1), `alerte` (2), `alerte_renforcée` (3), `crise` (4).
- Granularité fine : 1 110 zones d'alerte uniques après éclatement zone par zone.

### 2.2 Géocodage

- **Échelle département (vague 2)** : préfecture de chaque département via `geo.api.gouv.fr` (96 chefs-lieux). Le niveau retenu par département est le **maximum** des arrêtés actifs.
- **Échelle zone d'alerte (vague 3)** : centroïde du polygone de chaque zone via le shapefile `all_zones.shp` (8 535 polygones zones d'alerte historiques). Match `id_zone` ↔ `zone_id` ; `est_max_v=1` pour ne garder que la version la plus récente. **453 zones** sur 1 110 obtiennent un centroïde valide.

### 2.3 Données d'entrée H_local

Identiques au notebook `HlocalV3.ipynb` :

| Variable | Source | Résolution native |
|---|---|---|
| BWS (Baseline Water Stress) | WRI Aqueduct 4.0 (TIF) | ~10 km |
| GWS (Groundwater Stress) | WRI Aqueduct 4.0 (TIF) | ~10 km |
| SV (Seasonal Variability) | WRI Aqueduct 4.0 (GDB, sjoin) | polygones HydroBASIN |
| Aridité (P / PET annuel) | ERA5 1981–2010 baseline | 0,25° (~28 km) |
| SPEI-3, SPEI-12 | ERA5 1981–2010 + Fisk fit | 0,25° |

Optimisation pipeline : cache ERA5 par cellule unique (1 110 zones → 234 cellules ERA5 distinctes).

### 2.4 Modèle H_local

Formules **verbatim** de `HlocalV3.ipynb` :

```
H_v2 :
  S_struct = (w_bws·x_bws + w_a·x_a + w_gws·x_gws) / Σw   [moyenne arithmétique pondérée]
  S_conj   = Boltzmann(d3, d12, τ=5)                       [d_i = f_asym(SPEI_i)]
  H        = β·S_struct + (1−β)·S_conj   avec β = β₀(1−S_conj),  β₀=0.70

H_v3 :
  S_struct = exp(Σ w_i · log(x_i + ε)) − ε                 [moyenne géométrique pondérée + SV]
  S_conj   = max(d3, d12)                                   [max dur]
  H        = mêmes assemblage et β

Poids AHP (HlocalV3) :
  v2 : w_bws=0.50, w_a=0.36, w_gws=0.14
  v3 : w_bws=0.532, w_a=0.185, w_gws=0.185, w_sv=0.097
```

### 2.5 Validation statistique

- **Métrique principale** : ρ de Spearman entre H_local et le niveau Propluvia (1–4).
- **Bootstrap** : 5 000 ré-échantillonnages avec remise, seed=42, IC95 % par percentiles.
- **Différence appariée v2 vs v3** : Δρ calculé sur le **même** ré-échantillonnage à chaque réplicat.
- **Calibration des poids** : split stratifié 70/30 (train/test), `differential_evolution` avec contrainte de simplexe (Σw = 1), seed=42, 200 itérations, validation finale **uniquement sur le test set**.
- **Robustesse de la calibration** : optim répétée sur 5 splits indépendants (seeds 0–4).

---

## 3. Résultats détaillés

### 3.1 Vague 1 — n=8 villes

| Modèle | ρ médian (bootstrap) | IC95 % |
|---|---:|:---:|
| v2 | +0,475 | [−0,38, +0,94] |
| v3 | +0,113 | [−0,88, +0,80] |
| Δρ | +0,29 | [−0,27, +1,11] |

**Lecture :** zéro est dans tous les IC. Toute conclusion v2/v3 est du bruit. La "dégradation" annoncée du notebook initial n'a pas de support statistique à n=8.

### 3.2 Vague 2 — n=86 départements (max par dépt)

| Modèle | ρ ponctuel | IC95 % | p-value |
|---|---:|:---:|---:|
| v2 | +0,069 | [−0,16, +0,30] | 0,53 |
| v3 | −0,010 | [−0,24, +0,22] | 0,93 |
| Δρ | +0,077 | [+0,001, +0,163] | — |

**Lecture :** le ρ est devenu mesurable mais nul. À cette granularité (préfecture comme proxy d'un département de 5 000–10 000 km²), le modèle ne discrimine plus rien. Δρ techniquement non-nul mais effet microscopique.

### 3.3 Vague 3 — n=453 zones d'alerte

| Modèle | ρ ponctuel | IC95 % | p-value |
|---|---:|:---:|---:|
| **v2** | **+0,253** | **[+0,167, +0,338]** | **4,7 × 10⁻⁸** |
| **v3** | **+0,153** | **[+0,064, +0,238]** | **0,001** |
| Δρ | +0,099 | [+0,043, +0,158] | zéro hors IC |

**Lecture :**
- v2 et v3 capturent un signal **statistiquement très significatif** (p ≪ 10⁻⁵).
- v2 surclasse v3 de manière **statistiquement solide** (Δρ borné inférieurement à +0,043).
- L'hypothèse "granularité inadéquate" formulée à l'issue de la vague 2 est **confirmée** : zone d'alerte ≠ département.

### 3.4 Vague 4 — calibration empirique des poids v2

Optimisation `(w_bws, w_a, w_gws)` sur 70 % des zones, validation sur les 30 % restants.

| Poids | ρ_train | ρ_test |
|---|---:|---:|
| AHP `(0.50, 0.36, 0.14)` | +0,160 | +0,159 |
| Optimisés `(0.011, 0.857, 0.132)` | +0,310 | **+0,269** |
| **Gain test** | — | **+0,111** |

**Stabilité sur 5 splits indépendants :**

| seed | w_bws | w_a | w_gws | ρ_test optim | ρ_test AHP | gain |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0,015 | 0,863 | 0,122 | 0,350 | 0,286 | +0,064 |
| 1 | 0,037 | 0,756 | 0,208 | 0,346 | 0,237 | +0,109 |
| 2 | 0,010 | 0,750 | 0,240 | 0,288 | 0,189 | +0,099 |
| 3 | 0,018 | 0,729 | 0,253 | 0,296 | 0,209 | +0,087 |
| 4 | 0,010 | 0,745 | 0,245 | 0,217 | 0,087 | +0,129 |

**Gain médian = +0,099, range [+0,064, +0,129].**

L'optim **annule systématiquement w_bws** (≤ 0,04) et **transfère le poids sur l'aridité** (w_a ≥ 0,73). C'est un signal d'alerte sur la composante BWS.

### 3.5 Vague 4b — corrélation individuelle de chaque input

Méthode : Spearman entre chaque variable du modèle, prise **isolément**, et Propluvia (n=362).

| Variable | ρ | p-value | Sens attendu | Statut |
|---|---:|---:|:---:|:---:|
| **BWS Aqueduct** | **−0,253** | 1,1 × 10⁻⁶ | + | ❌ **inversé** |
| Aridité ERA5 | +0,321 | 4,2 × 10⁻¹⁰ | + | ✅ |
| GWS Aqueduct | +0,063 | 0,23 | + | ⚠️ **non significatif** |
| SV Aqueduct | −0,150 | 4,4 × 10⁻³ | + | ❌ **inversé** |
| SPEI-3 (inversé) | −0,215 | 3,7 × 10⁻⁵ | + | ✅ |
| SPEI-12 (inversé) | −0,174 | 8,7 × 10⁻⁴ | + | ✅ |

**Lecture :** sur les 6 inputs de H_local :
- **3 sont valides** (aridité, SPEI-3, SPEI-12) — tous issus de la chaîne ERA5.
- **2 sont inversés** (BWS, SV) — variables Aqueduct WRI.
- **1 est inutile** (GWS) — variable Aqueduct WRI, p=0,23.

L'optim de la vague 4 réduisait w_bws à zéro. Le diagnostic ici **explique pourquoi** : BWS contribue *positivement* dans la formule alors que le signal qu'il porte est *négatif* vis-à-vis de Propluvia.

---

## 4. Interprétation du résultat sur BWS Aqueduct

Trois explications, non exclusives :

1. **BWS mesure une variable économique, pas physique.** WRI Aqueduct définit BWS = `prélèvements / ressources renouvelables`. Une zone à BWS élevé est une zone qui *consomme* beaucoup d'eau, pas une zone en *manque* d'eau. Les agglomérations alimentées par adduction (Île-de-France, métropoles) ont BWS élevé sans subir de stress local.

2. **Inadéquation France métropolitaine.** Aqueduct est calibré globalement. Les variations fines entre bassins versants français peuvent être noyées dans la résolution ~10 km et la définition globale.

3. **Décalage temporel.** BWS Aqueduct 4.0 est une moyenne climatologique (multi-décennale). Propluvia 2022 reflète un événement climatique précis (été 2022 record). Le BWS d'un jour fixe ne capture pas la variabilité événementielle.

GWS et SV présentent les mêmes biais structurels (ρ ≈ 0 ou inversé), avec la même origine probable.

---

## 5. Conclusions actionnables

### 5.1 Ce qui est confirmé

- **H_local capture un signal physique réel** au niveau zone d'alerte (p < 10⁻⁷ à n=453).
- **v2 surclasse v3** sur Propluvia, de manière statistiquement solide (Δρ médian +0,10, IC95 hors zéro).
- **Le pouvoir prédictif maximal de H_v2 dans sa forme actuelle est ρ ≈ 0,27** sur test set hors échantillon (calibration empirique).

### 5.2 Ce qui doit être revu

- **Composantes Aqueduct (BWS, GWS, SV)** : à retirer ou à reformuler. En l'état elles diluent le signal des autres variables.
- **Validité de l'AHP** : les poids "expert" donnent ρ_test = +0,16, les poids appris donnent +0,27. L'AHP n'est pas optimal pour cet usage.
- **Granularité minimale** : la maille département est trop grossière pour valider H_local. Toute future validation doit se faire à la maille zone d'alerte ou plus fine.

### 5.3 Suites recommandées

| # | Action | Gain attendu sur ρ_test | Coût |
|:---:|---|:---:|:---:|
| 1 | Retirer BWS, GWS, SV ; H = aridité + SPEI uniquement | +0,05 à +0,10 | 1 j |
| 2 | Ajouter Hub'Eau (piézométrie temps réel) et anomalie de débit Banque Hydro | +0,05 à +0,15 | 3 j |
| 3 | Random Forest / Gradient Boosting sur les inputs valides, train/test stratifié | +0,15 à +0,25 | 2 j |
| 4 | Étendre la validation à 1100 zones (relax du filtre `est_max_v`) | confirmation | 0,5 j |
| 5 | Re-baseline SPEI sur 1991–2020 (vs 1981–2010 actuel) | +0,02 | 0,5 j |

L'objectif réaliste après actions 1+2+3 est **ρ_test ∈ [0,40, 0,55]**, à confirmer empiriquement.

---

## 6. Reproductibilité

Tous les scripts et données sont dans le repo :

- `data/propluvia/arretes.csv` (source : data.gouv VigiEau, mis à jour 2026-05-05)
- `data/propluvia/zones_2022_07_15.csv` (1 110 zones, généré par `wave3_explode_zones.py`)
- `data/propluvia/sites_dept_2022.csv` (86 départements)
- `data/propluvia/all_zones.shp` (8 535 polygones zones d'alerte)
- `data/wave2/sites_h_local_2022.csv` (n=86, H v2/v3 + Propluvia)
- `data/wave3/zones_h_local_2022.csv` (n=453, H v2/v3 + Propluvia)
- `notebooks/wave1_diagnostic.py` (bootstrap n=8, ablation décomposée)
- `notebooks/wave2_compute_h.py` (pipeline n=86)
- `notebooks/wave3_compute_h.py` (pipeline n=1110, cache ERA5)
- `notebooks/wave4_calibration_poids.ipynb` (calibration empirique des poids)
- `notebooks/wave4b_diag_bws.py` (corrélation individuelle des inputs)

Tous les scripts utilisent **seed=42** sauf le multi-seed (seeds 0–4) en vague 4. Les rapports markdown et figures sont générés automatiquement à `data/wave{1,2,3,4}/`.

---

*Fin du rapport.*
