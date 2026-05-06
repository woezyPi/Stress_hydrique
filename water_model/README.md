# GreenDC Water Stress Model

Modele de stress hydrique pour l'evaluation de sites datacenter en France.
Calcule un **WaterScore** (0-100) combinant stress de surface, stress souterrain et risque de secheresse
a rajouter plus tard le :
- WUE
- La predcition 

> **Le dossier `water_model/data/` est fourni en zip** car les fichiers sont trop volumineux pour le repo git (~1 Go).
> Si vous n'avez pas le zip, contactez l'equipe.

---

## Table des matieres

1. [Demarrage rapide](#demarrage-rapide)
2. [Installation detaillee](#installation-detaillee)
3. [Lancer la pipeline](#lancer-la-pipeline)
4. [API et dashboard](#api-et-dashboard)
5. [Sorties](#sorties)
6. [Modele mathematique](#modele-mathematique)
7. [Sources de donnees](#sources-de-donnees)
8. [Validation](#validation)
9. [Structure du projet](#structure-du-projet)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## Installation detaillee

### Prerequis

- Python 3.10+
- Linux / WSL recommande


```bash
cd greendc

# Creer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dependances
pip install -r water_model/requirements.txt

# Verifier que les donnees sont en place
ls water_model/data/raw/aqueduct/bws_raw.tif   # doit exister
ls water_model/data/raw/brgm/bdlisa_v3.gpkg     # doit exister
ls water_model/data/raw/era5/                    # doit contenir les .nc

le zip de data est normalement entier et deja ranger au moment du dezip.
```

### Structure attendue du dossier data

```
water_model/data/
  raw/
    aqueduct/     bws_raw.tif, gws_raw.tif, drr_raw.tif
    brgm/         bdlisa_v3.gpkg
    era5/         total_precipitation_2022.nc, ...
    hubeau/       stations_piezo.parquet (optionnel)
    bnpe/         prelevements.parquet   (optionnel)
  processed/      (genere automatiquement)
  outputs/        (genere automatiquement)
```

Les fichiers **obligatoires** sont :
- `aqueduct/bws_raw.tif`, `gws_raw.tif`, `drr_raw.tif` — rasters stress hydrique WRI
- `brgm/bdlisa_v3.gpkg` — aquiferes francais
- `era5/*.nc` — precipitations mensuelles

Les donnees Hub'Eau (piezometrie) et BNPE (prelevements) sont **optionnelles** : la pipeline les telecharge automatiquement via API, ou continue sans si l'API est indisponible.

---

## Lancer la pipeline

```bash
source venv/bin/activate

# Pipeline standard (donnees deja presentes dans data/)
python -m water_model.pipeline --year 2022 --skip-download

# Avec tentative de telechargement des donnees manquantes
python -m water_model.pipeline --year 2022

# Mode test rapide (donnees synthetiques, ~2 min)
python -m water_model.pipeline --test

# Forcer le recalcul complet (ignore le cache)
python -m water_model.pipeline --year 2022 --force
```

### Etapes de la pipeline

1. Telechargement des donnees brutes (Aqueduct, BRGM, ERA5, Hub'Eau)
2. Nettoyage des rasters (nodata, outliers)
3. Reprojection vers Lambert-93 (EPSG:2154)
4. Harmonisation sur une grille commune 5 km
5. Calcul SPI-12 (indice de secheresse standardise)
6. Piezometrie temps reel + prelevements BNPE (optionnels)
7. Stress souterrain (GWS) : fusion Aqueduct + BRGM + piezometrie
8. Calcul WSI : combinaison ponderee des 3 composantes
9. Projection vers grilles WGS84 (5/15/30 km)
10. Validation scientifique (6 tests)
11. Integration GreenDC (enrichissement des scores)

### Reprise apres crash

Si la pipeline plante a l'etape N, relancer la meme commande : les etapes 1 a N-1 sont automatiquement sautees grace au cache.

---

## API et dashboard

Une fois que tout est ok lance le uvicorn  pour l'api et le dashboard.

```bash
source venv/bin/activate
uvicorn water_model.api:app --reload --port 8001
```

Ouvrir http://localhost:8001 pour le dashboard interactif.

### Endpoints

| Endpoint | Description | Exemple |
|----------|-------------|---------|
| `GET /` | Dashboard HTML | |
| `GET /api/query?lat=48.8&lon=2.3` | Score pour un point | Paris |
| `GET /api/query?lat=48.8&lon=2.3&month=7` | Score mensuel (juillet) | |
| `GET /api/monthly?lat=48.8&lon=2.3` | Serie 12 mois | |
| `GET /api/stats` | Statistiques du dataset | |
| `GET /api/heatmap?step=10` | Grille pour visualisation | |
| `GET /api/validation` | Rapport de validation | |

---

## Sorties

### Fichiers NetCDF (3 resolutions)

| Fichier | Resolution | Usage |
|---------|-----------|-------|
| `water_stress_5km_2022.nc` | 5 km | Haute precision |
| `water_stress_15km_2022.nc` | 15 km | Resolution moyenne |
| `water_stress_30km_2022.nc` | 30 km | Vue d'ensemble |

**Variables dans chaque NetCDF :**

| Variable | Plage | Description |
|----------|-------|-------------|
| `water_score` | [0, 100] | Score principal (100 = pas de stress) |
| `wsi_mean` | [0, 1] | WSI brut |
| `bws_norm` | [0, 1] | Stress de surface normalise |
| `gws_norm` | [0, 1] | Stress souterrain normalise |
| `drought_norm_mean` | [0, 1] | Risque secheresse normalise |
| `wsi_uncertainty` | [0, 1] | Incertitude (1 sigma) |
| `wsi_p10`, `wsi_p90` | [0, 1] | Percentiles temporels |

### Interpretation du WaterScore

| Score | Niveau | Description |
|-------|--------|-------------|
| 80-100 | Excellent | Eau abondante, stress faible |
| 65-80 | Bon | Pression moderee |
| 50-65 | Modere | Stress notable, surtout en ete |
| 35-50 | Defavorable | Pression significative |
| 0-35 | Critique | Stress hydrique severe |

### GeoJSON d'integration GreenDC

```
water_model/data/outputs/greendc_scores_water/
  points_low_with_water.geojson      (30 km)
  points_medium_with_water.geojson   (15 km)
  points_high_with_water.geojson     (5 km)
```

Colonnes ajoutees : `water_score`, `wsi_mean`, `bws_norm`, `gws_norm`, `drought_norm`, `water_score_uncertainty`.

### Rapport de validation

`water_model/data/outputs/water_validation_report.txt`

---

## Modele mathematique

### Formule principale : Water Stress Index (WSI)

```
WSI = 0.60 x BWS_norm + 0.25 x GWS_norm + 0.15 x DroughtRisk_norm
```

```
WaterScore = 100 x (1 - clip(WSI, 0, 1))
```

| Composante | Poids | Source | Description |
|------------|-------|--------|-------------|
| **BWS** (Baseline Water Stress) | 0.60 | WRI Aqueduct 4.0 | Ratio prelevements / debit disponible (0-5) |
| **GWS** (Groundwater Stress) | 0.25 | Aqueduct + BRGM + Hub'Eau | Stress souterrain combine |
| **DroughtRisk** | 0.15 | ERA5 via SPI-12 | Risque de secheresse meteorologique |

### Stress souterrain (GWS) - Fusion de 3 sources

```
GWS = 0.40 x Aqueduct_GWS + 0.30 x BRGM_score + 0.30 x Piezo_stress
```

(Sans piezometrie : GWS = 0.60 x Aqueduct + 0.40 x BRGM)

**Aqueduct GWS** : Depletion des nappes (satellite GRACE + modeles hydrologiques)

```
gws_score = clip(max(gtd, 0) x 10, 0, 5)    # 0.5 m/an -> score 5 (critique)
```

**BRGM BDLISA v3** : Caracterisation geologique des aquiferes francais.

| Type (milieueh) | Score | Raison |
|------------------|-------|--------|
| Poreux libre (1) | 0.15 | Tres productif, faible stress |
| Fracture (2) | 0.35 | Productivite variable |
| Mixte (3) | 0.45 | Heterogene |
| Karstique (4) | 0.55 | Vulnerable, comportement imprevisible |
| Non-aquifere (5) | 0.85 | Pas de ressource souterraine |

**Hub'Eau Piezometrie** : Anomalie du niveau des nappes en temps reel.

```
anomalie = (niveau_actuel - moyenne_20ans) / ecart_type_20ans
stress = normalisation vers [0, 1]  (bas = stress eleve)
```

### Risque de secheresse : SPI-12 (McKee et al., 1993)

L'indice SPI (Standardized Precipitation Index) mesure l'ecart des precipitations par rapport a la normale.

**Calcul :**

1. Cumuler les precipitations sur 12 mois glissants
2. Ajuster une distribution Gamma (alpha, beta) sur la periode de reference 1981-2010
3. Calculer la CDF : `P(X <= precip)` (avec probabilite de precipitation nulle q0)
4. Convertir en loi normale standard : `SPI = inverse_normale(P)`

**Normalisation vers un risque de secheresse :**

```
DR_norm = clip((-SPI + 1.0) / 3.0, 0, 1)
```

### Propagation d'incertitude

Methode analytique (1er ordre) avec incertitudes relatives :
- sigma_BWS = 12%, sigma_GWS = 18%, sigma_DR = 22%

Sortie : `wsi_uncertainty` et `water_score_uncertainty` dans les NetCDF.

---

## Sources de donnees

| Source | Fournisseur | Resolution | Fourni dans le zip |
|--------|------------|------------|-------------------|
| **Aqueduct 4.0** | WRI | ~10 km | Oui (rasters .tif) |
| **BDLISA v3** | BRGM | Limites aquiferes | Oui (.gpkg) |
| **ERA5** | Copernicus / ECMWF | 0.25 deg (~28 km) | Oui (.nc) |
| **Hub'Eau Piezo** | BRGM / EauFrance | Stations ponctuelles | Non (API temps reel, optionnel) |
| **BNPE** | EauFrance | Communes | Non (API, optionnel) |

---

## Validation

6 tests automatiques sont executes a l'etape 10 de la pipeline :

| Test | Description | Critere |
|------|-------------|---------|
| SPI_NORMALITY | Distribution SPI proche de N(0, sigma) | \|moyenne\| < 2.5, 0.2 < sigma < 2.0 |
| BWS_SPATIAL_COHERENCE | Mediterranee > Bretagne (gradient nord-sud) | Gradient positif |
| NATIONAL_WSI_RANGE | Moyenne France WSI raisonnable | WSI moyen entre 0.10 et 0.50 |
| WATER_SCORE_RANGE | Scores dans [0, 100] avec variabilite | std >= 1.0 |
| UNCERTAINTY_BOUNDS | Incertitude WSI raisonnable | sigma entre 0.02 et 0.30 |
| ALPINE_WATER_RICHNESS | Alpes = score eleve (eau abondante) | score > 60 |

Le rapport complet est sauvegarde dans `water_model/data/outputs/water_validation_report.txt`.


## Configuration (config.yaml)

Les parametres cles sont dans `water_model/config.yaml` :

```yaml
wsi:
  weights:
    bws:     0.60    # Poids stress de surface
    gws:     0.25    # Poids stress souterrain
    drought: 0.15    # Poids secheresse

grids:
  high:   { resolution_km: 5,  interpolation: bilinear }
  medium: { resolution_km: 15, interpolation: bilinear }
  low:    { resolution_km: 30, interpolation: nearest  }

hubeau_piezo:
  gws_weights_v11:
    aqueduct: 0.40
    brgm:     0.30
    piezo:    0.30
```

## Avenir

Pour l'instant nous sommes sur une normlisation lineaire, pour la v2 comparer les reusltats avec un sigmoide et ce construire un dataset pour voir les vrais impact des normalisation.


## References

- Hofste, R. et al. (2019). "Aqueduct 3.0: Updated Decision-Relevant Global Water Risk Indicators." WRI.
- McKee, T. B. et al. (1993). "The relationship of drought frequency and duration to time scales." AMS.
- OMM (2012). "Standardized Precipitation Index User Guide." WMO-No. 1090.
- BRGM. "BDLISA v3 : Base de Donnees des Limites des Systemes Aquiferes."
