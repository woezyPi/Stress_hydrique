# Olara V0 — radar de risque hydrique DC

**Périmètre** : 4 sites représentatifs, 4 métriques physiques.
**Fenêtre temporelle** : 1991–2022 (baseline climato 1991–2020).

## Métriques
- **f1** : nombre d'années (sur 32) où le débit mensuel a chuté en dessous de z = −2 (étiage critique).
- **f2** : durée moyenne (en mois consécutifs) des épisodes z < −1.
- **f3** : tendance d'aggravation = pente du z_Q annuel par décennie. Négatif = aggravation.
- **f4** : nombre de mois (juin/juillet/août) sur 32 ans avec étiage z<−1 ET ≥3 jours consécutifs T_max > 35 °C.

## Résultats — 4 sites
| Site | Station Banque Hydro | f1 | f2 | f3 | f4 |
|---|---|---:|---:|---:|---:|
| Paris-Saint-Denis | F700000102 (La Seine à Paris - Austerlitz - station limni - secours CT) | 0 | 1.80 | +0.830 | 0 |
| Marseille-L2 | Y441403001 (L'Huveaune à Roquevaire) | 0 | 1.00 | -0.176 | 0 |
| Tours | K671091001 (Le Cher à Tours - Pont Saint-Sauveur) | 0 | 2.00 | -0.364 | 1 |
| Lyon-Venissieux | U472002001 () | nan | nan | +nan | nan |

## Lecture
Chaque axe du radar (cf. `data/v0/radar_v0.png`) est normalisé entre les 4 sites :
0 = meilleur des 4, 1 = pire des 4. C'est une **comparaison relative**, pas un score absolu.
Pour v1, prévoir un benchmark national (percentiles France) afin de produire une grille absolue.

## Limites assumées V0
- Une seule station Banque Hydro par site (la plus proche, ≤ 30 km).
- Open-Meteo archive utilise ERA5 → résolution 0,25° (~28 km).
- Pas de prise en compte régulation barrages, réseau, mitigation cooling.
- Tendance f3 sensible aux extrêmes (ex. 2022).

## Validation visuelle attendue
- Marseille en tête sur f1, f3, f4 (méditerranéen sec et chaud).
- Tours différencié par f2 (cluster où la gestion humaine domine — durée d'étiage cohérente avec le climat).
- Paris/Lyon similaires structurellement, différenciés par f4 (canicule urbaine + débits régulés).