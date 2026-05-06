"""Build the clean H_local notebook."""
import json

cells = []

def md(cell_id, source):
    cells.append({"cell_type": "markdown", "id": cell_id, "source": source, "metadata": {}})

def code(cell_id, source):
    cells.append({"cell_type": "code", "id": cell_id, "source": source,
                  "metadata": {}, "execution_count": None, "outputs": []})

# ═══════════════════════════════════════════════════════════════
# 1. HEADER + MATHEMATICAL DEFINITION
# ═══════════════════════════════════════════════════════════════
md("header", r"""# $H_{\text{local}}(x,t)$ — Score de stress hydrique territorial

## Objectif

Score décisionnel pour évaluer la **contrainte hydrique territoriale** d'un site candidat
à l'implantation d'un data center. Ce score ne modélise pas l'hydrologie ;
il agrège des indicateurs publics en un signal interprétable et stable.

---

## 1. Définition mathématique

$$H_{\text{local}}(x,t) \;=\; \beta \;\cdot\; S_{\text{struct}}(x) \;+\; (1-\beta) \;\cdot\; S_{\text{conj}}(x,t) \qquad \in [0,\,1]$$

où $0$ = favorable, $1$ = contrainte forte.

### 1.1 Composante structurelle (invariante temporelle)

$$S_{\text{struct}}(x) \;=\; \alpha_B \;\widetilde{BWS}(x) \;+\; \alpha_A \; A(x) \;+\; \alpha_G \;\widetilde{GWS}(x)$$

| Variable | Notation | Transformation | Source | Référence |
|---|---|---|---|---|
| Baseline Water Stress | $\widetilde{BWS} = \text{clip}\!\left(\frac{BWS_{\text{raw}}}{0.6},\;0,\;1\right)$ | Linéaire, rescalé sur max France | WRI Aqueduct 4.0 | Kuzma et al. (2023) |
| Aridity | $A = \frac{1}{1 + (AI / 0.65)^3}$, $\;AI = \overline{P} / \overline{PET}$ | Sigmoïde calibrée sur seuils UNEP | ERA5 baseline 1981–2010 | UNEP (1992), Middleton & Thomas (1997) |
| Groundwater Stress | $\widetilde{GWS} = \text{clip}\!\left(\frac{GWS_{\text{raw}}}{0.15},\;0,\;1\right)$ | Linéaire, rescalé sur max France | WRI Aqueduct 4.0 | Kuzma et al. (2023) |

### 1.2 Composante conjoncturelle (dynamique)

$$S_{\text{conj}}(x,t) \;=\; \max\!\Big(\,f\big(\text{SPEI-12}(x,t)\big),\;\; f\big(\text{SPEI-3}(x,t)\big)\,\Big)$$

$$f(\text{SPEI}) \;=\; \text{clip}\!\left(\frac{-\text{SPEI}^*}{3},\;0,\;1\right), \qquad \text{SPEI}^* = \text{clip}(\text{SPEI},\;-3,\;+3)$$

| Choix | Justification |
|---|---|
| $\max(\cdot)$ | Conservateur : retient le pire signal entre déficit cumulé (12 mois) et choc court (3 mois). Choix **produit**, pas scientifiquement neutre. |
| Cap $[-3, +3]$ | Au-delà, le SPEI est un artefact numérique du fit log-logistique (saturation de $\Phi^{-1}$). Stabilise les extrêmes. |
| Fit par mois calendaire | 12 distributions fisk ajustées séparément. Réf : Beguería et al. (2014). |

### 1.3 Paramètres

| Symbole | Valeur | Rôle |
|---|---|---|
| $\beta$ | $0.70$ | Part du structurel dans le score final |
| $\alpha_B$ | $0.50$ | Poids intra-bloc BWS |
| $\alpha_A$ | $0.36$ | Poids intra-bloc aridité |
| $\alpha_G$ | $0.14$ | Poids intra-bloc GWS |

**Tous les poids sont fixés à dire d'expert, non calibrés.**

### 1.4 Correction pixel côtier ERA5

ERA5 à 0.25° mélange terre et mer pour les pixels côtiers, ce qui dilue la PET.
Si $PET_{\text{centre}} < 60\%$ de $\max(PET_{3\times3})$, on reroute vers le gridpoint
continental voisin (PET max). Affecte Marseille et Montpellier.
Réf : Hersbach et al. (2020), QJRMS.

### 1.5 Références

| Ref | Citation |
|---|---|
| Kuzma et al. (2023) | *Aqueduct 4.0: Updated Decision-Relevant Global Water Risk Indicators.* WRI Technical Note. |
| Vicente-Serrano et al. (2010) | *A Multiscalar Drought Index Sensitive to Global Warming: The SPEI.* J. Climate, 23(7), 1696–1718. |
| Beguería et al. (2014) | *Standardized precipitation evapotranspiration index (SPEI) revisited.* Int. J. Climatol., 34, 3001–3023. |
| UNEP (1992) | *World Atlas of Desertification.* United Nations Environment Programme. |
| Middleton & Thomas (1997) | *World Atlas of Desertification.* 2nd ed., Arnold. |
| Hersbach et al. (2020) | *The ERA5 global reanalysis.* QJRMS, 146(730), 1999–2049. |
| McKee et al. (1993) | *The relationship of drought frequency and duration to time scales.* 8th Conf. Applied Climatology, AMS. |
| Vidal et al. (2010) | *A 50-year high-resolution atmospheric reanalysis over France.* Int. J. Climatol., 30(11), 1627–1644. |
| Bonnet et al. (2024) | *The 2022 drought in France.* INRAE report. |""")

# ═══════════════════════════════════════════════════════════════
# 2. SETUP
# ═══════════════════════════════════════════════════════════════
md("setup_md", "---\n## 2. Setup")

code("setup_code", """import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -- Chemins --
BASE = Path("../water_model/data")
RAW  = BASE / "raw"
AQUEDUCT_BWS = RAW / "aqueduct" / "bws_raw.tif"
AQUEDUCT_GWS = RAW / "aqueduct" / "gws_raw.tif"
ERA5_FULL    = RAW / "era5" / "era5_monthly_1981_2022.nc"

# -- Sites de validation --
SITES = {
    "Paris-Saclay":  (48.73, 2.17),
    "Lyon":          (45.76, 4.83),
    "Marseille":     (43.30, 5.37),
    "Strasbourg":    (48.57, 7.75),
    "Brest":         (48.39, -4.49),
    "Rennes":        (48.11, -1.68),
    "Bordeaux":      (44.84, -0.58),
    "Montpellier":   (43.61, 3.88),
}

# -- Parametres du modele --
BWS_MAX_FR = 0.60     # max France ~ 0.5, marge 20%
GWS_MAX_FR = 0.15     # max France ~ 0.1, marge 50%
SPEI_CAP   = 3.0      # [-3, +3] credible
ALPHA = {"bws": 0.50, "aridity": 0.36, "gws": 0.14}
BETA  = 0.70           # 70% structurel

VALIDATION_DATES = [
    (2015, 7, "2015 -- normale"),
    (2018, 7, "2018 -- canicule"),
    (2022, 7, "2022 -- secheresse"),
]

print("Setup OK")
for f in [AQUEDUCT_BWS, AQUEDUCT_GWS, ERA5_FULL]:
    print(f"  {f.name:30s} : {'OK' if f.exists() else 'MANQUANT'}")""")

# ═══════════════════════════════════════════════════════════════
# 3. CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════
md("core_md", "---\n## 3. Fonctions du modele")

code("core_code", """# ==================================================================
# 3.1 Extraction raster (Aqueduct)
# ==================================================================

def extract_raster_point(raster_path, lat, lon):
    \"\"\"Extrait la valeur au point (lat, lon) dans un GeoTIFF.\"\"\"
    with rasterio.open(raster_path) as src:
        vals = list(src.sample([(lon, lat)]))
        val = float(vals[0][0])
        if src.nodata is not None and val == src.nodata:
            return np.nan
        return val


# ==================================================================
# 3.2 SPEI -- fit log-logistique par mois calendaire
# ==================================================================
# Ref: Vicente-Serrano et al. (2010), Begueria et al. (2014)

def compute_spei(precip_mm, pet_mm, dates, window=12,
                 baseline_start=1981, baseline_end=2010):
    \"\"\"
    SPEI-N avec fit fisk PAR MOIS calendaire.
    12 distributions log-logistiques ajustees separement.
    \"\"\"
    D = precip_mm - pet_mm
    n = len(D)
    D_acc = np.full(n, np.nan)
    for i in range(window - 1, n):
        D_acc[i] = np.sum(D[i - window + 1 : i + 1])

    years, months = dates.year, dates.month
    mask_base = (years >= baseline_start) & (years <= baseline_end)

    monthly_params = {}
    for m in range(1, 13):
        mask_m = (months == m) & mask_base & ~np.isnan(D_acc)
        base_m = D_acc[mask_m]
        if len(base_m) < 10:
            monthly_params[m] = None
            continue
        shift_m = -np.min(base_m) + 1.0
        try:
            c, _, scale = stats.fisk.fit(base_m + shift_m, floc=0)
            _, ks_pval = stats.kstest(base_m + shift_m, 'fisk', args=(c, 0, scale))
            monthly_params[m] = (c, scale, shift_m, ks_pval, len(base_m))
        except Exception:
            monthly_params[m] = None

    spei = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(D_acc[i]):
            continue
        params = monthly_params.get(months[i])
        if params is None:
            continue
        c, scale, shift_m, _, _ = params
        prob = stats.fisk.cdf(D_acc[i] + shift_m, c, loc=0, scale=scale)
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        spei[i] = stats.norm.ppf(prob)

    ks_pvals = [p[3] for p in monthly_params.values() if p is not None]
    return {
        'spei': spei, 'D_monthly': D, 'D_acc': D_acc,
        'ks_pvalue': np.mean(ks_pvals) if ks_pvals else 0.0,
        'monthly_params': monthly_params,
    }


# ==================================================================
# 3.3 Aridity -- sigmoide UNEP
# ==================================================================
# Ref: UNEP (1992), Middleton & Thomas (1997)
# AI_mid = 0.65 (seuil sub-humide sec), k = 3

def aridity_stress(ai, ai_mid=0.65, k=3.0):
    \"\"\"Stress = 1 / (1 + (AI / AI_mid)^k).\"\"\"
    if ai <= 0:
        return 1.0
    return float(1.0 / (1.0 + (ai / ai_mid) ** k))

def compute_aridity(precip_mm, pet_mm, dates,
                    baseline_start=1981, baseline_end=2010):
    \"\"\"AI = P_annuel / PET_annuel sur baseline, puis stress sigmoide.\"\"\"
    years = dates.year
    mask = (years >= baseline_start) & (years <= baseline_end)
    n_years = baseline_end - baseline_start + 1
    p_ann = np.sum(precip_mm[mask]) / n_years
    pet_ann = np.sum(pet_mm[mask]) / n_years
    if pet_ann <= 0:
        return 0.0, 999.0
    ai = p_ann / pet_ann
    return aridity_stress(ai), ai


# ==================================================================
# 3.4 Correction pixel cotier ERA5
# ==================================================================
# Ref: Hersbach et al. (2020)

def find_land_gridpoint(lat, lon, ds, threshold=0.60):
    \"\"\"Detecte pixel cotier, reroute vers voisin continental.\"\"\"
    td = 'valid_time' if 'valid_time' in ds.dims else 'time'
    lats, lons = ds.latitude.values, ds.longitude.values
    i_lat = int(np.argmin(np.abs(lats - lat)))
    i_lon = int(np.argmin(np.abs(lons - lon)))
    times = pd.DatetimeIndex(ds[td].values)
    month_idx = next(k for k in range(len(times)-1, -1, -1) if times[k].month == 7)
    lat_sl = slice(max(0, i_lat-1), min(len(lats), i_lat+2))
    lon_sl = slice(max(0, i_lon-1), min(len(lons), i_lon+2))
    pev_3x3 = ds['pev'].isel({td: month_idx, 'latitude': lat_sl, 'longitude': lon_sl})
    pet_3x3 = np.maximum(-pev_3x3.values.astype(float), 0.0)
    pet_center = pet_3x3[min(1, i_lat), min(1, i_lon)]
    pet_max = np.max(pet_3x3)
    ratio = pet_center / pet_max if pet_max > 0 else 1.0
    if ratio >= threshold:
        return lats[i_lat], lons[i_lon], False
    local_lats, local_lons = lats[lat_sl], lons[lon_sl]
    max_pos = np.unravel_index(np.argmax(pet_3x3), pet_3x3.shape)
    return float(local_lats[max_pos[0]]), float(local_lons[max_pos[1]]), True


# ==================================================================
# 3.5 Score final
# ==================================================================

def spei_to_stress(spei_val, cap=SPEI_CAP):
    \"\"\"f(SPEI) = clip(-clip(SPEI, -cap, cap) / 3, 0, 1).\"\"\"
    if np.isnan(spei_val):
        return np.nan
    return float(np.clip(-np.clip(spei_val, -cap, cap) / 3.0, 0, 1))

print("Fonctions chargees")""")

# ═══════════════════════════════════════════════════════════════
# 4. PIPELINE
# ═══════════════════════════════════════════════════════════════
md("pipeline_md", "---\n## 4. Pipeline")

code("pipeline_code", """# ==================================================================
# 4. Calcul H_local multi-sites x multi-annees
# ==================================================================

ds = xr.open_dataset(ERA5_FULL)
td = 'valid_time' if 'valid_time' in ds.dims else 'time'

# 4.1 Detection pixels cotiers
coastal_map = {}
print("=== Detection pixels cotiers ===")
for name, (lat, lon) in SITES.items():
    blat, blon, coastal = find_land_gridpoint(lat, lon, ds)
    coastal_map[name] = (blat, blon, coastal)
    if coastal:
        print(f"  {name:14s} : COASTAL -> ({blat:.2f}, {blon:.2f})")
print()

# 4.2 Calcul
results = []
for name, (lat, lon) in SITES.items():
    bws_raw = extract_raster_point(AQUEDUCT_BWS, lat, lon)
    gws_raw = extract_raster_point(AQUEDUCT_GWS, lat, lon)
    pet_lat, pet_lon, is_coastal = coastal_map[name]

    tp = ds['tp'].sel(latitude=lat, longitude=lon, method='nearest')
    t = pd.DatetimeIndex(tp[td].values)
    dy = t.days_in_month.values.astype(float)
    p = np.maximum(tp.values.astype(float) * dy * 1000, 0)
    pv = ds['pev'].sel(latitude=pet_lat, longitude=pet_lon, method='nearest')
    pet = np.maximum(-pv.values.astype(float) * dy * 1000, 0)

    b = float(np.clip(bws_raw / BWS_MAX_FR, 0, 1)) if not np.isnan(bws_raw) else np.nan
    g = float(np.clip(gws_raw / GWS_MAX_FR, 0, 1)) if not np.isnan(gws_raw) else np.nan
    a_stress, ai = compute_aridity(p, pet, t)
    sr12 = compute_spei(p, pet, t, window=12)
    sr3  = compute_spei(p, pet, t, window=3)

    for year, month, label in VALIDATION_DATES:
        idx = next((i for i, tt in enumerate(t)
                    if tt.year == year and tt.month == month), None)
        if idx is None:
            continue

        # S_struct
        comps = {'bws': b, 'aridity': a_stress, 'gws': g}
        num = sum(ALPHA[k] * v for k, v in comps.items()
                  if v is not None and not np.isnan(v))
        den = sum(ALPHA[k] for k, v in comps.items()
                  if v is not None and not np.isnan(v))
        s_struct = num / den if den > 0 else np.nan

        # S_conj = max(f(SPEI-12), f(SPEI-3))
        d12 = spei_to_stress(sr12['spei'][idx])
        d3  = spei_to_stress(sr3['spei'][idx])
        if np.isnan(d12) and np.isnan(d3):
            s_conj = np.nan
        elif np.isnan(d3):
            s_conj = d12
        elif np.isnan(d12):
            s_conj = d3
        else:
            s_conj = max(d12, d3)

        # H = beta * S_struct + (1-beta) * S_conj
        if np.isnan(s_struct) and np.isnan(s_conj):
            H = np.nan
        elif np.isnan(s_struct):
            H = s_conj
        elif np.isnan(s_conj):
            H = s_struct
        else:
            H = BETA * s_struct + (1 - BETA) * s_conj
        H = float(np.clip(H, 0, 1)) if not np.isnan(H) else np.nan

        c_s = BETA * s_struct if not np.isnan(s_struct) else 0.0
        c_c = (1 - BETA) * s_conj if not np.isnan(s_conj) else 0.0
        c_t = c_s + c_c
        pct_s = c_s / c_t * 100 if c_t > 0 else np.nan

        results.append({
            'Site': name, 'year': year, 'label': label,
            'S_struct': s_struct, 'S_conj': s_conj, 'H': H,
            'BWS': b, 'Aridity': a_stress, 'GWS': g,
            'AI': ai, 'SPEI12': sr12['spei'][idx], 'SPEI3': sr3['spei'][idx],
            '%_struct': pct_s})

ds.close()
df = pd.DataFrame(results)
print(f"Pipeline termine : {len(df)} observations "
      f"({len(SITES)} sites x {len(VALIDATION_DATES)} dates)")""")

# ═══════════════════════════════════════════════════════════════
# 5. VALIDATION
# ═══════════════════════════════════════════════════════════════
md("valid_md", r"""---
## 5. Validation

Bloc minimal : **cohérence physique**, pas calibration.

**Attendus** (littérature + connaissance terrain) :

- **Gradient NW vers SE** : la France métropolitaine présente un gradient hydrique
  océanique (Brest) vers méditerranéen (Marseille). Le SE doit scorer plus haut.
  Réf : Vidal et al. (2010), *A 50-year high-resolution atmospheric reanalysis over France*, Int. J. Climatol.

- **Ordering temporel** : 2022 (sécheresse record, Bonnet et al. 2024) > 2018 (canicule courte) >= 2015 (normale).
  Note : SPEI-12 ne capte pas 2018 comme choc car le bilan annuel était positif ; SPEI-3 le détecte.

- **Marseille >> Brest** : AI Marseille = 0.4 (semi-aride), AI Brest = 1.8 (très humide).
  Réf : classification UNEP, confirmée par Météo-France normales 1991-2020.

- **Contributions** : en année sans crise, le structurel doit dominer (>50%). En crise, le conjoncturel monte.""")

code("valid_code", """# ==================================================================
# 5.1 Tableau par annee
# ==================================================================

for year, _, label in VALIDATION_DATES:
    sub = df[df['year'] == year].set_index('Site')
    print(f"=== {label} ===")
    print(sub[['S_struct', 'S_conj', 'H', '%_struct']].round(3).to_string())
    print()

# ==================================================================
# 5.2 Pivot H par annee
# ==================================================================

pivot = df.pivot(index='Site', columns='year', values='H')
print("=== H(x,t) par site et par annee ===\\n")
print(pivot.round(3).to_string())
print()
for year, _, label in VALIDATION_DATES:
    print(f"  {label:25s} : H moyen = {pivot[year].mean():.3f}")

# ==================================================================
# 5.3 Test : gradient NW -> SE
# ==================================================================

NW = ['Brest', 'Rennes']
SE = ['Marseille', 'Montpellier', 'Lyon']

print("\\n=== Gradient NW -> SE ===")
all_ok = True
for year, _, label in VALIDATION_DATES:
    sub = df[df['year'] == year].set_index('Site')
    nw_h = sub.loc[NW, 'H'].mean()
    se_h = sub.loc[SE, 'H'].mean()
    ok = se_h > nw_h
    all_ok &= ok
    print(f"  {label:25s} : NW={nw_h:.3f}, SE={se_h:.3f}, "
          f"ratio={se_h/nw_h:.2f}x  {'PASS' if ok else 'FAIL'}")
print(f"  => {'PASS' if all_ok else 'FAIL'} sur les 3 annees")

# ==================================================================
# 5.4 Test : Marseille vs Brest
# ==================================================================

print("\\n=== Marseille vs Brest ===")
for year, _, label in VALIDATION_DATES:
    sub = df[df['year'] == year].set_index('Site')
    hm = sub.loc['Marseille', 'H']
    hb = sub.loc['Brest', 'H']
    print(f"  {label:25s} : Marseille={hm:.3f}, Brest={hb:.3f}, "
          f"delta={hm-hb:+.3f}  {'PASS' if hm > hb else 'FAIL'}")

sub22 = df[df['year'] == 2022].set_index('Site')
sm, sb = sub22.loc['Marseille'], sub22.loc['Brest']
print(f"\\n  Detail 2022 :")
print(f"  {'':14s}  {'Marseille':>10s}  {'Brest':>10s}")
print(f"  {'AI':14s}  {sm['AI']:10.2f}  {sb['AI']:10.2f}")
print(f"  {'Aridity':14s}  {sm['Aridity']:10.3f}  {sb['Aridity']:10.3f}")
print(f"  {'S_struct':14s}  {sm['S_struct']:10.3f}  {sb['S_struct']:10.3f}")
print(f"  {'S_conj':14s}  {sm['S_conj']:10.3f}  {sb['S_conj']:10.3f}")

# ==================================================================
# 5.5 Contribution struct vs conj
# ==================================================================

print("\\n=== Contribution struct vs conj (moyenne) ===")
for year, _, label in VALIDATION_DATES:
    sub = df[df['year'] == year]
    ps = sub['%_struct'].mean()
    status = ('STRUCT domine' if ps > 60
              else ('EQUILIBRE' if ps > 40
              else 'CONJ domine'))
    print(f"  {label:25s} : struct={ps:.0f}%, conj={100-ps:.0f}%  {status}")""")

# ═══════════════════════════════════════════════════════════════
# 6. LIMITATIONS
# ═══════════════════════════════════════════════════════════════
md("limits_md", r"""---
## 6. Limitations

Ce score est un **prototype opérationnel**, pas un modèle hydrologique.

### Ce que $H_{\text{local}}$ est

- Un **score décisionnel territorial brut** : il agrège des indicateurs publics pour comparer des sites.
- Un signal **interprétable** : chaque composante a un sens physique documenté.
- Un point de départ pour une analyse plus fine (couche besoin DC, calibration).

### Ce que $H_{\text{local}}$ n'est pas

| Limitation | Détail |
|---|---|
| **Pas un modèle hydrologique** | Pas de bilan hydrique, pas de modélisation pluie-débit, pas de nappe phréatique dynamique. |
| **Pas calibré** | Tous les poids ($\alpha$, $\beta$) sont fixés à dire d'expert. Pas d'AHP, pas de régression, pas de validation croisée. |
| **Pas DC-aware** | Ne modélise pas le besoin du data center (MW vers m3, type de refroidissement, profil saisonnier, contraintes de prélèvement). |
| **Score décisionnel, pas neutre** | Le $\max(\text{SPEI-3}, \text{SPEI-12})$ est conservateur par design. Le cap à $[-3,+3]$ perd de l'information extrême. |
| **Contributions = design, pas physique** | Les % struct/conj dépendent de la normalisation et des poids, pas d'une vérité physique. |
| **8 sites, 3 dates** | Validation minimale. Pas de couverture spatiale exhaustive ni de validation temporelle complète. |
| **ERA5 = 28 km** | Résolution insuffisante pour les effets locaux (vallées, microclimats urbains). |
| **Aridity = baseline figée** | L'AI est calculé sur 1981-2010. Sous changement climatique, cette référence vieillit. |

### Prochaines étapes

1. **Couche besoin DC** : transformer le score territorial en score de compatibilité site x data center.
2. **Calibration** : ajuster les poids sur des données de terrain (débits, piézométrie, retours d'exploitants).
3. **Résolution spatiale** : descendre à 5 km avec rasters prétraités.
4. **Temporalité** : intégrer des scénarios climatiques (CMIP6 / DRIAS-2020) pour projection 2030-2050.""")

# ═══════════════════════════════════════════════════════════════
# BUILD NOTEBOOK
# ═══════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}

out_path = "water_stress_H_local.ipynb"
with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"OK -- {len(cells)} cells written to {out_path}")
