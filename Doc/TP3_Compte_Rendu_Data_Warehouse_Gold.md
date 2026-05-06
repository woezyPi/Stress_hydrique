# TP Seance 3 - Le Data Warehouse & la Zone Gold

**MaisonShop - Star Schema - Faits - Dimensions**

**Projet BigQuery** : `data-warehouse-491214`  
**Region** : EU  
**Datasets** : `bronze` | `silver` | `gold`

---

## Etape preliminaire - Creation du dataset Gold

Le dataset `gold` a ete cree dans BigQuery avec la commande :
```sql
-- Via CLI :
bq mk --location=EU --dataset data-warehouse-491214:gold
```

> **Capture d'ecran a ajouter** : panneau gauche BigQuery montrant les 3 datasets (bronze, silver, gold).

---

## Phase 0 - Completer la Silver (clients et produits)

### Requete 0.1 - silver.clients

```sql
CREATE OR REPLACE TABLE silver.clients AS
SELECT
  SAFE_CAST(client_id AS INT64) AS client_id,
  prenom,
  nom,
  INITCAP(ville) AS ville,
  SAFE.PARSE_DATE('%Y-%m-%d', date_inscription) AS date_inscription
FROM bronze.clients
WHERE SAFE_CAST(client_id AS INT64) IS NOT NULL;
```

### Requete 0.2 - silver.produits

```sql
CREATE OR REPLACE TABLE silver.produits AS
SELECT
  SAFE_CAST(produit_id AS INT64) AS produit_id,
  nom_produit,
  COALESCE(categorie, 'Non classe') AS categorie,
  SAFE_CAST(prix_unitaire AS FLOAT64) AS prix_unitaire
FROM bronze.produits
WHERE SAFE_CAST(produit_id AS INT64) IS NOT NULL;
```

### Q1 - Combien de lignes contient silver.clients ? Et silver.produits ?

| Table             | Nombre de lignes |
|-------------------|-----------------|
| bronze.clients    | 301             |
| **silver.clients**    | **300**         |
| bronze.produits   | 29              |
| **silver.produits**   | **28**          |

**Interpretation** :
- `silver.clients` contient **300 lignes** contre 301 dans bronze. La ligne en moins correspond a la ligne d'en-tete du CSV (ou `client_id` n'est pas un entier valide), filtree par `SAFE_CAST(...) IS NOT NULL`.
- `silver.produits` contient **28 lignes** contre 29 dans bronze. Meme raison : la ligne d'en-tete a ete exclue.

---

## Phase 1 - Creer gold.dim_date

### Requete 1.1 - gold.dim_date

```sql
CREATE OR REPLACE TABLE gold.dim_date AS
SELECT
  date,
  EXTRACT(YEAR FROM date) AS annee,
  CONCAT('T', CAST(EXTRACT(QUARTER FROM date) AS STRING)) AS trimestre,
  EXTRACT(MONTH FROM date) AS mois_num,
  FORMAT_DATE('%B', date) AS mois_nom,
  FORMAT_DATE('%A', date) AS jour_semaine,
  EXTRACT(DAYOFWEEK FROM date) IN (1, 7) AS est_weekend
FROM UNNEST(
  GENERATE_DATE_ARRAY('2023-01-01', '2024-12-31', INTERVAL 1 DAY)
) AS date;
```

> Note : la plage a ete etendue a 2023-2024 car les commandes couvrent les deux annees.

### Q2 - Combien de lignes contient gold.dim_date ?

**Resultat** : **731 lignes** (365 jours en 2023 + 366 jours en 2024).

Les lignes ou `est_weekend = true` representent les **samedis et dimanches** (jours non ouvres). Cela permet des analyses separant jours ouvres et week-ends.

### Q3 - SELECT * FROM gold.dim_date WHERE trimestre = 'T4' LIMIT 5

| date       | annee | trimestre | mois_num | mois_nom  | jour_semaine | est_weekend |
|------------|-------|-----------|----------|-----------|--------------|-------------|
| 2023-10-22 | 2023  | T4        | 10       | October   | Sunday       | true        |
| 2023-10-27 | 2023  | T4        | 10       | October   | Friday       | false       |
| 2023-11-09 | 2023  | T4        | 11       | November  | Thursday     | false       |
| 2023-11-12 | 2023  | T4        | 11       | November  | Sunday       | true        |
| 2023-12-05 | 2023  | T4        | 12       | December  | Tuesday      | false       |

**Interpretation** : Le champ `trimestre` permet de regrouper les donnees par trimestre (T1 = Jan-Mar, T2 = Avr-Jun, T3 = Jul-Sep, T4 = Oct-Dec) pour des analyses trimestrielles du chiffre d'affaires.

> **Capture d'ecran a ajouter** : resultat de la requete Q3 dans BigQuery.

---

## Phase 2 - Creer gold.dim_clients et gold.dim_produits

### Requete 2.1 - gold.dim_clients

```sql
CREATE OR REPLACE TABLE gold.dim_clients AS
SELECT
  client_id,
  prenom,
  nom,
  ville,
  date_inscription
FROM silver.clients;
```

### Requete 2.2 - gold.dim_produits

```sql
CREATE OR REPLACE TABLE gold.dim_produits AS
SELECT
  produit_id,
  nom_produit,
  categorie,
  prix_unitaire
FROM silver.produits
WHERE produit_id NOT IN (29, 30);
```

### Q4 - Combien de produits contient gold.dim_produits ?

**Resultat** : **28 produits**.

Les IDs 29 et 30 sont exclus car ce sont des **produits fantomes** (produits de test ou errones qui ne correspondent pas a de vrais articles du catalogue MaisonShop). Leur exclusion garantit l'integrite des analyses.

### Q5 - Combien de categories distinctes ?

**Resultat** : **7 categories distinctes** :

| categorie      |
|----------------|
| Bureau         |
| Chambre        |
| Cuisine        |
| Jardin         |
| Non classe     |
| Salle de bain  |
| Salon          |

> **Capture d'ecran a ajouter** : resultat des categories distinctes.

---

## Phase 3 - Creer gold.fact_ventes

### Requete 3.1 - gold.fact_ventes

```sql
CREATE OR REPLACE TABLE gold.fact_ventes AS
SELECT
  ROW_NUMBER() OVER (ORDER BY c.commande_id) AS vente_id,
  c.commande_id,
  c.client_id,
  c.produit_id,
  c.date_commande,
  c.montant_ttc,
  c.statut
FROM silver.commandes c
WHERE c.produit_id IN (
  SELECT produit_id FROM silver.produits
  WHERE produit_id NOT IN (29, 30)
);
```

### Q6 - Combien de lignes contient gold.fact_ventes ?

| Table             | Nombre de lignes |
|-------------------|-----------------|
| silver.commandes  | 481             |
| **gold.fact_ventes**  | **474**         |

**Interpretation** : 481 - 474 = **7 lignes exclues**. Ces 7 commandes concernaient les produits fantomes (IDs 29 et 30) qui ont ete filtres pour garantir la coherence du Star Schema.

### Q7 - Verification de l'integrite referentielle

| Verification                  | Resultat |
|-------------------------------|----------|
| Orphelins clients (client_id) | **0**    |
| Orphelins produits (produit_id) | **0**  |
| Dates manquantes (date_commande) | **60** |

**Interpretation** :
- **0 orphelins clients** : tous les `client_id` de fact_ventes existent dans dim_clients.
- **0 orphelins produits** : tous les `produit_id` de fact_ventes existent dans dim_produits.
- **60 dates manquantes** : ces 60 commandes ont une `date_commande` a NULL dans les donnees source. Elles ne peuvent pas etre jointes a dim_date. Cela n'affecte pas l'integrite du schema mais signale un probleme de qualite en amont (donnees bronze).

---

## Phase 4 - Requetes analytiques sur Gold

### Requete 4.1 - CA mensuel 2023

```sql
SELECT
  d.mois_num, d.mois_nom,
  COUNT(f.vente_id) AS nb_commandes,
  ROUND(SUM(f.montant_ttc), 2) AS ca_mensuel
FROM gold.fact_ventes f
JOIN gold.dim_date d ON f.date_commande = d.date
GROUP BY d.mois_num, d.mois_nom
ORDER BY d.mois_num;
```

| Mois      | Nb commandes | CA mensuel |
|-----------|-------------|------------|
| January   | 29          | 2 721,89   |
| February  | 27          | 1 885,99   |
| March     | 36          | 3 073,23   |
| April     | 33          | 2 338,76   |
| May       | 42          | 3 788,50   |
| June      | 45          | 3 655,75   |
| July      | 44          | 3 712,74   |
| August    | 35          | 2 442,98   |
| September | 27          | 2 319,16   |
| October   | 39          | 3 050,67   |
| November  | 41          | 2 935,65   |
| December  | 16          | 1 396,71   |

### Q8 - Quel est le mois avec le CA le plus eleve ? Et le moins de commandes ?

- **CA le plus eleve** : **Mai (May)** avec **3 788,50 EUR** (42 commandes)
- **Mois avec le moins de commandes** : **Decembre (December)** avec seulement **16 commandes** (1 396,71 EUR)

> **Capture d'ecran a ajouter** : resultat de la requete 4.1.

---

### Requete 4.2 - Top 5 produits par CA

```sql
SELECT
  dp.nom_produit, dp.categorie,
  COUNT(f.vente_id) AS nb_ventes,
  ROUND(SUM(f.montant_ttc), 2) AS ca_total
FROM gold.fact_ventes f
JOIN gold.dim_produits dp ON f.produit_id = dp.produit_id
GROUP BY dp.nom_produit, dp.categorie
ORDER BY ca_total DESC
LIMIT 5;
```

| Produit                        | Categorie  | Nb ventes | CA total  |
|--------------------------------|------------|-----------|-----------|
| Miroir rond dore 60cm          | Salon      | 20        | 4 720,94  |
| Lampe de bureau LED flexible   | Non classe | 26        | 3 252,16  |
| Lampe de salon tactile         | Salon      | 18        | 2 560,33  |
| Parure de lit 240x260          | Chambre    | 17        | 2 059,07  |
| Poele antiadhesive 28cm        | Cuisine    | 18        | 1 826,11  |

### Q9 - Quelle categorie genere le plus de CA ?

- La categorie **Salon** domine le top 5 avec 2 produits (Miroir + Lampe tactile) totalisant **7 281,27 EUR**.
- **Categorie surprenante** : **"Non classe"** apparait en 2e position (Lampe de bureau LED). Ce produit n'a pas de categorie attribuee dans les donnees source, ce qui constitue un probleme de qualite des donnees a corriger en amont.

> **Capture d'ecran a ajouter** : resultat du top 5 produits.

---

### Requete 4.3 - CA par ville

```sql
SELECT
  dc.ville,
  COUNT(f.vente_id) AS nb_commandes,
  ROUND(SUM(f.montant_ttc), 2) AS ca_total
FROM gold.fact_ventes f
JOIN gold.dim_clients dc ON f.client_id = dc.client_id
GROUP BY dc.ville
ORDER BY ca_total DESC
LIMIT 10;
```

| Ville       | Nb commandes | CA total  |
|-------------|-------------|-----------|
| Bordeaux    | 47          | 3 905,66  |
| Marseille   | 45          | 3 224,73  |
| Lille       | 34          | 3 155,91  |
| Nantes      | 34          | 3 084,53  |
| Nice        | 36          | 3 050,44  |
| Tours       | 31          | 2 456,87  |
| Rennes      | 29          | 2 393,86  |
| Strasbourg  | 27          | 2 383,54  |
| Paris       | 37          | 2 188,56  |
| Montpellier | 31          | 2 162,30  |

### Q10 - Quelle est la ville qui genere le plus de CA ?

- **Bordeaux** est la ville n.1 avec **3 905,66 EUR** de CA (47 commandes).
- **6 villes** representent environ 50% du CA total (37 422,53 EUR) :
  - Bordeaux (10,4%), Marseille (19,1%), Lille (27,5%), Nantes (35,7%), Nice (43,9%), **Tours (50,4%)**

> **Capture d'ecran a ajouter** : resultat du CA par ville.

---

## Phase 5 - Controle qualite Gold

### Check 1 - Integrite referentielle clients

```sql
SELECT COUNT(*) AS orphelins_clients
FROM gold.fact_ventes f
LEFT JOIN gold.dim_clients dc ON f.client_id = dc.client_id
WHERE dc.client_id IS NULL;
```

**Resultat obtenu** : **0**

**Interpretation** : Aucune commande ne reference un client inexistant. L'integrite referentielle entre `fact_ventes` et `dim_clients` est parfaite.

---

### Check 2 - Integrite referentielle produits

```sql
SELECT COUNT(*) AS orphelins_produits
FROM gold.fact_ventes f
LEFT JOIN gold.dim_produits dp ON f.produit_id = dp.produit_id
WHERE dp.produit_id IS NULL;
```

**Resultat obtenu** : **0**

**Interpretation** : Aucune commande ne reference un produit inexistant. L'exclusion des produits fantomes (IDs 29, 30) a ete appliquee de maniere coherente dans `dim_produits` et `fact_ventes`.

---

### Check 3 - Integrite referentielle dates

```sql
SELECT COUNT(*) AS dates_manquantes
FROM gold.fact_ventes f
LEFT JOIN gold.dim_date d ON f.date_commande = d.date
WHERE d.date IS NULL;
```

**Resultat obtenu** : **60**

**Interpretation** : 60 commandes ont une `date_commande` a NULL dans les donnees source (zone bronze). Ces lignes ne trouvent pas de correspondance dans `dim_date`. Ce n'est pas un defaut du Star Schema mais un probleme de qualite des donnees en amont qui devrait etre traite dans la zone Silver (imputation ou exclusion des dates manquantes).

---

## Synthese du Star Schema Gold

```
                    +----------------+
                    | gold.dim_date  |
                    |----------------|
                    | date (PK)      |
                    | annee          |
                    | trimestre      |
                    | mois_num       |
                    | mois_nom       |
                    | jour_semaine   |
                    | est_weekend    |
                    +-------+--------+
                            |
                            | date_commande = date
                            |
+------------------+   +----+---------------+   +-------------------+
| gold.dim_clients |   | gold.fact_ventes   |   | gold.dim_produits |
|------------------|   |--------------------|   |-------------------|
| client_id (PK)   +---+ vente_id (PK)      +---+ produit_id (PK)   |
| prenom           |   | commande_id        |   | nom_produit       |
| nom              |   | client_id (FK)     |   | categorie         |
| ville            |   | produit_id (FK)    |   | prix_unitaire     |
| date_inscription |   | date_commande (FK) |   +-------------------+
+------------------+   | montant_ttc        |
                       | statut             |
                       +--------------------+
```

### Resume des livrables

| Element                           | Statut |
|-----------------------------------|--------|
| Dataset `gold` cree               | OK     |
| `gold.dim_date` (731 lignes)      | OK     |
| `gold.dim_clients` (300 lignes)   | OK     |
| `gold.dim_produits` (28 lignes)   | OK     |
| `gold.fact_ventes` (474 lignes)   | OK     |
| 3 checks integrite referentielle  | OK     |
| Reponses Q1 a Q10                 | OK     |
| Mois CA le plus eleve             | **Mai - 3 788,50 EUR** |
| Top 3 villes                      | **Bordeaux, Marseille, Lille** |

---

> **IMPORTANT** : Ce compte rendu necessite que tu ajoutes les captures d'ecran depuis l'interface BigQuery Console. Les emplacements sont indiques par "Capture d'ecran a ajouter".
