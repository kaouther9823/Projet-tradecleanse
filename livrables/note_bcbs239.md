# NOTE TECHNIQUE — CONFORMITÉ BCBS 239
## Pipeline TradeCleanse · QuantAxis Capital, Paris
### À l'attention du : Risk Officer — Département Risque de Contrepartie
### Rédigée par : Data Engineer / Chef de Projet IA
### Date : 2024

---

## Objet

La présente note technique démontre la conformité du pipeline de qualité de données
TradeCleanse aux Principes BCBS 239 (Basel Committee on Banking Supervision —
"Principles for effective risk data aggregation and risk reporting", janvier 2013).

Ce pipeline traite le dataset consolidé `tradecleanse_raw.csv` (8 742 observations,
20 colonnes, 3 sources hétérogènes : Bloomberg Terminal, Murex, Refinitiv Eikon)
avant tout entraînement du modèle de scoring de risque de contrepartie.

---

## Principe 2 — Exactitude et intégrité des données

> *"Une banque devrait être en mesure de produire des données de risque exactes
> et fiables pour répondre aux demandes de reporting, tant dans des conditions
> normales qu'en période de crise."*

### Transformations garantissant l'exactitude

**Remplacement des valeurs sentinelles (Étape 1)**
Les exports Bloomberg, Murex et Refinitiv produisent des marqueurs non-standard
("N/A", "#VALUE!", "99999", "0.0" sur volatility_30d) qui masquent des données
réellement absentes. Le pipeline détecte et neutralise 22 patterns de sentinelles
textuelles et 7 patterns numériques colonne par colonne, transformant chaque
occultation en NaN explicite avant toute analyse.

**Correction des incohérences structurelles financières (Étape 5)**
Six règles métier sont appliquées mécaniquement à chaque ligne :
- `settlement_date < trade_date` → NaT (violation règle T+2 CSDR)
- `bid ≥ ask` → NaN sur les deux valeurs (fourchette physiquement impossible)
- `mid_price ≠ (bid+ask)/2` → recalcul systématique (donnée dérivée Bloomberg)
- `price` hors fourchette ±0.5% → substitution par `mid_price`
- `notional_eur ≤ 0` → valeur absolue (inversion de signe à l'export)
- Rating investment-grade + `default_flag=1` → NaN sur le rating (contradiction
  factuelle : un émetteur en défaut avéré ne peut pas être noté AAA/AA/A)

**Validation des domaines (Étape 6)**
Chaque colonne est contrôlée contre son domaine de validité réglementaire :
`country_risk ∈ [0,100]`, `volatility_30d ∈ (0.1, 200]`, `default_flag ∈ {0,1}`,
`quantity > 0`. Toute valeur hors domaine est isolée avant d'alimenter le modèle.

**Score de qualité mesuré** : le Data Quality Score (DQS = complétude×0.6 +
unicité×0.4) est calculé avant et après chaque run et versé dans le fichier
`tradecleanse_pipeline.log` pour audit ultérieur.

---

## Principe 3 — Complétude des données

> *"Une banque devrait être en mesure de capturer et d'agréger toutes les données
> de risque significatives à l'échelle du groupe."*

### Taux de complétude atteint

Le pipeline distingue trois régimes d'imputation selon le taux de NaN observé :

| Régime | Seuil | Traitement | Colonnes concernées |
|--------|-------|------------|---------------------|
| Critique | NaN sur clé primaire | Suppression de ligne | `trade_id` |
| Standard | < 70% NaN | Médiane (numérique) / Mode (catégoriel) + flag `_was_missing` | `price`, `bid`, `ask`, `volatility_30d`, `asset_class`... |
| Irrécupérable | > 70% NaN | Suppression de colonne | (cas non observé sur ce dataset) |

**Cas particuliers documentés :**

- `settlement_date` NaT → recalcul T+2 jours ouvrés (`pd.offsets.BusinessDay(2)`)
  plutôt qu'imputation par mode, car la convention de marché est déterministe.

- `credit_rating` NaN → attribution de `"d"` si `default_flag=1` (fait observé
  prime sur la note non encore révisée), sinon `"nr"` (not rated, traité comme
  BB dans les modèles internes pondérés). L'imputation par mode (`"BBB"`) est
  volontairement exclue : attribuer une note investment-grade à une contrepartie
  inconnue fausserait les pondérations RWA bâloises.

**Traçabilité de l'imputation :** chaque colonne imputée génère une colonne
binaire `colonne_was_missing` (0/1) permettant au modèle de distinguer les
valeurs observées des valeurs inférées, et au Risk Officer de quantifier
l'exposition aux données synthétiques dans chaque rapport de risque.

**Objectif de complétude** : le pipeline vise un taux de complétude global > 90%
sur le dataset nettoyé, mesuré et loggé à chaque exécution. Le seuil de 95%
sur les colonnes obligatoires (`trade_id`, `price`, `quantity`, `notional_eur`,
`trade_date`, `asset_class`) est validé par la suite Great Expectations
(Expectation 13).

---

## Principe 6 — Adaptabilité

> *"Une banque devrait être en mesure de générer des données de risque agrégées
> pour répondre à un large éventail de demandes ad hoc de la direction."*

### Architecture d'adaptabilité du pipeline

**Détection automatique du format source**
Le module d'extraction (`extract_raw` / Étape 1 Cellule 1) détecte automatiquement
le séparateur CSV via `csv.Sniffer`, gère les encodages BOM Windows (`utf-8-sig`)
et latin-1, et applique `errors='coerce'` sur tous les casts — le pipeline ne plante
pas si une nouvelle source introduit un format légèrement différent.

**Dictionnaire de mapping extensible**
Le référentiel `ASSET_CLASS_MAP` (Étape 4) est un dictionnaire Python versionné
séparément du code de traitement. L'ajout d'une nouvelle variante (ex : un nouveau
système source utilisant "EQT" pour les actions) se fait en une ligne sans modifier
la logique de traitement.

**Sentinelles configurables**
La liste `NA_VALUES` et le dictionnaire `NUMERIC_SENTINELS` (Étape 1) sont
déclarés en tête de pipeline. L'intégration d'une quatrième source de données
(ex : FactSet) ne nécessite que l'ajout de ses sentinelles spécifiques dans ces
structures, sans refactorisation.

**Orchestration Prefect (Bonus 4)**
Le pipeline est structuré en tâches Prefect indépendantes. L'ajout d'une nouvelle
étape de nettoyage (ex : validation d'un nouveau référentiel réglementaire) s'effectue
en déclarant une `@task` supplémentaire et en l'insérant dans le `@flow` — sans
modifier les étapes existantes. Le retry automatique (retries=3 sur l'extraction,
retries=1 sur les transformations) garantit la résilience face aux indisponibilités
transitoires des sources Bloomberg et Refinitiv.

**Suite de validation extensible**
Les 14 expectations Great Expectations (Notebook 03) constituent un contrat de
qualité versionné. L'ajout d'une nouvelle règle réglementaire (ex : contrôle du LEI
de la contrepartie sous EMIR) se traduit par l'ajout d'une `expect()` sans modifier
les règles existantes.

---

## Traçabilité — Audit trail complet

Chaque exécution du pipeline produit :

**`tradecleanse_pipeline.log`** — journal horodaté de chaque transformation :
nombre de lignes avant/après, nombre de valeurs modifiées, DQS avant/après.
Format : `YYYY-MM-DD HH:MM:SS | INFO | [Étape] message`

**Colonnes `_was_missing`** — présentes dans `tradecleanse_clean.csv`, elles
permettent de remonter à l'origine de chaque valeur imputée.

**`ge_validation_report.csv`** — rapport des 14 expectations avec statut PASS/FAIL
et nombre de lignes en erreur, généré à chaque run de validation.

**Pseudonymisation SHA-256 + salt** (Étape 9) — les colonnes PII
(`counterparty_name`, `trader_id`, `counterparty_id`) sont remplacées par leur
hash SHA-256 salé (salt lu depuis `CLEANSE_SALT` en variable d'environnement,
jamais dans le code). La table de correspondance n'est jamais persistée dans le
pipeline — conformément à l'Article 4(5) RGPD (pseudonymisation irréversible
sans la table de mapping).

**Immuabilité du dataset brut** — le fichier `tradecleanse_raw.csv` n'est jamais
modifié. Toutes les transformations opèrent sur `df = df_raw.copy()` (contrainte
non négociable du projet, documentée en commentaire dans chaque module).

---

## Conclusion

Le pipeline TradeCleanse répond aux exigences des Principes 2, 3 et 6 de BCBS 239 :
les données alimentant le modèle de scoring sont exactes (incohérences corrigées),
complètes (NaN imputés avec traçabilité) et produites par un système adaptable
à de nouvelles sources sans refactorisation majeure. Chaque décision de nettoyage
est justifiée par une règle métier financière, loggée de manière horodatée, et
validée par une suite d'expectations automatisées.

Ce pipeline est prêt pour audit réglementaire.

---
*Document confidentiel — QuantAxis Capital — Usage interne Risk Department*
