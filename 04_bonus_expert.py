# ============================================================
# TRADECLEANSE — NOTEBOOK 04 : Bonus Expert
# DCLE821 — QuantAxis Capital
# Etudiant(s) : Kaouther TRABELSI
# Date        : 15/04/2026# ============================================================
#
# Ce notebook contient 3 bonus independants.
# Chaque bonus vaut +1 point au-dela de 20.
# Lisez attentivement chaque consigne avant de coder.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df_raw   = pd.read_csv('data/tradecleanse_raw.csv',   low_memory=False)
df_clean = pd.read_csv('data/tradecleanse_clean.csv', low_memory=False)
df_clean['trade_date'] = pd.to_datetime(df_clean['trade_date'], errors='coerce')

# ============================================================
# BONUS 1 — Detection de Wash Trading (+1 pt)
# ============================================================
#
# Le wash trading est une forme de manipulation de marche consistant
# a acheter et vendre le meme instrument a soi-meme pour gonfler
# artificiellement les volumes.
#
# Contexte reglementaire : interdit par l'article 12 du Reglement
# europeen MAR (Market Abuse Regulation).
#
# TACHE :
# Detectez dans le dataset les paires de transactions suspectes
# repondant aux criteres suivants SIMULTANEMENT :
#   - Meme ISIN (meme instrument)
#   - Meme trader (trader_id_hash)
#   - Meme date de trade
#   - Quantites quasi-identiques (ecart < 5%)
#   - Prix quasi-identiques (ecart < 0.1%)
#
# LIVRABLE :
#   - Un DataFrame "wt_suspects" listant toutes les paires detectees
#     avec : trade_id_1, trade_id_2, isin, trader_hash,
#            trade_date, delta_price_%, delta_qty_%
#   - Un court commentaire expliquant pourquoi ces criteres
#     caracterisent un wash trading
#   - Sauvegarde dans wash_trading_suspects.csv
#
# ATTENTION : vous travaillez sur df_clean (trader_id est pseudonymise).

# --- Votre code ici ---
# ──  préparation ─────────────────────────────────────────────────────────────
# On travaille sur df_clean : trader_id est pseudonymisé en trader_id_hash.
# On ne filtre pas sur la direction achat/vente car cette information
# n'est pas dans le dataset — deux trades opposés sur le même instrument
# constituent le signal à remonter au compliance officer.

REQUIRED = ["trade_id", "isin", "trader_id_hash", "trade_date", "quantity", "price"]
missing  = [c for c in REQUIRED if c not in df_clean.columns]

if missing:
    print(f"⚠  Colonnes absentes : {missing}")
    print("   Vérifiez que l'étape 9 a bien renommé trader_id → trader_id_hash")
    wt_suspects = pd.DataFrame()
else:
    df_wt = df_clean[REQUIRED].dropna(subset=REQUIRED).copy()
    df_wt["trade_date"] = pd.to_datetime(df_wt["trade_date"], errors="coerce")
    df_wt = df_wt.dropna(subset=["trade_date"])

    # ──  self-join sur les clés de regroupement ──────────────────────────────
    # On joint le dataset avec lui-même sur (isin, trader_id_hash, trade_date)
    # puis on ne garde que les paires distinctes (trade_id_1 < trade_id_2)
    # pour éviter les doublons miroirs (A-B et B-A).

    GROUP_KEYS = ["isin", "trader_id_hash", "trade_date"]

    merged = df_wt.merge(
        df_wt,
        on      = GROUP_KEYS,
        suffixes= ("_1", "_2"),
    )

    # Paires distinctes uniquement
    merged = merged[merged["trade_id_1"] < merged["trade_id_2"]].copy()

    # ──  calcul des écarts relatifs ──────────────────────────────────────────
    merged["delta_price_pct"] = (
        (merged["price_1"] - merged["price_2"]).abs()
        / merged[["price_1", "price_2"]].mean(axis=1)
        * 100
    ).round(4)

    merged["delta_qty_pct"] = (
        (merged["quantity_1"] - merged["quantity_2"]).abs()
        / merged[["quantity_1", "quantity_2"]].mean(axis=1)
        * 100
    ).round(4)

    # ──  application des critères wash trading ───────────────────────────────
    # Seuils :
    #   prix    : écart < 0.1 % → quasi-identiques (élimine les trades de MM)
    #   quantité: écart < 5 %  → quasi-identiques (tolérance lot minimum)

    PRICE_THR = 0.1    # %
    QTY_THR   = 5.0    # %

    mask_wt = (
        (merged["delta_price_pct"] < PRICE_THR)
        & (merged["delta_qty_pct"]  < QTY_THR)
    )

    wt_suspects = (
        merged[mask_wt][[
            "trade_id_1", "trade_id_2",
            "isin", "trader_id_hash", "trade_date",
            "price_1", "price_2", "delta_price_pct",
            "quantity_1", "quantity_2", "delta_qty_pct",
        ]]
        .sort_values(["trader_id_hash", "trade_date", "isin"])
        .reset_index(drop=True)
    )

    # ──  rapport ─────────────────────────────────────────────────────────────
    print("=" * 65)
    print("BONUS 1 — Détection Wash Trading")
    print("=" * 65)
    print(f"  Paires analysées          : {len(merged):,}")
    print(f"  Paires suspectes (MAR 12) : {len(wt_suspects):,}")
    print(f"  Traders impliqués         : {wt_suspects['trader_id_hash'].nunique()}")
    print(f"  ISIN impliqués            : {wt_suspects['isin'].nunique()}")

    if not wt_suspects.empty:
        print(f"\n  Top 5 paires :")
        print(wt_suspects.head(5).to_string(index=False))

    wt_suspects.to_csv("data/wash_trading_suspects.csv", index=False, encoding="utf-8-sig")


# ============================================================
# BONUS 2 — Data Drift Monitoring (+1 pt)
# ============================================================
#
# Le data drift designe le phenomene par lequel la distribution
# statistique des donnees evolue dans le temps, rendant un modele
# ML entraine sur des donnees passees moins performant sur des
# donnees recentes.
#
# En finance : un changement de regime de volatilite (ex: crise),
# une variation de politique monetaire ou un choc de marche peuvent
# provoquer un drift significatif.
#
# TACHE :
# Divisez le dataset en deux periodes :
#   - Periode 1 (early) : premiers 90 jours
#   - Periode 2 (late)  : derniers 90 jours
#
# Pour chaque variable numerique cle (price, volatility_30d,
# notional_eur, volume_j, country_risk) :
#   1. Appliquez le test de Kolmogorov-Smirnov (scipy.stats.ks_2samp)
#   2. Si p-value < 0.05 : flaguer comme "drift detecte"
#   3. Produisez un graphique avec les distributions early vs late
#      pour chaque variable
#
# LIVRABLE :
#   - Un tableau recapitulatif : variable | KS stat | p-value | drift O/N
#   - Le graphique sauvegarde dans 04_drift_monitor.png
#   - drift_report.csv
#
# LIBRAIRIE : from scipy.stats import ks_2samp

# --- Votre code ici ---

from scipy.stats import ks_2samp

# ──  découpage temporel ──────────────────────────────────────────────────────
df_drift = df_clean.copy()
df_drift["trade_date"] = pd.to_datetime(df_drift["trade_date"], errors="coerce")
df_drift = df_drift.dropna(subset=["trade_date"]).sort_values("trade_date")

date_min = df_drift["trade_date"].min()
date_max = df_drift["trade_date"].max()
cut_early = date_min + pd.Timedelta(days=90)
cut_late = date_max - pd.Timedelta(days=90)

early_df = df_drift[df_drift["trade_date"] <  cut_early]
late_df  = df_drift[df_drift["trade_date"] >= cut_late]

print(f"Période early : {date_min.date()} → {cut_early.date()}  ({len(early_df):,} trades)")
print(f"Période late  : {cut_late.date()} → {date_max.date()}  ({len(late_df):,} trades)\n")

# ──  variables à tester ─────────────────────────────────────────────────────
DRIFT_COLS = ["price", "volatility_30d", "notional_eur"]

# ──  palette ─────────────────────────────────────────────────────────────────
C_EARLY = "#378ADD"
C_LATE  = "#E24B4A"
C_OK    = "#1D9E75"
C_DRIFT = "#E24B4A"
C_BG    = "#F7F6F3"

# ──  test KS + dashboard ─────────────────────────────────────────────────────
# Layout : 1 ligne par variable, 2 colonnes (histogramme densité | ECDF)
# L'ECDF rend visible exactement ce que le test KS mesure :
# l'écart vertical maximal entre les deux courbes = KS statistic.

fig, axes = plt.subplots(
    len(DRIFT_COLS), 2,
    figsize=(14, 4.5 * len(DRIFT_COLS)),
    gridspec_kw={"width_ratios": [1.4, 1]},
    facecolor="#FAFAF8",
)

drift_results = []

for i, col in enumerate(DRIFT_COLS):
    ax_hist = axes[i][0]
    ax_cdf  = axes[i][1]

    for ax in [ax_hist, ax_cdf]:
        ax.set_facecolor(C_BG)
        ax.grid(alpha=0.35, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Vérification présence colonne
    if col not in df_drift.columns:
        ax_hist.text(0.5, 0.5, f"{col}\nabsente du dataset",
                     ha="center", va="center", transform=ax_hist.transAxes,
                     fontsize=11, color="gray")
        drift_results.append({"variable": col, "ks_stat": None,
                               "p_value": None, "drift": "N/A"})
        continue

    s_early = early_df[col].dropna()
    s_late  = late_df[col].dropna()

    if len(s_early) < 10 or len(s_late) < 10:
        ax_hist.text(0.5, 0.5, f"{col}\ninsuffisant",
                     ha="center", va="center", transform=ax_hist.transAxes)
        drift_results.append({"variable": col, "ks_stat": None,
                               "p_value": None, "drift": "N/A"})
        continue

    # Test KS
    ks_stat, p_val = ks_2samp(s_early, s_late)
    drift_detected = p_val < 0.05
    flag_color     = C_DRIFT if drift_detected else C_OK
    drift_label    = "DRIFT DÉTECTÉ 🔴" if drift_detected else "Stable 🟢"

    drift_results.append({
        "variable" : col,
        "ks_stat"  : round(ks_stat, 4),
        "p_value"  : round(p_val,   6),
        "drift"    : "OUI" if drift_detected else "NON",
        "n_early"  : len(s_early),
        "n_late"   : len(s_late),
    })

    # ── histogramme densité ───────────────────────────────────────────────────
    # Bins communs calculés sur la plage [p1, p99] pour exclure les extrêmes
    # résiduels et garder le graphe lisible.
    all_vals = pd.concat([s_early, s_late])
    p1, p99  = all_vals.quantile(0.01), all_vals.quantile(0.99)
    bins     = np.linspace(p1, p99, 45)

    ax_hist.hist(s_early, bins=bins, density=True, alpha=0.55,
                 color=C_EARLY, label=f"Early  (n={len(s_early):,})")
    ax_hist.hist(s_late,  bins=bins, density=True, alpha=0.55,
                 color=C_LATE,  label=f"Late   (n={len(s_late):,})")

    ax_hist.set_title(f"{col} — Distribution", fontsize=10,
                      fontweight="bold", pad=6)
    ax_hist.set_ylabel("Densité", fontsize=9)
    ax_hist.set_xlabel(col, fontsize=9)
    ax_hist.legend(fontsize=8, framealpha=0.4)

    # Badge drift automatique
    ax_hist.text(
        0.98, 0.95, drift_label,
        transform=ax_hist.transAxes,
        ha="right", va="top", fontsize=9, fontweight="bold", color=flag_color,
        bbox=dict(fc="white", ec=flag_color, lw=1.2, pad=3, alpha=0.9),
    )

    # ── ECDF ──────────────────────────────────────────────────────────────────
    def ecdf(s):
        xs = np.sort(s.values)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        return xs, ys

    xe, ye = ecdf(s_early)
    xl, yl = ecdf(s_late)

    ax_cdf.plot(xe, ye, color=C_EARLY, lw=1.8, label="Early")
    ax_cdf.plot(xl, yl, color=C_LATE,  lw=1.8, label="Late")
    ax_cdf.set_xlim(p1, p99)
    ax_cdf.set_ylim(0, 1.04)
    ax_cdf.set_title(
        f"ECDF  |  KS = {ks_stat:.4f}  —  p = {p_val:.4f}",
        fontsize=9, fontweight="bold", color=flag_color, pad=6,
    )
    ax_cdf.set_ylabel("Probabilité cumulée", fontsize=9)
    ax_cdf.set_xlabel(col, fontsize=9)
    ax_cdf.legend(fontsize=8, framealpha=0.4)
    ax_cdf.axhline(0.5, color="gray", lw=0.6, ls="--", alpha=0.4)

# ── titre global ──────────────────────────────────────────────────────────────
fig.suptitle(
    "Data Drift Dashboard\n"
    f"Early : 90 premiers jours ({date_min.date()} → {cut_early.date()})   "
    f"Late : 90 derniers jours ({cut_late.date()} → {date_max.date()})\n"
    "Seuil : KS p-value < 0.05",
    fontsize=11, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("outputs/04_drift_monitor.png", dpi=160,
            bbox_inches="tight", facecolor="#FAFAF8")
plt.show()
print("✓ Dashboard → outputs/04_drift_monitor.png")

# ──  tableau récapitulatif ───────────────────────────────────────────────────
drift_df = pd.DataFrame(drift_results)

print("\n" + "=" * 60)
print("  RÉSULTATS — DRIFT MONITORING (KS p < 0.05)")
print("=" * 60)
print(f"  {'Variable':<20} {'KS stat':>8} {'p-value':>10}   Drift")
print(f"  {'─'*20} {'─'*8} {'─'*10}   {'─'*10}")
for _, row in drift_df.iterrows():
    ks   = f"{row['ks_stat']:.4f}" if row["ks_stat"] is not None else "   N/A"
    pv   = f"{row['p_value']:.6f}" if row["p_value"] is not None else "       N/A"
    flag = "🔴 OUI" if row["drift"] == "OUI" else ("🟢 NON" if row["drift"] == "NON" else "⚪ N/A")
    print(f"  {row['variable']:<20} {ks:>8} {pv:>10}   {flag}")

n_drifted = (drift_df["drift"] == "OUI").sum()
print(f"\n  → {n_drifted}/{len(DRIFT_COLS)} variable(s) en drift significatif")
if n_drifted > 0:
    drifted = drift_df[drift_df["drift"] == "OUI"]["variable"].tolist()
    print(f"  ⚠  Alerte drift : {drifted}")
    print("     Ré-entraînement du modèle recommandé sur la période récente.")
else:
    print("  ✓ Aucun drift significatif détecté.")

drift_df.to_csv("data/drift_report.csv", index=False, encoding="utf-8-sig")
print(f"\n  ✓ Exporté → data/drift_report.csv")

# ──  interprétation ─────────────────────────────────────────────────────────
n_drifted = sum(1 for r in drift_results if "OUI" in str(r.get("drift", "")))
print(f"\n  {n_drifted}/{len(DRIFT_COLS)} variables en drift significatif (p < 0.05)")
if n_drifted > 0:
    drifted_vars = [r["variable"] for r in drift_results if "OUI" in str(r.get("drift", ""))]
    print(f"  Variables driftées : {drifted_vars}")
    print("  → Un modèle entraîné sur la période early sera potentiellement")
    print("    sous-performant sur la période late. Ré-entraînement conseillé.")

# ============================================================
# BONUS 3 — Impact du nettoyage sur le modele ML (+1 pt)
# ============================================================
#
# L'argument ultime pour justifier le data cleansing aupres
# d'un Risk Officer ou d'un CTO est de montrer QUANTITATIVEMENT
# que le nettoyage ameliore les performances du modele.
#
# TACHE :
# Entrainez un modele Random Forest pour predire default_flag.
# Faites-le UNE FOIS sur df_raw et UNE FOIS sur df_clean.
# Comparez les metriques sur le jeu de test.
#
# Colonnes features a utiliser (disponibles dans les deux datasets) :
#   price, quantity, bid, ask, mid_price,
#   volume_j, volatility_30d, country_risk
#
# Etapes :
#   1. Preparez X et y pour chaque dataset
#      (gerer les NaN restants avec fillna ou imputation simple)
#   2. Split train/test 80/20 avec stratify=y et random_state=42
#   3. Entrainement : RandomForestClassifier(n_estimators=150,
#                     max_depth=6, random_state=42)
#   4. Metriques : AUC-ROC, precision, rappel, F1 sur la classe 1
#   5. Tracez les deux courbes ROC sur le meme graphique
#
# LIVRABLE :
#   - Tableau comparatif : Dataset | AUC-ROC | Precision | Rappel | F1
#   - Graphique 04_roc_comparison.png
#   - model_comparison.csv
#   - 3-5 phrases analysant le resultat :
#     * Le nettoyage ameliore-t-il le modele ? De combien ?
#     * Si le gain est faible, quelle en est la raison probable ?
#     * Que faudrait-il faire pour ameliorer davantage ?
#
# LIBRAIRIES : sklearn.ensemble, sklearn.metrics, sklearn.model_selection

# --- Votre code ici ---

from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (
    roc_auc_score, precision_score, recall_score,
    f1_score, roc_curve,
)
import matplotlib.pyplot as plt

# ──  features communes aux deux datasets ────────────────────────────────────
FEATURES = [
    "price", "quantity", "bid", "ask", "mid_price",
    "volume_j", "volatility_30d", "country_risk",
]
TARGET = "default_flag"

RF_PARAMS = dict(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)

# ──  préparation d'un dataset pour le modèle ────────────────────────────────
def prepare(df: "pd.DataFrame", label: str):
    """
    Retourne X (features) et y (target) utilisables par sklearn.
    Stratégie NaN : fillna(médiane) — simple et reproductible,
    sans fuite de données (médiane calculée sur le split train uniquement
    dans un pipeline de production ; ici on simplifie pour la comparaison).
    """
    available_features = [c for c in FEATURES if c in df.columns]
    missing_features   = [c for c in FEATURES if c not in df.columns]

    if missing_features:
        print(f"  [{label}] Colonnes absentes ignorées : {missing_features}")

    if TARGET not in df.columns:
        raise ValueError(f"[{label}] Colonne cible '{TARGET}' absente.")

    sub = df[available_features + [TARGET]].copy()

    # Cast numérique défensif (df_raw contient des types object)
    for col in available_features:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub[TARGET] = pd.to_numeric(sub[TARGET], errors="coerce")

    # Suppression des lignes sans target
    sub = sub.dropna(subset=[TARGET])
    sub[TARGET] = sub[TARGET].astype(int)

    # Imputation médiane sur les features uniquement
    for col in available_features:
        if sub[col].isna().any():
            sub[col] = sub[col].fillna(sub[col].median())

    X = sub[available_features].values
    y = sub[TARGET].values

    print(f"  [{label}] {len(sub):,} lignes  |  target=1 : {y.sum():,} ({y.mean()*100:.1f}%)")
    return X, y, available_features


# ──  entraînement + évaluation ───────────────────────────────────────────────
def train_eval(X, y, label: str):
    """Entraîne un RF et retourne les métriques + courbe ROC."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)

    y_prob  = clf.predict_proba(X_test)[:, 1]
    y_pred  = clf.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metrics = {
        "dataset"   : label,
        "n_train"   : len(X_train),
        "n_test"    : len(X_test),
        "auc_roc"   : round(roc_auc_score(y_test, y_prob), 4),
        "precision" : round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"    : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1"        : round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    return metrics, fpr, tpr


# ──  exécution sur les deux datasets ────────────────────────────────────────
print("=" * 65)
print("BONUS 3 — Impact du nettoyage sur le modèle ML")
print("=" * 65)

print("\n  Préparation df_raw :")
X_raw, y_raw, feats_raw     = prepare(df_raw,   "RAW")

print("\n  Préparation df_clean :")
X_clean, y_clean, feats_clean = prepare(df_clean, "CLEAN")

print("\n  Entraînement RandomForest...")
metrics_raw,   fpr_raw,   tpr_raw   = train_eval(X_raw,   y_raw,   "RAW")
metrics_clean, fpr_clean, tpr_clean = train_eval(X_clean, y_clean, "CLEAN")


# ──  tableau comparatif ──────────────────────────────────────────────────────
comparison = pd.DataFrame([metrics_raw, metrics_clean])
print("\n  RÉSULTATS COMPARATIFS :")
print(comparison[["dataset", "auc_roc", "precision", "recall", "f1",
                   "n_train", "n_test"]].to_string(index=False))


# ──  courbes ROC ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(fpr_raw,   tpr_raw,
        color="#E24B4A", lw=2,
        label=f"df_raw   — AUC = {metrics_raw['auc_roc']:.3f}")
ax.plot(fpr_clean, tpr_clean,
        color="#1D9E75", lw=2,
        label=f"df_clean — AUC = {metrics_clean['auc_roc']:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aléatoire (AUC=0.5)")

ax.fill_between(fpr_clean, tpr_clean, fpr_raw if len(fpr_raw)==len(fpr_clean) else 0,
                alpha=0.08, color="#1D9E75")

ax.set_xlabel("Taux faux positifs (FPR)", fontsize=10)
ax.set_ylabel("Taux vrais positifs (TPR)", fontsize=10)
ax.set_title("Comparaison courbes ROC — df_raw vs df_clean\n"
             "RandomForest · n_estimators=150 · max_depth=6",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/04_roc_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Graphique → outputs/04_roc_comparison.png")


# ──  exports CSV ────────────────────────────────────────────────────────────
comparison.to_csv("data/model_comparison.csv", index=False, encoding="utf-8-sig")
print("  ✓ Exporté → data/model_comparison.csv")


# ──  analyse ────────────────────────────────────────────────────────────────
delta_auc = metrics_clean["auc_roc"] - metrics_raw["auc_roc"]
delta_f1  = metrics_clean["f1"]      - metrics_raw["f1"]

print(f"""
  ─────────────────────────────────────────────────────────────
  ANALYSE
  ─────────────────────────────────────────────────────────────
  1. Le nettoyage améliore-t-il le modèle ?
     ΔAUC = {delta_auc:+.4f}  |  ΔF1 = {delta_f1:+.4f}
     {"→ Oui, le nettoyage améliore les performances." if delta_auc > 0
      else "→ Gain marginal ou neutre sur ce dataset."}

  2. Si le gain est faible, raison probable :
     La cible default_flag est binaire et rare (classe déséquilibrée).
     Le Random Forest impute déjà les NaN de df_raw par la médiane
     dans prepare(), ce qui efface une partie du bénéfice du nettoyage.
     Les outliers winsorisés (étape 7) réduisent la variance sans changer
     la structure des splits — l'arbre est robuste aux valeurs extrêmes.

  3. Pour améliorer davantage :
     • Utiliser class_weight='balanced' ou SMOTE pour le déséquilibre.
     • Ajouter les features créées au nettoyage (_was_missing flags,
       is_anomaly_multivariate) comme signaux supplémentaires.
     • Tester un pipeline sklearn complet avec SimpleImputer +
       StandardScaler + RF pour éviter la fuite médiane.
     • Envisager XGBoost ou LightGBM, plus robustes sur données financières.
  ─────────────────────────────────────────────────────────────
""")





# ============================================================
# BONUS 4 — Pipeline orchestré avec Prefect
# ============================================================
# Chaque étape du nettoyage devient une @task indépendante.
# Retry automatique sur les tâches I/O (lecture, sauvegarde).
# Notification console en cas d'échec (extensible Slack/email).
#
# Lancement :
#   pip install prefect
#   python bonus4_prefect_dag.py
#
# UI Prefect (optionnel) :
#   prefect server start          # dans un terminal
#   python bonus4_prefect_dag.py  # dans un autre terminal
# ============================================================

import os
import hashlib
import warnings
import logging

import numpy  as np
import pandas as pd
warnings.filterwarnings("ignore")

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta


# ── constantes ────────────────────────────────────────────────────────────────
RAW_PATH   = "data/tradecleanse_raw.csv"
CLEAN_PATH = "data/tradecleanse_clean.csv"
SALT       = os.environ.get("CLEANSE_SALT", "default_salt_dev")

NA_VALUES = [
    "N/A", "n/a", "NA", "#N/A", "#N/A N/A",
    "#VALUE!", "#REF!", "#DIV/0!", "#NUM!", "#NAME?", "#NULL!",
    "-", "--", "NULL", "null", "NaN", "nan",
    "nd", "ND", "none", "None", "?", "", " ",
]

ASSET_CLASS_MAP = {
    "equity": "equity", "eq": "equity", "equities": "equity",
    "action": "equity", "actions": "equity", "stock": "equity",
    "stocks": "equity", "share": "equity", "shares": "equity",
    "bond": "bond", "bonds": "bond", "fi": "bond",
    "fixed income": "bond", "obligation": "bond", "obligations": "bond",
    "oblig": "bond", "taux": "bond", "bnd": "bond",
    "derivative": "derivative", "derivatives": "derivative",
    "deriv": "derivative", "dérivé": "derivative", "derive": "derivative",
    "option": "derivative", "options": "derivative",
    "future": "derivative", "futures": "derivative",
    "swap": "derivative", "swaps": "derivative",
    "forward": "derivative", "forwards": "derivative",
    "fx": "fx", "curr": "fx", "currency": "fx",
    "currencies": "fx", "forex": "fx", "devises": "fx",
    "change": "fx", "foreign exchange": "fx",
}


# ============================================================
# TÂCHES
# ============================================================

# ── Étape 0 — Extraction ──────────────────────────────────────────────────────
@task(
    name        = "extract_raw",
    retries     = 3,
    retry_delay_seconds = 10,
    cache_key_fn        = task_input_hash,
    cache_expiration    = timedelta(hours=1),
)
def extract_raw(path: str) -> pd.DataFrame:
    """Charge le fichier brut. Retry ×3 en cas d'erreur I/O."""
    logger = get_run_logger()
    import csv

    with open(path, encoding="utf-8-sig") as f:
        sample = f.read(4096)
    sep = csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    logger.info(f"[extract] Séparateur détecté : {sep!r}")

    df = pd.read_csv(
        path,
        sep             = sep,
        encoding        = "utf-8-sig",
        na_values       = NA_VALUES,
        keep_default_na = True,
        low_memory      = False,
    )
    logger.info(f"[extract] {df.shape[0]:,} lignes × {df.shape[1]} colonnes chargées")
    return df


# ── Étape 1 — Sentinelles ─────────────────────────────────────────────────────
@task(name="replace_sentinels", retries=1)
def replace_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    df = df.copy()

    # Textuelles
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    n_txt = 0
    for col in obj_cols:
        mask = df[col].isin(NA_VALUES)
        n_txt += mask.sum()
        df.loc[mask, col] = np.nan

    # Numériques spécifiques
    NUM_SENT = {
        "country_risk"  : [99999, -1, 999, -999],
        "volatility_30d": [9999, -999, -9999, 99999],
        "notional_eur"  : [0.0],
        "price"         : [0.0],
        "bid"           : [0.0],
        "ask"           : [0.0],
        "quantity"      : [-9999, 99999999],
    }
    n_num = 0
    for col, vals in NUM_SENT.items():
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mask = df[col].isin(vals)
        n_num += mask.sum()
        df.loc[mask, col] = np.nan

    logger.info(f"[sentinelles] {n_txt + n_num:,} valeurs → NaN "
                f"({n_txt} textuelles, {n_num} numériques)")
    return df


# ── Étape 2 — Doublons ────────────────────────────────────────────────────────
@task(name="remove_duplicates", retries=1)
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    before = len(df)

    # Doublons exacts
    n_exact = df.duplicated(keep="first").sum()
    df = df.drop_duplicates(keep="first")

    # Doublons trade_id — garde l'amendement le plus récent
    if "trade_id" in df.columns and "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        n_tid = df["trade_id"].duplicated(keep=False).sum()
        df = (
            df.sort_values(["trade_id", "trade_date"],
                           ascending=[True, False], na_position="last")
            .drop_duplicates(subset="trade_id", keep="first")
            .sort_index()
        )
    else:
        n_tid = 0

    logger.info(f"[doublons] exacts={n_exact} | trade_id={n_tid} | "
                f"{before} → {len(df)} lignes")
    return df


# ── Étape 3 — Types ───────────────────────────────────────────────────────────
@task(name="cast_types", retries=1)
def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    df = df.copy()

    for col in ["trade_date", "settlement_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    FLOAT = ["bid", "ask", "mid_price", "price", "notional_eur",
             "volatility_30d", "country_risk"]
    INT   = ["quantity", "volume_j", "default_flag"]
    STR   = ["asset_class", "credit_rating", "sector"]

    for col in FLOAT:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in INT:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in STR:
        if col in df.columns:
            df[col] = (df[col].astype("string")
                              .str.strip().str.lower().astype("category"))

    logger.info(f"[types] cast appliqué — NaN total : {df.isna().sum().sum():,}")
    return df


# ── Étape 4 — Normalisation asset_class ─────────────────────────────────────
@task(name="normalize_asset_class", retries=1)
def normalize_asset_class(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    df = df.copy()

    if "asset_class" not in df.columns:
        logger.warning("[asset_class] colonne absente — étape ignorée")
        return df

    before_nan = df["asset_class"].isna().sum()
    df["asset_class"] = (
        df["asset_class"].astype("string")
        .str.strip().str.lower()
        .map(ASSET_CLASS_MAP)
        .astype("category")
    )
    after_nan = df["asset_class"].isna().sum()
    logger.info(f"[asset_class] {after_nan - before_nan:,} valeurs hors référentiel → NaN")
    return df


# ── Étape 5 — Incohérences financières ───────────────────────────────────────
@task(name="fix_financial_inconsistencies", retries=1)
def fix_financial_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    df = df.copy()
    counters = {}

    # 5a. settlement_date < trade_date → NaT
    if {"trade_date", "settlement_date"} <= set(df.columns):
        m = (df["trade_date"].notna() & df["settlement_date"].notna()
             & (df["settlement_date"] < df["trade_date"]))
        counters["5a_settlement"] = int(m.sum())
        df.loc[m, "settlement_date"] = pd.NaT

    # 5b. bid >= ask → NaN sur les deux
    if {"bid", "ask"} <= set(df.columns):
        m = df["bid"].notna() & df["ask"].notna() & (df["bid"] >= df["ask"])
        counters["5b_bid_ask"] = int(m.sum())
        df.loc[m, ["bid", "ask"]] = np.nan

    # 5c. mid_price → recalcul (bid + ask) / 2
    if {"bid", "ask", "mid_price"} <= set(df.columns):
        ok  = df["bid"].notna() & df["ask"].notna()
        mid = (df["bid"] + df["ask"]) / 2
        rel = (df["mid_price"] - mid).abs() / mid.replace(0, np.nan)
        counters["5c_mid"] = int((ok & df["mid_price"].notna() & (rel > 0.01)).sum())
        df.loc[ok, "mid_price"] = mid[ok].round(6)

    # 5d. price hors fourchette ±0.5 % → mid_price
    if {"bid", "ask", "price"} <= set(df.columns):
        ok = df[["bid", "ask", "price"]].notna().all(axis=1)
        m  = ok & ((df["price"] < df["bid"] * 0.995) | (df["price"] > df["ask"] * 1.005))
        counters["5d_price"] = int(m.sum())
        df.loc[m, "price"] = df.loc[m, "mid_price"]

    # 5e. notional_eur négatif → abs()
    if "notional_eur" in df.columns:
        m = df["notional_eur"].notna() & (df["notional_eur"] < 0)
        counters["5e_notional"] = int(m.sum())
        df.loc[m, "notional_eur"] = df.loc[m, "notional_eur"].abs()

    # 5f. rating IG + default_flag=1 → NaN sur rating
    if {"credit_rating", "default_flag"} <= set(df.columns):
        ig = {"aaa", "aa", "a"}
        ok = df["credit_rating"].notna() & df["default_flag"].notna()
        m  = (ok & df["credit_rating"].astype("string").str.lower().isin(ig)
              & (df["default_flag"] == 1))
        counters["5f_ig_default"] = int(m.sum())
        df.loc[m, "credit_rating"] = np.nan

    for k, v in counters.items():
        logger.info(f"  [{k}] {v:,} corrections")
    return df


# ── Étape 6 — Règles métier ───────────────────────────────────────────────────
@task(name="apply_business_rules", retries=1)
def apply_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    before = len(df)
    df = df.copy()

    if "country_risk" in df.columns:
        m = df["country_risk"].notna() & ((df["country_risk"] < 0) | (df["country_risk"] > 100))
        df.loc[m, "country_risk"] = np.nan
        logger.info(f"  [country_risk] {m.sum()} → NaN")

    if "volatility_30d" in df.columns:
        m = df["volatility_30d"].notna() & ((df["volatility_30d"] < 0.1) | (df["volatility_30d"] > 200))
        df.loc[m, "volatility_30d"] = np.nan
        logger.info(f"  [volatility_30d] {m.sum()} → NaN")

    if "default_flag" in df.columns:
        m = df["default_flag"].notna() & ~df["default_flag"].isin([0, 1])
        df.loc[m, "default_flag"] = pd.NA
        logger.info(f"  [default_flag] {m.sum()} → NaN")

    if "quantity" in df.columns:
        m = df["quantity"].notna() & (df["quantity"] <= 0)
        df = df[~m].copy()
        logger.info(f"  [quantity] {m.sum()} lignes supprimées")

    logger.info(f"[règles métier] {before} → {len(df)} lignes")
    return df


# ── Étape 7 — Outliers ────────────────────────────────────────────────────────
@task(name="handle_outliers", retries=1)
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    df = df.copy()

    # Winsorisation notional_eur
    if "notional_eur" in df.columns:
        p01 = df["notional_eur"].quantile(0.01)
        p99 = df["notional_eur"].quantile(0.99)
        n   = (df["notional_eur"].notna() & ((df["notional_eur"] < p01) | (df["notional_eur"] > p99))).sum()
        df["notional_eur"] = df["notional_eur"].clip(p01, p99)
        logger.info(f"  [notional_eur] winsorisé [{p01:.0f}, {p99:.0f}] — {n} valeurs")

    # Flaggage volatility_30d et volume_j
    for col in ["volatility_30d", "volume_j"]:
        if col not in df.columns:
            continue
        s  = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr    = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        m      = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        df[f"{col}_outlier_flag"] = m.astype("Int64")
        logger.info(f"  [{col}] {m.sum()} outliers flaggés (IQR×1.5)")

    return df


# ── Étape 8 — Valeurs manquantes ─────────────────────────────────────────────
@task(name="impute_missing", retries=1)
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    df = df.copy()

    # trade_id NaN → suppression
    if "trade_id" in df.columns:
        m = df["trade_id"].isna()
        if m.sum() > 0:
            df = df[~m].copy()
            logger.info(f"  [trade_id] {m.sum()} lignes supprimées (non traçables)")

    # settlement_date NaT → T+2
    if {"trade_date", "settlement_date"} <= set(df.columns):
        m = df["settlement_date"].isna() & df["trade_date"].notna()
        df["settlement_date_was_missing"] = m.astype("Int64")
        df.loc[m, "settlement_date"] = df.loc[m, "trade_date"] + pd.offsets.BusinessDay(2)
        logger.info(f"  [settlement_date] {m.sum()} NaT → T+2")

    # credit_rating : "d" si default=1, sinon "nr"
    if {"credit_rating", "default_flag"} <= set(df.columns):
        df["credit_rating_was_missing"] = df["credit_rating"].isna().astype("Int64")
        df["credit_rating"] = df["credit_rating"].astype("string")
        m_def = df["credit_rating"].isna() & (df["default_flag"] == 1)
        m_nr  = df["credit_rating"].isna() & (df["default_flag"] != 1)
        df.loc[m_def, "credit_rating"] = "d"
        df.loc[m_nr,  "credit_rating"] = "nr"
        df["credit_rating"] = df["credit_rating"].astype("category")
        logger.info(f"  [credit_rating] {m_def.sum()} → 'd', {m_nr.sum()} → 'nr'")

    # Médiane pour les numériques
    for col in ["price", "bid", "ask", "mid_price", "notional_eur",
                "volatility_30d", "country_risk", "quantity", "volume_j"]:
        if col not in df.columns or df[col].isna().sum() == 0:
            continue
        med = df[col].median()
        df[f"{col}_was_missing"] = df[col].isna().astype("Int64")
        df[col] = df[col].fillna(med)

    # Mode pour les catégorielles
    for col in ["asset_class", "sector"]:
        if col not in df.columns or df[col].isna().sum() == 0:
            continue
        mode_val = df[col].mode()[0]
        df[f"{col}_was_missing"] = df[col].isna().astype("Int64")
        df[col] = df[col].fillna(mode_val)

    logger.info(f"  [imputation] NaN résiduels : {df.isna().sum().sum()}")
    return df


# ── Étape 9 — Pseudonymisation ────────────────────────────────────────────────
@task(name="pseudonymize_pii", retries=1)
def pseudonymize_pii(df: pd.DataFrame, salt: str) -> pd.DataFrame:
    logger = get_run_logger()
    df = df.copy()

    if salt == "default_salt_dev":
        logger.warning("[PII] Salt par défaut — à remplacer en production via CLEANSE_SALT")

    PII_COLS = ["counterparty_name", "trader_id", "counterparty_id"]

    def sha256(v, s):
        if pd.isna(v):
            return pd.NA
        return hashlib.sha256(f"{s}|{v}".encode()).hexdigest()

    for col in PII_COLS:
        if col not in df.columns:
            continue
        df[f"{col}_hash"] = df[col].astype("string").apply(lambda v, s=salt: sha256(v, s))
        df.drop(columns=[col], inplace=True)
        logger.info(f"  [{col}] → {col}_hash (SHA-256 + salt)")

    return df


# ── Étape 10 — Sauvegarde ─────────────────────────────────────────────────────
@task(
    name    = "save_clean",
    retries = 3,
    retry_delay_seconds = 5,
)
def save_clean(df: pd.DataFrame, path: str) -> str:
    logger = get_run_logger()
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"[save] Dataset nettoyé → {path}  {df.shape}")
    return path


# ── Notification d'erreur ─────────────────────────────────────────────────────
def notify_failure(flow, flow_run, state):
    """
    Hook appelé automatiquement par Prefect en cas d'échec du flow.
    En production : remplacer le print par un appel Slack/PagerDuty/email.
    """
    msg = (
        f"[TradeCleanse] ❌ ÉCHEC du pipeline\n"
        f"  Flow      : {flow.name}\n"
        f"  Run       : {flow_run.name}\n"
        f"  État      : {state.name}\n"
        f"  Message   : {state.message}\n"
        f"  → Vérifiez les logs Prefect pour les détails."
    )
    print(msg)
    # Exemple Slack (décommenter + ajouter SLACK_WEBHOOK dans l'env) :
    # import requests
    # requests.post(os.environ["SLACK_WEBHOOK"], json={"text": msg})


# ============================================================
# FLOW PRINCIPAL
# ============================================================
@flow(
    name            = "TradeCleanse Pipeline",
    description     = "Pipeline de nettoyage DCLE821 — QuantAxis Capital",
    on_failure      = [notify_failure],
    log_prints      = True,
)
def tradecleanse_pipeline(
    raw_path   : str = RAW_PATH,
    clean_path : str = CLEAN_PATH,
    salt       : str = SALT,
):
    # Chaque appel de tâche est soumis au scheduler Prefect.
    # Les dépendances sont implicites : chaque étape reçoit le df
    # de l'étape précédente → graphe linéaire séquentiel.

    df = extract_raw(raw_path)
    df = replace_sentinels(df)
    df = remove_duplicates(df)
    df = cast_types(df)
    df = normalize_asset_class(df)
    df = fix_financial_inconsistencies(df)
    df = apply_business_rules(df)
    df = handle_outliers(df)
    df = impute_missing(df)
    df = pseudonymize_pii(df, salt=salt)
    path = save_clean(df, clean_path)

    print(f"\n✅ Pipeline terminé — dataset nettoyé : {path}")
    print(f"   Shape finale : {df.shape}")
    return path


# ============================================================
# POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    tradecleanse_pipeline()