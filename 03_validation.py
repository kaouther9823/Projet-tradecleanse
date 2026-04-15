# ============================================================
# TRADECLEANSE — NOTEBOOK 03 : Validation du Dataset Nettoye
# DCLE821 — QuantAxis Capital
# Etudiant(s) : Kaouther TRABELSI
# Date        : 15/04/2026
# ============================================================
#
# CONSIGNE GENERALE :
# Ce notebook valide que votre pipeline a correctement nettoye le dataset.
# Vous devez implementer au minimum 14 tests de validation (expectations).
#
# Deux approches possibles :
#   A) Utiliser la librairie Great Expectations (recommande en entreprise)
#      pip install great_expectations
#      Documentation : https://docs.greatexpectations.io
#
#   B) Implementer vos propres tests avec pandas + assertions Python
#      (acceptable si vous documentez clairement chaque test)
#
# Pour chaque test, affichez clairement : [PASS] ou [FAIL] + le detail.
# A la fin, affichez un score : X/14 tests passes.
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Chargement du dataset nettoye
# ATTENTION : ce fichier doit avoir ete genere par 02_cleaning_pipeline.py
df = pd.read_csv('data/tradecleanse_clean.csv', low_memory=False)
print(f"Dataset nettoye charge : {df.shape[0]} lignes x {df.shape[1]} colonnes\n")
# ── Helpers ───────────────────────────────────────────────────────────────────
results: list[dict] = []

def expect(
    test_id   : int,
    name      : str,
    passed    : bool,
    n_fail    : int   = 0,
    detail    : str   = "",
) -> None:
    """Enregistre le résultat d'une expectation et l'affiche."""
    status = "PASS" if passed else "FAIL"
    icon   = "✅" if passed else "❌"
    msg    = f"[{status}] E{test_id:02d} — {name}"
    if not passed:
        msg += f"  →  {n_fail:,} ligne(s) en erreur"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append({
        "expectation_id"  : test_id,
        "name"            : name,
        "status"          : status,
        "n_failing_rows"  : n_fail,
        "detail"          : detail,
    })

print("=" * 65)
print("  VALIDATION — 14 EXPECTATIONS")
print("=" * 65 + "\n")

# ── Pré-traitement des dates pour les tests temporels ─────────────────────────
for col in ["trade_date", "settlement_date"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


# ============================================================
# EXPECTATION 1 — Unicite de trade_id
# ============================================================
# Verifie qu'il n'existe aucun doublon sur la cle metier.

# --- Votre code ici ---
n_dup = df["trade_id"].duplicated().sum()
expect(1, "Unicité trade_id", n_dup == 0, n_fail=n_dup,
       detail=f"{df['trade_id'].nunique():,} valeurs uniques / {len(df):,} lignes")

# ============================================================
# EXPECTATION 2 — Colonnes obligatoires non nulles
# ============================================================
# Les colonnes suivantes ne doivent contenir aucun NaN :
# trade_id, counterparty_id, isin, trade_date,
# asset_class, price, quantity, default_flag

# --- Votre code ici ---
MANDATORY = [
    "trade_id", "counterparty_id_hash", "isin", "trade_date",
    "asset_class", "price", "quantity", "default_flag",
]
# On accepte la version hashée de counterparty_id si l'originale est absente
mandatory_present = [c for c in MANDATORY if c in df.columns]
mandatory_missing = [c for c in MANDATORY if c not in df.columns]

n_null_mandatory = df[mandatory_present].isna().sum().sum()
detail_e2 = ""
if mandatory_missing:
    detail_e2 = f"colonnes absentes du dataset : {mandatory_missing}"

expect(2, "Colonnes obligatoires non nulles",
       n_null_mandatory == 0 and not mandatory_missing,
       n_fail=n_null_mandatory,
       detail=detail_e2 or f"{len(mandatory_present)} colonnes vérifiées")
# ============================================================
# EXPECTATION 3 — settlement_date >= trade_date
# ============================================================
# Un reglement ne peut pas etre anterieur au trade.

# --- Votre code ici ---
if {"trade_date", "settlement_date"} <= set(df.columns):
    both = df["trade_date"].notna() & df["settlement_date"].notna()
    n_inv = (both & (df["settlement_date"] < df["trade_date"])).sum()
    expect(3, "settlement_date >= trade_date", n_inv == 0, n_fail=n_inv)
else:
    expect(3, "settlement_date >= trade_date", False, detail="colonnes absentes")


# ============================================================
# EXPECTATION 4 — bid < ask sur toutes les lignes
# ============================================================
# La fourchette de prix doit toujours etre dans le bon sens.

# --- Votre code ici ---
if {"bid", "ask"} <= set(df.columns):
    both = df["bid"].notna() & df["ask"].notna()
    n_inv = (both & (df["bid"] >= df["ask"])).sum()
    expect(4, "bid < ask", n_inv == 0, n_fail=n_inv)
else:
    expect(4, "bid < ask", False, detail="colonnes absentes")


# ============================================================
# EXPECTATION 5 — price dans la fourchette [bid * 0.995, ask * 1.005]
# ============================================================
# Un prix d'execution ne peut pas etre en dehors de la fourchette.

# --- Votre code ici ---
if {"bid", "ask", "price"} <= set(df.columns):
    all_k = df[["bid", "ask", "price"]].notna().all(axis=1)
    out   = all_k & (
        (df["price"] < df["bid"] * 0.995) |
        (df["price"] > df["ask"] * 1.005)
    )
    n_out = out.sum()
    expect(5, "price dans fourchette [bid*0.995, ask*1.005]",
           n_out == 0, n_fail=n_out)
else:
    expect(5, "price dans fourchette", False, detail="colonnes absentes")


# ============================================================
# EXPECTATION 6 — mid_price coherent avec (bid + ask) / 2
# ============================================================
# Tolerance : ecart < 1% du mid theorique.

# --- Votre code ici ---
if {"bid", "ask", "mid_price"} <= set(df.columns):
    all_k   = df[["bid", "ask", "mid_price"]].notna().all(axis=1)
    theo    = (df["bid"] + df["ask"]) / 2
    rel_err = (df["mid_price"] - theo).abs() / theo.replace(0, np.nan)
    n_inco  = (all_k & (rel_err > 0.01)).sum()
    expect(6, "mid_price cohérent avec (bid+ask)/2 ±1%",
           n_inco == 0, n_fail=n_inco)
else:
    expect(6, "mid_price cohérent", False, detail="colonnes absentes")

# ============================================================
# EXPECTATION 7 — asset_class dans le referentiel normalise
# ============================================================
# Seules ces valeurs sont acceptees : equity, bond, derivative, fx

# --- Votre code ici ---

VALID_ASSET = {"equity", "bond", "derivative", "fx"}
if "asset_class" in df.columns:
    known = df["asset_class"].notna()
    invalid = known & ~df["asset_class"].astype(str).str.lower().isin(VALID_ASSET)
    n_inv   = invalid.sum()
    cats    = sorted(df["asset_class"].dropna().unique().tolist())
    expect(7, "asset_class dans {equity, bond, derivative, fx}",
           n_inv == 0, n_fail=n_inv,
           detail=f"catégories présentes : {cats}")
else:
    expect(7, "asset_class référentiel", False, detail="colonne absente")


# ============================================================
# EXPECTATION 8 — Pas de contradiction rating investissement + defaut
# ============================================================
# credit_rating AAA, AA ou A avec default_flag = 1 est impossible.

# --- Votre code ici ---

IG = {"aaa", "aa", "a"}
if {"credit_rating", "default_flag"} <= set(df.columns):
    both  = df["credit_rating"].notna() & df["default_flag"].notna()
    is_ig = df["credit_rating"].astype(str).str.lower().isin(IG)
    n_co  = (both & is_ig & (df["default_flag"] == 1)).sum()
    expect(8, "rating IG incompatible avec default_flag=1",
           n_co == 0, n_fail=n_co)
else:
    expect(8, "rating IG / default_flag", False, detail="colonnes absentes")


# ============================================================
# EXPECTATION 9 — notional_eur > 0
# ============================================================
# Le montant notionnel doit etre strictement positif.

# --- Votre code ici ---
if "notional_eur" in df.columns:
    n_neg = (df["notional_eur"].notna() & (df["notional_eur"] <= 0)).sum()
    expect(9, "notional_eur > 0", n_neg == 0, n_fail=n_neg,
           detail=f"min={df['notional_eur'].min():.2f}")
else:
    expect(9, "notional_eur > 0", False, detail="colonne absente")


# ============================================================
# EXPECTATION 10 — country_risk dans [0, 100]
# ============================================================

# --- Votre code ici ---
if "country_risk" in df.columns:
    known  = df["country_risk"].notna()
    n_out  = (known & ((df["country_risk"] < 0) | (df["country_risk"] > 100))).sum()
    expect(10, "country_risk dans [0, 100]", n_out == 0, n_fail=n_out)
else:
    expect(10, "country_risk [0,100]", False, detail="colonne absente")


# ============================================================
# EXPECTATION 11 — Format ISIN valide
# ============================================================
# Un ISIN est compose de 2 lettres majuscules + 10 caracteres alphanumeriques.
# Regex : ^[A-Z]{2}[A-Z0-9]{10}$

# --- Votre code ici ---
if "isin" in df.columns:
    known     = df["isin"].notna()
    pattern   = r"^[A-Z]{2}[A-Z0-9]{10}$"
    n_invalid = (known & ~df["isin"].astype(str).str.match(pattern)).sum()
    expect(11, "ISIN conforme ISO 6166 (^[A-Z]{2}[A-Z0-9]{10}$)",
           n_invalid == 0, n_fail=n_invalid)
else:
    expect(11, "Format ISIN", False, detail="colonne absente")


# ============================================================
# EXPECTATION 12 — volatility_30d dans [0.1, 200]
# ============================================================

# --- Votre code ici ---
if "volatility_30d" in df.columns:
    known = df["volatility_30d"].notna()
    n_out = (known & (
        (df["volatility_30d"] < 0.1) | (df["volatility_30d"] > 200)
    )).sum()
    expect(12, "volatility_30d dans [0.1, 200]", n_out == 0, n_fail=n_out)
else:
    expect(12, "volatility_30d [0.1,200]", False, detail="colonne absente")


# ============================================================
# EXPECTATION 13 — Completude globale > 90%
# ============================================================
# Le taux de completude global (toutes colonnes confondues) doit
# etre superieur a 90%.

# --- Votre code ici ---
completude = 1 - df.isna().mean().mean()
expect(13, "Complétude globale > 90%",
       completude > 0.90,
       detail=f"complétude mesurée = {completude * 100:.2f}%")


# ============================================================
# EXPECTATION 14 — Absence de PII en clair
# ============================================================
# Les colonnes counterparty_name et trader_id ne doivent PAS
# exister dans le dataset final (remplacees par leurs versions hashees).

# --- Votre code ici ---

PII_RAW  = ["counterparty_name", "trader_id", "counterparty_id"]
pii_found = [c for c in PII_RAW if c in df.columns]
pii_hashed_missing = [
    f"{c}_hash" for c in PII_RAW if f"{c}_hash" not in df.columns
]

expect(14, "Absence de PII en clair",
       len(pii_found) == 0,
       n_fail=len(pii_found),
       detail=(
           f"colonnes PII résiduelles : {pii_found}" if pii_found
           else f"colonnes hash présentes : {[c+'_hash' for c in PII_RAW if c+'_hash' in df.columns]}"
       ))

# ============================================================
# SCORE FINAL
# ============================================================
# Affichez : "Score : X/14 expectations passees"
# Exportez les resultats dans un fichier ge_validation_report.csv

# --- Votre code ici ---

print("\n" + "=" * 65)
n_pass  = sum(1 for r in results if r["status"] == "PASS")
n_total = len(results)
score   = f"{n_pass}/{n_total}"

print(f"  SCORE : {score} expectations passées")
if n_pass == n_total:
    print("  🏆 Pipeline validé — dataset prêt pour la modélisation")
elif n_pass >= n_total * 0.85:
    print("  ⚠️  Qualité acceptable — investiguer les FAIL avant production")
else:
    print("  🔴 Pipeline à corriger — trop d'expectations échouées")
print("=" * 65)

# ── Détail des FAILs ─────────────────────────────────────────────────────────
fails = [r for r in results if r["status"] == "FAIL"]
if fails:
    print("\n  DÉTAIL DES FAILS :")
    for r in fails:
        print(f"  E{r['expectation_id']:02d} — {r['name']}")
        print(f"       {r['n_failing_rows']:,} lignes  |  {r['detail']}")


# ── Export CSV du rapport ────────────────────────────────────────────────────
report_df = pd.DataFrame(results)
report_df.to_csv("data/ge_validation_report.csv", index=False, encoding="utf-8-sig")
print(f"\n  Rapport exporté → data/ge_validation_report.csv")
print(f"  Shape finale dataset : {df.shape}")
