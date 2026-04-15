# ============================================================
# ge_report_generator.py — Rapport HTML Great Expectations
# Génère ge_validation_report.html à partir du dataset nettoyé
# ============================================================

import pandas as pd
import numpy  as np
from datetime import datetime

# ── Chargement ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/tradecleanse_clean.csv", low_memory=False)
df["trade_date"]      = pd.to_datetime(df["trade_date"],      errors="coerce")
df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce")
print(f"Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# ── Runner d'expectations ─────────────────────────────────────────────────────
results = []

def expect(test_id, name, passed, n_fail=0, detail=""):
    results.append({
        "id"      : test_id,
        "name"    : name,
        "status"  : "PASS" if passed else "FAIL",
        "n_fail"  : n_fail,
        "detail"  : detail,
    })

n = len(df)

# E1 — Unicité trade_id
dup = df["trade_id"].duplicated().sum()
expect(1, "Unicité de trade_id",
       dup == 0, n_fail=dup,
       detail=f"{df['trade_id'].nunique():,} valeurs uniques")

# E2 — Colonnes obligatoires non nulles
MANDATORY = ["trade_id","isin","trade_date","asset_class","price","quantity","default_flag"]
mandatory_present = [c for c in MANDATORY if c in df.columns]
n_null = df[mandatory_present].isna().sum().sum()
expect(2, "Colonnes obligatoires non nulles",
       n_null == 0, n_fail=n_null,
       detail=str(mandatory_present))

# E3 — settlement_date >= trade_date
if {"trade_date","settlement_date"} <= set(df.columns):
    both = df["trade_date"].notna() & df["settlement_date"].notna()
    n_inv = int((both & (df["settlement_date"] < df["trade_date"])).sum())
    expect(3, "settlement_date >= trade_date", n_inv == 0, n_inv)
else:
    expect(3, "settlement_date >= trade_date", False, detail="colonnes absentes")

# E4 — bid < ask
if {"bid","ask"} <= set(df.columns):
    both = df["bid"].notna() & df["ask"].notna()
    n_inv = int((both & (df["bid"] >= df["ask"])).sum())
    expect(4, "bid < ask", n_inv == 0, n_inv)
else:
    expect(4, "bid < ask", False, detail="colonnes absentes")

# E5 — price dans fourchette [bid*0.995, ask*1.005]
if {"bid","ask","price"} <= set(df.columns):
    ok = df[["bid","ask","price"]].notna().all(axis=1)
    n_out = int((ok & ((df["price"] < df["bid"]*0.995)|(df["price"] > df["ask"]*1.005))).sum())
    expect(5, "price dans [bid*0.995, ask*1.005]", n_out == 0, n_out)
else:
    expect(5, "price dans fourchette", False, detail="colonnes absentes")

# E6 — mid_price cohérent ±1%
if {"bid","ask","mid_price"} <= set(df.columns):
    ok   = df[["bid","ask","mid_price"]].notna().all(axis=1)
    theo = (df["bid"] + df["ask"]) / 2
    rel  = (df["mid_price"] - theo).abs() / theo.replace(0, np.nan)
    n_ko = int((ok & (rel > 0.01)).sum())
    expect(6, "mid_price cohérent avec (bid+ask)/2 ±1%", n_ko == 0, n_ko)
else:
    expect(6, "mid_price cohérent", False, detail="colonnes absentes")

# E7 — asset_class dans référentiel
VALID = {"equity","bond","derivative","fx"}
if "asset_class" in df.columns:
    known = df["asset_class"].notna()
    n_inv = int((known & ~df["asset_class"].astype(str).str.lower().isin(VALID)).sum())
    cats  = sorted(df["asset_class"].dropna().unique().tolist())
    expect(7, "asset_class dans {equity,bond,derivative,fx}",
           n_inv == 0, n_inv, detail=str(cats))
else:
    expect(7, "asset_class référentiel", False, detail="colonne absente")

# E8 — rating IG incompatible avec default=1
IG = {"aaa","aa","a"}
if {"credit_rating","default_flag"} <= set(df.columns):
    both = df["credit_rating"].notna() & df["default_flag"].notna()
    is_ig = df["credit_rating"].astype(str).str.lower().isin(IG)
    n_co  = int((both & is_ig & (df["default_flag"] == 1)).sum())
    expect(8, "rating IG incompatible avec default_flag=1", n_co == 0, n_co)
else:
    expect(8, "rating IG / default_flag", False, detail="colonnes absentes")

# E9 — notional_eur > 0
if "notional_eur" in df.columns:
    n_neg = int((df["notional_eur"].notna() & (df["notional_eur"] <= 0)).sum())
    expect(9, "notional_eur > 0", n_neg == 0, n_neg,
           detail=f"min={df['notional_eur'].min():.2f}")
else:
    expect(9, "notional_eur > 0", False, detail="colonne absente")

# E10 — country_risk dans [0, 100]
if "country_risk" in df.columns:
    known = df["country_risk"].notna()
    n_out = int((known & ((df["country_risk"] < 0)|(df["country_risk"] > 100))).sum())
    expect(10, "country_risk dans [0, 100]", n_out == 0, n_out)
else:
    expect(10, "country_risk [0,100]", False, detail="colonne absente")

# E11 — Format ISIN
if "isin" in df.columns:
    known = df["isin"].notna()
    n_inv = int((known & ~df["isin"].astype(str).str.match(r"^[A-Z]{2}[A-Z0-9]{10}$")).sum())
    expect(11, "ISIN conforme ISO 6166 (^[A-Z]{2}[A-Z0-9]{10}$)", n_inv == 0, n_inv)
else:
    expect(11, "Format ISIN", False, detail="colonne absente")

# E12 — volatility_30d dans [0.1, 200]
if "volatility_30d" in df.columns:
    known = df["volatility_30d"].notna()
    n_out = int((known & ((df["volatility_30d"] < 0.1)|(df["volatility_30d"] > 200))).sum())
    expect(12, "volatility_30d dans [0.1, 200]", n_out == 0, n_out)
else:
    expect(12, "volatility_30d [0.1,200]", False, detail="colonne absente")

# E13 — Complétude globale > 90%
completude = 1 - df.isna().mean().mean()
expect(13, "Complétude globale > 90%",
       completude > 0.90, detail=f"{completude*100:.2f}%")

# E14 — Absence de PII en clair
PII_RAW = ["counterparty_name","trader_id","counterparty_id"]
pii_found = [c for c in PII_RAW if c in df.columns]
expect(14, "Absence de colonnes PII en clair",
       len(pii_found) == 0, n_fail=len(pii_found),
       detail=str(pii_found) if pii_found else "toutes pseudonymisées ✓")

# ── Génération du rapport HTML ────────────────────────────────────────────────
n_pass  = sum(1 for r in results if r["status"] == "PASS")
n_total = len(results)
score_pct = n_pass / n_total * 100
ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

rows_html = ""
for r in results:
    color  = "#1D9E75" if r["status"] == "PASS" else "#E24B4A"
    bg     = "#E1F5EE" if r["status"] == "PASS" else "#FCEBEB"
    icon   = "✅" if r["status"] == "PASS" else "❌"
    rows_html += f"""
    <tr style="background:{bg}">
      <td style="padding:8px 12px;font-weight:500;color:#2C2C2A">E{r['id']:02d}</td>
      <td style="padding:8px 12px;color:#2C2C2A">{r['name']}</td>
      <td style="padding:8px 12px;text-align:center;font-weight:700;color:{color}">{icon} {r['status']}</td>
      <td style="padding:8px 12px;text-align:right;color:#5F5E5A">{r['n_fail']:,}</td>
      <td style="padding:8px 12px;color:#5F5E5A;font-size:12px">{r['detail']}</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TradeCleanse — Validation Report</title>
<style>
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#F7F6F3;margin:0;padding:2rem;color:#2C2C2A}}
  .card{{background:#fff;border-radius:12px;border:0.5px solid #D3D1C7;padding:2rem;margin-bottom:1.5rem;max-width:1000px;margin-left:auto;margin-right:auto}}
  h1{{font-size:22px;font-weight:500;margin:0 0 4px}}
  .meta{{font-size:13px;color:#888780;margin-bottom:1.5rem}}
  .score-big{{font-size:48px;font-weight:700;color:{("#1D9E75" if n_pass==n_total else "#E24B4A")}}}
  .score-sub{{font-size:14px;color:#5F5E5A;margin-top:4px}}
  .dqs-bar{{height:10px;background:#D3D1C7;border-radius:5px;overflow:hidden;margin:8px 0}}
  .dqs-fill{{height:100%;background:{("#1D9E75" if n_pass==n_total else "#EF9F27")};width:{score_pct:.1f}%;border-radius:5px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{padding:10px 12px;text-align:left;background:#F1EFE8;font-weight:500;font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:#5F5E5A}}
  tr:nth-child(even){{filter:brightness(0.97)}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600}}
  .pass{{background:#E1F5EE;color:#085041}}.fail{{background:#FCEBEB;color:#791F1F}}
  footer{{text-align:center;font-size:12px;color:#B4B2A9;margin-top:2rem}}
</style>
</head>
<body>
<div class="card">
  <h1>TradeCleanse — Rapport de Validation</h1>
  <div class="meta">QuantAxis Capital · Généré le {ts} · Dataset : tradecleanse_clean.csv · {n:,} lignes</div>
  <div style="display:flex;gap:3rem;align-items:flex-start;flex-wrap:wrap">
    <div>
      <div class="score-big">{n_pass}/{n_total}</div>
      <div class="score-sub">expectations passées</div>
      <div class="dqs-bar"><div class="dqs-fill"></div></div>
      <div style="font-size:12px;color:#888780">{score_pct:.1f}% de réussite</div>
    </div>
    <div style="flex:1;min-width:200px">
      <div style="font-size:13px;margin-bottom:8px;font-weight:500">Résumé</div>
      <div style="font-size:13px;color:#5F5E5A;line-height:1.8">
        <span class="badge pass">PASS</span> {n_pass} expectations<br>
        <span class="badge fail">FAIL</span> {n_total-n_pass} expectations<br>
        Complétude : {completude*100:.2f}%
      </div>
    </div>
  </div>
</div>
<div class="card">
  <table>
    <thead>
      <tr>
        <th>ID</th><th>Expectation</th><th style="text-align:center">Statut</th>
        <th style="text-align:right">Lignes KO</th><th>Détail</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
<footer>TradeCleanse Pipeline · DCLE821 · QuantAxis Capital Paris · Conforme BCBS 239</footer>
</body>
</html>"""

with open("data/ge_validation_report.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✓ Rapport HTML → data/ge_validation_report.html")
print(f"  Score : {n_pass}/{n_total} ({score_pct:.1f}%)")

# Export CSV aussi
pd.DataFrame(results).to_csv(
    "data/ge_validation_report.csv", index=False, encoding="utf-8-sig"
)
print("✓ Rapport CSV  → data/ge_validation_report.csv")