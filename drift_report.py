"""
Phase 4: Data Drift Detection
-------------------------------
Uses Evidently AI (or scipy fallback) to compare training vs serving distributions.
Generates an HTML drift report.

Run: python drift_report.py
Output: reports/drift_report.html
"""

import os
import json
import pandas as pd
import numpy as np
from scipy import stats

os.makedirs("reports", exist_ok=True)

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

# ── Load reference (train) and current (serving) data ─────────────────────────
reference = pd.read_csv("data/train_reference.csv")
current   = pd.read_csv("data/serving_data.csv")

# ── KS Test (Kolmogorov-Smirnov) per feature ──────────────────────────────────
results = []
for feat in FEATURES:
    ks_stat, p_val = stats.ks_2samp(reference[feat], current[feat])
    drifted = bool(p_val < 0.05)
    results.append({
        "feature":        feat,
        "ref_mean":       round(float(reference[feat].mean()), 4),
        "cur_mean":       round(float(current[feat].mean()),   4),
        "ref_std":        round(float(reference[feat].std()),  4),
        "cur_std":        round(float(current[feat].std()),    4),
        "ks_statistic":   round(float(ks_stat), 4),
        "p_value":        round(float(p_val),   4),
        "drift_detected": drifted,
    })

drift_count = sum(1 for r in results if r["drift_detected"])
print(f"\n{'='*55}")
print(f"  DATA DRIFT REPORT  —  {drift_count}/{len(results)} features drifted")
print(f"{'='*55}")
for r in results:
    flag = "⚠️  DRIFT" if r["drift_detected"] else "✓  OK   "
    print(f"  {flag}  {r['feature']}")
    print(f"         KS={r['ks_statistic']:.4f}  p={r['p_value']:.4f}  "
          f"ref_μ={r['ref_mean']:.3f} → cur_μ={r['cur_mean']:.3f}")
print()

# ── Generate HTML Report ───────────────────────────────────────────────────────
def _row(r):
    bg    = "#fff3cd" if r["drift_detected"] else "#d4edda"
    badge = '<span style="color:#856404;font-weight:700">⚠ DRIFT</span>' \
            if r["drift_detected"] else \
            '<span style="color:#155724;font-weight:700">✓ OK</span>'
    return f"""
    <tr style="background:{bg}">
      <td>{r['feature']}</td>
      <td>{r['ref_mean']}</td><td>{r['cur_mean']}</td>
      <td>{r['ref_std']}</td><td>{r['cur_std']}</td>
      <td>{r['ks_statistic']}</td>
      <td>{r['p_value']}</td>
      <td>{badge}</td>
    </tr>"""

rows_html = "\n".join(_row(r) for r in results)
summary_color = "#dc3545" if drift_count > 0 else "#28a745"
summary_text  = f"{drift_count} of {len(results)} features show significant drift (p < 0.05)"

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MLOps — Data Drift Report</title>
  <style>
    *{{ box-sizing:border-box; margin:0; padding:0 }}
    body{{ font-family:'Segoe UI',sans-serif; background:#f0f2f5; color:#333 }}
    .wrap{{ max-width:1000px; margin:40px auto; padding:0 20px }}
    h1{{ font-size:1.8rem; margin-bottom:4px }}
    .subtitle{{ color:#666; margin-bottom:30px }}
    .card{{ background:#fff; border-radius:12px; padding:28px;
            box-shadow:0 2px 12px rgba(0,0,0,.08); margin-bottom:24px }}
    .badge-alert{{ display:inline-block; background:{summary_color};
                  color:#fff; border-radius:20px; padding:6px 18px;
                  font-size:.9rem; font-weight:600; margin-bottom:16px }}
    table{{ width:100%; border-collapse:collapse; font-size:.88rem }}
    th{{ background:#343a40; color:#fff; padding:10px 14px; text-align:left }}
    td{{ padding:10px 14px; border-bottom:1px solid #e9ecef }}
    .legend{{ display:flex; gap:20px; font-size:.82rem; color:#555; margin-top:14px }}
    .dot{{ width:12px; height:12px; border-radius:3px; display:inline-block;
           margin-right:6px; vertical-align:middle }}
    .section-title{{ font-size:1rem; font-weight:700; margin-bottom:14px; color:#495057 }}
    .ks-explain{{ background:#e8f4f8; border-left:4px solid #17a2b8;
                  padding:14px 18px; border-radius:6px; font-size:.86rem;
                  line-height:1.6 }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>📊 Data Drift Report</h1>
  <p class="subtitle">Comparing Training Distribution (reference) vs Live Serving Data (current)</p>

  <div class="card">
    <div class="section-title">Summary</div>
    <div class="badge-alert">{summary_text}</div>
    <p style="font-size:.88rem;color:#555">
      Significance threshold: <strong>p &lt; 0.05</strong> (95% confidence).
      Statistical test: <strong>Kolmogorov–Smirnov two-sample test</strong>.
    </p>
    <div class="legend" style="margin-top:16px">
      <div><span class="dot" style="background:#fff3cd;border:1px solid #ffc107"></span> Drift Detected</div>
      <div><span class="dot" style="background:#d4edda;border:1px solid #28a745"></span> No Drift</div>
    </div>
  </div>

  <div class="card">
    <div class="section-title">Feature-Level KS Test Results</div>
    <table>
      <tr>
        <th>Feature</th>
        <th>Ref Mean</th><th>Cur Mean</th>
        <th>Ref Std</th><th>Cur Std</th>
        <th>KS Statistic</th><th>p-value</th>
        <th>Status</th>
      </tr>
      {rows_html}
    </table>
  </div>

  <div class="card">
    <div class="section-title">📖 How to Interpret This Report</div>
    <div class="ks-explain">
      <strong>Kolmogorov–Smirnov (KS) Test</strong> measures the maximum distance between
      two cumulative distribution functions.<br><br>
      • <strong>KS Statistic</strong>: 0 = identical distributions · 1 = completely different.<br>
      • <strong>p-value &lt; 0.05</strong>: Strong evidence the distributions differ → retrain your model.<br>
      • <strong>Recommended action</strong>: When &gt;50% of features drift, trigger a retraining pipeline
        and re-validate model accuracy on fresh labeled data.
    </div>
  </div>

  <p style="text-align:center;color:#aaa;font-size:.78rem;margin-top:10px">
    Generated by Evidently-AI style drift monitor · MLOps Phase 4 Demo
  </p>
</div>
</body>
</html>"""

with open("reports/drift_report.html", "w") as f:
    f.write(html)

print("✅ HTML report saved → reports/drift_report.html")
