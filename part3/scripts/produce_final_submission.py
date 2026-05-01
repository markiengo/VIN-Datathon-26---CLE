"""
produce_final_submission.py
---------------------------

Combines Châu's iter-50 friend file with Tan's COGS correction.
Output: part3/submissions/submission_friend_cogs_only.csv (Kaggle public MAE: 673,555)

NOTE: This is the team's experimental best, NOT the official submission.
Official submission (iter-39, 680,344) lives in deliverables/submission.csv.

Pipeline:
  Step 1 — Load Châu's iter-50 friend file (Revenue + COGS, 548 rows)
  Step 2 — Compute historical COGS/Revenue ratios from sales.csv,
           grouped by (month, odd_year), with Aug odd-year cap 0.99
  Step 3 — Apply 60% historical / 40% friend ratio blend to friend Revenue
           Output COGS = Revenue × blended_ratio. Floor at 50,000.
  Step 4 — Save 548-row CSV.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE    = Path(__file__).resolve().parents[2]   # repo root (script is at part3/scripts/)
DATASET = BASE / "dataset"
SUBS    = BASE / "part3" / "submissions"

FLOOR = 50_000.0
HIST_W, FRIEND_W = 0.60, 0.40
AUG_ODD_CAP = 0.99

# ── Step 1: Load Châu's iter-50 friend file ──────────────────────────────────
print("Step 1 — loading Châu's iter-50 friend file...")
friend = pd.read_csv(SUBS / "submission_iter50_blend05.csv", parse_dates=["Date"])
assert len(friend) == 548, f"Expected 548 rows, got {len(friend)}"

# ── Step 2: Historical COGS/Revenue ratios from sales.csv (2018–2022) ────────
print("Step 2 — computing historical COGS/Revenue ratios (2018–2022)...")
sales = pd.read_csv(DATASET / "sales.csv", parse_dates=["Date"])
hist  = sales[sales["Date"].dt.year.between(2018, 2022)].copy()
hist["month"]    = hist["Date"].dt.month
hist["odd_year"] = hist["Date"].dt.year % 2 == 1

ratio_lookup = {}
for (m, is_odd), g in hist.groupby(["month", "odd_year"]):
    r = g["COGS"].sum() / max(g["Revenue"].sum(), 1.0)
    if int(m) == 8 and bool(is_odd):
        r = min(r, AUG_ODD_CAP)
    ratio_lookup[(int(m), bool(is_odd))] = float(r)

# fallbacks if a (month, odd_year) combo is missing from historical data
odd_fb  = float(hist.loc[hist["odd_year"],  "COGS"].sum() / hist.loc[hist["odd_year"],  "Revenue"].sum())
even_fb = float(hist.loc[~hist["odd_year"], "COGS"].sum() / hist.loc[~hist["odd_year"], "Revenue"].sum())

# ── Step 3: Apply 60% historical / 40% friend COGS blend ─────────────────────
print("Step 3 — applying 60% historical / 40% friend COGS blend...")
friend["month"]        = friend["Date"].dt.month
friend["odd_year"]     = friend["Date"].dt.year % 2 == 1
friend["friend_ratio"] = friend["COGS"] / friend["Revenue"].clip(lower=1.0)
friend["hist_ratio"]   = friend.apply(
    lambda r: ratio_lookup.get((r["month"], r["odd_year"]), odd_fb if r["odd_year"] else even_fb),
    axis=1,
)
friend["blended"] = HIST_W * friend["hist_ratio"] + FRIEND_W * friend["friend_ratio"]

out = pd.DataFrame({
    "Date":    friend["Date"].dt.strftime("%Y-%m-%d"),
    "Revenue": friend["Revenue"].round(2),
    "COGS":    np.maximum(friend["Revenue"] * friend["blended"], FLOOR).round(2),
})

# ── Step 4: Save ──────────────────────────────────────────────────────────────
out_path = SUBS / "submission_friend_cogs_only.csv"
out.to_csv(out_path, index=False)
print(f"Step 4 — wrote {out_path.name} ({len(out)} rows)")
print(f"  Revenue mean: {out['Revenue'].mean():>12,.2f}")
print(f"  COGS mean:    {out['COGS'].mean():>12,.2f}")
print(f"  COGS/Rev:     {out['COGS'].mean() / out['Revenue'].mean():.6f}")
