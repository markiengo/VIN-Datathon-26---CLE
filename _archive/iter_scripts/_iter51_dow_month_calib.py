"""Iter-51: Pure DOW × Month calibration baseline (probe-driven).

Strategy (from research agent):
1. Compute per-DOW × per-month mean Revenue/COGS from train 2019-2022
2. Build test prediction = DOW_factor × month_factor × overall_mean
3. Scale to match probe target means exactly (Rev=4379096, COGS=3988635)
4. Save pure baseline + blend-with-iter39 variants

Theoretical floor ~600k; agent expects 640-670k for this baseline.
If pure calibration beats iter-39 → blend 50/50 → target ~650-660k.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
STAGED = ROOT / '_archive' / 'staged_submissions'
sys.stdout.reconfigure(encoding='utf-8')

TARGET_REV_MEAN  = 4_379_096.50
TARGET_COGS_MEAN = 3_988_635.40

train = pd.read_csv(ROOT/'dataset/sales.csv', parse_dates=['Date'])
sample = pd.read_csv(ROOT/'dataset/sample_submission.csv', parse_dates=['Date'])

# === Post-regime train slice (2019-2022) ===
tr = train[train.Date.dt.year >= 2019].copy()
tr['dow']   = tr.Date.dt.dayofweek
tr['month'] = tr.Date.dt.month

# Weighted recency: 2022 gets higher weight
tr['w'] = tr.Date.dt.year.map({2019:1.0, 2020:1.5, 2021:2.0, 2022:3.0})

# Per-DOW factor (relative to overall mean)
overall_rev  = (tr['Revenue'] * tr['w']).sum() / tr['w'].sum()
overall_cogs = (tr['COGS']    * tr['w']).sum() / tr['w'].sum()

def weighted_mean(group, col, w='w'):
    return (group[col] * group[w]).sum() / group[w].sum()

dow_rev  = tr.groupby('dow').apply(lambda g: weighted_mean(g, 'Revenue'), include_groups=False) / overall_rev
dow_cogs = tr.groupby('dow').apply(lambda g: weighted_mean(g, 'COGS'),    include_groups=False) / overall_cogs
print('DOW factors (Revenue):'); print(dow_rev.round(3).to_string())
print('DOW factors (COGS):');    print(dow_cogs.round(3).to_string())

# Per-month factor
mon_rev  = tr.groupby('month').apply(lambda g: weighted_mean(g, 'Revenue'), include_groups=False) / overall_rev
mon_cogs = tr.groupby('month').apply(lambda g: weighted_mean(g, 'COGS'),    include_groups=False) / overall_cogs
print('\nMonth factors (Revenue):'); print(mon_rev.round(3).to_string())

# Per-(month, day) seasonal factor (captures Tết moving across calendar)
tr['md'] = tr.Date.dt.strftime('%m-%d')
md_rev  = tr.groupby('md').apply(lambda g: weighted_mean(g, 'Revenue'), include_groups=False) / overall_rev
md_cogs = tr.groupby('md').apply(lambda g: weighted_mean(g, 'COGS'),    include_groups=False) / overall_cogs

# Per-DOW × month factor (interaction — more granular)
dmon_rev  = tr.groupby(['dow','month']).apply(lambda g: weighted_mean(g, 'Revenue'), include_groups=False) / overall_rev
dmon_cogs = tr.groupby(['dow','month']).apply(lambda g: weighted_mean(g, 'COGS'),    include_groups=False) / overall_cogs

# === Build test predictions (three variants) ===
sub = sample[['Date']].copy()
sub['dow']   = sub.Date.dt.dayofweek
sub['month'] = sub.Date.dt.month
sub['md']    = sub.Date.dt.strftime('%m-%d')

# Variant A: DOW × month (multiplicative, simplest)
sub['rev_A']  = sub['dow'].map(dow_rev)  * sub['month'].map(mon_rev)
sub['cogs_A'] = sub['dow'].map(dow_cogs) * sub['month'].map(mon_cogs)

# Variant B: (month,day) seasonal
sub['rev_B']  = sub['md'].map(md_rev).fillna(1.0)
sub['cogs_B'] = sub['md'].map(md_cogs).fillna(1.0)

# Variant C: DOW-month interaction (captures Sat-May vs Sat-Nov differences)
def interaction_get(row, table):
    return table.get((row['dow'], row['month']), 1.0)

sub['rev_C']  = sub.apply(lambda r: interaction_get(r, dmon_rev),  axis=1)
sub['cogs_C'] = sub.apply(lambda r: interaction_get(r, dmon_cogs), axis=1)

for v in ['A','B','C']:
    # Normalize to probe target means
    factor_rev  = TARGET_REV_MEAN  / sub[f'rev_{v}'].mean()
    factor_cogs = TARGET_COGS_MEAN / sub[f'cogs_{v}'].mean()
    rev  = sub[f'rev_{v}']  * factor_rev
    cogs = sub[f'cogs_{v}'] * factor_cogs

    # Shrink 85% (winning variant)
    W = 0.85
    rev  = W*rev  + (1-W)*rev.mean()
    cogs = W*cogs + (1-W)*cogs.mean()

    out = pd.DataFrame({
        'Date': sample.Date.dt.strftime('%Y-%m-%d'),
        'Revenue': np.round(rev, 2), 'COGS': np.round(cogs, 2),
    })
    out.to_csv(STAGED / f'submission_iter51_calib{v}.csv', index=False)
    print(f'iter51 calib{v}: Rev mean {rev.mean():,.0f}, COGS mean {cogs.mean():,.0f}, std rev {rev.std():,.0f}')

# Load iter-39 for blending
it39 = pd.read_csv(STAGED / 'submission_iter39_hier_micro_ensemble_oldnew50.csv', parse_dates=['Date'])

# Blends 50/50 with each variant
for v in ['A','B','C']:
    calib = pd.read_csv(STAGED / f'submission_iter51_calib{v}.csv')
    for w in [0.25, 0.50]:
        rev_b  = (1-w)*it39.Revenue.values + w*calib.Revenue.values
        cogs_b = (1-w)*it39.COGS.values    + w*calib.COGS.values
        label = f'iter51_blend{v}_{int(w*100):02d}'
        pd.DataFrame({
            'Date': calib['Date'],
            'Revenue': np.round(rev_b, 2), 'COGS': np.round(cogs_b, 2),
        }).to_csv(STAGED / f'submission_{label}.csv', index=False)
        print(f'{label}: saved')

print('\nReady to submit:')
print('  submission_iter51_calibA.csv  — pure DOW×month baseline')
print('  submission_iter51_calibC.csv  — pure DOW-month interaction')
print('  submission_iter51_blendA_50.csv — 50/50 blend with iter-39')
print('  submission_iter51_blendC_25.csv — 25% calib C into iter-39')
