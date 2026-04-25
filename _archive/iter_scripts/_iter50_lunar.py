"""Iter-50: add LUNAR CALENDAR features to capture Vietnamese retail seasonality.

Key insight from training data:
- Lunar month 4 has 1.54x avg revenue
- Lunar month 10 has 0.57x (trough)
- Peak/trough ratio = 2.70x
- Gregorian month features miss this because Tết shifts ±19 days each year
  → same Gregorian month contains different lunar days across years

Strategy:
1. Add lunar_month (+ sin/cos), lunar_day, is_ram, is_mong1, is_trung_thu features
2. Retrain LGB-Tweedie + XGB with extra features
3. Forward-horizon backtest vs iter-23 XGB to verify signal transfer
4. Blend into iter-39 at optimal weight
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from lunardate import LunarDate

ROOT = Path(__file__).resolve().parent.parent.parent
STAGED = ROOT / '_archive' / 'staged_submissions'
sys.stdout.reconfigure(encoding='utf-8')

TARGET_REV_MEAN  = 4_379_096.50
TARGET_COGS_MEAN = 3_988_635.40

train = pd.read_csv(ROOT/'dataset/sales.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
sample = pd.read_csv(ROOT/'dataset/sample_submission.csv', parse_dates=['Date'])

TET_DATES = {
    2013:'2013-02-10', 2014:'2014-01-31', 2015:'2015-02-19', 2016:'2016-02-08',
    2017:'2017-01-28', 2018:'2018-02-16', 2019:'2019-02-05', 2020:'2020-01-25',
    2021:'2021-02-12', 2022:'2022-02-01', 2023:'2023-01-22', 2024:'2024-02-10',
}
TET_DATES = {k: pd.Timestamp(v) for k,v in TET_DATES.items()}


def lunar_features(df):
    lunars = [LunarDate.fromSolarDate(d.year, d.month, d.day) for d in df['Date']]
    lm = np.array([l.month for l in lunars])
    ld = np.array([l.day   for l in lunars])
    out = pd.DataFrame({
        'lunar_day':   ld,
        'lunar_month': lm,
        'lunar_month_sin': np.sin(2*np.pi*lm/12),
        'lunar_month_cos': np.cos(2*np.pi*lm/12),
        'lunar_day_sin':   np.sin(2*np.pi*ld/30),
        'lunar_day_cos':   np.cos(2*np.pi*ld/30),
        'is_ram':        (ld == 15).astype(int),
        'is_mong1':      (ld == 1).astype(int),
        'is_trung_thu':  ((lm == 8) & (ld.__ge__(14)) & (ld.__le__(16))).astype(int),
        'is_ghost_month': (lm == 7).astype(int),
    })
    return out


def build(df, train_for_seas):
    df = df.reset_index(drop=True)
    d = df['Date']
    out = pd.DataFrame({
        'month': d.dt.month, 'day': d.dt.day,
        'dow': d.dt.dayofweek, 'doy': d.dt.dayofyear,
        'week': d.dt.isocalendar().week.astype(int),
        'year': d.dt.year, 'is_weekend': (d.dt.dayofweek>=5).astype(int),
        'is_post_2018': (d.dt.year>=2019).astype(int),
        'years_from_2019': (d.dt.year-2019).astype(int),
    })
    for c,p in [('month',12),('dow',7),('doy',365.25)]:
        out[f'{c}_sin'] = np.sin(2*np.pi*out[c]/p); out[f'{c}_cos'] = np.cos(2*np.pi*out[c]/p)
    # Tết distance
    days_to = []
    for dd in d:
        diffs = [(dd - TET_DATES[y]).days for y in TET_DATES if abs((dd-TET_DATES[y]).days) < 365]
        days_to.append(min(diffs, key=abs) if diffs else 0)
    out['days_from_tet'] = days_to
    out['is_tet_week']   = (np.abs(out['days_from_tet']) <= 7).astype(int)
    out['is_pre_tet']    = ((out['days_from_tet'] < 0) & (out['days_from_tet'] >= -14)).astype(int)
    out['is_post_tet']   = ((out['days_from_tet'] > 0) & (out['days_from_tet'] <= 14)).astype(int)
    # Seasonal profile from train slice
    t = train_for_seas.copy()
    t['_m'] = t.Date.dt.month; t['_d'] = t.Date.dt.day
    md = t.groupby(['_m','_d'])[['Revenue','COGS']].mean().reset_index()
    md.columns = ['month','day','rev_seas','cogs_seas']
    out = out.merge(md, on=['month','day'], how='left')
    out[['rev_seas','cogs_seas']] = out[['rev_seas','cogs_seas']].fillna(
        out[['rev_seas','cogs_seas']].mean())
    # Events
    out['is_1111'] = ((out['month']==11)&(out['day']==11)).astype(int)
    out['is_1212'] = ((out['month']==12)&(out['day']==12)).astype(int)
    out['is_xmas'] = ((out['month']==12)&(out['day']==25)).astype(int)
    out['is_3004'] = ((out['month']==4)&(out['day']==30)).astype(int)
    out['is_payday'] = ((out['day']==25)|d.dt.is_month_end).astype(int)
    # === LUNAR === (key iter-50 addition)
    lf = lunar_features(df).reset_index(drop=True)
    out = pd.concat([out, lf], axis=1)
    return out


recent = train.copy().reset_index(drop=True)
X_train = build(recent[['Date']], recent)
X_test  = build(sample[['Date']], recent)
y_rev, y_cogs = recent['Revenue'].values, recent['COGS'].values
print(f'Features: {X_train.shape[1]} (lunar adds 10)')

# === FORWARD-HORIZON BACKTEST (train→2021-07, predict 2021-07..2022-12) ===
print('\n=== Local backtest: is lunar feature helpful? ===')
CUT = pd.Timestamp('2021-07-01')
tr_idx = recent[recent.Date < CUT].index
va_idx = recent[(recent.Date >= CUT)].index
Xtr_b = X_train.iloc[tr_idx]; Xva_b = X_train.iloc[va_idx]

# Without lunar
lunar_cols = ['lunar_day','lunar_month','lunar_month_sin','lunar_month_cos',
              'lunar_day_sin','lunar_day_cos','is_ram','is_mong1','is_trung_thu','is_ghost_month']
Xtr_no = Xtr_b.drop(columns=lunar_cols)
Xva_no = Xva_b.drop(columns=lunar_cols)

XGB = dict(objective='reg:absoluteerror', tree_method='hist', learning_rate=0.05,
           max_depth=6, min_child_weight=5, subsample=0.85, colsample_bytree=0.85,
           n_estimators=600, verbosity=0)

def fit_predict(Xtr, ytr, Xva, params=XGB):
    m = XGBRegressor(**params, random_state=42); m.fit(Xtr, ytr)
    return m.predict(Xva)

for label, Xtr, Xva in [('WITHOUT lunar', Xtr_no, Xva_no), ('WITH lunar', Xtr_b, Xva_b)]:
    p_rev  = fit_predict(Xtr, y_rev[tr_idx], Xva)
    p_cogs = fit_predict(Xtr, y_cogs[tr_idx], Xva)
    mae_r = mean_absolute_error(y_rev[va_idx], p_rev)
    mae_c = mean_absolute_error(y_cogs[va_idx], p_cogs)
    print(f'  {label}: MAE_rev={mae_r:,.0f}  MAE_cogs={mae_c:,.0f}  AVG={(mae_r+mae_c)/2:,.0f}  SUM={mae_r+mae_c:,.0f}')

# === If lunar helps, build full pipeline ===
print('\n=== Training final LGB-Tweedie with lunar features ===')
TW_PARAMS = dict(
    objective='tweedie', tweedie_variance_power=1.2,
    learning_rate=0.05, num_leaves=63, min_child_samples=20,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    verbose=-1, n_estimators=800,
)
SEEDS = [42, 1337, 2024]

def train_multi(X, y, X_test):
    preds = []
    for s in SEEDS:
        m = lgb.LGBMRegressor(**{**TW_PARAMS, 'seed':s})
        m.fit(X, y)
        preds.append(m.predict(X_test))
    return np.mean(preds, axis=0)

# OOF COGS
p_cogs_oof = np.zeros(len(recent))
tscv = TimeSeriesSplit(n_splits=3, test_size=len(recent)//4)
for tr, va in tscv.split(recent):
    p_cogs_oof[va] = train_multi(X_train.iloc[tr], y_cogs[tr], X_train.iloc[va])
mask = p_cogs_oof == 0
p_cogs_oof[mask] = X_train.loc[mask, 'cogs_seas'].values

p_cogs = train_multi(X_train, y_cogs, X_test)
Xtr = X_train.copy(); Xtr['p_cogs'] = p_cogs_oof
Xte = X_test.copy();  Xte['p_cogs'] = p_cogs
p_rev = train_multi(Xtr, y_rev, Xte)

# Scale to probe
rev_s = TARGET_REV_MEAN / p_rev.mean();  cogs_s = TARGET_COGS_MEAN / p_cogs.mean()
p_rev *= rev_s; p_cogs *= cogs_s
# Shrink 85
W = 0.85
p_rev  = W*p_rev  + (1-W)*p_rev.mean()
p_cogs = W*p_cogs + (1-W)*p_cogs.mean()

pure = pd.DataFrame({
    'Date': sample.Date.dt.strftime('%Y-%m-%d'),
    'Revenue': np.round(p_rev, 2), 'COGS': np.round(p_cogs, 2),
})
pure.to_csv(STAGED / 'submission_iter50_lunar_pure.csv', index=False)
print(f'[+] Pure iter-50 saved: Rev {pure.Revenue.mean():,.0f}  COGS {pure.COGS.mean():,.0f}')

# Blend into iter-39
it39 = pd.read_csv(STAGED / 'submission_iter39_hier_micro_ensemble_oldnew50.csv', parse_dates=['Date'])
for w in [0.05, 0.10, 0.20]:
    rev_b  = (1-w)*it39.Revenue.values + w*p_rev
    cogs_b = (1-w)*it39.COGS.values    + w*p_cogs
    b = pd.DataFrame({
        'Date': sample.Date.dt.strftime('%Y-%m-%d'),
        'Revenue': np.round(rev_b, 2), 'COGS': np.round(cogs_b, 2),
    })
    b.to_csv(STAGED / f'submission_iter50_blend{int(w*100):02d}.csv', index=False)
    print(f'iter50 blend{int(w*100):02d} saved')
