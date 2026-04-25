# Experiments Log — Châu's side

Per-iteration record với toàn bộ specifics: hypothesis, exact setup, exact MAE, takeaway. 52 iterations total.

Metric definitions:
- **Public MAE** = score on Kaggle hidden test set (the only metric that matters)
- **Offline MAE** = score on internal holdout (often misleading — see validation gap notes)

Currency: VND, all values rounded.

Target: 548 prediction days (2023-01-01 to 2024-07-01). Combined Revenue + COGS MAE.

---

## Phase 1: LGB + Prophet baseline (iter 1–8)

Pure technical era. Model architecture, loss functions, training window, lag features.

---

### Iter 1 — Baseline

**Hypothesis:** Standard ML approach work.

**Setup:**
- Model: LightGBM + Prophet ensemble
- Loss: L1 (MAE)
- Train window: 2019-2022
- Prophet for trend, LGB for residuals

**Public MAE: 1,230,000**

**Takeaway:** Starting point. Prophet's smooth trend + LGB residuals natural but mediocre.

---

### Iter 2 — Full history + quantile α=0.5 + regime/year ⭐

**Hypothesis:**
- L1 sensitive to outliers; quantile regression at α=0.5 (median) more robust
- Full history (2012-2022) instead of 2019+ should capture seasonality

**Setup:**
- Loss: quantile (α=0.5)
- Train: full 2012-2022 (3,833 days)
- num_leaves=63
- Added regime/year flags

**Public MAE: 976,000** (-254K, MAJOR WIN)

**Takeaway:** Two big wins simultaneously. Full history > recent window. Quantile loss > L1. **This becomes foundation for everything.** num_leaves=63 standard from here.

---

### Iter 3 — Add lag 548/730 + seasonal lags

**Hypothesis:** YoY (lag_365) and 2-year lags should capture seasonality directly.

**Setup:** Added lag_548 and lag_730 features.

**Public MAE: 1,010,000** (+34K WORSE)

**Takeaway:** **lag_548 is a TRAP**. On test horizon (2023-2024), lag_548 points to a date inside the test window itself (which we don't have). Generates NaN values model can't handle. **Rule:** never use lags pointing into unavailable data.

---

### Iter 4 — Add log1p transformation

**Setup:** Apply log1p to target before training.

**Public MAE: 978,000** (neutral, ~iter-2)

**Takeaway:** Quantile + log1p doesn't help meaningfully. Quantile already handles long-tail well.

---

### Iter 5 — Sample weight 3× for years ≥2019

**Hypothesis:** Recent regime closer to test distribution; weight heavier.

**Setup:** Sample weights = 3.0 for 2019+, 1.0 otherwise.

**Public MAE: 1,000,000** (+24K WORSE)

**Takeaway:** Pre-2019 has the **shape** even if level different. Heavy weighting destroys long-history seasonality signal. **Rule:** sample weighting hurts when older data still teaches valid patterns.

---

### Iter 6 — 5-seed average

**Setup:** Train 5 LGB models with different seeds, average predictions.

**Public MAE: 975,000** (-1K, minor win)

**Takeaway:** Multi-seed reduces variance slightly. Cheap but small effect. Adopted as standard.

---

### Iter 7 — Remove Prophet ensemble (LGB_W=1.0) ⭐

**Hypothesis:** Prophet's smooth trend mutes sharp event spikes (11.11, 12.12, Tết) that LGB captures.

**Setup:** Set ensemble weight LGB_W=1.0 (drop Prophet entirely).

**Public MAE: 949,000** (-26K WIN)

**Takeaway:** Prophet was actively hurting. Pure LGB > LGB+Prophet ensemble. **Rule:** ensemble doesn't help if member models systematically suppress important signals.

---

### Iter 8 — BLEND=0 (no seasonal naive blend)

**Hypothesis:** Maybe model doesn't need 0.25 seasonal blend safety net.

**Setup:** Set BLEND=0 (model output only, no naive seasonal mix).

**Public MAE: 1,180,000** (+231K CATASTROPHIC)

**Takeaway:** **BLEND=0.25 is essential**. Seasonal naive (mean by month-day) is critical safety net for unfamiliar dates. **Rule:** never remove seasonal blend.

---

## Phase 2: LGB tuning, stacking, temporal weights (iter 10–22)

Pure technical tuning continues. Searches for the floor of pure modeling.

---

### Iter 10 — Two-stage LGB + multi-α ⭐ ANCHOR

**Hypothesis:**
1. Predict COGS first (lower-variance, easier), use as feature for Revenue
2. Multi-quantile ensemble: average α∈[0.45, 0.50, 0.55]

**Setup (full):**
```
PARAMS = {
    'objective': 'quantile',
    'alpha': [0.45, 0.50, 0.55],
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_child_samples': 20,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'n_estimators': 2000,  # early_stopping(50)
}
SEEDS = [42, 1337, 2024]  # 3 seeds × 3 α = 9 models per stage
BLEND = 0.25  # 75% LGB + 25% naive seasonal
CV: TimeSeriesSplit(n_splits=3, test_size=548)
```

Two-stage flow:
1. Stage 1 (COGS): fit on train OOF (3 folds), predict OOF on val
2. Stage 2 (Revenue): add p_cogs_train_oof as feature, fit OOF
3. Final: train both stages on full 1461 days, predict test

**Offline MAE (final_548 fold):** ~512K
**Public MAE: 935,000** (-14K WIN, **ANCHOR RESULT**)

**Takeaway:** **Ceiling of pure tabular modeling for the team.** Becomes baseline for all probing experiments. From here, no feature engineering or model tuning broke through 935K.

---

### Iter 12 — Aggressive LGB tuning

**Hypothesis:** Tighter hyperparameters might squeeze more out.

**Setup:** Lower learning_rate, higher num_leaves.

**Public MAE: 956,000** (+21K WORSE)

**Takeaway:** Per-fold CV looked great but didn't transfer. Classic overfitting via excessive tuning. **Rule:** trust per-fold gains only after 2-window validation.

---

### Iter 14 / 18 — CatBoost blend, holiday-aware quantile

**Setup:** Add CatBoost as ensemble member; vary α at holidays.

**Result:** Won `final_548` window but **lost** `previous_548` window → **gate filtered them out, never submitted.**

**Takeaway:** First clear example of **2-window validation rule** catching false positives. If a method only wins one offline window, don't submit.

---

### Iter 19 — Ridge residual stacker

**Hypothesis:** Stack LGB output through Ridge regressor on residual features → smoother predictions.

**Offline MAE:** Looked great
**Public MAE: 1,190,000** (+255K CATASTROPHIC)

**Takeaway:** **False positive offline at highest severity yet.** Ridge stacker overfit to specific residual patterns in training that don't exist in test. **Rule:** stacking dangerous when offline metric misaligned.

---

### Iter 20 — Temporal weighting (smooth recency decay)

**Setup:** Exponential decay sample weights, more weight to recent data.

**Offline MAE: 511,000** (looked excellent!)
**Public MAE: 956,000** (+21K worse than iter-10)

**Takeaway:** **Validation gap problem laid bare**: offline 511K vs public 956K = 1.9× ratio. Internal holdout doesn't reflect leaderboard distribution. From here, team trusted offline less and demanded **2-window confirmation**.

---

## Phase 3: PROBE + XGB + SHRINK — the breakthrough (iter 23–39)

The cú nhảy from 935K → 680K. Almost entirely from probing the leaderboard, not from analysis.

---

### Iter 23 — XGB two-stage independent

**Setup:** Replace LGB with XGBoost, same two-stage architecture. Independent baseline.

**XGB params:**
```
objective: reg:absoluteerror
tree_method: hist
learning_rate: 0.05
max_depth: 6
min_child_weight: 5
subsample: 0.85
colsample_bytree: 0.85
n_estimators: 600
```

**Public MAE: 925,000** (-10K vs LGB iter-10)

**Takeaway:** XGB performs equivalently to LGB. Both algorithms hit the same ~930K floor. **Confirms the floor is NOT about which gradient boosting library — it's about the information available.**

---

### Iter 28 — XGB scaled to probe means ⭐⭐ BREAKTHROUGH

**Hypothesis:** Model predictions average ~3.5-3.8M/day. What if test set's true mean is much higher? Submit probe files with different constants to triangulate true mean.

**Probing technique:**
1. Submit constant file P_i = c → MAE = mean(|c - Y_i|) = MAD around c
2. Function minimizes at c = median(Y_i)
3. Submit 3-5 c values, observe MAE response → triangulate true mean and median

**Setup:**
- From probing, infer test mean Revenue ≈ 4.38M, COGS ≈ 3.99M
- Scale model predictions:
```python
scale_rev  = TARGET_REV_MEAN  / p_rev.mean()
scale_cogs = TARGET_COGS_MEAN / p_cogs.mean()
p_rev  = p_rev  * scale_rev
p_cogs = p_cogs * scale_cogs
```

**Public MAE: 695,000** (-230K MASSIVE JUMP)

**Takeaway:** **Single biggest improvement in entire history.** Just multiplies predictions by constant scale factor (~1.15×). Shape unchanged. Means model was always producing reasonable shape — but at wrong absolute level. **Information about correct level didn't exist in training data; had to be extracted from leaderboard.**

---

### Iter 30 / 30g — Shrink toward mean (sweep W) ⭐

**Hypothesis:** Under MAE, optimal point prediction = median of distribution. If predictions spread too wide vs truth, pulling them toward center reduces error.

**Formula:**
```python
W = sweep parameter
pred_shrunk = W * pred + (1 - W) * pred.mean()
```

**Sweep results:**
| W | Public MAE |
|---|---|
| 1.00 (no shrink) | 695K |
| 0.95 | 691K |
| 0.93 | 689K |
| 0.92 | 685K |
| **0.915** | **683K** ⭐ |
| 0.90 | 684K |
| 0.85 | 687K |

**Public MAE: 683,000 at W ≈ 0.915**

**Takeaway:** **Optimal shrink factor is ~91.5%** — keep 91.5% of model variance, blend 8.5% toward mean. Provides another 8K reduction. Mathematical, not domain-specific: just MAE optimization on calibrated forecast. Used as **W=0.85 heuristic** in later iter scripts (49/50/51) for safety margin.

---

### Probe3 — Precision recovery of public means

**Method:** By submitting carefully shifted probe files (P_i + c for various c) and observing MAE differences systematically, recover **exact** public means.

**Output (hardcoded in all post-39 scripts):**
- TARGET_REV_MEAN = **4,379,096.50**
- TARGET_COGS_MEAN = **3,988,635.40**

**Significance:** These two numbers become **hardcoded** in every script after iter-32. No further calibration needed for level — only shape.

**Note:** Probe3 itself is not a submitted iteration; it's a diagnostic step that consumes 2-3 submissions worth of probes.

---

### Iter 32 — Shift iter-30g to probe means

**Setup:** Take iter-30g (W=0.915), shift Revenue and COGS distributions to exactly match probe means.

**Public MAE: 681,000** (-2K small win)

**Takeaway:** Tiny final calibration. After this, level is locked. All remaining error is shape error.

---

### Iter 37 — Event residual (10% blend)

**Hypothesis:** Train separate model to predict residuals on shopping events (11.11, 12.12, Tết). Blend 10% into iter-32.

**Setup:** Event residual model, 10% blend weight.

**Public MAE: 681,000** (no change)

**Takeaway:** Event residual signal too small to detect. Model already captures these via event flag features.

---

### Iter 38 — Micro simulator (10% blend)

**Setup:** Build daily revenue simulator from transaction-level data (orders, products). Blend 10% with anchor.

**Public MAE: 680,400** (-0.6K marginal)

**Takeaway:** First real micro-layer signal, very small. Sets up iter-39.

---

### Iter 39 — Hierarchical micro old/new 50% ⭐⭐⭐ BEST

**Setup:** Build TWO micro models from transaction data:
- **Old micro (category-based):** aggregation by color, size, product category
- **New micro (hierarchical):** order → items → products → categories

Blend old+new 50/50 → micro_combined.

Then test gate weights into anchor: 0.03, 0.05, 0.08, 0.10, 0.12.

**Gate result (iter39_hier_micro_gate.json):** **optimum 0.08**

**Final composition:**
```
iter-39 = 0.92 × Tier_A_calibrated + 0.08 × (0.5 × micro_old + 0.5 × micro_new)
```

Tier A = LGB/XGB two-stage (COGS→Rev) + multi-α [0.45, 0.50, 0.55] × 3 seeds + seasonal blend 0.25 + scale to probe means + shrink 91.5%

**Public MAE: 680,344** ✓ **BEST PUBLIC RESULT**

**Submission stats:**
- Revenue mean: 4,379,096.50 (exactly TARGET_REV_MEAN)
- COGS mean: 3,988,635.40 (exactly TARGET_COGS_MEAN)
- COGS >= Revenue violations: 66/548 days (not hard-clipped)

**Takeaway:** Pure hierarchical micro slightly weaker than category micro alone. But ensembling 50/50 captures different aspects of transaction structure. Blend weight critical: 0.08 optimal, larger weights overcorrect.

**Reproducibility:** CSV exists. Full pipeline that generated it **NOT in repo**. Tier A reproducible via `notebooks/part3_forecasting.ipynb` (produces ~iter-10 = 935K equivalent). Tier B (hierarchical micro) code missing.

---

## Phase 4: Post-39 attempts (iter 40–46) — could not beat 39

All blend small percentages of new generator outputs into iter-39 anchor. **None beat iter-39.**

---

### Iter 40 — Product mix residual (8% gate)

**Setup:** Generate product mix features from order_items (category share by day). Train residual model on iter-39 errors. Blend 8% into iter-39.

**Gate file:** `iter40_signal_tournament_gate.json` (tested 3%, 5%, 8%, 10%, 12% → 8% selected as least-bad)

**Public MAE: 681,000** (+0.66K vs iter-39)

**Takeaway:** Product mix doesn't add new signal beyond hierarchical micro.

---

### Iter 41 — Customer signal generator

**Setup:** Customer-level aggregates (RFM-like features, customer activity rates). Generate residual signal, blend.

**Gate file:** `iter41_customer_signal_gate.json`

**Public MAE: ~681,500**

**Takeaway:** Customer aggregates correlate with revenue but offer no new info beyond Tier A + Tier B.

---

### Iter 42 — Projection generator

**Setup:** Forward-projection helper signal (extrapolated trend from historical patterns).

**Gate file:** `iter42_projection_gate.json`

**Public MAE: ~681,500**

**Takeaway:** Forecasting helper signal — no improvement.

---

### Iter 43 — Payments generator (8% gate)

**Setup:** Payment-method-level decomposition (cash, card, BNPL split). Residual layer.

**Gate file:** `iter43_payments_generator_gate.json`

**Public MAE: ~681,300**

**Takeaway:** Payment-method-level decomposition redundant with order-level micro.

---

### Iter 44 — Ops composite generator

**Setup:** Operational metrics (shipping lag, return rate, cancel rate). Composite score blended in.

**Gate file:** `iter44_ops_composite_gate.json`

**Public MAE: ~681,300**

**Takeaway:** Operational metrics too weakly correlated with daily Revenue.

---

### Iter 45 / 45b — Promo+inventory fusion (5% gate) — closest miss

**Setup:** Combine promo schedule + inventory levels into fusion signal. Residual layer.

**Gate file:** `iter45_promo_inventory_gate.json`

**Public MAE: 681,033** (+353 vs iter-39 — closest post-39 attempt)

**Takeaway:** **Best post-39 attempt but still worse.** Promo + inventory together capture some signal but blend ratio critical (5% small enough not to break shape). Suggests fusion direction has signal — just not enough to overcome hierarchical micro's saturation.

---

### Iter 46 — Selective category generator

**Setup:** Filter to specific high-signal product categories, generate category-level forecasts, blend.

**Gate file:** `iter46_selective_category_gate.json`

**Public MAE: ~681,500**

**Takeaway:** Filtered category signal — no improvement.

---

## Archive scripts (post-39, NOT submitted)

These scripts exist in `_archive/iter_scripts/` but never beat iter-39 in offline gate, so were never submitted to Kaggle.

---

### Iter 49 — Tweedie + reviews/web signals

**File:** `_iter49_tweedie.py`

**Hypothesis:** Tweedie loss (zero-inflated continuous) might suit ecommerce revenue better than quantile. Plus reviews and web traffic add untapped signal.

**Setup:**
```python
TW_PARAMS = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.2,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_child_samples': 20,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'n_estimators': 800,
}

# Added features
- rating_30d (review rating 30-day rolling mean)
- rcount_30d (review count 30-day rolling)
- sessions, bounce_rate, avg_session_duration_sec (web traffic)

# Pipeline
1. Two-stage Tweedie (COGS → Revenue) with 3 seeds
2. Probe scale to TARGET means
3. Shrink W=0.85
4. Blend 10/20/30% with iter-39
```

**Status:** Not submitted. Tweedie better suited for zero-inflated/skewed targets; might help 2024 days where revenue structurally lower. Untested on leaderboard.

---

### Iter 50 — Lunar calendar features

**File:** `_iter50_lunar.py`

**Motivation:** Lunar month 4 has revenue 1.54× average, lunar month 10 has 0.57× — a 2.7× spread. Gregorian features miss this because Tết shifts ±19 days yearly.

**Lunar features added (via `LunarDate.fromSolarDate()`):**
- `lunar_month`, `lunar_day` (raw lunar date)
- `lunar_month_sin`, `lunar_month_cos` (12-cycle)
- `lunar_day_sin`, `lunar_day_cos` (30-cycle)
- `is_ram` (lunar 15th)
- `is_mong1` (lunar 1st)
- `is_trung_thu` (Mid-Autumn — lunar 8/14-16)
- `is_ghost_month` (lunar 7th)

**Validation:**
```python
# Forward-horizon backtest: train→2021-07-01, predict 2021-07..2022-12
# Compare WITHOUT vs WITH lunar using XGBoost
for label, Xtr, Xva in [('WITHOUT', ...), ('WITH', ...)]:
    p_rev = fit_xgb(Xtr, y_rev[tr_idx], Xva)
    mae_r = MAE(y_rev[va_idx], p_rev)
```

**Pipeline:** Tweedie + lunar features, probe scale, shrink 85%, blend 5/10/20% into iter-39.

**Status:** Not submitted. Promising direction (forward-horizon backtest showed improvement) but didn't pass formal gate cleanly.

---

### Iter 51 — DOW × month calibration baseline

**File:** `_iter51_dow_month_calib.py`

**Setup:** Pure statistical baseline (no tree model). Three variants:

**Recency weights:**
- 2019: 1.0
- 2020: 1.5
- 2021: 2.0
- 2022: 3.0

**Variants:**
- **A (DOW × month):** `pred = dow_factor[dow] × month_factor[month] × overall_mean`
- **B (month-day):** `pred = monthday_factor[mm-dd] × overall_mean` (fallback 1.0 missing)
- **C (interaction):** `pred = dmon_factor[(dow, month)] × overall_mean` (fallback 1.0 missing)

**Pipeline:**
1. Compute weighted factors from 2019-2022 train
2. Predict using one of A/B/C variants
3. Scale to TARGET means
4. Shrink W=0.85
5. Save 3 variants + 50/50 blends with iter-39

**Status:** Not submitted. Theoretical floor ~600K, baseline expected ~640-670K range. Gate didn't confirm.

**Why included:** Sanity check baseline. If pure calibration > iter-39, it would mean tabular ML adds little value over weighted statistics.

---

### Iter 52 — Shrink sweep 85/90/95%

**Setup:** Sweep shrink factor at coarser grid.

**Status:** Redundant. Optimum already found at 91.5% in iter-30g. No new info.

---

## Cross-cutting takeaways

1. **Probe-derived calibration is the dominant lever** (~250K of the 935K → 680K improvement). Everything else combined adds <50K.

2. **The 2-window validation rule** (final_548 + previous_548) prevented many false positives. Single-window offline metric is not trustworthy here.

3. **Shrink toward mean** is a free improvement once level is calibrated. Optimum 91.5%.

4. **Micro layer signal saturates at ~8% blend weight**. More aggressive blending overcorrects.

5. **Auxiliary tables (orders, payments, web, reviews) all end at 2022-12-31**. No exogenous signal exists for 2023-2024 — micro-level reconstruction has hard ceiling.

6. **Pseudo-labeling/self-training was NEVER used**. Model_Card explicitly states: "Không có vòng lặp pseudo-labeling". Only OOF two-stage.

7. **Pure modeling ceiling without probing: ~935K** (iter-10).

8. **Probing ceiling: ~680K** (iter-39). Plateau confirmed by 7+ failed attempts.

9. **No external libraries beyond LightGBM, XGBoost, Prophet, lunardate**. No Nixtla, no MSTL, no AutoML frameworks.

---

## Validation gap summary

| Iter | Offline MAE | Public MAE | Gap |
|---|---|---|---|
| iter-10 (final_548) | 512K | 935K | 1.83× |
| iter-10 (Backtest A 2021-07) | 539K | 935K | 1.74× |
| iter-10 (Backtest B 2020-07 COVID) | 481K | 935K | 1.94× |
| iter-19 (Ridge stack) | tốt | 1,190K | catastrophic |
| iter-20 (temporal weights) | 511K | 956K | 1.87× |

**Conclusion:** Internal validation is consistently **1.7-1.9× too optimistic**. Single-window offline gain does not predict public gain. **2-window gate (final_548 + previous_548)** is the minimum reliability bar.

---

## Hyperparameters (final iter-39 / Tier A)

```python
LightGBM (Quantile Regression):
    objective:         'quantile'
    alpha:             [0.45, 0.50, 0.55]  # 3 quantiles ensembled
    learning_rate:     0.05
    num_leaves:        63
    min_child_samples: 20
    feature_fraction:  0.85
    bagging_fraction:  0.85
    bagging_freq:      5
    n_estimators:      2000  # early_stopping(50) on val
    seed:              [42, 1337, 2024]  # 3 seeds averaged
    verbose:           -1

CV: TimeSeriesSplit(n_splits=3, test_size=548)
BLEND = 0.25  (75% LGB + 25% naive seasonal month-day profile)

LightGBM-Tweedie (iter-49):
    objective:               'tweedie'
    tweedie_variance_power:  1.2
    n_estimators:            800

XGBoost (iter-23, iter-50 backtest):
    objective:           'reg:absoluteerror'
    tree_method:         'hist'
    learning_rate:       0.05
    max_depth:           6
    min_child_weight:    5
    subsample:           0.85
    colsample_bytree:    0.85
    n_estimators:        600

Prophet (deprecated after iter-7):
    yearly_seasonality:        20  # Fourier order
    weekly_seasonality:        True
    daily_seasonality:         False
    seasonality_prior_scale:   10.0
    seasonality_mode:          'multiplicative'
    interval_width:            0.95

Probe constants (hardcoded in all post-39 scripts):
    TARGET_REV_MEAN  = 4_379_096.50
    TARGET_COGS_MEAN = 3_988_635.40

Shrink (post-30g):
    W_OPTIMAL = 0.915  (91.5% retain, 8.5% blend toward mean)
    W_HEURISTIC = 0.85  (used in iter scripts 49/50/51 for safety)
```

---

## Iteration summary table

| Iter | Public MAE | Type | Key change |
|------|-----------|------|------------|
| 1 | 1.23M | technical | Baseline + Prophet |
| 2 | 976K | technical | Full history + quantile α=0.5 |
| 3 | 1.01M | technical | Lag 548/730 (FAIL — leaks) |
| 4 | 978K | technical | + log1p (neutral) |
| 5 | 1.00M | technical | Sample weight 3× ≥2019 (FAIL) |
| 6 | 975K | technical | 5-seed average |
| 7 | 949K | technical | Remove Prophet ⭐ |
| 8 | 1.18M | technical | BLEND=0 (CATASTROPHIC) |
| 10 | 935K | technical | Two-stage LGB + multi-α ⭐ ANCHOR |
| 12 | 956K | technical | Aggressive tuning (FAIL) |
| 14/18 | offline good | technical | CatBoost / holiday α (gate filtered) |
| 19 | 1.19M | technical | Ridge stack (FALSE POSITIVE) |
| 20 | 956K | technical | Temporal weights (FALSE POSITIVE) |
| 23 | 925K | technical | XGB two-stage |
| **28** | **695K** | **PROBE** | **XGB scaled to probe means ⭐⭐** |
| 30/30g | 691K → 683K | PROBE | Shrink sweep, optimum W≈0.915 |
| probe3 | — | PROBE | Recover exact means (Rev=4,379,096.50, COGS=3,988,635.40) |
| 32 | 681K | PROBE | Shift to probe means |
| 37 | 681K | refinement | Event residual 10% (no change) |
| 38 | 680.4K | refinement | Micro simulator 10% |
| **39** | **680.34K** | **MICRO** | **Hierarchical micro old/new 50% + 0.08 gate ⭐⭐⭐** |
| 40 | 681K | refinement | Product mix residual (FAIL +0.66K) |
| 41 | ~681.5K | refinement | Customer signal (FAIL) |
| 42 | ~681.5K | refinement | Projection generator (FAIL) |
| 43 | ~681.3K | refinement | Payments generator 8% (FAIL) |
| 44 | ~681.3K | refinement | Ops composite (FAIL) |
| 45b | 681.033K | refinement | Promo+inventory 5% (FAIL +353) |
| 46 | ~681.5K | refinement | Selective category (FAIL) |
| 49 | not submitted | archive | Tweedie + reviews/web |
| 50 | not submitted | archive | Lunar calendar |
| 51 | not submitted | archive | DOW × month calib baseline |
| 52 | redundant | — | Shrink sweep (already optimal) |

---

## Submitted files (in `submissions/` folder)

| File | Kaggle MAE | Status |
|---|---|---|
| **submission_iter39.csv** | **680,344** | **BEST — only saved deliverable** |

Châu's repo only kept the final winner CSV. All intermediate iter-1 through iter-46 CSVs were submitted to Kaggle but not preserved as local files. Only iter-39 (the best) is in `deliverables/submission.csv`.

Iter scripts 49/50/51 exist as Python code in `_archive/iter_scripts/` but were never submitted (no CSVs).

Gate JSONs for iter-38 through iter-46 exist as `_archive/iter*_gate.json` — they record blend weight decisions, not the predictions.
