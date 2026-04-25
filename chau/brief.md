# Context Handoff — Datathon Forecasting (Châu's side)

Audience: basic ML + decent math. Mixed EN/VI.

---

## 1. Bài toán

Cùng cuộc thi như Tan: dự báo **Revenue** và **COGS** cho ecommerce store Việt Nam — **548 ngày** (2023-01-01 đến 2024-07-01). Metric **MAE tổng hợp**.

Training: ~3800 ngày (2012–2022). Test set ẩn.

---

## 2. Best result — 680,344 MAE (iter-39)

Là kết quả tốt nhất của team. Thấp hơn ~250K so với best pure modeling (iter-10: 935K). Cú nhảy này không phải từ business insight hay model architecture — chủ yếu từ **leaderboard probing**.

---

## 3. Cấu trúc 2 tầng — phải hiểu để không nhầm

Châu's pipeline có **2 tầng riêng biệt**:

### Tầng A — Tabular backbone (LGB/XGB)
- **Input:** features từ sales history
- **Output:** daily Revenue/COGS predictions
- **Best result tầng A đơn lẻ:** ~935K (iter-10)
- **Reproducible** — code có sẵn, có thể chạy lại

### Tầng B — Hierarchical micro / generator
- **Input:** transaction-level data (orders, order_items, products)
- **Output:** daily Revenue/COGS reconstructed bottom-up
- **Đóng góp:** ~10% blend vào tầng A
- **Reproducibility kém** — final iter-39 code không nằm trong repo public

iter-39 = **0.92 × tầng A (sau probing) + 0.08 × tầng B (hierarchical micro)**

Khi nói cải tiến, phải nói rõ đang sửa tầng A, tầng B, hay cách nối.

---

## 4. Validation gap — vấn đề cốt lõi của team

Cùng bug mà Tan gặp: offline metric **không** transfer ra leaderboard.

| Method | Offline MAE | Public MAE | Gap |
|---|---|---|---|
| iter-10 (best tabular) | ~512K (final_548 fold) | 935K | 1.8× |
| iter-20 (temporal weighting) | ~511K | 956K | 1.9× |
| iter-19 (Ridge stacking) | offline tốt | 1.19M | 2.3× |

Nguyên nhân: holdout window (final 548 ngày của train) có distribution gần với 2021–2022, không phản ánh 2023–2024 (regime đã shift, business level đã jump 25%).

**Hậu quả:** nhiều ý tưởng "thắng offline" nhưng tệ trên leaderboard. Rule team áp dụng: cần **2 holdout windows** (final_548 + previous_548) trước khi tin một experiment.

---

## 5. Hành trình experiments — 4 giai đoạn

### Phase 1 (iter 1–8): LGB + Prophet baseline → 949K

Pure technical work. Thử các option cơ bản:
- L1 vs quantile loss
- Train window (2019+ vs full history)
- Lag features (lag_365, lag_548, lag_730)
- Sample weighting
- Ensemble với Prophet
- Seasonal blend

**Wins:** quantile (α=0.5) tốt hơn L1, full history tốt hơn recent window, tắt Prophet (Prophet làm mượt quá), giữ blend 0.25 với seasonal naive.

**Losses:** lag_548 leak vào test horizon, sample weight nặng 2019+ làm mất pre-regime shape, BLEND=0 thảm họa (mất safety net).

Best: **iter-7: 949K** (remove Prophet).

### Phase 2 (iter 10–22): LGB tuning + stacking → 935K floor

Tiếp tục technical tuning:
- Two-stage architecture (predict COGS first, dùng làm feature cho Revenue)
- Multi-quantile ensemble (α=0.45, 0.50, 0.55)
- Multi-seed (3 seeds)
- Heavy hyperparam tuning
- Ridge residual stacker
- CatBoost blend
- Temporal weighting

**Win:** iter-10 ở 935K — two-stage LGB + multi-α = ceiling của pure tabular modeling.

**Losses:** 
- Heavy tuning (iter-12) → 956K, overfit fold
- Ridge stack (iter-19) → 1.19M, false positive offline cực mạnh
- Temporal weighting (iter-20) → 956K, gap offline-public lớn
- CatBoost blend (iter-14/18) → thắng 1 fold, fail fold khác → gate loại

Đây là chỗ team nhận ra: technical tuning đã hết juice ở ~935K. Không có path để vượt qua bằng feature engineering hay model tuning.

### Phase 3 (iter 23–39): PROBE + XGB + SHRINK → **680K**

Đây là **breakthrough phase**. Cú nhảy 935K → 680K trong vài iteration.

**Mechanism (chi tiết ở section 6):**
1. iter-23: XGB two-stage độc lập (925K — tương đương LGB)
2. iter-28: XGB scaled theo probe means → **695K** (jump 230K)
3. iter-30g: Shrink toward mean với W ≈ 91.5% → 683K
4. probe3: Recover được public means: **Rev = 4,379,096.50**, **COGS = 3,988,635.40**
5. iter-32: Shift iter-30g theo probe means → 681K
6. iter-37: Event residual 10% blend → 681K (no change)
7. iter-38: Micro simulator 10% blend → 680.4K
8. **iter-39: Hierarchical micro old/new 50% + 0.08 gate → 680.34K** ✓

Best public score. Tất cả từ đây dùng probe means làm ground truth.

### Phase 4 (iter 40–52): Cố vượt 39, không thành công

Sau 39 thử nhiều generator/residual layers:
- iter-40: Product mix residual (8% blend) → 681K
- iter-41: Customer signal generator → ~681.5K
- iter-43: Payments generator (gate 8% weight) → ~681.3K
- iter-44: Ops composite generator → ~681.3K
- iter-45b: Promo+inventory fusion (gate 5% weight) → 681.033K (gần nhất, +353 MAE worse)
- iter-46: Selective category generator → ~681.5K

Tất cả đều tệ hơn 39 dù chỉ một chút. Ý nghĩa: micro signal đã saturated, residual layers thêm noise nhiều hơn signal.

Iter 49 (Tweedie + reviews/web), 50 (lunar calendar), 51 (DOW×month calibration) — script lưu trong archive, **chưa được submit**. Lý do: không qua được gate 2-window validation.

iter-52 (shrink sweep 85/90/95%) — redundant vì optimum ở 91.5% rồi.

---

## 6. Probing — cơ chế chi tiết

Đây là kỹ thuật quan trọng nhất, làm cho team nhảy từ 935K xuống 680K.

### Insight cốt lõi

Mỗi lần nộp predictions P_i lên Kaggle, mình nhận được:

```
MAE = (1/548) × Σ |P_i - Y_i|
```

MAE là một function của ground truth Y_i. Bằng cách thiết kế các submissions đặc biệt, có thể **reverse-engineer** thông tin về Y_i.

### Probe step 1: Discover the level

Cách đơn giản nhất: nộp một file constant.

Submit P_i = c (constant) → MAE = mean(|c - Y_i|) = MAD của Y_i quanh c.

Function này minimize ở c = median(Y_i). Bằng cách nộp 3-5 giá trị c khác nhau và quan sát MAE response, có thể triangulate được median và mean của test distribution.

### Probe step 2: Scale predictions

Sau khi biết true mean (Rev = 4.38M, COGS = 3.99M):

```
scale_factor = TRUE_MEAN / pred.mean()
pred_calibrated = pred × scale_factor
```

Apply vào XGB output → MAE 925K → 695K. Đây là cú jump 230K mạnh nhất trong toàn bộ history.

### Probe step 3: Shrink toward mean

Dưới MAE, optimal constant prediction = median. Nếu predictions của mình spread quá rộng so với truth, kéo chúng về center sẽ giảm error.

Formula:
```
pred_shrunk = W × pred + (1 - W) × pred.mean()
```

Empirical sweep từ W = 100% xuống 85%, optimum ở **W ≈ 91.5%** (95% retain, 8.5% blend toward mean). Giảm thêm từ 691K xuống 683K.

### Probe step 4: Probe3 (precision recovery)

Bằng cách nộp nhiều file shifted với offset c khác nhau và observe MAE differences một cách có hệ thống, recover được **chính xác**:
- TARGET_REV_MEAN = **4,379,096.50**
- TARGET_COGS_MEAN = **3,988,635.40**

Các con số này được hardcode trong tất cả scripts post-39.

### Cost
Toàn bộ probing chain consume khoảng **5–10 submissions** trên public quota. Đây là deliberate trade-off: mỗi probe là một query vào ground truth distribution, đắt nhưng rất informative.

---

## 7. Iter-39 — best result composition

```
iter-39 (680.34K) = 0.92 × Tier_A_calibrated + 0.08 × Tier_B_micro
```

**Tier A (calibrated tabular):**
- LGB/XGB two-stage (COGS → Revenue)
- Multi-α quantile [0.45, 0.50, 0.55] × 3 seeds = 9 models averaged
- Seasonal blend 0.25 với (month, day) profile
- Scale theo probe means (4.38M, 3.99M)
- Shrink 91.5% toward mean

**Tier B (hierarchical micro):**
- Reconstruct daily Revenue/COGS từ orders/order_items/products
- 50/50 blend của 2 micro variants:
  - **Old:** category-based aggregation (color, size, product category)
  - **New:** hierarchical structure (order → items → products → categories)
- Pure new tệ hơn pure old, nhưng 50/50 ensemble thắng cả hai
- Gate test 5 weights (3%, 5%, 8%, 10%, 12%) → optimum **0.08**

**Composition logic:** Tier A đã đúng level và shape gần đúng (sau probing). Tier B add một layer detail từ transaction data. Blend nhỏ (8%) vì lớn hơn sẽ overcorrect.

---

## 8. Tại sao không vượt được 680K

Sau iter-39 thử ~10 generator/residual layers (product mix, customer, payments, ops, promo+inventory, category) — không cái nào beat. Lý do structural:

1. **Probing đã đủ accurate**: target means được recover xuống tới 2 decimal places. Không còn level error nào để fix.
2. **Tier A shape đã optimal cho tabular features**: thêm shape correction = thêm noise.
3. **Tier B saturated ở 8% blend**: micro signal có upper bound, push nhiều hơn = overcorrect.
4. **Auxiliary tables ngừng ở 2022-12-31**: orders, payments, web_traffic, reviews đều không có data 2023–2024 → không có exogenous signal cho test horizon.

**Honest ceiling cho approach này:** ~680K. Không có path tăng từ tighter calibration hay better feature engineering trên cùng information set.

---

## 9. Khác biệt giữa Châu's approach và Tan's approach

| Aspect | Châu | Tan |
|---|---|---|
| Best score | 680K | 673K (= Châu's file + COGS correction) |
| Cách biết level 4.38M | Probing leaderboard ~5–10 lần | Business analytics (regime recovery + AOV CAGR) |
| Architecture | LGB/XGB two-stage + hierarchical micro + probe + shrink | Post-process trên Châu's file |
| Probing budget | Sử dụng tích cực | Không dùng |
| Reproducibility | Tier A có, Tier B partial | Code đầy đủ |
| Validation gap | Same problem (offline 512K vs public 935K) | Same problem (offline 511K vs public 935K) |

Tan added một business insight cụ thể (COGS correction dựa trên historical odd/even year ratio) lên trên Châu's file → 680K → 673K.

---

## 10. Không gian "big idea" — kết hợp 2 approaches

Combined dead ends (đã prove):
- Historical DOM/DOW shape patterns
- Uniform Revenue scaling
- Pseudo-labeling từ historical model
- Heavy ensemble weights cho micro signals
- Stacking/temporal weighting
- Promo overrides (Urban Blowout)

Combined neutral space (chưa được prove out):
- **Temporal hierarchy reconciliation** — forecast monthly first (level đúng), distribute to weeks, then days. Tách level estimation khỏi shape estimation thực sự.
- **Multi-seasonality decomposition** (MSTL/STL) — separate weekly + annual seasonality + trend trước khi model
- **Tweedie loss** với zero-inflated structure (iter-49 chưa submit)
- **Lunar calendar features** — Châu identify lunar month 4 = 1.54× revenue, lunar 10 = 0.57× revenue. Gregorian features miss this. Chưa submit (iter-50).
- **DOW×month calibration baseline** — pure statistical baseline, weighted recency. Iter-51 expected 640–670K range nếu được tune đúng.

**Honest ceiling không probe:** ~750–800K. Châu's 680K cộng trùng với khoảng này (Châu's 680K = 935K modeling - 255K từ probing).

**Để beat 673K legitimately:** cần một structural insight về 2023–2024 daily shape mà không phụ thuộc historical patterns hay leaderboard feedback.

---

## 11. TL;DR

| Câu hỏi | Câu trả lời |
|---|---|
| Best score Châu? | iter-39 = 680.34K |
| Cấu trúc iter-39? | 92% tabular (LGB/XGB two-stage + probe + shrink) + 8% hierarchical micro |
| Châu biết level 4.38M bằng cách nào? | Probing ~5-10 submissions |
| Probe means cụ thể? | Rev = 4,379,096.50, COGS = 3,988,635.40 |
| Shrink factor optimal? | W ≈ 91.5% (kéo 8.5% về mean) |
| Pure modeling ceiling? | iter-10 ở 935K |
| Probing contribution? | ~250K MAE reduction (935K → 683K) |
| Pseudo-labeling có dùng? | Không. Chỉ OOF two-stage |
| Auxiliary data 2023-2024 có không? | Không có. Tất cả tables ngừng ở 2022-12-31 |
| Iter-39 reproducible đầy đủ? | Tier A có. Tier B (hierarchical micro) code không trong repo public |
| Tại sao không vượt 680K? | Probing đã saturate level/shape; auxiliary signals saturated; auxiliary data ngừng 2022 |
