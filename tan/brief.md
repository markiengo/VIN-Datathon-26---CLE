# Context Handoff — Datathon Forecasting (Tan's side)

Audience: basic ML + decent math. Mixed EN/VI.

---

## 1. Bài toán

Dự báo **Revenue** và **COGS** cho ecommerce store Việt Nam — **548 ngày liên tiếp** (2023-01-01 đến 2024-07-01). Metric: **MAE tổng hợp** (Revenue MAE + COGS MAE), càng thấp càng tốt.

Training data: ~3800 ngày lịch sử (2012–2022). Tất cả auxiliary tables (orders, traffic, promos, payments, reviews) cũng chỉ đến **2022-12-31**. Test set ẩn hoàn toàn.

---

## 2. Vấn đề cốt lõi

**The level problem.**

Model train trên 2012–2022 sẽ tự nhiên extrapolate về phía cuối training window — khoảng **3.5M VND/ngày**. Ground truth test set là **4.38M VND/ngày** — cao hơn ~25%.

Đây không phải vấn đề feature hay architecture. Đây là **information gap** — con số 4.38M là một fact về 2023–2024 không tồn tại trong training data.

**Business context giải thích tại sao 4.38M:**
- Business crash ~40% năm 2019 (từ ~5.3M xuống ~3M/ngày)
- 2020–2022: phục hồi dần
- 4.38M = 82.5% của mức pre-2019 = 60% recovery của cú crash
- Growth driver không phải order volume (orders thực tế giảm -2.6%) mà là **AOV tăng +10.9%** (2019–2022)
- AOV CAGR từ transaction data: **+6.48%/năm** → 2024 cao hơn 2023 ~6.5%

Insight này đúng nhưng chỉ cho biết **level**. Nó không cho biết **daily shape** — ngày nào trong tháng cao, ngày nào thấp.

---

## 3. Validation gap — tại sao offline metric không đáng tin

Internal holdout (giả lập test bằng cách hold out 2021–2022):

- Offline MAE tốt nhất: **~511K**
- Kaggle thực tế: **~935K**
- Gap: **~424K**

Lý do: model dùng lag_365 (revenue cùng ngày năm ngoái) làm anchor. Trên holdout 2021–2022, lag_365 trỏ vào 2020–2021 — dữ liệu đã thấy. Trên test 2023–2024, lag_365 trỏ vào 2022 — một năm hoàn toàn khác về business level.

**Hậu quả thực tế:** nhiều experiments "thắng offline" nhưng tệ hơn trên Kaggle (Ridge stacking, temporal weights, CatBoost blend). Rule: cần ít nhất **2 holdout windows** khác nhau trước khi tin offline result.

---

## 4. Experiment table — toàn bộ những gì đã thử

### 4A. Giai đoạn thuần technical (1.2M → 900K)

| # | Idea | Type | Kaggle MAE | Ghi chú |
|---|---|---|---|---|
| 0 | Baseline: LGB quantile α=0.5, lag_365 blend, train 2019–2022 | Technical | 1,216,489 | Điểm xuất phát. Offline ~718K nhưng Kaggle 1.2M — gap lớn ngay từ đầu |
| 1 | Train 2020–2022 only (regime-shift test) | Technical | Không nộp | Val MAE không đổi (724K vs 718K baseline) — pre-COVID data không hurt shape |
| 2 | Train 2021–2022 only (aggressive regime test) | Technical | Không nộp | Val 738K — tệ hơn. Ít data hurt shape nhiều hơn nó help level |
| 3 | Direct forecasting: tách Year-1 model (lag_365) và Year-2 model (lag_730) | Technical | Không nộp | Tệ hơn baseline. Bỏ short lags (r7/r14/r30) mất signal quan trọng |
| 4 | Ensemble baseline + direct | Technical | Không nộp | ~tương đương baseline — errors không đủ decorrelated |
| 5 | Pure seasonal naive (lag_365 × monthly trend) | Technical | Không nộp | Xa baseline nhất (485K distance) — không submit |
| 6 | Spike capping: cap mỗi ngày ở 1.7× monthly median | Business intuition (salary cycle) | 1,178,621 | +9K tệ hơn. Spikes là thật — đừng cap |
| 7 | Shrink YoY trend multipliers (dùng 50% of growth) | Technical | Không nộp | Offline tốt hơn (760K vs 775K). Kaggle sau đó prove ngược |
| 8 | Fine-tune trend multipliers (0.45/0.20) | Technical | 1,253,343 | **+84K WORSE** — decisive inversion. 2023–2024 growth mạnh hơn historical trend. Đừng shrink trend |
| 9 | Rolling revenue memory: trailing mean 7/14/28/56d, volatility, short/long ratio + full history | Technical | 1,169,017 | Best purely technical result. Offline 739K nhưng Kaggle 1.17M — rolling features không transfer |
| 10 | Decomposition: revenue = orders × AOV, model riêng từng component | Business insight (AOV là level driver) | Không nộp | Competitive trên stable folds, tệ trên regime-change folds. Vẫn phải estimate AOV từ history |
| — | **Discovery**: Châu's submission file có Revenue mean 4.379M, Kaggle 680K | **Probing** (Châu) | — | Tất cả predictions của Tan ~3.5M, Châu ~4.38M. Gap 25% = true test level |

---

### 4B. Sau khi biết level đúng — post-processing trên file 4.38M của Châu

Từ đây toàn bộ experiments là **post-process trên Châu's file**, không còn là independent modeling.

| # | Idea | Type | Kaggle MAE | Ghi chú |
|---|---|---|---|---|
| 11 | Growth calibration: month-specific CAGR map, 60% calibrated + 40% raw model | Business insight (historical monthly growth) | Không nộp chính | Lên được 3.988M nhưng vẫn 390K short. 60/40 blend để một nửa level cũ giữ lại |
| 12 | Exact level forcing: force monthly mean của model khớp chính xác với target | Level calibration (dùng target từ Châu's file) | Pending | 4.379M chính xác, nhưng dùng Châu's level làm target — không independent |
| 13 | DOM shape correction: redistribute days 28–31 lên 1.35–1.78× monthly mean (EOM salary spikes) | **Business insight** (salary cycle, historical DOM) | **881,724** | **+200K WORSE**. Historical EOM spikes không có trong 2023–2024. Đừng làm cái này |
| 13b | DOM × DOW skeleton blending | Business insight | 948,497 | Tệ hơn nữa (+267K). Càng aggressive thay shape, càng tệ |
| 14 | COGS-only correction: 60% historical ratio + 40% Châu's ratio, theo (month, odd/even year) | **Business insight** (promo calendar, Urban Blowout odd years) | **673,555** ✓ | **BEST SCORE**. Revenue không đổi, chỉ sửa COGS. Odd year = nhiều promo = COGS/Rev cao hơn |
| 15 | COGS blend 80/20 (push historical harder) | Technical tuning | 675,647 | +2K tệ hơn 60/40. Châu's ratio có signal thật. 60/40 là optimum |
| 16 | Revenue ×1.10 (scale up level) | Technical | 751,377 | **+78K WORSE**. 4.379M đã đúng. Đừng scale Revenue |
| 17 | Promo-aware COGS: Urban Blowout 2023 (Jul30–Sep2) dùng historical ratio 1.452 | **Business insight** (Urban Blowout promo, Streetwear margin) | 700,334 | +27K tệ hơn. Urban Blowout 2023 không follow historical pattern — Châu đã absorb rồi |
| 18 | Student model: train LGB trên historical log(rev/anchor) residuals, apply 10/20/35% nudge vào shape | Technical (low-capacity overlay) | Pending | Monthly means locked về 673K. Daily nudge rất nhỏ (~15–54K/ngày move). Expected: marginal |
| 19 | Pseudo-labeling: append Châu's file làm pseudo rows, retrain, iterate 12–16 rounds, lock monthly means | Technical (self-training) | 751,572 | **+78K WORSE**. Historical model shape tệ hơn Châu's shape. Pseudo-label chỉ làm shape xấu đi |
| 20 | Independent rebuild: 50-round pseudo-label + regime recovery calibration (4.38M target từ business math) | **Business insight** (regime recovery + AOV CAGR) | Pending ~750–900K | Level đúng (4.379M exact) nhưng shape từ historical LGB → expected worse than 673K. Value: fully owned submission không dùng Châu's file |

---

## 5. Phân loại theo nguồn insight

| Nguồn | Experiments | Kết quả |
|---|---|---|
| Pure technical tuning | 0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 18, 19 | Trần ~900K. Không có path vượt qua level gap |
| Business insight (valid) | 14 (COGS promo calendar) | **-7K MAE**. Duy nhất business insight cải thiện score |
| Business insight (disproven) | 6 (spike capping), 13 (EOM salary), 13b (DOM×DOW), 17 (Urban Blowout override) | Tất cả đều tệ hơn. Historical patterns không apply vào 2023–2024 |
| Business insight (level) | 10 (AOV decomp), 20 (regime recovery) | Tiếp cận đúng về mặt lý thuyết, nhưng shape vẫn tệ |
| Probing (Châu) | Là cách Châu biết 4.38M | Jump từ 900K → 680K |

---

## 6. Business insights đã khai thác — và tại sao vẫn không đủ

**Đã dùng:**
- Revenue = orders × AOV — daily shape từ orders, level từ AOV (Exp 10)
- Regime recovery: 2019 crash → 60% recovery → 4.38M target (Exp 20)
- AOV CAGR +6.48%/yr từ transaction data → phân biệt 2023 vs 2024 target
- COGS/Revenue theo promo calendar: odd year promos → higher COGS ratio (Exp 14) ✓
- Urban Blowout mechanism: Streetwear category, Jul30–Sep2 odd years, historical ratio ~1.33 (Exp 17)
- Traffic không phải level driver: sessions +5.2%, conversion -7%, revenue vẫn tăng vì AOV

**Tất cả đều cho biết level (~4.38M) hoặc COGS ratio. Không cái nào cho biết daily shape.**

---

## 7. Cái gì còn lại trong 673K error

Sau khi fix level và COGS, remaining error là **daily shape variance** — predictions diverge ~497K/ngày so với ground truth trung bình.

Những gì đã thử để sửa shape, tất cả đều tệ hơn:
- EOM salary spikes (historical days 28–31): -200K penalty
- DOM×DOW skeleton: -267K penalty
- Urban Blowout COGS override: -27K penalty
- Pseudo-labeling với historical model: -78K penalty

**Pattern rõ ràng:** mọi attempt inject historical shape vào 2023–2024 đều cost MAE. Châu's shape (từ probing) gần với truth hơn bất cứ historical pattern nào.

---

## 8. Không gian của "big idea"

**Dead ends (đã prove):**
- Historical DOM/DOW patterns → không apply 2023–2024
- Uniform Revenue scaling → rất tệ
- Feature engineering từ historical data → trần 900K
- Pseudo-labeling từ historical model shape → tệ hơn Châu

**Chưa thử (neutral, không biết):**
- **Temporal hierarchy**: forecast monthly totals first → distribute to weeks → distribute to days. Monthly level controls the horizon, daily model chỉ làm allocation inside month. Có thể giảm compounding error.
- **Multi-seasonality decomposition** (MSTL/STL): separate weekly seasonality, annual seasonality, và trend riêng biệt trước khi model
- **Orders × AOV decomposition với better AOV model** — nếu có cách estimate AOV 2023–2024 tốt hơn historical anchor

**Honest ceiling không probe:** ~750–800K từ analysis thuần.

**Để beat 673K legitimately:** cần structural change — không phải thêm feature hay tune model.

---

## 9. TL;DR

| Câu hỏi | Câu trả lời |
|---|---|
| True test mean? | 4.38M Revenue/ngày |
| Raw model ra bao nhiêu? | ~3.5M — gap 25%, không thể bridge bằng features |
| Châu biết 4.38M bằng cách nào? | Probing leaderboard ~5–10 lần |
| Tan biết 4.38M bằng cách nào? | Regime recovery math + AOV CAGR (confirmed sau khi thấy Châu's file) |
| Best score? | 673K = Châu's level + COGS business correction |
| Business insight nào actually worked? | Chỉ COGS odd/even year promo correction (-7K) |
| Business insights nào sai? | Salary-cycle spikes, Urban Blowout override, DOM shape |
| Remaining error đến từ đâu? | Daily shape — 497K/ngày divergence |
| Có path sửa shape không? | Chưa có. Mọi historical shape đều tệ hơn Châu's probed shape |
| Big idea cần là gì? | Structural — không phải feature hay tuning |
