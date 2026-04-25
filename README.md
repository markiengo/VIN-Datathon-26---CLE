# Handoff Folder — Datathon Forecasting

Bro, đọc cái này trước khi đụng vào folder bên trong. Ngắn thôi.

---

## Bài toán nhanh

- **Cuộc thi:** Vin Datathon 2026 Round 1 — Kaggle private
- **Dữ liệu:** ecommerce store Việt Nam, sales từ 2012-07-04 đến 2022-12-31 (~3800 ngày)
- **Cần dự báo:** Daily Revenue và COGS cho 548 ngày (2023-01-01 đến 2024-07-01)
- **Metric:** MAE tổng hợp = Revenue MAE + COGS MAE. Càng thấp càng tốt.
- **Best score hiện tại:** **673,555** (Tan's submission, lai từ Châu's file)
- **Pure model ceiling không probing:** ~900K (cả Tan và Châu đều plateau chỗ này)

---

## Cấu trúc folder

```
handoff/
├── README.md                  ← bro đang đọc
├── tan/                       ← Tan's work
│   ├── brief.md               ← context tổng quát Tan's side
│   ├── experiments.md         ← log chi tiết từng experiment Tan đã thử
│   └── submissions/           ← 9 file CSV Tan đã submit hoặc generate
└── chau/                      ← Châu's work
    ├── brief.md               ← context tổng quát Châu's side
    ├── experiments.md         ← log chi tiết 52 iterations Châu đã thử
    └── submissions/           ← chỉ 1 file (Châu chỉ giữ file best)
```

---

## Đọc theo thứ tự nào?

**Nếu m muốn nắm context nhanh nhất (15 phút):**
1. Phần "Cốt lõi" của README này (ngay dưới)
2. `tan/brief.md` đoạn TL;DR ở cuối
3. `chau/brief.md` đoạn TL;DR ở cuối

**Nếu m muốn hiểu sâu (1-2 tiếng):**
1. README này (full)
2. `tan/brief.md` (full)
3. `chau/brief.md` (full)
4. `tan/experiments.md` — focus vào Exp 14 (best score), Exp 13/16/17/19 (dead ends quan trọng)
5. `chau/experiments.md` — focus vào iter 28 (probing), iter 30g (shrink), iter 39 (best)

**Nếu m muốn đề xuất "big idea":**
1. Tất cả ở trên + section "Không gian big idea" trong cả 2 brief
2. Đặc biệt: `tan/experiments.md` Phase B (Exp 11-20 đã exhausted nhiều direction)

---

## Cốt lõi — đọc dùm tao

### Vấn đề chính: **the level problem**

Model train trên 2012-2022 sẽ tự nhiên dự đoán level ~3.5M VND/ngày. Test set thực tế là **4.38M VND/ngày** — cao hơn 25%.

Đây là **information gap**, không phải feature problem. Con số 4.38M là một fact về 2023-2024 không tồn tại trong training data. Mọi pure modeling đều đụng trần ~900K vì cùng lý do này.

### 2 con đường đến level đúng:

| Path | Ai dùng | Cách biết level 4.38M |
|---|---|---|
| **Probing** | Châu | Submit nhiều file lên Kaggle với constant + offset, đọc MAE response để reverse-engineer test mean. Tốn 5-10 submissions. |
| **Business analytics** | Tan | Regime recovery (2019 crash → 60% recovery → 4.38M) + AOV CAGR (+6.48%/yr). Không tốn submission, từ math thuần. |

### Best score 673K được tạo thế nào:

```
1. Châu probing được level → predictions calibrated → 935K → 695K
2. Châu shrink toward mean (W=0.915) → 695K → 683K
3. Châu add hierarchical micro 8% blend → 683K → 680.34K (iter-39)
4. Tan post-process Châu's iter-39 với COGS correction (60% historical / 40% Châu, 
   theo (month, odd_year)) → 680.85K → 673,555 ✓ BEST
```

**Iter-39 = 0.92 × Tier_A_calibrated + 0.08 × Tier_B_micro**
- Tier A = LGB/XGB two-stage + multi-α [0.45, 0.50, 0.55] × 3 seeds + seasonal blend 0.25 + scale to probe means + shrink 91.5%
- Tier B = 50/50 of (category-based micro) + (hierarchical order→item→product→category)

### Cái 7K cuối từ Tan là duy nhất business insight thực sự work:

Mọi business insight khác Tan thử đều TỆ HƠN:
- Salary-cycle EOM spikes (-200K)
- DOM×DOW skeleton (-267K)
- Urban Blowout COGS override (-27K)
- Revenue scaling up ×1.10 (-78K)
- Pseudo-labeling từ historical model (-78K)

**Pattern:** Mọi historical pattern (2012-2022) đều fail khi inject vào 2023-2024 shape. Châu's probed shape gần truth hơn bất cứ historical reconstruction nào.

---

## Hard rules — đừng vi phạm trừ khi có lý do mạnh

1. **Đừng scale Revenue.** 4.379M/ngày đã đúng. ×1.10 cost +78K (Tan Exp 16).
2. **Đừng redistribute days within a month.** Historical DOM patterns không apply 2023-2024 (Tan Exp 13: -200K).
3. **Đừng apply promo COGS overrides.** Urban Blowout 2023 không follow historical pattern (Tan Exp 17: -27K).
4. **COGS blend phải là 60% historical / 40% Châu.** 80/20 worse, 100/0 worse (Tan Exp 15: -2K).
5. **Đừng pseudo-label với historical LightGBM.** Daily shape tệ hơn Châu's (Tan Exp 19: -78K).
6. **Đừng tin offline metric một mình.** Cần 2 holdout windows (final_548 + previous_548). Single-window false positives cost rất nhiều (Châu iter 14/18/19/20).
7. **Đừng remove seasonal blend.** BLEND=0 thảm họa (Châu iter 8: +231K). Giữ 0.25.
8. **Đừng dùng lag pointing into test horizon.** lag_548 trỏ vào unknown data → NaN/leakage (Châu iter 3: +34K).

---

## Submitted files reference

### Tan's submissions (`tan/submissions/`)

| File | Kaggle MAE | Note |
|---|---|---|
| current.csv | 1,171,475 | Tan's prior baseline (pre-collaboration) |
| submission_iter50_blend05.csv | 680,854 | Châu's friend file — **foundation for everything** |
| **submission_friend_cogs_only.csv** | **673,555** | **CURRENT BEST. 60/40 COGS on Châu Revenue.** |
| submission_plan_round1_exact_level.csv | Pending | Exp 12 — exact level forcing |
| submission_student_friend_s10.csv | Pending | Exp 18 — 10% nudge overlay |
| submission_student_friend_s20.csv | Pending | Exp 18 — 20% nudge overlay |
| submission_student_friend_s35.csv | Pending | Exp 18 — 35% nudge overlay |
| submission_pseudo50_calibrated.csv | Pending | Exp 20 — fully independent rebuild với regime recovery |
| submission_pseudo50_scratch.csv | Pending | Earlier scratch version of Exp 20 (uncalibrated) |

### Châu's submissions (`chau/submissions/`)

| File | Kaggle MAE | Note |
|---|---|---|
| **submission_iter39.csv** | **680,344** | **Best Châu — chỉ file duy nhất Châu giữ lại.** Là Châu's iter-39 (`deliverables/submission.csv` trong repo gốc) |

Châu chỉ giữ file best. Iter-1 → iter-46 đã submit lên Kaggle nhưng không lưu CSV local. Iter scripts 49/50/51 chỉ có Python code, chưa submit.

---

## Sample submission format

CSV format Kaggle yêu cầu:
```
Date,Revenue,COGS
2023-01-01,4325000.00,3840000.00
2023-01-02,4012000.00,3712000.00
...
2024-07-01,5121000.00,4523000.00
```

- 548 rows exactly
- Date format: YYYY-MM-DD
- Revenue và COGS round to 2 decimal places
- Floor: 50,000.0 cho cả Revenue và COGS
- Không có constraint COGS ≤ Revenue (historical data có nhiều ngày COGS > Revenue do promo)

---

## Validation gap warning

Cả Tan và Châu đều gặp cùng một vấn đề:

| Method | Offline MAE | Public MAE | Gap |
|---|---|---|---|
| Tan best modeling (Exp 9) | 739K | 1,169K | 1.6× |
| Châu iter-10 | 512K | 935K | 1.83× |
| Châu iter-19 (Ridge) | offline tốt | 1,190K | catastrophic |
| Châu iter-20 (temporal) | 511K | 956K | 1.87× |

**Internal validation luôn quá lạc quan 1.7-1.9×.** Đừng trust một offline win nếu chưa qua 2 holdout windows.

---

## "Big idea" space — chưa có ai prove out

Để beat 673K legitimately (không probe thêm), cần một structural change. Những direction chưa được test:

1. **Temporal hierarchy reconciliation** — forecast monthly first, distribute to weeks, then days. Tách level estimation khỏi shape estimation.
2. **Multi-seasonality decomposition** (MSTL/STL) — separate weekly + annual seasonality + trend trước khi model.
3. **Tweedie loss với zero-inflated** — Châu iter-49, untested on Kaggle.
4. **Lunar calendar features** — Châu iter-50, untested on Kaggle. Lunar month 4 = 1.54× revenue, lunar 10 = 0.57×.
5. **Orders × AOV decomposition với better AOV model** — Tan Exp 10 đã tried một version, có thể refine.

**Honest ceiling không probe: ~750-800K.** Châu's 680K cộng = 935K modeling - 255K từ probing.

**Để beat 673K legitimately:** cần một insight về 2023-2024 daily shape mà không phụ thuộc historical patterns hoặc leaderboard feedback. Đây là chỗ "big idea" cần xuất hiện.

Good luck. The level problem is the real enemy. Everything else is decoration.
