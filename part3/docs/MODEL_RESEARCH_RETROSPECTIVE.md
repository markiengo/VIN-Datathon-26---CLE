# Retrospective — Thí nghiệm forecasting & hạn chế model (Vin Datathon 2026, Round 1)

Tài liệu dành cho **toàn team**: tổng hợp những gì đã thử, kết quả trên leaderboard (public MAE) hoặc offline gate, **hạn chế** của stack hiện tại, và **hướng suy nghĩ** khi cải tiến — tránh lặp lại cùng một vòng thử sai.

**Tham chiếu nhanh:** [`MODEL_CARD.md`](../MODEL_CARD.md) (định nghĩa hai tầng: submission **iter-39** vs pipeline **LightGBM Part 3**), [`notebooks/part3_forecasting.ipynb`](../notebooks/part3_forecasting.ipynb), [`_gap_report.md`](../_gap_report.md). Nhật ký chi tiết từng lần nộp nằm trong `_iter_log.md` (file cục bộ, có thể bị `.gitignore` khi push GitHub).

---

## 1. Bối cảnh & mục tiêu

| Mục | Chi tiết |
|-----|----------|
| **Target** | Dự báo `Revenue` và `COGS` cho **548 ngày**: 2023-01-01 → 2024-07-01 (template `dataset/sample_submission.csv`). |
| **Metric chính** | **MAE** (tổng hợp revenue + COGS theo quy địch BTC). |
| **Train quan sát được** | `dataset/sales.csv` — **3.833 ngày**, 2012-07-04 → 2022-12-31. |
| **Điểm tham chiếu** | Top public kickoff ~**658k** MAE (team khác). **Best team** sau phiên thi: **iter-39** — public MAE **~680,344** (`deliverables/submission.csv`). |
| **Ràng buộc đề** | Không dùng target test làm feature; không dùng dữ liệu ngoài phát hành; các bảng phụ **không** kéo dài qua 2023 trong public — exogenous cho horizon test hạn chế. |

---

## 2. Hai “tầng” model cần phân biệt rõ

| Tầng | Vai trò | Reproducibility trong repo |
|------|---------|----------------------------|
| **A — Tabular / LGB / XGB** | Backbone ngày: calendar, seasonal profile, Tết, sale events; two-stage **COGS → Revenue**; multi-α; blend naive seasonal; sau này **XGB + probe mean + shrink**. | **Có** — chủ yếu `part3_forecasting.ipynb` + các script thử trong `_archive/iter_scripts/`. |
| **B — Micro / generator / residual** | Tận dụng `orders`, `order_items`, `products`, … để tái cấu trúc doanh thu–COGS; blend nhẹ vào anchor; iter-39 = **hierarchical micro old/new 50%** + ~**10%** trọng số micro, nhấn **COGS**. | **Một phần** — artifact CSV winner; **full notebook sinh iter-39 có thể không còn khớp 100%** trong repo (xem `MODEL_CARD.md`). |

Mọi thảo luận cải tiến nên nói rõ đang sửa **tầng A**, **tầng B**, hay **cách nối hai tầng**.

---

## 3. Dòng thời gian thí nghiệm (tóm tắt)

### 3.1 Giai đoạn đầu — LGB + Prophet, chỉnh feature & blend

| Iter (public) | MAE ~ | Ý chính | Kết luận |
|---------------|-------|---------|----------|
| 1 | 1.23M | Baseline 2019–2022 + L1 + Prophet | Điểm xuất phát |
| 2 | 976k | Full history + quantile α=0.5 + regime/year + `num_leaves=63` | **Win lớn** — full history + quantile |
| 3 | 1.01M | Lag 548/730 + seasonal gần | **Fail** — lag-548 NaN trên test 2024 |
| 4 | 978k | + log1p | Trung tính |
| 5 | 1.00M | Sample weight 3× năm ≥2019 | **Fail** — mất tín hiệu pre-regime |
| 6 | 975k | 5-seed average | Win nhỏ |
| **7** | **949k** | Bỏ Prophet (`LGB_W=1`) | Prophet làm mượt quá / lệch horizon |
| 8 | 1.18M | `BLEND=0` (bỏ naive seasonal) | **Fail lớn** — blend seasonal là lưới an toàn |

**Bài học:** quantile > L1 cho use case; **Prophet tắt** tốt hơn ensemble với LGB trên board; **blend 0.25** với seasonal `(month,day)` cực kỳ quan trọng; lag target trùng horizon test là **bẫy**.

### 3.2 Sau iter-10 — chỉnh LGB, stacking, weight thời gian (nhiều regression public)

- **iter-12:** tune LGB “cứng” hơn → public **~956k** (tệ hơn iter-10 ~935k) — CV fold đẹp nhưng **không transfer**.
- **iter-19:** Ridge residual stacker → public **~1.19M** — **false positive offline nghiêm trọng**.
- **iter-20:** temporal weighting mượt → public **~956k** — offline ~511k avg nhưng board không khớp.
- **iter-14 / 18 (không nộp):** CatBoost blend, holiday-aware alpha — thắng `final_548` nhưng thua `previous_548` → **gate robustness** loại.

**Bài học:** screen offline (một cửa sổ 548 ngày) **dễ lạc quan**; cần ít nhất **hai cửa sổ** (`previous_548` + `final_548`) trước khi tin “win offline”.

### 3.3 Giai đoạn “probe + XGB + shrink” (nhảy vọc ~935k → ~680k)

| Iter | Public MAE ~ | Ý chính |
|------|--------------|---------|
| 10 | **935k** | Two-stage LGB + multi-α — neo chất lượng tabular |
| 23 | **925k** | XGB two-stage độc lập |
| 28 | **695k** | XGB **scale** theo mean từ probe (chẩn đoán public mean) |
| 30–30g | **691k → 683k** | **Shrink** về mean (tối ưu ~91.5%) |
| probe3 | (probe) | Khôi phục **public target means** Rev / COGS |
| 32 | **681k** | Shift **iter30g** theo mean probe |
| 37 | **681k** | Event residual **10%** vào iter32 — tín hiệu rất nhỏ |
| 38 | **680.4k** | Micro simulator **10%** blend vào iter37 |
| **39** | **680.34k** | **Hierarchical micro old/new 50%** — **best public** |
| 40–45b | 681k–682k | Residual / generator (product, customer, payments, ops, promo+inventory…) | Không vượt 39; **45b** gần nhất (~**+353** MAE so với 39) khi blend **3%** vào đúng anchor 39 |

**Bài học:** (1) **Calibration mean** (probe) là bước nhảy lớn; (2) **shrink** quanh mean public giúp ổn định split Rev/COGS; (3) micro/generator có tín hiệu nhưng **blend mạnh** dễ overcorrect (**38b** 20% thua); (4) residual nhỏ trên anchor tốt vẫn có thể **false positive offline** (40–42).

### 3.4 Sau iter-39 (đã archive script)

| Nhãn | Ý tưởng | Ghi chú |
|------|---------|---------|
| **49** | LGB **Tweedie** + reviews + web traffic, blend 10–30% vào 39 | Tín hiệu zero-inflated / chất lượng web; cần gate 2 cửa sổ trước nộp |
| **50** | **Lunar** calendar + Tweedie, blend 5–20% vào 39 | Phù hợp bối cảnh VN; verify backtest forward |
| **51** | DOW×month / mm-dd / interaction calibration + blend 25–50% | Baseline rẻ; không thay shape XGB đầy đủ lịch sử |
| **52** | Shrink 85/90/95% | Gia đình iter-30 đã có optimum ~91.5%; sweep thô ít giá trị |

---

## 4. Hạn chế của model hiện tại (để teammate không kỳ vọng sai)

### 4.1 Khoảng cách validation vs public (core issue)

- Backtest **iter-10**: `final_548` ~**512k**, `previous_548` ~**775k**, trong khi **public ~935k** — gap lớn (`_gap_report.md` so sánh offline ~511k vs public 935k).
- **Hệ quả:** mọi ý tưởng chỉ thắng một cửa sổ offline dễ **ảo**; đã xảy ra với CatBoost, alpha có điều kiện ngày lễ, Ridge stack, temporal weights.

**Hàm ý cho cải tiến:** ưu tiên **làm validation bám board** (nhiều cut-time, stress COVID, hoặc proxy distribution shift) hơn là thêm model phức tạp.

### 4.2 Dữ liệu ngoài `sales.csv` cho test horizon

- Các bảng phụ **kết thúc 2022-12-31** — không có exogenous “đúng ngày” cho 2023–2024 trong public repo.
- Model tabular phải dựa vào **calendar + seasonal profile** + micro **tái cấu trúc từ lịch sử** — không có “ảnh hưởng thực tế 2023” từ inventory/promo sau 2022.

### 4.3 Regime change 2018→2019

- Doanh thu sụt ~40%; full history vẫn tốt hơn cắt sớm **2019+** cho **shape** (iter-35 thua lớn khi chỉ post-2019 XGB).

### 4.4 Lag target trên horizon dài

- Lag 548 trỏ vào **vùng test** → NaN / leakage conceptual.
- Lag 365 “lý tưởng” nhưng **không có** năm 2023–2024 trong train — đã chuyển sang seasonal mean `(month,day)` + LOO trên train.

### 4.5 iter-39 — mức trần gần như “plateau”

- Cải tiện từ 38→39 chỉ **~22** MAE public; nhiều generator/residual sau đó **không** vượt được.
- Pure hierarchical micro **yếu hơn** category cũ — cần **ensemble 50/50** và trọng số micro nhỏ.

### 4.6 Không có pseudo-labeling / vòng self-training

- Pipeline chính = **OOF two-stage**, multi-α, blend seasonal — không phải iterative pseudo-label (xem `MODEL_CARD.md`). Đừng kỳ vọng “thêm round pseudo-label” đã được thử nếu chưa code.

### 4.7 Khả năng tái lập iter-39

- CSV winner có; **toàn bộ cell** sinh ra đúng blend cuối có thể thiếu hoặc lệch trong notebook hiện tại — rủi ro cho teammate port code.

---

## 5. Ma trận “đã thử → không nên lặp vô điều kiện”

| Hướng | Kết quả đại thể | Ghi chú |
|-------|-----------------|---------|
| Bỏ blend seasonal (`BLEND=0`) | Thảm họa public | Giữ blend ~0.25 (hoặc tune trong gate 2 cửa sổ) |
| Lag 548/730 naïve trên test | NaN / noise | Đã bỏ; seasonal profile thay lag YoY |
| Sample weight chỉ nặng 2019+ | Tệ | Mất pre-regime shape |
| Prophet cùng LGB (early) | Tệ hơn LGB-only | Có thể xem lại nếu validation mới chứng minh ngược |
| LGB tune quá mạnh (LR↓, leaves↑) | Public tệ hơn | Overfit fold |
| Ridge / temporal weight chỉ thắng offline | Public kém | Ưu tiên align validation |
| CatBoost / alpha lễ chỉ thắng `final_548` | Bị gate loại | Luôn check `previous_548` |
| Post-2019-only XGB + calib | ~832k | **Cần full history** cho shape ngày |
| Profile tháng từ train blend mạnh vào 32 | ~685k | Shape hiện tại tốt hơn profile lịch sử |
| Total+Ratio manifold blend >15% | ~682k | Corr ~0.997 nhưng blend vẫn hại |
| Micro blend 20%+ | Public regress | Giữ ~10% và bracket nhỏ |
| Residual product / customer / payments | Public kém | Offline false positive |
| Generator payments / ops / promo | Cải thiện dần nhưng **chưa** beat 39 | **45b** gần nhất — fusion khó, cần ý mới |

---

## 6. Hướng cải tiến gợi ý (ưu tiên logic)

### 6.1 Ưu tiên cao — “đo đúng trước khi sửa model”

1. **Hiệu chỉnh validation** để MAE offline cùng cỡ public (hoặc biết hệ số lệch ổn định).
2. **Stress test** trên cửa sổ có COVID và cửa sổ gần 2022→2023.
3. **Audit feature** theo ngày: mọi merge phải không “nhìn thấy” tương lai (đã có iter-15 cơ bản — lặp khi thêm cột mới).

### 6.2 Cải tiến backbone (tầng A)

- **Quantile α** grid nhỏ (0.47–0.52) **chỉ khi** cả hai backtest 548 ngày không tệ hơn anchor.
- **Margin / ratio** làm feature giai đoạn 2 thay vì chỉ `p_cogs` (chưa khai thác sâu).
- **Rolling 30/60/90d seasonal** (iter-11 track) — thử lại **sau** validation ổn định.
- **CatBoost** như thành viên ensemble **rất nhẹ** + gate 2 cửa sổ nghiêm.

### 6.3 Cải tiến micro / generator (tầng B)

- Tiếp tục **cấu trúc giao dịch** (promo lifecycle + inventory đã cho tín hiệu) nhưng tập trung **fusion không over-rotate** (bài học 45b: blend nhỏ vào đúng anchor 39).
- **Tweedie / lunar / web & reviews** (iter 49–50): giá trị khi có target validation tốt; tránh blend mù vào public quota.

### 6.4 Tránh lãng phí quota / thời gian

- Sweep shrink thô 85/90/95 không giả thuyết mới.
- Residual “mạnh offline” không qua `previous_548`.
- Mean shift theo **năm** đã chứng minh tệ hơn shift **global** (iter-33).

---

## 7. Checklist trước khi teammate mở PR / nộp Kaggle mới

- [ ] Ghi rõ **anchor** (iter-32? 37? 39?) và **delta** MAE trên **cả hai** cửa sổ 548 ngày (nếu có pipeline gate).
- [ ] Không dùng target test làm feature; kiểm tra merge ngày.
- [ ] So sánh **distribution** Rev/COGS submission vs train gần đoạn 2022 (và probe means nếu vẫn dùng).
- [ ] Cập nhật `_iter_log.md` cục bộ (nếu team giữ log) — không rely vào memory.

---

## 8. Tài liệu liên quan trong repo

| File | Nội dung |
|------|----------|
| [`MODEL_CARD.md`](../MODEL_CARD.md) | Hai tầng model, tham số LGB reproducible, “không pseudo-label” |
| [`LITERATURE_ML_INSIGHTS.md`](./LITERATURE_ML_INSIGHTS.md) | Insight literature **hướng ML** + **danh mục tên bài** để trích trong report |
| [`notebooks/part3_forecasting.ipynb`](../notebooks/part3_forecasting.ipynb) | EDA, `build_features`, CV, two-stage multi-α, submission path tabular |
| [`_gap_report.md`](../_gap_report.md) | Phân phối theo năm + backtest A/B vs gap public |
| `_iter_log.md` | Bảng đầy đủ iter + learnings dài (local) |
| `_archive/iter*_*.json` | Gate offline (trọng số blend đã thử) |
| `_archive/iter_scripts/` | Script thử iter 49–51 (Tweedie, lunar, calibration) |

---

*Tài liệu tổng hợp từ nhật ký phiên thi và artifact trong repo; số MAE làm tròn theo bảng log — luôn đối chiếu leaderboard chính thức khi trích dẫn ngoài team.*
