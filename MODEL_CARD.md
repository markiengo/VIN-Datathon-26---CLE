# Model card — Sales forecasting (Part 3)

Tài liệu này mô tả **hai tầng** mô hình liên quan bài thi:

1. **Submission đang nộp** (`deliverables/submission.csv`) — tương đương **iter-39** (hierarchical microdata ensemble, nhánh *old/new 50*). Đây là artifact **mạnh nhất theo public leaderboard** đã ghi trong nhật ký phiên thi.
2. **Pipeline LightGBM được mô tả đầy đủ trong code** — `notebooks/part3_forecasting.ipynb`, nhánh **two-stage + multi-quantile-α** (trong notebook được gắn nhãn tiến hoá **iter-10**). Đây là nơi **có thể tái chạy từng dòng** và trích tham số chính xác.

---

## 1. Training data & date range

### A. Iter-39 (submission canonical)

| Mục | Mô tả (theo artifact + mô tả định tính trong repo) |
|-----|-----------------------------------------------------|
| Chuỗi ngày gốc | Cùng universe với `dataset/sales.csv`: **2012-07-04 → 2022-12-31** (3.833 ngày), align với các bảng micro (`orders`, `order_items`, `products`, …). |
| Cách dùng | Thành phần **micro / hierarchical** tái tạo cấu trúc doanh thu–COGS từ giao dịch; **không** dùng target của tập test Kaggle làm đặc trưng. |
| Giới hạn repo | **Không** có notebook/script trong repo lặp lại **đúng** toàn bộ pipeline sinh ra file iter-39; chỉ còn CSV (và khi clone đầy đủ: bản sao trong `_archive/staged_submissions/` nếu bạn giữ file đó cục bộ). |

### B. Part 3 notebook (LightGBM two-stage)

| Mục | Giá trị trong code |
|-----|---------------------|
| `recent` | `train.copy()` — **toàn bộ** `dataset/sales.csv` đã sort theo `Date`. |
| Khoảng ngày | **2012-07-04 → 2022-12-31** (khớp README / cell setup). |

---

## 2. Features fed to the model

### A. Iter-39

- **Không** có danh sách cột cố định trong file `.py`/notebook đính kèm iter-39 trong repo.
- Định tính (từ mô tả phiên thi): mix **phân khúc / kích cỡ / màu**, kênh đơn, vùng, thanh toán, trạng thái đơn; ensemble **micro category cũ** vs **hierarchical mới**; tổng trọng số micro ~**10%**, ưu tiên **điều chỉnh nhẹ COGS** so với anchor trước đó.

### B. Part 3 — `build_features()` (tabular)

Gộp các nhóm sau (không dùng lag target 365/730 trên nhánh iter-4 hiện tại):

- **Calendar:** `month`, `day`, `dow`, `doy`, `week`, `quarter`, `is_weekend`, `is_month_start`, `is_month_end`, `year`, `years_from_2019`, `is_post_2018`, `is_pre_regime`
- **Cyclic:** `month_sin/cos`, `dow_sin/cos`, `doy_sin/cos`
- **Seasonal means:** `(month, day)` → `rev_seas_md`, `cogs_seas_md`; `(month, dow)` → `rev_seas_mdow`, `cogs_seas_mdow`
- **Tết:** `days_from_tet`, `is_tet_week`, `is_pre_tet`, `is_pre_tet2`, `is_post_tet`, `is_post_tet2`, `tet_x_weekend`
- **Sự kiện mua sắm / lễ:** `is_1111`, `is_1212`, `is_8_3`, `is_20_10`, `is_20_11`, `is_xmas`, `is_blackfriday`, `is_natlday`, `is_reunif`, `is_labor`, `is_payday`
- **Giai đoạn Revenue:** thêm **`p_cogs`** (OOF COGS trên train, hoặc dự báo COGS trên test từ giai đoạn 1)

**Lưu ý leakage:** khi CV, `train_for_seasonal` cho tập val **chỉ** lấy từ fold train (không dùng val để tính profile theo ngày).

---

## 3. LightGBM hyperparameters (Part 3)

Dùng chung object `PARAMS` (quantile regression):

```text
objective          = quantile
alpha              = 0.5   (bị ghi đè lần lượt bởi 0.45, 0.50, 0.55 trong khối multi-α)
learning_rate      = 0.05
num_leaves         = 63
min_child_samples  = 20
feature_fraction   = 0.85
bagging_fraction   = 0.85
bagging_freq       = 5
verbose            = -1
```

- **CV / early stopping:** `n_estimators=2000`, callback `early_stopping(50)` trên tập validation từng fold.
- **Khối train final multi-α:** `LGBMRegressor(..., n_estimators=max(iter_rev|iter_cogs, 100))` — fit **full** train cho đủ số cây (không early stopping trong khối đó).

**Iter-39:** tham số LightGBM (hoặc mô hình con) của nhánh micro **không** được lưu trong repo.

---

## 4. Pseudo-labeling loop — **không áp dụng**

Trong repo **không có** vòng lặp pseudo-labeling (train → dự báo test → gán nhãn giả → train lại nhiều round).

### Thay thế trong Part 3 (cần trình bày đúng tên)

| Khái niệm | Thực tế trong code |
|------------|-------------------|
| Vòng lặp | **Một lần OOF** cho COGS: `TimeSeriesSplit(n_splits=3, test_size=len(recent)//4)` để điền `p_cogs_train_oof` trên train. |
| Blend LGB vs seasonal | `BLEND = 0.25` → `0.75 × LGB + 0.25 ×` naive theo `(month, day)` (`rev_seas_md` / `cogs_seas_md`) sau khi đã gộp multi-α + multi-seed. |
| Trọng số “real vs pseudo” | **Chỉ có hàng thật** của `sales.csv`; không có pseudo-row. |
| “Monthly means locked?” | **Không khóa.** Trên train với `oof=True`, seasonal dùng **leave-one-year-out** theo `(month, day, year)`. Trên test (`oof=False`), profile `(month, day)` tính trên **toàn bộ** train làm `train_for_seasonal`. |
| Lag tại inference từng round | **Không** có lag target động theo round; không có lặp predict-test-retrain. `p_cogs` test: **một lần** fit trên full `X_train_full` rồi `predict(X_test)`. |

**Multi-quantile + multi-seed:** `ALPHAS = [0.45, 0.50, 0.55]`, `SEEDS = [42, 1337, 2024]` → **9** mô hình mỗi lần gọi `train_predict_multi_alpha_seed`; kết quả = **trung bình** 9 vector dự báo.

### Iter-39 (để tránh nhầm với pseudo-label)

- Là **ensemble / blend** giữa dự báo anchor (chuỗi cải tiến trước iter-39) và thành phần **hierarchical microdata**, có **cửa offline** (gate). File `_archive/iter39_hier_micro_gate.json` ghi các ứng viên trọng số **0.03 … 0.12** và **chọn 0.08** — đây là **trọng số blend**, không phải siêu tham số LightGBM.

---

## 5. How COGS was generated

### Part 3 (theo notebook)

1. `X_train_full = build_features(recent[['Date']], recent, oof=True)`, `X_test = build_features(sample_sub[['Date']], recent, oof=False)`.
2. **OOF COGS:** với mỗi fold `(tr_idx, va_idx)`, `train_predict_multi_alpha_seed(Xtr, y_cogs[tr], Xva, n_estimators)` ghi vào `p_cogs_train_oof[va_idx]`.
3. Ô còn 0: điền `X_train_full.loc[mask, 'cogs_seas_md']`.
4. **Test:** `p_cogs_test_raw = train_predict_multi_alpha_seed(X_train_full, y_cogs, X_test, n_estimators)`.
5. **Blend seasonal:** `p_lgb_cogs = (1 - BLEND) * p_cogs_test_raw + BLEND * X_test['cogs_seas_md']`.
6. **Hệ số LGB vs Prophet (nếu bật grid):** `p_cogs = LGB_W_COGS * p_lgb_cogs` (+ Prophet khi `LGB_W_COGS < 1`). Với cấu hình `RUN_GRID=False` và `LGB_W_COGS = 1.0` trong notebook, **bước cuối không cộng Prophet**.

### Iter-39

- COGS là kết quả **ensemble** sau khi nhánh hierarchical micro **chỉnh nhẹ** so với anchor; công thức đại số từng ngày **không** có trong repo.

---

## 6. Post-processing

### Part 3

- `Revenue` / `COGS`: `np.round(..., 2)`.
- Kiểm tra số dòng vi phạm `COGS >= Revenue` (in log); **không** có bước ép ràng buộc cứng trong đoạn code đã trích.

### Iter-39

- Tên artifact **`oldnew50`**: tỉ lệ **50% / 50%** giữa thành phần micro **category cũ** và **hierarchical mới** trong phần micro; kết hợp thêm lớp blend theo gate (ví dụ weight **0.08** trong JSON iter-39).
- Chi tiết làm tròn / clip bổ sung (nếu có) **không** có trong script công khai.

---

## Tóm tắt cho ban tổ chức

| Câu hỏi | Iter-39 (submission) | Part 3 notebook (LGB) |
|---------|-------------------------|------------------------|
| Code đầy đủ trong repo? | Không — chỉ artifact + mô tả | Có — `part3_forecasting.ipynb` |
| Pseudo-labeling? | Không mô tả | **Không có** |
| COGS | Ensemble micro + anchor | Two-stage + multi-α + seasonal blend |

**Khuyến nghị:** phần methodology trong PDF báo cáo nên **phân tách rõ** đoạn “LightGBM tabular two-stage (reproducible trong notebook)” và đoạn “ensemble iter-39 (artifact cuối)” để tránh hiểu nhầm một pipeline duy nhất.
