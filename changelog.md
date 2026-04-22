# Changelog

## Mốc so sánh

- So với repo gốc: `https://github.com/chaulemuoichin/VIN-Datathon-26---CLE`
- Mốc so sánh: branch `main`
- File này chỉ ghi delta của repo nộp bài. Bỏ qua đống tooling local trong `.claude/`.

## Delta 2026-04-22

### 1. Dọn lại structure

- Tách repo theo đúng đồ nộp:
- `notebooks/`
- `report/`
- `deliverables/`
- `references/`
- `scripts/`

- Rename mấy file chính:
- `eda_round1.ipynb` -> `notebooks/part2_analytics.ipynb`
- `forecasting.ipynb` -> `notebooks/part3_forecasting.ipynb`
- `mcq_solve.ipynb` -> `notebooks/part1_mcq.ipynb`
- `submission.csv` -> `deliverables/submission.csv`
- `report_assets/` -> `report/assets/`
- `build_round1_eda_notebook.py` -> `scripts/build_part2_analytics_notebook.py`
- `build_round1_report_assets.py` -> `scripts/build_report_assets.py`

### 2. EDA -> Full analytics. thêm customer retention

- Nguồn chuẩn cho Part 2 là `notebooks/part2_analytics.ipynb`.
- Script build chuẩn là `scripts/build_part2_analytics_notebook.py`.
- Notebook này cover đủ các block đang dùng trong report, gồm cả retention/cohort.

### 3. Report lên bản nộp cuối

- Thay metadata thật của team:
- team `claude and em`
- email `tamhuyle.lht@gmail.com`

- Thêm figure còn thiếu:
- `cohort_retention.png`
- `margin_channel.png`

- Sửa bố cục report để figure/table đứng đúng chỗ, caption/ref sạch hơn.
- Sync `deliverables/round1_report.pdf` từ bản compile trong `report/`.

### 4. Asset report gọn lại

- `scripts/build_report_assets.py` chỉ build vào `report/assets/`.
- Giữ lại asset đang dùng.
- Bỏ asset chết:
- `action_backlog.png`
- `traffic_vs_orders.png`

- `summary_metrics.json` có thêm metric để khớp notebook + report.

### 5. Docs sync lại cho đúng repo hiện tại

- Viết lại `README.md` theo structure mới.
- Sync `references/đề.md` theo `references/đề_og_btc.pdf`.
- Thêm `requirements.txt`.

### 6. Bổ sung artifact nộp bài

- Có đủ:
- `deliverables/round1_report.pdf`
- `deliverables/submission.csv`
- `deliverables/mcq_answers.md`

- Có thêm reference/schema:
- `references/đề.md`
- `references/đề_og_btc.pdf`
- `references/schemas/ddl_simple.sql`
- `references/schemas/ERD_simple.png`

- Có thêm notebook phụ:
- `notebooks/part1_data_validation.ipynb`
- `notebooks/part3_baseline.ipynb`
