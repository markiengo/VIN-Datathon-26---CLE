# changelog

## mốc so sánh

- so với repo gốc: `https://github.com/chaulemuoichin/VIN-Datathon-26---CLE`
- mốc so sánh: branch `main`
- file này chỉ ghi phần delta của repo nộp bài, bỏ qua tooling local trong `.claude/`

## delta 2026-04-22

### 1. dọn lại structure

- tách lại repo để khớp đúng structure nộp bài:
  - `notebooks/`
  - `report/`
  - `deliverables/`
  - `references/`
  - `scripts/`

- rename file để đồng bộ tên gọi theo từng phần nộp:
  - `eda_round1.ipynb` -> `notebooks/part2_analytics.ipynb`
  - `forecasting.ipynb` -> `notebooks/part3_forecasting.ipynb`
  - `mcq_solve.ipynb` -> `notebooks/part1_mcq.ipynb`
  - `submission.csv` -> `deliverables/submission.csv`
  - `report_assets/` -> `report/assets/`
  - `build_round1_eda_notebook.py` -> `scripts/build_part2_analytics_notebook.py`
  - `build_round1_report_assets.py` -> `scripts/build_report_assets.py`

### 2. nâng eda thành full analytics

- gộp eda và business analysis để part 2 bớt rời rạc
- thêm retention analysis để đọc vòng đời khách hàng
- thêm cohort heatmaps để thấy cohort decay rõ hơn
- thêm channel-quality analysis để đánh giá chất lượng kênh
- thêm profit-depth analysis để nhìn lợi nhuận thật theo danh mục

### 3. nâng report lên bản nộp cuối

- thay metadata team để khớp thông tin nộp bài thật
- thêm figure còn thiếu để đủ ý cho phần retention và channel economics
- sửa layout report để figure, table, caption và reference gọn hơn
- sync pdf cuối để deliverables bám đúng bản compile mới nhất

### 4. gọn lại asset report

- gom asset về `report/assets/` để build flow gọn hơn
- giữ lại asset đang dùng để repo đỡ rác
- xoá asset chết để tránh lệch notebook với report
- cập nhật `summary_metrics.json` để khớp số liệu giữa notebook và report

### 5. sync lại docs

- viết lại `README.md` để khớp structure repo mới
- sync `references/đề.md` để bám sát file đề gốc
- thêm `requirements.txt` để setup môi trường rõ hơn

### 6. bổ sung artifact nộp bài

- chốt đủ file nộp để repo sẵn sàng submit:
  - `deliverables/round1_report.pdf`
  - `deliverables/submission.csv`
  - `deliverables/mcq_answers.md`

- thêm reference và schema để hỗ trợ trace logic và dữ liệu:
  - `references/đề.md`
  - `references/đề_og_btc.pdf`
  - `references/schemas/ddl_simple.sql`
  - `references/schemas/ERD_simple.png`

- thêm notebook phụ để hỗ trợ validation và baseline:
  - `notebooks/part1_data_validation.ipynb`
  - `notebooks/part3_baseline.ipynb`