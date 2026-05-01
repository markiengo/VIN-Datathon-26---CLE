# VinUni Datathon 2026 — Vòng 1

**Team:** claude and em &nbsp;|&nbsp; Lê Huy Tâm · Ngô Nhật Tân · Lê Minh Châu

---

## Source of Truth

- `references/đề_og_btc.pdf` là bản đề cập nhật mới nhất đang có trong repo.
- `dataset/*.csv` là nguồn chuẩn cho tên file, số dòng, cột, và khoảng thời gian dữ liệu thực tế phát hành.
- `report/assets/summary_metrics.json` là metric build lại từ CSV và nên được ưu tiên hơn các con số chép tay trong prose cũ.

Ghi chú: đề gốc vẫn dùng cách gọi `sales_train.csv` / `sales_test.csv` ở một số đoạn. Trong repo này, train series public là `dataset/sales.csv`, còn horizon test public được biểu diễn qua `dataset/sample_submission.csv` với 548 dòng từ `2023-01-01` đến `2024-07-01`.

---

## Nên Xem Gì Trước

| Thứ tự | File | Tại sao |
|--------|------|---------|
| 1 | [`deliverables/round1_report.pdf`](./deliverables/round1_report.pdf) | toàn bộ phân tích, kết quả, methodology trong 4 trang nội dung chính |
| 2 | [`deliverables/submission.csv`](./deliverables/submission.csv) | file nộp Kaggle, 548 dòng, giữ đúng thứ tự `sample_submission.csv` |
| 3 | [`deliverables/mcq_answers.md`](./deliverables/mcq_answers.md) | đáp án Part 1 theo đề cập nhật |
| 4 | [`part2/part2_analytics.ipynb`](./part2/part2_analytics.ipynb) | notebook chuẩn cho Part 2, là nguồn của các section analytics trong report |
| 5 | [`part3/notebooks/part3_forecasting.ipynb`](./part3/notebooks/part3_forecasting.ipynb) | pipeline Part 3: feature engineering, CV theo thời gian, SHAP, submission |
| 6 | [`MODEL_CARD.md`](./MODEL_CARD.md) | thẻ mô hình: iter-39 (submission) vs LightGBM two-stage trong Part 3, không pseudo-label |
| 7 | [`part3/docs/MODEL_RESEARCH_RETROSPECTIVE.md`](./part3/docs/MODEL_RESEARCH_RETROSPECTIVE.md) | toàn bộ thí nghiệm đã chạy, hạn chế model, hướng cải tiến cho team |
| 8 | [`part3/docs/LITERATURE_ML_INSIGHTS.md`](./part3/docs/LITERATURE_ML_INSIGHTS.md) | insight từ paper (ML) + references để chèn report |

---

## Sơ Đồ Quan Hệ Bảng

![ERD giản lược](references/schemas/ERD_simple.png)

---

## Cấu Trúc Repo

```text
.
├── dataset/                        # 14 file CSV phát hành trong repo, không chỉnh sửa
│   ├── sales.csv                   # train target — 3,833 dòng, 2012-07-04 → 2022-12-31
│   ├── sample_submission.csv       # template submission — 548 dòng, 2023-01-01 → 2024-07-01
│   └── *.csv                       # orders, products, customers, payments, ...
│
├── deliverables/                   # artifacts nộp bài (KHÔNG chỉnh sửa)
│   ├── round1_report.pdf
│   ├── submission.csv              # official submission — iter-39, MAE 680,344
│   └── mcq_answers.md
│
├── part1/                          # MCQ + Data Validation
│   ├── README.md
│   ├── part1_mcq.ipynb
│   └── part1_data_validation.ipynb
│
├── part2/                          # Analytics
│   ├── README.md
│   ├── part2_analytics.ipynb
│   └── build_part2_analytics_notebook.py
│
├── part3/                          # Forecasting
│   ├── README.md
│   ├── notebooks/
│   │   └── part3_forecasting.ipynb          # from-scratch LGB (~935K)
│   ├── scripts/
│   │   └── produce_final_submission.py      # Châu friend file + Tan COGS correction (673K)
│   ├── submissions/
│   │   ├── submission_iter50_blend05.csv    # Châu iter-50 (680,854)
│   │   ├── submission_friend_cogs_only.csv  # experimental best (673,555)
│   │   └── _archive/                        # gate JSON files iter 38–46
│   └── docs/
│       ├── MODEL_RESEARCH_RETROSPECTIVE.md
│       ├── LITERATURE_ML_INSIGHTS.md
│       └── _gap_report.md
│
├── report/
│   ├── round1_report.tex           # LaTeX source
│   ├── round1_refs.bib
│   ├── round1_report.pdf           # bản build local nếu đã compile
│   └── assets/                     # figures + summary_metrics.json do script generate
│
├── references/
│   ├── đề.md
│   ├── đề_og_btc.pdf
│   └── schemas/
│       ├── ddl_simple.sql
│       └── ERD_simple.png
│
└── scripts/
    └── build_report_assets.py      # generates report/assets/ figures
```

---

## Reproduce

**Cài môi trường**

```bash
pip install -r requirements.txt
```

**Build lại analytics notebook**

```bash
python part2/build_part2_analytics_notebook.py --overwrite
# -> part2/part2_analytics.ipynb
```

**Build lại figures và metric cho report**

```bash
python scripts/build_report_assets.py
# -> report/assets/
```

**Compile lại report PDF**

```bash
cd report
xelatex round1_report.tex
bibtex round1_report
xelatex round1_report.tex
xelatex round1_report.tex
```

---

## Ghi Chú

- Mọi notebook tự tìm repo root nên đọc `dataset/` đúng dù mở từ thư mục nào.
- `part2_analytics.ipynb` là nguồn chuẩn cho Part 2; report chỉ lấy một phần các section trong notebook này.    
- Toàn bộ stochastic operations dùng seed `42`.
