# VinUniversity Datathon 2026 - Round 1

Repo này chứa toàn bộ bài làm cho vòng 1, gồm:

- `Phần 1`: notebook giải câu hỏi trắc nghiệm
- `Phần 2`: notebook EDA và các hình dùng trong báo cáo
- `Phần 3`: notebook forecasting và file `submission.csv`
- báo cáo LaTeX/PDF để review hoặc nộp

## Nên Xem Gì Trước

Nếu chỉ có `10-15` phút để review, nên đi theo thứ tự này:

1. Đọc [round1_report.pdf](./round1_report.pdf) để nắm bức tranh tổng thể.
2. Mở [round1_report.tex](./round1_report.tex) nếu muốn góp ý trực tiếp vào nội dung báo cáo.
3. Xem [eda_round1.ipynb](./eda_round1.ipynb) cho phần phân tích kinh doanh.
4. Xem [forecasting.ipynb](./forecasting.ipynb) cho phần mô hình dự báo.
5. Kiểm tra [submission.csv](./submission.csv) là file nộp cuối cho bài forecasting.

## Cấu Trúc Repo

```text
.
├── dataset/                       # Dữ liệu gốc do đề bài cung cấp
├── report_assets/                 # Hình và metrics dùng cho báo cáo
├── eda_round1.ipynb               # Notebook Part 2 để review
├── forecasting.ipynb              # Notebook Part 3 để review
├── mcq_solve.ipynb                # Notebook Part 1
├── submission.csv                 # File submit cuối cho forecasting
├── round1_report.tex              # Source báo cáo chính
├── round1_report.pdf              # Bản PDF để đọc nhanh
├── round1_refs.bib                # Tài liệu tham khảo
├── build_round1_eda_notebook.py   # Script sinh notebook EDA
└── build_round1_report_assets.py  # Script sinh hình cho report
```

## Các File Quan Trọng

- [round1_report.pdf](./round1_report.pdf): bản review nhanh, phù hợp để góp ý tổng thể.
- [round1_report.tex](./round1_report.tex): nguồn chính của báo cáo.
- [eda_round1.ipynb](./eda_round1.ipynb): notebook EDA đã Việt hóa để đồng bộ với report.
- [forecasting.ipynb](./forecasting.ipynb): pipeline forecast cuối cùng, đã có validation và lưu `submission.csv`.
- [submission.csv](./submission.csv): output nộp cuối.

## Cách Tái Lập

### 1. Sinh lại notebook EDA

```powershell
python build_round1_eda_notebook.py
```

### 2. Sinh lại hình cho report

```powershell
$env:PYTHONIOENCODING='utf-8'
python build_round1_report_assets.py
```

### 3. Compile lại báo cáo PDF

```powershell
xelatex -interaction=nonstopmode -halt-on-error round1_report.tex
bibtex round1_report
xelatex -interaction=nonstopmode -halt-on-error round1_report.tex
xelatex -interaction=nonstopmode -halt-on-error round1_report.tex
```

## Ghi Chú Cho Reviewer

- `round1_report.tex` và `round1_report.pdf` là nguồn report chính.
- `report_assets/` được commit để mở report là thấy đúng figure hiện tại, không cần build lại ngay.
- `.gitignore` đã loại các file build LaTeX, cache notebook và thư mục cấu hình cục bộ.
- `dataset/submission.csv` đã được bỏ khỏi repo review để tránh nhầm với [submission.csv](./submission.csv) ở root.

## Deliverables Hiện Có

- notebook MCQ
- notebook EDA tiếng Việt
- notebook forecasting
- file submit cuối
- báo cáo PDF + LaTeX + BibTeX
