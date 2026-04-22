# VinUniversity Datathon 2026 — Round 1

Repo này chứa toàn bộ phần làm việc cho vòng 1, gồm:

- `Phần 1`: notebook giải MCQ
- `Phần 2`: notebook EDA và các hình dùng cho báo cáo
- `Phần 3`: notebook forecasting và file `submission.csv`
- báo cáo LaTeX/PDF để review hoặc nộp

## Review Nhanh

Nếu team chỉ có `10-15` phút để review, nên đi theo thứ tự này:

1. Đọc [round1_report.pdf](./round1_report.pdf) để nắm bức tranh tổng thể.
2. Mở [round1_report.tex](./round1_report.tex) nếu muốn góp ý trực tiếp vào narrative.
3. Xem [eda_round1.ipynb](./eda_round1.ipynb) cho `Phần 2`.
4. Xem [forecasting.ipynb](./forecasting.ipynb) cho `Phần 3`.
5. Kiểm tra [submission.csv](./submission.csv) là file nộp cuối cho forecasting.

## Cấu Trúc Repo

```text
.
├── dataset/                       # dữ liệu gốc do đề bài cung cấp
├── report_assets/                 # hình và metrics dùng cho report
├── eda_round1.ipynb               # notebook Part 2 để review
├── forecasting.ipynb              # notebook Part 3 để review
├── mcq_solve.ipynb                # notebook Part 1
├── submission.csv                 # file submit cuối cho forecasting
├── round1_report.tex              # source báo cáo
├── round1_report.pdf              # bản PDF để đọc nhanh
├── round1_report_scaffold.md      # khung nháp/supporting notes
├── round1_refs.bib                # tài liệu tham khảo
├── build_round1_eda_notebook.py   # script sinh notebook EDA
└── build_round1_report_assets.py  # script sinh hình cho report
```

## File Quan Trọng

- [round1_report.pdf](./round1_report.pdf): bản đọc nhanh, phù hợp để team review nội dung.
- [round1_report.tex](./round1_report.tex): bản nguồn chính của báo cáo.
- [eda_round1.ipynb](./eda_round1.ipynb): bản EDA đã Việt hoá để đồng bộ với report.
- [forecasting.ipynb](./forecasting.ipynb): pipeline forecast cuối cùng, đã có validation và save `submission.csv`.
- [submission.csv](./submission.csv): output nộp cuối.

## Tái Lập Kết Quả

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

- `round1_report_scaffold.md` là file khung nháp để giữ note và talking points; bản chính để review là `round1_report.tex` và `round1_report.pdf`.
- `report_assets/` được commit để mọi người mở report là thấy đúng figure hiện tại, không cần build lại ngay.
- `.gitignore` đã loại các file build LaTeX, cache notebook và thư mục cấu hình cục bộ.
- `dataset/submission.csv` được bỏ khỏi repo review để tránh nhầm với [submission.csv](./submission.csv) ở root.

## Deliverables Hiện Có

- MCQ notebook
- EDA notebook tiếng Việt
- Forecasting notebook
- File submit cuối
- Báo cáo PDF + LaTeX + BibTeX

