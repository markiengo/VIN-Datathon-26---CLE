# VinUniversity Datathon 2026 - Round 1

Repo nay chua toan bo bai lam vong 1, gom:

- `Phan 1`: notebook giai MCQ
- `Phan 2`: notebook EDA va cac hinh dung cho bao cao
- `Phan 3`: notebook forecasting va file `submission.csv`
- bao cao LaTeX/PDF de review hoac nop

## Review Nhanh

Neu team chi co `10-15` phut de review, nen di theo thu tu nay:

1. Doc [round1_report.pdf](./round1_report.pdf) de nam buc tranh tong the.
2. Mo [round1_report.tex](./round1_report.tex) neu muon gop y truc tiep vao noi dung.
3. Xem [eda_round1.ipynb](./eda_round1.ipynb) cho `Phan 2`.
4. Xem [forecasting.ipynb](./forecasting.ipynb) cho `Phan 3`.
5. Kiem tra [submission.csv](./submission.csv) la file nop cuoi cho forecasting.

## Cau Truc Repo

```text
.
├── dataset/                       # du lieu goc do de bai cung cap
├── report_assets/                 # hinh va metrics dung cho report
├── eda_round1.ipynb               # notebook Part 2 de review
├── forecasting.ipynb              # notebook Part 3 de review
├── mcq_solve.ipynb                # notebook Part 1
├── submission.csv                 # file submit cuoi cho forecasting
├── round1_report.tex              # source bao cao chinh
├── round1_report.pdf              # ban PDF de doc nhanh
├── round1_refs.bib                # tai lieu tham khao
├── build_round1_eda_notebook.py   # script sinh notebook EDA
└── build_round1_report_assets.py  # script sinh hinh cho report
```

## File Quan Trong

- [round1_report.pdf](./round1_report.pdf): ban doc nhanh, phu hop de team review noi dung.
- [round1_report.tex](./round1_report.tex): ban nguon chinh cua bao cao.
- [eda_round1.ipynb](./eda_round1.ipynb): ban EDA da Viet hoa de dong bo voi report.
- [forecasting.ipynb](./forecasting.ipynb): pipeline forecast cuoi cung, da co validation va save `submission.csv`.
- [submission.csv](./submission.csv): output nop cuoi.

## Tai Lap Ket Qua

### 1. Sinh lai notebook EDA

```powershell
python build_round1_eda_notebook.py
```

### 2. Sinh lai hinh cho report

```powershell
$env:PYTHONIOENCODING='utf-8'
python build_round1_report_assets.py
```

### 3. Compile lai bao cao PDF

```powershell
xelatex -interaction=nonstopmode -halt-on-error round1_report.tex
bibtex round1_report
xelatex -interaction=nonstopmode -halt-on-error round1_report.tex
xelatex -interaction=nonstopmode -halt-on-error round1_report.tex
```

## Ghi Chu Cho Reviewer

- `round1_report.tex` va `round1_report.pdf` la nguon report chinh.
- `report_assets/` duoc commit de mo report la thay dung figure hien tai, khong can build lai ngay.
- `.gitignore` da loai cac file build LaTeX, cache notebook va thu muc cau hinh cuc bo.
- `dataset/submission.csv` duoc bo khoi repo review de tranh nham voi [submission.csv](./submission.csv) o root.

## Deliverables Hien Co

- MCQ notebook
- EDA notebook tieng Viet
- Forecasting notebook
- File submit cuoi
- Bao cao PDF + LaTeX + BibTeX
