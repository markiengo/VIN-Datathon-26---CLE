# Part 3 — Forecasting

Daily Revenue and COGS forecasting for 548 days (2023-01-01 → 2024-07-01).

## Reproducible paths

### Path A — From-scratch baseline (~935K MAE)

`notebooks/part3_forecasting.ipynb` trains a LightGBM two-stage model (COGS first, then Revenue) from raw `dataset/sales.csv`. No external CSV required. Produces a ~935K submission.

```bash
jupyter notebook part3/notebooks/part3_forecasting.ipynb
```

### Path B — Experimental best (673,555 MAE)

`scripts/produce_final_submission.py` applies Tan's COGS correction on top of Châu's iter-50 pseudo-labeled friend file. Requires `part3/submissions/submission_iter50_blend05.csv` (checked in).

```bash
python part3/scripts/produce_final_submission.py
# Writes: part3/submissions/submission_friend_cogs_only.csv
```

**Note:** Path B is the team's experimental best but NOT the official submission. The official submission (iter-39, 680,344 MAE) is in `deliverables/submission.csv`. Its Tier B hierarchical micro code was not preserved in this repository.

## Folder structure

```
part3/
├── notebooks/
│   └── part3_forecasting.ipynb     — from-scratch LGB model (~935K)
├── scripts/
│   └── produce_final_submission.py — Châu's friend file + Tan's COGS correction (673K)
├── submissions/
│   ├── submission_iter50_blend05.csv      — Châu's iter-50 pseudo-labeled file (680,854)
│   ├── submission_friend_cogs_only.csv    — output of produce_final_submission.py (673,555)
│   └── _archive/                          — gate JSON files from iter 38–46
└── docs/
    ├── MODEL_RESEARCH_RETROSPECTIVE.md
    ├── LITERATURE_ML_INSIGHTS.md
    └── _gap_report.md
```

## Score history

| File | Kaggle MAE | Method |
|---|---|---|
| `deliverables/submission.csv` | **680,344** | Châu iter-39: LGB/XGB two-stage + probing + shrink + 8% hierarchical micro |
| `submissions/submission_friend_cogs_only.csv` | **673,555** | iter-50 friend file + Tan's 60/40 COGS correction |
| `submissions/submission_iter50_blend05.csv` | 680,854 | Châu iter-50: 50-round pseudo-labeling, 0.5 blend |
| *(notebooks/part3_forecasting.ipynb output)* | ~935,000 | From-scratch LGB two-stage, no probing |
