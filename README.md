# Credit Card Fraud Detection (Deep Project, 2026)

## Problem
Detect fraudulent credit card transactions in an **extremely imbalanced** dataset (fraud is rare).
The key challenge is that **accuracy is misleading**: a model can achieve ~99.8% accuracy by predicting "not fraud" for every case.

## Goals
- Build a robust baseline fraud detection pipeline
- Evaluate with **appropriate metrics** for imbalance (Precision, Recall, F1, PR-AUC)
- Perform **threshold tuning** to match business constraints (e.g., high recall or cost minimization)
- Provide **error analysis** and a business-ready summary

## Why this is not "just a Kaggle notebook"
This repo is structured like a real analytics/ML project:
- `src/` contains reusable modules (metrics, thresholding, evaluation)
- `notebooks/` are narrative reports (data understanding → modeling → business/error analysis)
- `reports/` contains an executive summary for non-technical stakeholders

## Dataset
Expected schema (classic Kaggle credit card fraud dataset):
- features: `V1..V28`, `Time`, `Amount`
- target: `Class` (1 = fraud, 0 = normal)

> Note: data is **not** committed to git. Place the CSV in `data/raw/`.

## Repository structure

data/ raw + processed data (ignored by git)
notebooks/ narrative analysis notebooks
src/ reusable python modules (project core)
reports/ executive summary and figures


## Method overview
1) **Data quality & imbalance analysis**
   - class ratio, missing values, distributions
   - demonstrate why accuracy is not acceptable

2) **Baseline modeling**
   - Logistic Regression with class weights
   - evaluate with Precision/Recall/F1 and PR-AUC

3) **Threshold tuning**
   - choose threshold for:
     - minimum precision OR
     - minimum recall OR
     - **minimum expected cost** (FN vs FP trade-off)

4) **Error analysis**
   - investigate false positives / false negatives
   - translate outcomes into business impact (review load, potential loss)

## Key metrics (focus)
- Precision, Recall, F1
- **PR-AUC (Average Precision)** — preferred under extreme imbalance
- Confusion Matrix at selected threshold

## How to run
1) Create environment and install dependencies:
```bash
pip install -r requirements.txt

Put the dataset CSV into:
data/raw/creditcard.csv

Run notebooks in order:

notebooks/01_data_understanding.ipynb

notebooks/02_modeling_thresholding.ipynb

notebooks/03_error_analysis_business.ipynb

Deliverables

Reusable evaluation toolkit in src/

Notebooks with clear narrative

reports/executive_summary.md containing decision-ready conclusions

Talking points for interviews (90 seconds)

Why accuracy fails in fraud detection

Why PR-AUC + precision/recall are more meaningful

How you selected threshold based on business constraints/cost

What patterns you found in FP/FN and what you’d do next (monitoring, new features, calibration)

---

## 5) `reports/executive_summary.md` (шаблон)
```md
# Executive Summary — Credit Card Fraud Detection (Deep Project)

## Context
Fraud events are rare; the dataset is highly imbalanced. Therefore, accuracy is not meaningful.

## Objective
Maximize fraud detection effectiveness while controlling false alerts (manual review load).

## Results (fill after modeling)
| Metric | Value |
|---|---:|
| PR-AUC |  |
| Precision (selected threshold) |  |
| Recall (selected threshold) |  |
| F1 (selected threshold) |  |
| False Positives (alerts) |  |
| False Negatives (missed fraud) |  |

## Threshold decision
Chosen threshold: **T =**
Reason:
- (e.g.) Recall ≥ 0.80 with acceptable precision
- or cost-minimization under FN/FP costs

## Business interpretation
- Expected missed fraud loss (FN): ___
- Expected review cost (FP): ___
- Recommended action: (roll into monitoring / retrain / add features / collect labels)

## Next steps
- Calibrate probabilities (Platt/Isotonic)
- Try gradient boosting (XGBoost/LightGBM) and compare PR-AUC
- Monitoring plan: drift + alert rate + weekly PR-AUC on fresh labels


