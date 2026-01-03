# Credit Card Fraud Detection (Deep Project)

## Problem
Detect fraudulent credit card transactions in an **extremely imbalanced** dataset where fraud is rare.
The key challenge is that **accuracy is misleading**: a model can achieve ~99.8% accuracy by predicting
“not fraud” for every transaction while detecting zero fraud cases.

## Goals
- Build a robust baseline fraud detection pipeline
- Evaluate models with **metrics suitable for class imbalance**
  (Precision, Recall, F1, PR-AUC)
- Perform **decision threshold tuning** based on business constraints
  (high recall vs. alert quality vs. cost minimization)
- Provide **error analysis** and a business-ready executive summary

## Why this is not “just a Kaggle notebook.”
This repository is structured like a real analytics / ML project:
- `src/` contains reusable Python modules (metrics, thresholding, evaluation)
- `notebooks/` contain narrative analysis (data understanding → modeling → business impact)
- `reports/` contains an executive summary for non-technical stakeholders

## Dataset
Classic Kaggle credit card fraud dataset:
- Features: `V1..V28`, `Time`, `Amount`
- Target: `Class` (1 = fraud, 0 = normal)

> **Note:** The dataset is **not committed to GitHub**.  
> Place `creditcard.csv` in `data/raw/`.

## Repository Structure
 - data/ raw and processed data (ignored by git)
 - notebooks/ narrative analysis notebooks
 - src/ reusable Python modules (project core)
 - reports/ executive summary and figures


## Method Overview
1. **Data quality & imbalance analysis**
   - Class ratio, missing values, feature distributions
   - Demonstration of the “accuracy trap” in fraud detection

2. **Baseline modeling**
   - Logistic Regression with class weighting
   - Evaluation using Precision, Recall, F1, and PR-AUC

3. **Threshold tuning**
   - Selection of decision thresholds based on:
     - minimum precision
     - minimum recall
     - **minimum expected business cost** (false negative vs. false positive trade-off)

4. **Error analysis**
   - Detailed inspection of false positives and false negatives
   - Translation of model errors into business impact
     (alert volume, missed fraud loss)

## Key Metrics (Focus)
- Precision, Recall, F1-score
- **PR-AUC (Average Precision)** — preferred under extreme class imbalance
- Confusion Matrix at the selected decision threshold

## How to Run
1. Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```
2. Place the dataset:
data/raw/creditcard.csv

3. Run notebooks in order:
- `notebooks/01_data_understanding.ipynb`
- `notebooks/02_modeling_thresholding.ipynb`
- `notebooks/03_error_analysis_business.ipynb`

## Deliverables:
- Reusable evaluation and thresholding toolkit in src/
- Three notebooks with a clear analytical narrative
- reports/executive_summary.md with decision-ready conclusions

## Interview Talking Points (90 seconds):
- Why accuracy fails in fraud detection
- Why PR-AUC and precision/recall are more informative
- How the decision threshold was selected using business cost trade-offs
- What patterns were observed in false positives / false negatives
- What would be done next (stronger models, calibration, monitoring)

🔙 [Back to Portfolio](https://github.com/BlladeRunner)
