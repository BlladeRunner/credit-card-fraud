from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)

def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """PR-AUC (Average Precision) is often more informative than ROC-AUC for extreme imbalance."""
    return float(average_precision_score(y_true, y_prob))

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

def cm(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Returns confusion matrix in order [[TN, FP],[FN, TP]]."""
    return confusion_matrix(y_true, y_pred)

def expected_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float,
    cost_fp: float,
) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(fn * cost_fn + fp * cost_fp)
