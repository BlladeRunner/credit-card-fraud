from __future__ import annotations
import numpy as np
from .metrics import pr_auc, classification_metrics, cm, expected_cost
from .thresholding import apply_threshold

def evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_fn: float | None = None,
    cost_fp: float | None = None,
) -> dict:
    y_pred = apply_threshold(y_prob, threshold)

    out = {
        "threshold": float(threshold),
        "pr_auc": pr_auc(y_true, y_prob),
        **classification_metrics(y_true, y_pred),
        "confusion_matrix": cm(y_true, y_pred).tolist(),
    }

    if cost_fn is not None and cost_fp is not None:
        out["expected_cost"] = expected_cost(y_true, y_pred, cost_fn=cost_fn, cost_fp=cost_fp)

    return out
