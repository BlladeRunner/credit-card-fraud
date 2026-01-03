from __future__ import annotations
import numpy as np
from sklearn.metrics import precision_score, recall_score
from .metrics import expected_cost

def apply_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    return (y_prob >= threshold).astype(int)

def find_threshold_for_min_precision(y_true, y_prob, min_precision: float) -> float:
    thresholds = np.linspace(0.0, 1.0, 1001)
    best = None
    for t in thresholds:
        y_pred = apply_threshold(y_prob, t)
        p = precision_score(y_true, y_pred, zero_division=0)
        if p >= min_precision:
            best = t
            break
    return 0.5 if best is None else float(best)

def find_threshold_for_min_recall(y_true, y_prob, min_recall: float) -> float:
    thresholds = np.linspace(0.0, 1.0, 1001)
    best = None
    for t in thresholds:
        y_pred = apply_threshold(y_prob, t)
        r = recall_score(y_true, y_pred, zero_division=0)
        if r >= min_recall:
            best = t
            break
    return 0.5 if best is None else float(best)

def find_threshold_min_cost(y_true, y_prob, cost_fn: float, cost_fp: float) -> float:
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_t = 0.5
    best_cost = float("inf")
    for t in thresholds:
        y_pred = apply_threshold(y_prob, t)
        c = expected_cost(y_true, y_pred, cost_fn=cost_fn, cost_fp=cost_fp)
        if c < best_cost:
            best_cost = c
            best_t = t
    return float(best_t)
