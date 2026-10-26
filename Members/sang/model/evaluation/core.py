"""Common patient-level prediction validation and binary classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def to_numpy(value, *, dtype):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype).reshape(-1)


def validate_scores(target, probability) -> tuple[np.ndarray, np.ndarray]:
    raw_y = to_numpy(target, dtype=np.float64)
    score = to_numpy(probability, dtype=np.float64)
    if raw_y.size == 0 or raw_y.size != score.size:
        raise ValueError("Targets and scores must be non-empty arrays of equal length")
    if not np.isfinite(raw_y).all() or not np.isin(raw_y, (0, 1)).all():
        raise ValueError("Targets must contain only binary values 0 and 1")
    y = raw_y.astype(np.int64)
    if np.unique(y).size != 2:
        raise ValueError("ROC-AUC evaluation requires positive and negative targets")
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Prediction scores must be finite probabilities in [0, 1]")
    return y, score


def threshold_metrics(target, probability, threshold: float) -> dict[str, float]:
    y, score = validate_scores(target, probability)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite probability in [0, 1]")
    predicted = score >= threshold
    positive = y == 1
    negative = ~positive
    return {
        "threshold": float(threshold),
        "sensitivity": float(predicted[positive].mean()),
        "specificity": float((~predicted[negative]).mean()),
    }


def binary_metrics(target, probability, threshold: float) -> dict[str, float | int]:
    y, score = validate_scores(target, probability)
    result: dict[str, float | int] = {
        "n": int(y.size),
        "scd": int(y.sum()),
        "auroc": float(roc_auc_score(y, score)),
        "auprc_average_precision": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
    }
    result.update(threshold_metrics(y, score, threshold))
    return result
