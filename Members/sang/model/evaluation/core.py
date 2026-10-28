"""Common patient-level prediction validation and binary classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


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
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Prediction scores must be finite probabilities in [0, 1]")
    return y, score


def threshold_metrics(target, probability, threshold: float) -> dict[str, float]:
    y, score = validate_scores(target, probability)
    threshold = validate_threshold(threshold)
    predicted = score >= threshold
    positive = y == 1
    negative = ~positive
    return {
        "threshold": float(threshold),
        "sensitivity": float(predicted[positive].mean()) if positive.any() else None,
        "specificity": float((~predicted[negative]).mean()) if negative.any() else None,
    }


def validate_threshold(value) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError("A numeric classification threshold in [0, 1] is required")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("A numeric classification threshold in [0, 1] is required") from error
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"Classification threshold must be in [0, 1], got {value!r}")
    return threshold


def binary_metrics(target, probability, threshold: float) -> dict[str, float | int]:
    y, score = validate_scores(target, probability)
    threshold = validate_threshold(threshold)
    predicted = (score >= threshold).astype(np.int64)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    negatives, positives = np.bincount(y, minlength=2)
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    reasons = {}
    if len(np.unique(y)) == 2:
        auroc = float(roc_auc_score(y, score))
        ap = float(average_precision_score(y, score))
    else:
        auroc = None
        ap = None
        reason = "only_one_target_class_in_evaluation_split"
        reasons.update({"auroc": reason, "average_precision": reason})
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    precision = tp / precision_denominator if precision_denominator else None
    recall = tp / recall_denominator if recall_denominator else None
    f1 = 2 * tp / f1_denominator if f1_denominator else None
    if precision is None:
        reasons["precision"] = "no_positive_predictions"
    if recall is None:
        reasons["recall"] = reasons["sensitivity"] = "no_positive_targets"
    if f1 is None:
        reasons["f1"] = "f1_denominator_is_zero"
    if not negatives:
        reasons["specificity"] = "no_negative_targets"
    result: dict[str, float | int] = {
        "n": int(y.size),
        "scd": int(y.sum()),
        "negative": int(negatives),
        "auroc": auroc,
        "average_precision": ap,
        "auprc_average_precision": ap,
        "brier": float(brier_score_loss(y, score)),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision) if precision is not None else None,
        "recall": float(recall) if recall is not None else None,
        "sensitivity": float(recall) if recall is not None else None,
        "specificity": float(matrix[0, 0] / negatives) if negatives else None,
        "f1": float(f1) if f1 is not None else None,
        "confusion_matrix": matrix.tolist(),
        "metric_reasons": reasons,
    }
    result["threshold"] = float(threshold)
    return result
