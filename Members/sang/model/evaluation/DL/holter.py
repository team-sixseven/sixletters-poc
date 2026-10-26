"""Metrics and validation-only threshold selection for Holter ECG models."""

from __future__ import annotations

import numpy as np

from ..core import binary_metrics, threshold_metrics, validate_scores


def metrics(target, probability) -> dict[str, float | int]:
    """Compute threshold-independent patient-level metrics for one split."""
    y, score = validate_scores(target, probability)
    return {
        key: value
        for key, value in binary_metrics(y, score, threshold=0.5).items()
        if key not in {"threshold", "sensitivity", "specificity"}
    }


def choose_youden_threshold(target, probability) -> dict[str, float]:
    """Choose the validation threshold maximizing Youden J.

    Ties prefer higher sensitivity, then the lower threshold, matching the
    previous Holter evaluator's selection rule.
    """
    y, score = validate_scores(target, probability)
    candidates = np.unique(np.concatenate(([0.0, 1.0], score)))
    scored = []
    for threshold in candidates:
        values = threshold_metrics(y, score, float(threshold))
        youden_j = values["sensitivity"] + values["specificity"] - 1
        scored.append((youden_j, values["sensitivity"], -values["threshold"], values))
    return max(scored, key=lambda row: row[:3])[3]


def evaluate_holter_predictions(target, probability, *, threshold: float) -> dict[str, float | int]:
    """Evaluate one split with a threshold selected on validation data."""
    return binary_metrics(target, probability, threshold=threshold)
