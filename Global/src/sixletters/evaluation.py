"""Shared validation metrics. No fitting or threshold selection happens here."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    if not {"patient_id", "target", "score"}.issubset(predictions.columns):
        raise ValueError("Predictions need patient_id, target and score")
    if predictions["patient_id"].isna().any() or predictions["patient_id"].duplicated().any():
        raise ValueError("Predictions must contain one row per patient")
    if predictions["target"].isna().any() or set(predictions["target"]) != {0, 1}:
        raise ValueError("Validation metrics require both target classes, 0 and 1")
    scores = predictions["score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Predicted scores must be finite probabilities between 0 and 1")
    target = predictions["target"]
    return {
        "roc_auc": float(roc_auc_score(target, scores)),
        "average_precision": float(average_precision_score(target, scores)),
        "brier_score": float(brier_score_loss(target, scores)),
    }
