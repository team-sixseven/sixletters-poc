"""Clinical prediction metrics using a caller-selected operating threshold."""

from __future__ import annotations

from ..core import binary_metrics


def evaluate_clinical_predictions(
    target, probability, threshold: float = 0.5
) -> dict[str, float | int]:
    """Evaluate one clinical model's probability vector at a configured threshold."""
    return binary_metrics(target, probability, threshold=threshold)
