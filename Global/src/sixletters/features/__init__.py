"""Feature-set dispatch for independent experiments on a shared Dataset."""

from sixletters.contracts import Dataset
from sixletters.features import clinical, combined, ecg


def select_features(dataset: Dataset, feature_set: str) -> tuple[str, ...]:
    selectors = {
        "clinical": clinical.select_features,
        "ecg": ecg.select_features,
        "combined": combined.select_features,
    }
    if feature_set not in selectors:
        raise ValueError(f"unknown feature_set {feature_set!r}; choose clinical, ecg, or combined")
    return selectors[feature_set](dataset)


__all__ = ["select_features"]
