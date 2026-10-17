"""Select declared clinical inputs without changing the cohort or split."""

from sixletters.contracts import Dataset


def select_features(dataset: Dataset) -> tuple[str, ...]:
    dataset.validate()
    return dataset.feature_groups["clinical"]
