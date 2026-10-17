"""Join declared input groups; patient inclusion is decided before selection."""

from sixletters.contracts import Dataset


def select_features(dataset: Dataset) -> tuple[str, ...]:
    dataset.validate()
    return dataset.feature_groups["clinical"] + dataset.feature_groups["ecg"]
