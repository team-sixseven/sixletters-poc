"""Deterministic synthetic inputs for integration tests, not clinical evidence."""

import numpy as np
import pandas as pd

from sixletters.contracts import Dataset


def make_demo_dataset() -> Dataset:
    """Return 180 artificial patients with shared, balanced demo splits.

    The labels, feature names, relationships and split sizes are synthetic test
    fixtures. They do not define MUSIC eligibility, endpoints or evaluation policy.
    """
    rng = np.random.default_rng(42)
    split_sizes = {"train": 108, "validation": 36, "test": 36}
    target = np.concatenate(
        [rng.permutation(np.tile([0, 1], size // 2)) for size in split_sizes.values()]
    )
    total = len(target)
    frame = pd.DataFrame(
        {
            "patient_id": [f"DEMO-{number:04d}" for number in range(1, total + 1)],
            "target": target,
            "split": np.repeat(list(split_sizes), list(split_sizes.values())),
            "clinical_1": rng.normal(size=total) + 0.65 * target,
            "clinical_2": rng.normal(size=total) - 0.30 * target,
            "ecg_1": rng.normal(size=total) + 0.55 * target,
            "ecg_2": rng.normal(size=total) + 0.45 * target,
        }
    )
    groups = {"clinical": ("clinical_1", "clinical_2"), "ecg": ("ecg_1", "ecg_2")}
    for columns in groups.values():
        for column in columns:
            frame.loc[rng.choice(total, size=18, replace=False), column] = np.nan
    dataset = Dataset(
        frame=frame,
        feature_groups=groups,
        dataset_id="synthetic-demo-v1",
        target_definition="Synthetic binary target for integration checks; NOT MUSIC or a clinical endpoint.",
    )
    dataset.validate()
    return dataset
