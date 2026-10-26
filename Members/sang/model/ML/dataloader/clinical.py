"""Load clinical features from clinical_clean.csv and apply saved patient splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_clinical_splits(clinical_csv: Path, split_dir: Path) -> dict[str, pd.DataFrame]:
    """Use the clean table as the feature source and split CSVs only for membership.

    The supplied split CSVs are expected to cover every clean-table patient once.
    Their labels are checked against the clean table before any model fitting.
    """
    clinical = pd.read_csv(clinical_csv, dtype={"Patient ID": "string"})
    clinical.columns = clinical.columns.str.strip()
    if clinical.columns.duplicated().any():
        duplicated = clinical.columns[clinical.columns.duplicated()].tolist()
        raise ValueError(
            f"clinical_clean.csv has duplicate columns after trimming whitespace: {duplicated}"
        )
    required = {"Patient ID", "SCD"}
    if not required.issubset(clinical.columns):
        raise ValueError("clinical_clean.csv must contain Patient ID and SCD columns")
    if clinical["Patient ID"].isna().any() or clinical["Patient ID"].duplicated().any():
        raise ValueError("clinical_clean.csv Patient ID values must be present and unique")
    if clinical["SCD"].isna().any() or not clinical["SCD"].isin((0, 1)).all():
        raise ValueError("clinical_clean.csv SCD labels must contain binary values 0 and 1")
    features = [name for name in clinical.columns if name not in required]
    if not features:
        raise ValueError("clinical_clean.csv has no feature columns")
    if clinical[features].isna().any().any():
        missing = clinical[features].isna().sum()
        raise ValueError(
            f"clinical_clean.csv has missing features: {missing[missing > 0].to_dict()}"
        )
    if not all(pd.api.types.is_numeric_dtype(clinical[name]) for name in features):
        raise ValueError("clinical_clean.csv feature columns must be numeric")

    indexed = clinical.set_index("Patient ID", drop=False)
    split_ids: dict[str, list[str]] = {}
    split_frames = {}
    for split in ("train", "val", "test"):
        split_path = split_dir / f"{split}.csv"
        membership = pd.read_csv(
            split_path, usecols=["Patient ID", "SCD"], dtype={"Patient ID": "string"}
        )
        if membership["Patient ID"].isna().any() or membership["Patient ID"].duplicated().any():
            raise ValueError(f"{split_path}: Patient ID values must be present and unique")
        unknown = set(membership["Patient ID"]) - set(indexed.index)
        if unknown:
            raise ValueError(f"{split_path}: IDs are absent from clinical_clean.csv")
        labels = indexed.loc[membership["Patient ID"], "SCD"].to_numpy()
        if not (labels == membership["SCD"].to_numpy()).all():
            raise ValueError(f"{split_path}: SCD labels disagree with clinical_clean.csv")
        if membership["SCD"].nunique() != 2:
            raise ValueError(f"{split_path}: both SCD classes are required")
        split_ids[split] = membership["Patient ID"].astype(str).tolist()
        split_frames[split] = indexed.loc[membership["Patient ID"]].reset_index(drop=True)

    all_ids = [patient_id for ids in split_ids.values() for patient_id in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Patient leakage: train/val/test split IDs overlap")
    if set(all_ids) != set(indexed.index.astype(str)):
        raise ValueError("Saved splits must cover every patient in clinical_clean.csv exactly once")
    return split_frames
