"""Shared, patient-level inputs for independent experiments."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype, is_numeric_dtype

RESERVED_COLUMNS = ("patient_id", "target", "split")
SPLITS = frozenset({"train", "validation", "test"})
FEATURE_GROUPS = frozenset({"clinical", "ecg"})


def _hash_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class Dataset:
    """One row per patient, with one shared assignment to train/validation/test.

    Missing feature values are allowed; a training-only imputer handles them.
    Only explicitly declared clinical/ECG features may be selected for modeling.
    """

    frame: pd.DataFrame
    feature_groups: dict[str, tuple[str, ...]]
    dataset_id: str
    target_definition: str
    categorical_features: tuple[str, ...] = ()

    def validate(self) -> None:
        """Reject malformed identities, split assignments and unsafe feature groups."""
        for name in ("dataset_id", "target_definition"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.frame, pd.DataFrame) or self.frame.empty:
            raise ValueError("frame must be a non-empty pandas DataFrame")
        if self.frame.columns.has_duplicates:
            raise ValueError("frame has duplicate column names")
        missing = set(RESERVED_COLUMNS) - set(self.frame.columns)
        if missing:
            raise ValueError(f"frame is missing required columns: {sorted(missing)}")

        ids = self.frame["patient_id"]
        if ids.isna().any() or not ids.map(lambda value: isinstance(value, str)).all():
            raise ValueError("patient_id must contain non-missing strings")
        if ids.str.strip().eq("").any() or ids.ne(ids.str.strip()).any():
            raise ValueError("patient_id must not be blank or padded with whitespace")
        if ids.duplicated().any():
            raise ValueError("patient_id must be unique: a patient cannot span rows or splits")

        splits = self.frame["split"]
        if splits.isna().any() or not splits.isin(SPLITS).all():
            raise ValueError("split must be train, validation, or test for every patient")
        if set(splits) != SPLITS:
            raise ValueError("split must include non-empty train, validation, and test sets")

        target = self.frame["target"]
        if (
            not is_numeric_dtype(target.dtype)
            or is_complex_dtype(target.dtype)
            or target.isna().any()
            or not target.isin([0, 1]).all()
        ):
            raise ValueError("target must contain non-missing numeric 0/1 values")

        for split in ("train", "validation"):
            split_target = target.loc[splits.eq(split)]
            counts = {label: int(split_target.eq(label).sum()) for label in (0, 1)}
            if 0 in counts.values():
                raise ValueError(
                    f"{split} split requires both target classes 0 and 1 "
                    f"(target=0: {counts[0]}, target=1: {counts[1]})"
                )

        if not isinstance(self.feature_groups, dict) or set(self.feature_groups) != FEATURE_GROUPS:
            raise ValueError("feature_groups must define clinical and ecg groups")
        if (
            not isinstance(self.categorical_features, tuple)
            or any(not isinstance(column, str) for column in self.categorical_features)
            or len(set(self.categorical_features)) != len(self.categorical_features)
        ):
            raise ValueError("categorical_features must be a tuple of unique feature names")
        seen: set[str] = set()
        for group in sorted(FEATURE_GROUPS):
            columns = self.feature_groups[group]
            if not isinstance(columns, tuple) or not columns:
                raise ValueError(f"feature group {group} must be a non-empty tuple")
            for column in columns:
                if not isinstance(column, str) or not column.strip():
                    raise ValueError(f"feature group {group} contains an invalid column name")
                if column in RESERVED_COLUMNS:
                    raise ValueError(f"reserved column {column} cannot be used as a feature")
                if column in seen:
                    raise ValueError(f"duplicate feature {column} within or across groups")
                seen.add(column)
                if column not in self.frame:
                    raise ValueError(f"feature column {column} is missing from frame")
                values = self.frame[column]
                if column in self.categorical_features:
                    if not all(isinstance(value, str) for value in values.dropna()):
                        raise ValueError(
                            f"categorical feature {column} must contain strings or missing values"
                        )
                    continue
                if not is_numeric_dtype(values.dtype) or is_complex_dtype(values.dtype):
                    raise ValueError(f"feature {column} must be numeric and real-valued")
                numeric = values.to_numpy(dtype=float, na_value=np.nan)
                if np.isinf(numeric).any():
                    raise ValueError(f"feature {column} must not contain infinity")
        if not set(self.categorical_features).issubset(seen):
            raise ValueError("categorical_features must refer to declared features")

    def fingerprint(self) -> dict[str, str]:
        """Fingerprint the declared contract, independent of row/index/column order.

        Unselected metadata columns are not model inputs and are not hashed.
        Missing feature values are encoded as null; numeric dtypes normalize to float.
        Categorical values retain their strings, including empty strings and literal NA.
        Feature tuple order is part of the feature schema, since selectors preserve it.
        """
        self.validate()
        ordered = self.frame.sort_values("patient_id").reset_index(drop=True)
        ids = ordered["patient_id"].tolist()
        targets = ordered["target"].astype(int).tolist()
        assignments = ordered["split"].tolist()
        feature_names = sorted(
            column for columns in self.feature_groups.values() for column in columns
        )
        feature_columns = []
        for name in feature_names:
            if name in self.categorical_features:
                values = [None if pd.isna(value) else value for value in ordered[name]]
            else:
                values = [
                    None if np.isnan(value) else (0.0 if value == 0 else float(value))
                    for value in ordered[name].to_numpy(dtype=float, na_value=np.nan)
                ]
            feature_columns.append(values)
        feature_rows = list(zip(*feature_columns, strict=True))
        rows = [
            [patient_id, target, split, *values]
            for patient_id, target, split, values in zip(
                ids, targets, assignments, feature_rows, strict=True
            )
        ]
        return {
            "dataset_id": self.dataset_id,
            "target_definition": self.target_definition,
            "data_hash": _hash_json({"columns": [*RESERVED_COLUMNS, *feature_names], "rows": rows}),
            "cohort_hash": _hash_json(ids),
            "split_hash": _hash_json(list(zip(ids, assignments, strict=True))),
            "target_hash": _hash_json(list(zip(ids, targets, strict=True))),
            "feature_schema_hash": _hash_json(
                {
                    "feature_groups": self.feature_groups,
                    "categorical_features": sorted(self.categorical_features),
                }
            ),
        }
