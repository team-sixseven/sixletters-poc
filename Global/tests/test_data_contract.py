from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from sixletters.data.demo import make_demo_dataset
from sixletters.features import select_features


@pytest.fixture
def dataset():
    return make_demo_dataset()


def test_demo_is_reproducible_patient_level_and_explicitly_synthetic(dataset):
    other = make_demo_dataset()
    pd.testing.assert_frame_equal(dataset.frame, other.frame)
    assert dataset.fingerprint() == other.fingerprint()
    assert dataset.dataset_id.startswith("synthetic-")
    assert "NOT MUSIC" in dataset.target_definition
    assert len(dataset.frame) == dataset.frame.patient_id.nunique() == 180
    assert dataset.frame.groupby("split")["target"].nunique().eq(2).all()
    features = select_features(dataset, "combined")
    assert dataset.frame[list(features)].isna().any().all()


def test_fingerprints_ignore_row_index_column_and_dictionary_order(dataset):
    reordered = replace(
        dataset,
        frame=dataset.frame.sample(frac=1, random_state=7).iloc[:, ::-1].reset_index(drop=True),
        feature_groups=dict(reversed(list(dataset.feature_groups.items()))),
    )
    assert dataset.fingerprint() == reordered.fingerprint()
    result = dataset.fingerprint()
    assert set(result) == {
        "dataset_id",
        "target_definition",
        "data_hash",
        "cohort_hash",
        "split_hash",
        "target_hash",
        "feature_schema_hash",
    }
    assert all(len(value) == 64 for key, value in result.items() if key.endswith("_hash"))


def test_numeric_nullable_dtypes_have_the_same_fingerprint(dataset):
    frame = dataset.frame.copy()
    for column in select_features(dataset, "combined"):
        frame[column] = frame[column].astype("Float64")
    assert replace(dataset, frame=frame).fingerprint() == dataset.fingerprint()


@pytest.mark.parametrize("invalid_id", [None, np.nan, "", "  ", " DEMO-1 ", 42])
def test_invalid_patient_id_is_rejected(dataset, invalid_id):
    dataset.frame.loc[0, "patient_id"] = invalid_id
    with pytest.raises(ValueError, match="patient_id"):
        dataset.validate()


def test_same_patient_in_two_splits_is_rejected(dataset):
    train_id = dataset.frame.loc[dataset.frame.split.eq("train"), "patient_id"].iloc[0]
    test_row = dataset.frame.index[dataset.frame.split.eq("test")][0]
    dataset.frame.loc[test_row, "patient_id"] = train_id
    with pytest.raises(ValueError, match="unique"):
        dataset.validate()


@pytest.mark.parametrize("invalid_split", [None, "", "holdout"])
def test_unassigned_or_unknown_split_is_rejected(dataset, invalid_split):
    dataset.frame.loc[0, "split"] = invalid_split
    with pytest.raises(ValueError, match="split"):
        dataset.validate()


def test_missing_split_is_rejected(dataset):
    dataset.frame = dataset.frame.loc[dataset.frame.split.ne("test")]
    with pytest.raises(ValueError, match="non-empty"):
        dataset.validate()


@pytest.mark.parametrize("invalid_target", [None, np.nan, -1, 2, np.inf, "1"])
def test_invalid_target_is_rejected(dataset, invalid_target):
    if isinstance(invalid_target, str):
        dataset.frame["target"] = dataset.frame["target"].astype(object)
    else:
        dataset.frame["target"] = dataset.frame["target"].astype(float)
    dataset.frame.loc[0, "target"] = invalid_target
    with pytest.raises(ValueError, match="target"):
        dataset.validate()


@pytest.mark.parametrize("reserved", ["patient_id", "target", "split"])
def test_identity_label_and_split_cannot_leak_into_features(dataset, reserved):
    dataset.feature_groups["clinical"] += (reserved,)
    for feature_set in ("clinical", "ecg", "combined"):
        with pytest.raises(ValueError, match="reserved"):
            select_features(dataset, feature_set)


@pytest.mark.parametrize("group", ["clinical", "ecg"])
def test_duplicate_features_within_or_across_groups_are_rejected(dataset, group):
    dataset.feature_groups[group] += ("clinical_1",)
    with pytest.raises(ValueError, match="duplicate feature"):
        dataset.validate()


def test_missing_feature_column_is_rejected(dataset):
    dataset.frame = dataset.frame.drop(columns="clinical_1")
    with pytest.raises(ValueError, match="missing"):
        dataset.validate()


def test_duplicate_frame_columns_are_rejected(dataset):
    dataset.frame = pd.concat([dataset.frame, dataset.frame[["clinical_1"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate column"):
        dataset.validate()


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_infinite_features_are_rejected_while_nan_is_allowed(dataset, bad_value):
    dataset.validate()
    dataset.frame.loc[0, "ecg_1"] = bad_value
    with pytest.raises(ValueError, match="infinity"):
        dataset.validate()


def test_numeric_strings_are_not_silently_coerced(dataset):
    dataset.frame["clinical_1"] = dataset.frame["clinical_1"].astype(str)
    with pytest.raises(ValueError, match="numeric"):
        dataset.validate()


def test_declared_categorical_strings_and_missing_values_are_accepted(dataset):
    dataset.frame["clinical_1"] = pd.Series(["low", "NA", "", None] * 45, dtype="string")
    dataset.categorical_features = ("clinical_1",)
    dataset.validate()
    before = dataset.fingerprint()
    dataset.frame.loc[0, "clinical_1"] = "high"
    after = dataset.fingerprint()
    assert before["data_hash"] != after["data_hash"]
    assert before["cohort_hash"] == after["cohort_hash"]
    assert before["split_hash"] == after["split_hash"]


@pytest.mark.parametrize("value", [0, np.inf, complex(1, 2), True])
def test_categorical_mixed_types_are_rejected(dataset, value):
    dataset.frame["clinical_1"] = pd.Series(["low"] * len(dataset.frame), dtype=object)
    dataset.frame.loc[0, "clinical_1"] = value
    dataset.categorical_features = ("clinical_1",)
    with pytest.raises(ValueError, match="strings"):
        dataset.validate()


@pytest.mark.parametrize("categories", [("missing",), ("target",), ("ecg_1", "ecg_1"), ["ecg_1"]])
def test_categorical_declarations_must_be_unique_declared_features(dataset, categories):
    dataset.categorical_features = categories
    with pytest.raises(ValueError, match="categorical"):
        dataset.validate()


def test_categorical_missing_representations_have_the_same_fingerprint(dataset):
    dataset.frame["clinical_1"] = pd.Series(["NA", "", None, np.nan] * 45, dtype=object)
    dataset.categorical_features = ("clinical_1",)
    other = replace(dataset, frame=dataset.frame.copy())
    other.frame["clinical_1"] = other.frame["clinical_1"].astype("string")
    assert dataset.fingerprint() == other.fingerprint()


def test_categorical_schema_changes_are_detected_even_for_all_missing_features(dataset):
    dataset.frame["clinical_1"] = np.nan
    categorical = replace(dataset, categorical_features=("clinical_1",))
    assert dataset.fingerprint()["data_hash"] == categorical.fingerprint()["data_hash"]
    assert (
        dataset.fingerprint()["feature_schema_hash"]
        != categorical.fingerprint()["feature_schema_hash"]
    )


def test_feature_selection_keeps_all_patients_and_shared_splits(dataset):
    before = dataset.frame.copy(deep=True)
    fingerprints = dataset.fingerprint()
    assert select_features(dataset, "clinical") == ("clinical_1", "clinical_2")
    assert select_features(dataset, "ecg") == ("ecg_1", "ecg_2")
    assert select_features(dataset, "combined") == (
        "clinical_1",
        "clinical_2",
        "ecg_1",
        "ecg_2",
    )
    pd.testing.assert_frame_equal(dataset.frame, before)
    assert dataset.fingerprint() == fingerprints
    with pytest.raises(ValueError, match="unknown feature_set"):
        select_features(dataset, "unknown")


def test_feature_change_is_detected_without_changing_patient_or_split_hashes(dataset):
    before = dataset.fingerprint()
    row = dataset.frame.index[dataset.frame.clinical_1.notna()][0]
    dataset.frame.loc[row, "clinical_1"] += 1
    after = dataset.fingerprint()
    assert before["data_hash"] != after["data_hash"]
    assert {key for key in before if before[key] != after[key]} == {"data_hash"}


def test_patient_split_label_and_feature_schema_changes_are_distinguishable(dataset):
    before = dataset.fingerprint()
    frame = dataset.frame.copy()
    frame.loc[0, "patient_id"] = "DEMO-NEW"
    assert replace(dataset, frame=frame).fingerprint()["cohort_hash"] != before["cohort_hash"]

    frame = dataset.frame.copy()
    frame.loc[0, "split"] = "test"
    changed_split = replace(dataset, frame=frame).fingerprint()
    assert changed_split["split_hash"] != before["split_hash"]
    assert changed_split["cohort_hash"] == before["cohort_hash"]
    assert changed_split["target_hash"] == before["target_hash"]

    frame = dataset.frame.copy()
    frame.loc[0, "target"] = 1 - frame.loc[0, "target"]
    assert replace(dataset, frame=frame).fingerprint()["target_hash"] != before["target_hash"]

    groups = {"clinical": ("clinical_1",), "ecg": ("clinical_2", "ecg_1", "ecg_2")}
    changed_schema = replace(dataset, feature_groups=groups).fingerprint()
    assert changed_schema["feature_schema_hash"] != before["feature_schema_hash"]
    assert changed_schema["data_hash"] == before["data_hash"]
