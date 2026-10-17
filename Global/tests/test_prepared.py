import csv
import json

import numpy as np
import pandas as pd
import pytest

from sixletters.contracts import _hash_json
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import (
    import_csv_dataset,
    load_prepared_dataset,
    save_prepared_dataset,
)


@pytest.fixture
def dataset():
    value = make_demo_dataset()
    value.frame.loc[0, "patient_id"] = "0001"
    value.frame.loc[1, "patient_id"] = "NA"
    value.frame.loc[0, "clinical_1"] = np.nextafter(1.0, 2.0)
    value.frame.loc[1, "clinical_1"] = 1.2345678901234567e-20
    value.frame["clinical_2"] = pd.Series(["NA", "", "정상", None] * 45, dtype=object)
    value.categorical_features = ("clinical_2",)
    return value


def test_bundle_round_trip_preserves_precision_strings_missing_values_and_fingerprint(
    tmp_path, dataset
):
    path = save_prepared_dataset(dataset, tmp_path / "공통 데이터" / "v1", synthetic=False)
    assert path.name == "dataset.json"
    loaded, synthetic = load_prepared_dataset(path)
    assert synthetic is False
    assert loaded.fingerprint() == dataset.fingerprint()
    expected = dataset.frame.copy()
    expected["clinical_2"] = expected["clinical_2"].fillna(np.nan)
    pd.testing.assert_frame_equal(loaded.frame, expected)
    directory_loaded, _ = load_prepared_dataset(path.parent)
    assert directory_loaded.fingerprint() == loaded.fingerprint()


def test_bundle_only_saves_declared_inputs(tmp_path, dataset):
    dataset.frame["unreviewed_metadata"] = "do not export"
    path = save_prepared_dataset(dataset, tmp_path / "data", synthetic=True)
    loaded, synthetic = load_prepared_dataset(path)
    assert synthetic is True
    assert "unreviewed_metadata" not in loaded.frame
    assert "do not export" not in path.read_text(encoding="utf-8")
    assert loaded.fingerprint() == dataset.fingerprint()


def test_all_missing_numeric_and_categorical_columns_round_trip(tmp_path, dataset):
    dataset.frame["clinical_1"] = np.nan
    dataset.frame["clinical_2"] = pd.Series([pd.NA] * len(dataset.frame), dtype="string")
    path = save_prepared_dataset(dataset, tmp_path / "data", synthetic=False)
    loaded, _ = load_prepared_dataset(path)
    assert loaded.fingerprint() == dataset.fingerprint()


def test_save_never_overwrites_existing_directory(tmp_path, dataset):
    output = tmp_path / "data"
    path = save_prepared_dataset(dataset, output, synthetic=True)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_prepared_dataset(dataset, output, synthetic=False)
    assert path.read_bytes() == before
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        save_prepared_dataset(dataset, empty, synthetic=False)


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("label", [0, 1])
def test_save_rejects_single_class_training_or_validation_before_creating_bundle(
    tmp_path, dataset, split, label
):
    mask = dataset.frame.split.eq(split)
    dataset.frame.loc[mask, "target"] = label
    output = tmp_path / "data"
    with pytest.raises(ValueError, match=rf"{split} split requires both target classes") as error:
        save_prepared_dataset(dataset, output, synthetic=False)
    assert f"target={label}: {int(mask.sum())}" in str(error.value)
    assert f"target={1 - label}: 0" in str(error.value)
    assert all(patient_id not in str(error.value) for patient_id in dataset.frame.patient_id)
    assert not output.exists()


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("label", [0, 1])
def test_load_rejects_single_class_training_or_validation_even_with_valid_bundle_hash(
    tmp_path, dataset, split, label
):
    path = save_prepared_dataset(dataset, tmp_path / "data", synthetic=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    split_position = payload["columns"].index("split")
    target_position = payload["columns"].index("target")
    count = 0
    for row in payload["rows"]:
        if row[split_position] == split:
            row[target_position] = label
            count += 1
    payload.pop("bundle_hash")
    payload["bundle_hash"] = _hash_json(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"{split} split requires both target classes") as error:
        load_prepared_dataset(path)
    assert f"target={label}: {count}" in str(error.value)
    assert f"target={1 - label}: 0" in str(error.value)


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_test_set_remains_valid_and_round_trips(tmp_path, dataset, label):
    dataset.frame.loc[dataset.frame.split.eq("test"), "target"] = label
    path = save_prepared_dataset(dataset, tmp_path / "data", synthetic=False)
    loaded, _ = load_prepared_dataset(path)
    assert loaded.fingerprint() == dataset.fingerprint()
    assert set(loaded.frame.loc[loaded.frame.split.eq("test"), "target"]) == {label}


@pytest.mark.parametrize("column", ["patient_id", "target", "split", "clinical_1", "clinical_2"])
def test_changed_patient_data_labels_splits_or_features_are_rejected(tmp_path, dataset, column):
    path = save_prepared_dataset(dataset, tmp_path / "data", synthetic=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    replacements = {
        "patient_id": "new-id",
        "target": 1,
        "split": "test",
        "clinical_1": 42,
        "clinical_2": "new-category",
    }
    position = payload["columns"].index(column)
    current = payload["rows"][0][position]
    payload["rows"][0][position] = 1 - current if column == "target" else replacements[column]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_prepared_dataset(path)


def test_data_fingerprint_is_checked_even_when_bundle_hash_is_recomputed(tmp_path, dataset):
    path = save_prepared_dataset(dataset, tmp_path / "data", synthetic=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0][payload["columns"].index("clinical_1")] += 1
    payload.pop("bundle_hash")
    payload["bundle_hash"] = _hash_json(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_prepared_dataset(path)


@pytest.mark.parametrize("field", ["synthetic", "schema", "fingerprint"])
def test_metadata_changes_are_rejected(tmp_path, dataset, field):
    path = save_prepared_dataset(dataset, tmp_path / "data", synthetic=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if field == "synthetic":
        payload[field] = True
    elif field == "schema":
        payload[field]["target_definition"] = "different endpoint"
    else:
        payload[field]["dataset_id"] = "new-version"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_prepared_dataset(path)


@pytest.fixture
def csv_inputs(tmp_path):
    schema = {
        "dataset_id": "reviewed-v1",
        "target_definition": "Reviewed binary endpoint",
        "feature_groups": {"clinical": ["age", "status"], "ecg": ["rr"]},
        "categorical_features": ["status"],
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    csv_path = tmp_path / "공통.csv"
    csv_path.write_text(
        "patient_id,target,split,age,status,rr,extra\n"
        "0001,0,train,54,NA,1.0000000000000002,private\n"
        "NA,1,validation,62,,0.8123456789012345,private\n"
        "0003,0,test,,정상,,private\n"
        "0004,1,train,58,정상,0.9,private\n"
        "0005,0,validation,60,NA,0.8,private\n",
        encoding="utf-8-sig",
    )
    return csv_path, schema_path


def test_csv_import_preserves_ids_literal_na_and_precision_and_uses_explicit_splits(csv_inputs):
    dataset = import_csv_dataset(*csv_inputs)
    assert dataset.frame.patient_id.tolist() == ["0001", "NA", "0003", "0004", "0005"]
    assert dataset.frame.split.tolist() == ["train", "validation", "test", "train", "validation"]
    assert dataset.frame.status.iloc[0] == "NA"
    assert pd.isna(dataset.frame.status.iloc[1])
    assert pd.isna(dataset.frame.age.iloc[2])
    assert pd.isna(dataset.frame.rr.iloc[2])
    assert dataset.frame.rr.iloc[0] == np.nextafter(1.0, 2.0)
    assert "extra" not in dataset.frame


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("label", [0, 1])
def test_csv_import_rejects_single_class_training_or_validation(csv_inputs, split, label):
    csv_path, schema_path = csv_inputs
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    for row in rows:
        if row["split"] == split:
            row["target"] = str(label)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match=rf"{split} split requires both target classes") as error:
        import_csv_dataset(csv_path, schema_path)
    assert f"target={label}: 2" in str(error.value)
    assert f"target={1 - label}: 0" in str(error.value)


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        ("0001,0,train", ",0,train", "patient_id"),
        ("0001,0,train", " 0001,0,train", "patient_id"),
        ("0003,0,test", "0001,0,test", "unique"),
        ("0001,0,train", "0001,,train", "target"),
        ("0001,0,train", "0001,2,train", "target"),
        ("0003,0,test", "0003,0,train", "non-empty"),
        ("54,NA", "NA,NA", "real numbers"),
        ("54,NA", "NaN,NA", "real numbers"),
        ("54,NA", "inf,NA", "infinity"),
        ("age,status", "age,age", "duplicate"),
    ],
)
def test_csv_import_rejects_invalid_identity_labels_splits_or_features(
    csv_inputs, original, replacement, message
):
    csv_path, schema_path = csv_inputs
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8-sig").replace(original, replacement), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=message):
        import_csv_dataset(csv_path, schema_path)


def test_csv_import_does_not_guess_missing_split_column(csv_inputs):
    csv_path, schema_path = csv_inputs
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8-sig").replace("split,", "group,"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing required"):
        import_csv_dataset(csv_path, schema_path)
