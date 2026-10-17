import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.smoke_check import (
    FEATURES,
    fit_model,
    make_synthetic_data,
    run_smoke,
    split_patients,
)


def test_synthetic_patient_split_is_disjoint_complete_and_reproducible():
    frame = make_synthetic_data()
    pd.testing.assert_frame_equal(frame, make_synthetic_data())
    assert frame[FEATURES].isna().any().any()
    splits = split_patients(frame)
    ids = {name: set(part["patient_id"]) for name, part in splits.items()}
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])
    assert set.union(*ids.values()) == set(frame["patient_id"])
    assert {name: len(values) for name, values in ids.items()} == {
        "train": 72,
        "validation": 24,
        "test": 24,
    }
    for name, part in split_patients(frame).items():
        pd.testing.assert_frame_equal(part, splits[name])
        assert part.groupby("patient_id").size().eq(3).all()
        assert part.drop_duplicates("patient_id")["target"].mean() == 0.5


@pytest.mark.parametrize("missing_id", [None, "", "  "])
def test_missing_patient_id_is_rejected(missing_id):
    frame = make_synthetic_data()
    frame.loc[0, "patient_id"] = missing_id
    with pytest.raises(ValueError, match="patient_id"):
        split_patients(frame)


def test_conflicting_patient_labels_are_rejected():
    frame = make_synthetic_data()
    frame.loc[0, "target"] = 1 - frame.loc[0, "target"]
    with pytest.raises(ValueError, match="consistent target"):
        split_patients(frame)


def test_preprocessing_is_fitted_only_on_training_data():
    splits = split_patients(make_synthetic_data())
    train = splits["train"]
    model = fit_model(train)
    expected_medians = train[FEATURES].median().to_numpy()
    imputer = model.named_steps["simpleimputer"]
    scaler = model.named_steps["standardscaler"]
    np.testing.assert_allclose(imputer.statistics_, expected_medians)
    imputed_train = train[FEATURES].fillna(train[FEATURES].median())
    np.testing.assert_allclose(scaler.mean_, imputed_train.mean().to_numpy())

    # An extreme held-out set must not change already fitted preprocessing.
    held_out = splits["test"][FEATURES].copy() * 10_000
    held_out.iloc[0, 0] = np.nan
    model.predict_proba(held_out)
    np.testing.assert_allclose(imputer.statistics_, expected_medians)
    np.testing.assert_allclose(scaler.mean_, imputed_train.mean().to_numpy())


def test_smoke_writes_explicitly_synthetic_results(tmp_path):
    result = run_smoke(tmp_path)
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved == result
    assert "NOT MUSIC" in saved["purpose"]
    assert saved["missing_values_before_imputation"] > 0
    assert set(saved["metrics"]) == {"validation", "test"}
    for metrics in saved["metrics"].values():
        assert all(0 <= value <= 1 for value in metrics.values())
    assert len(pd.read_csv(tmp_path / "synthetic_input.csv")) == 360
    assert (tmp_path / "synthetic_metrics.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_cli_supports_korean_output_with_legacy_windows_encoding(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "smoke_check.py"
    output = tmp_path / "한글결과"
    completed = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output)],
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        capture_output=True,
        check=True,
    )
    assert "환경 점검 완료" in completed.stdout.decode("utf-8")
    assert (output / "metrics.json").is_file()
