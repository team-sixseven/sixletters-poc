"""Execute the personal tutorial with fictional patients and no external services."""

import csv
import hashlib
import io
import json

import matplotlib
import nbformat
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from test_music import csv_bytes, synthetic_rows

from sixletters.data import music
from sixletters.data.prepared import load_prepared_dataset
from sixletters.workspace import init_practice

matplotlib.use("Agg")


def _table_bytes(rows, *, encoding):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=";")
    writer.writerows(rows)
    return buffer.getvalue().encode(encoding)


def _snapshot(directory):
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_personal_music_practice_runs_offline_and_keeps_test_patients_out_of_models(
    tmp_path, monkeypatch
):
    root = tmp_path / "팀 실습 저장소"
    (root / "Global").mkdir(parents=True)
    (root / "Global/pyproject.toml").touch()
    raw = root / "Global/data/raw/music-1.0.1"
    raw.mkdir(parents=True)
    base_rows = synthetic_rows()
    rows = [
        {**base_rows[index % len(base_rows)], "Patient ID": f"SYN{index:04}"}
        for index in range(992)
    ]
    payloads = {
        "subject-info.csv": csv_bytes(rows),
        "subject-info_codes.csv": _table_bytes(
            [
                ["Variable", "Column", "Notes", "Codes"],
                ["Exit of the study", "D", "Fictional clasificación", "0: survivor"],
                ["", "", "", "1: lost to follow-up"],
                ["", "", "", "2: cardiac transplantation"],
                ["", "", "", "3: death"],
                ["Cause of death", "E", "Fictional codes", "0: survivor"],
                ["", "", "", "1: non-cardiac death"],
                ["", "", "", "3: SCD"],
                ["", "", "", "6-7: pump-failure death"],
                ["HF etiology - Diagnosis", "N", "Fictional category", "1: example"],
            ],
            encoding="cp1252",
        ),
        "subject-info_definitions.csv": _table_bytes(
            [
                ["Variable ", "Column", "Definition"],
                ["NYHA class", "K", "Fictional definition: 예제"],
            ],
            encoding="utf-8-sig",
        ),
    }
    for name, content in payloads.items():
        (raw / name).write_bytes(content)
    monkeypatch.setattr(
        music,
        "SOURCE_FILES",
        {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()},
    )

    def reject_external_effect(*args, **kwargs):
        raise AssertionError("Personal practice must not contact a server or save a formal run")

    monkeypatch.setattr("urllib.request.urlopen", reject_external_effect)
    monkeypatch.setattr("requests.sessions.Session.request", reject_external_effect)
    monkeypatch.setattr("sixletters.experiment.Experiment.save", reject_external_effect)
    bundle = music.prepare_music(root, offline=True)
    dataset, _ = load_prepared_dataset(bundle)
    original_data = _snapshot(root / "Global/data")
    features = list(dataset.feature_groups["clinical"])
    expected = {
        split: dataset.frame.loc[dataset.frame.split.eq(split)].set_index("patient_id")
        for split in ("train", "validation", "test")
    }
    training_calls, prediction_calls = [], []
    original_fit_transform = ColumnTransformer.fit_transform
    original_predict_proba = Pipeline.predict_proba

    def checked_fit_transform(self, X, y=None, **params):
        pd.testing.assert_frame_equal(X, expected["train"][features])
        np.testing.assert_array_equal(y, expected["train"].target.astype(int))
        training_calls.append(tuple(X.index))
        return original_fit_transform(self, X, y, **params)

    def checked_predict_proba(self, X, **params):
        pd.testing.assert_frame_equal(X, expected["validation"][features])
        prediction_calls.append(tuple(X.index))
        return original_predict_proba(self, X, **params)

    monkeypatch.setattr(ColumnTransformer, "fit_transform", checked_fit_transform)
    monkeypatch.setattr(Pipeline, "predict_proba", checked_predict_proba)
    monkeypatch.setattr("IPython.display.display", lambda *args, **kwargs: None)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    notebook_path = init_practice("실습 팀원", root=root / "Members")
    monkeypatch.chdir(notebook_path.parent)
    notebook = nbformat.read(notebook_path, as_version=4)
    namespace = {"__name__": "__main__"}
    with matplotlib.rc_context():
        try:
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    exec(
                        compile(cell.source, f"{notebook_path.name}:cell-{index}", "exec"),
                        namespace,
                    )
        finally:
            plt.close("all")

    assert len(training_calls) == len(prediction_calls) == 3
    assert _snapshot(root / "Global/data") == original_data
    assert namespace["subject"].shape == (992, len(music.EXPECTED_HEADERS))
    assert "clasificación" in namespace["codes"].Notes.iloc[0]
    assert "예제" in namespace["definitions"].Definition.iloc[0]
    assert namespace["codes"]["Variable"].notna().all()

    output = root / "Global/artifacts/실습 팀원/practice/clinical"
    assert {path.name for path in output.iterdir()} == {
        "raw_column_summary.csv",
        "missing_values.png",
        "training_class_balance.png",
        "validation_roc_pr.png",
        "validation_confusion_matrices.png",
        "validation_metrics.csv",
        "split_summary.csv",
        "practice_summary.json",
    }
    metrics = pd.read_csv(output / "validation_metrics.csv", index_col="model")
    assert set(metrics.index) == {"Dummy prior", "Logistic", "Logistic balanced"}
    assert {
        "accuracy",
        "recall",
        "precision",
        "roc_auc",
        "average_precision",
        "pr_auc_trapezoid",
    }.issubset(metrics.columns)
    assert metrics.loc["Dummy prior", "roc_auc"] == 0.5
    assert np.isclose(
        metrics.loc["Dummy prior", "average_precision"], expected["validation"].target.mean()
    )
    assert (
        metrics.loc["Dummy prior", "pr_auc_trapezoid"]
        != metrics.loc["Dummy prior", "average_precision"]
    )
    summary = json.loads((output / "practice_summary.json").read_text(encoding="utf-8"))
    assert summary["owner"] == "실습 팀원"
    assert summary["test_scored"] is False
    assert summary["mlflow_uploaded"] is False
    assert summary["dataset_fingerprint"] == dataset.fingerprint()
    assert summary["model_settings"]["Logistic"]["class_weight"] is None
    assert summary["model_settings"]["Logistic balanced"]["class_weight"] == "balanced"
    for path in output.iterdir():
        assert path.stat().st_size > 0
        if path.suffix in {".csv", ".json"}:
            contents = path.read_text(encoding="utf-8")
            assert not any(patient_id in contents for patient_id in dataset.frame.patient_id)
    assert all(
        path.is_relative_to(output)
        for path in (root / "Global/artifacts").rglob("*")
        if path.is_file()
    )
