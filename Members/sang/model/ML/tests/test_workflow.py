"""Synthetic clinical fits with account-aware tracking and notebook evaluation."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import nbformat
import numpy as np
import optuna
import pandas as pd
import pytest
import yaml
from Members.sang.model.config import load_basic_config, load_yaml
from Members.sang.model.evaluation.evaluation import evaluate_saved_model
from Members.sang.model.ML.optuna_experiment import ClinicalOptunaExperiment
from Members.sang.model.ML.training import train_clinical

from sixletters.local_settings import connection, write_settings

MODEL = Path(__file__).resolve().parents[2]


@pytest.fixture
def clinical_setup(tmp_path, monkeypatch):
    import Members.sang.model.evaluation.evaluation as evaluation
    import Members.sang.model.ML.training as training
    import Members.sang.model.tracking.check_connection as preflight
    import Members.sang.model.tracking.log_best as tracking
    import Members.sang.model.tracking.log_evaluation as evaluation_tracking

    basic = load_basic_config(MODEL / "basic.yaml")
    basic_path = tmp_path / "basic.yaml"
    basic_path.write_text(yaml.safe_dump(basic))
    clinical = pd.DataFrame(
        {
            "patient_id": [f"synthetic-{i:02}" for i in range(16)],
            "SCD": [i % 2 for i in range(16)],
            "Gender (male=1)": [i % 2 for i in range(16)],
            "Age": np.arange(16) + 50,
            "LVEF (%)": 55 - np.arange(16),
            "NYHA class": [i % 3 + 1 for i in range(16)],
        }
    )
    cohorts = {}
    for split, rows in (
        ("train", slice(0, 8)),
        ("validation", slice(8, 12)),
        ("test", slice(12, 16)),
    ):
        cohorts[split] = clinical.iloc[rows].copy().assign(split=split)
        cohorts[split]["y_true"] = cohorts[split].SCD
    audit = {
        "counts": {"clinical_patients": 16},
        "exclusions": [],
        "available_patient_ids": {
            name: {s: f.patient_id.tolist() if name == "ml" else [] for s, f in cohorts.items()}
            for name in ("ml", "dl", "fusion")
        },
    }

    def load(*_):
        return clinical, {"ml": cohorts, "dl": cohorts}, audit, {}

    monkeypatch.setattr(training, "load_cohorts", load)
    monkeypatch.setattr(evaluation, "load_cohorts", load)
    models = {
        "XGBoost": {"n_estimators": 4, "max_depth": 2, "n_jobs": 1, "random_state": 42},
        "CatBoost": {
            "iterations": 4,
            "depth": 2,
            "thread_count": 1,
            "random_seed": 42,
            "allow_writing_files": False,
            "verbose": False,
        },
        "Random Forest": {
            "n_estimators": 4,
            "max_depth": 2,
            "n_jobs": 1,
            "random_state": 42,
        },
    }
    config_paths = {}
    for i, (name, parameters) in enumerate(models.items()):
        parameter_path = tmp_path / f"parameters_{i}.yaml"
        parameter_path.write_text(
            yaml.safe_dump(
                {
                    "model": parameters,
                    "training": {
                        "seed": 42,
                        "early_stopping_rounds": 2,
                        "early_stopping": name != "Random Forest",
                    },
                    "classification": {"threshold": 0.5},
                }
            )
        )
        config_paths[name] = str(parameter_path)
    choice_path = tmp_path / "choice.yaml"
    choice_path.write_text(
        yaml.safe_dump(
            {
                "basic_config": str(basic_path),
                "models": list(models),
                "model_configs": config_paths,
                "model_output_dir": str(tmp_path / "models"),
            }
        )
    )
    for key in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_EXPERIMENT_NAME",
    ):
        monkeypatch.delenv(key, raising=False)
    write_settings(
        tmp_path,
        {
            "owner": "alice",
            "username": "alice",
            "password": "private-test-secret",
            "uri": "http://tracking.invalid:5000",
            "experiment": "synthetic-ml",
        },
    )
    monkeypatch.setattr(tracking, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(evaluation_tracking, "REPO_ROOT", tmp_path)
    checks, uploaded = [], []
    monkeypatch.setattr(preflight, "check_mlflow_connection", lambda: checks.append("alice"))
    uploader = importlib.import_module("sixletters.upload")

    def upload(root, directory, model=None):
        assert connection(root)["username"] == "alice"
        assert directory.is_relative_to(root / "Members/alice")
        _, params, files = uploader.bundle(directory, model)
        text = "\n".join(
            path.read_text() for path in files.values() if path.suffix in {".json", ".yaml", ".csv"}
        )
        assert "private-test-secret" not in text and "synthetic-00" not in text
        assert all("predictions" not in name and "patients" not in name for name in files)
        if model is not None:
            assert any(name.startswith("models/best_model") for name in files)
        run_id = directory.name
        uploaded.append(
            {
                "run_id": run_id,
                "params": params,
                "hashes": {name: uploader.digest(path) for name, path in files.items()},
            }
        )
        return {"run_id": run_id, "url": f"http://tracking.invalid/{run_id}"}

    monkeypatch.setattr(tracking, "upload", upload)
    monkeypatch.setattr(uploader, "upload", upload)
    return tmp_path, choice_path, basic_path, checks, uploaded


def evaluate(run, setup):
    root, _, basic_path, _, _ = setup
    config = {
        "basic_config": str(basic_path),
        "evaluation": {
            "target": "ml",
            "artifact_dir": str(run),
            "split": "test",
            "mlflow_logging": True,
            "threshold_override": None,
            "results_directory": str(root / "results"),
            "metrics": ["auroc", "average_precision"],
        },
    }
    path = root / "eval.yaml"
    path.write_text(yaml.safe_dump(config))
    result = evaluate_saved_model(path)
    assert len(pd.read_csv(result / "predictions.csv")) == 4
    assert json.loads((result / "mlflow_status.json").read_text())["status"] == "FINISHED"
    return result


@pytest.mark.parametrize("model", ["XGBoost", "CatBoost", "Random Forest"])
def test_ordinary_training_upload_reuse_and_final_evaluation(clinical_setup, model):
    _, choice, _, checks, uploaded = clinical_setup
    run = train_clinical(choice, model_names=[model], model_version=1, mlflow_logging=True)[0]
    assert checks == ["alice"] and len(uploaded) == 1
    first = uploaded[0]
    assert first["params"]["stage"] == "training"
    assert json.loads((run / "mlflow_status.json").read_text())["status"] == "FINISHED"
    reused = train_clinical(choice, model_names=[model], model_version=1, mlflow_logging=True)[0]
    assert reused == run and uploaded[1] == first
    evaluate(run, clinical_setup)
    assert uploaded[-1]["params"]["source_mlflow_run_id"] == first["run_id"]


def test_optuna_only_uploads_selected_best_then_evaluates(clinical_setup):
    _, choice, _, checks, uploaded = clinical_setup
    with ClinicalOptunaExperiment(choice, "Random Forest", 1, mlflow_logging=True) as experiment:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: experiment.run_trial(
                trial, {"n_estimators": trial.suggest_int("n_estimators", 2, 3)}
            ),
            n_trials=2,
        )
        assert not uploaded  # Intermediate trials stay offline.
        run = experiment.save_best(study)
    assert checks == ["alice"] and len(uploaded) == 1
    assert uploaded[0]["params"]["n_trials"] == "2"
    evaluate(run, clinical_setup)
    assert uploaded[-1]["params"]["source_mlflow_run_id"] == uploaded[0]["run_id"]


def test_failed_upload_preserves_model_and_retries_without_fit(clinical_setup, monkeypatch):
    import Members.sang.model.tracking.log_best as tracking

    _, choice, _, _, uploaded = clinical_setup
    original = tracking.upload

    def fail(*_args, **_kwargs):
        raise ConnectionError("private-test-secret")

    monkeypatch.setattr(tracking, "upload", fail)
    with pytest.warns(RuntimeWarning, match="전송 미완료"):
        run = train_clinical(
            choice, model_names=["Random Forest"], model_version=1, mlflow_logging=True
        )[0]
    assert json.loads((run / "run_status.json").read_text())["status"] == "COMPLETE"
    assert json.loads((run / "mlflow_status.json").read_text()) == {
        "status": "PENDING",
        "error_type": "ConnectionError",
    }
    monkeypatch.setattr(tracking, "upload", original)
    tracking.log_ml_training_to_mlflow(run)
    assert len(uploaded) == 1
    assert json.loads((run / "mlflow_status.json").read_text())["status"] == "FINISHED"


@pytest.mark.parametrize("mode", ["yaml", "optuna"])
def test_ml_notebook_selected_training_path_and_final_evaluation(clinical_setup, monkeypatch, mode):
    import Members.sang.model.evaluation.evaluation as evaluation
    import Members.sang.model.ML.optuna_experiment as experiments
    import Members.sang.model.ML.training as training

    root, choice, _, _, _ = clinical_setup
    run = train_clinical(choice, model_names=["Random Forest"], model_version=1, reuse_saved=False)[
        0
    ]
    (run / "mlflow_status.json").write_text('{"status":"FINISHED"}')
    (run / "summary.json").write_text(
        json.dumps({"mlflow": {"status": "FINISHED", "url": "http://tracking.invalid/run"}})
    )
    pd.DataFrame({"trial": [0]}).to_csv(run / "trials.csv", index=False)
    trained, evaluated, optimized = [], [], []

    def fake_train(*args, **kwargs):
        assert kwargs["mlflow_logging"] is True
        trained.append("yaml")
        return [run]

    class FakeExperiment:
        def __init__(self, *args, **kwargs):
            assert kwargs["mlflow_logging"] is True
            self.parameter_path = choice
            self.best_directory = run

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def run_trial(self, trial, parameters):
            optimized.append(trial.number)
            return 0.5

        def save_best(self, study):
            return run

    def fake_evaluation(path):
        settings = load_yaml(path)["evaluation"]
        assert settings["target"] == "ml" and settings["split"] == "test"
        assert settings["artifact_dir"] == str(run)
        assert settings["mlflow_logging"] is True and settings["threshold_override"] is None
        evaluated.append(run)
        result = root / "notebook_eval"
        result.mkdir()
        (result / "metrics.json").write_text('{"split":"test"}')
        (result / "mlflow_status.json").write_text('{"status":"FINISHED"}')
        pd.DataFrame({"p_positive": [0.5]}).to_csv(result / "predictions.csv", index=False)
        return result

    monkeypatch.setattr(training, "train_clinical", fake_train)
    monkeypatch.setattr(experiments, "ClinicalOptunaExperiment", FakeExperiment)
    monkeypatch.setattr(evaluation, "evaluate_saved_model", fake_evaluation)
    notebook = nbformat.read(MODEL / "ML/model_experiment_only_ML.ipynb", as_version=4)
    namespace = {"display": lambda *_: None}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            source = cell.source.replace(
                'training_mode = "optuna"', f'training_mode = "{mode}"'
            ).replace("n_trials = 100", "n_trials = 2")
            exec(compile(source, "clinical notebook", "exec"), namespace)
    assert evaluated == [run]
    assert trained == (["yaml"] if mode == "yaml" else [])
    assert len(optimized) == (2 if mode == "optuna" else 0)
