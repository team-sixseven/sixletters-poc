"""Train configured clinical models and persist validation-selected run artifacts."""

from __future__ import annotations

import importlib.metadata
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm
from xgboost import callback

from ..config import (
    load_experiment_config,
    load_yaml,
    resolve_config_path,
    resolve_data_path,
)
from ..data import clinical_feature_columns, load_cohorts, write_audit
from ..evaluation.core import binary_metrics, validate_threshold
from .artifacts import configured_model_run, save_model_path
from .feature_versions import get_feature_columns
from .models import (
    PREPROCESSING_DEFAULTS,
    build_estimator,
    make_preprocessor,
    prepare_categorical_frame,
)

SELECTION_METRIC = "average_precision"


def _run_id(name: str, override: str | None) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.lower()).strip("_.-")
    value = override or f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{slug}_{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"Invalid run_id {value!r}; use letters, digits, '.', '_' or '-'")
    return value


def _positive_probability(estimator, x: pd.DataFrame | np.ndarray) -> np.ndarray:
    classes = np.asarray(estimator.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError(f"Expected exactly one positive class 1, got {classes.tolist()}")
    probability = np.asarray(estimator.predict_proba(x), dtype=np.float64)[:, positive[0]]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Estimator returned invalid positive-class probabilities")
    return probability


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _xgboost_average_precision(y_true: np.ndarray, y_probability: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_probability))


class _CatBoostAveragePrecision:
    """CatBoost metric matching the validation selection metric exactly."""

    def is_max_optimal(self) -> bool:
        return True

    def evaluate(self, approxes, target, weight):
        logits = np.asarray(approxes[0], dtype=np.float64)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        return float(average_precision_score(np.asarray(target, dtype=int), probability)), 1.0

    def get_final_error(self, error: float, weight: float) -> float:
        return error / weight

    def get_description(self) -> str:
        return "AveragePrecision"


def _history_frame(estimator, name: str) -> pd.DataFrame:
    if name == "XGBoost":
        results = estimator.evals_result()
        metrics = results["validation_0"]
        dataset_name = "validation_0"
    else:
        results = estimator.get_evals_result()
        metrics = results["validation"]
        dataset_name = "validation"
    matching = [
        metric for metric in metrics if "averageprecision" in metric.lower().replace("_", "")
    ]
    if len(matching) != 1:
        raise ValueError(f"Expected one validation Average Precision history, got {list(metrics)}")
    metric_name = matching[0]
    if name == "XGBoost":
        return pd.DataFrame(
            {
                "iteration": np.arange(len(results[dataset_name][metric_name])),
                "validation_metric": results[dataset_name][metric_name],
                "metric_name": metric_name,
            }
        )
    return pd.DataFrame(
        {
            "iteration": np.arange(len(results[dataset_name][metric_name])),
            "validation_metric": results[dataset_name][metric_name],
            "metric_name": metric_name,
        }
    )


def _limit_train(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame
    if limit < 2:
        raise ValueError("limit_train_patients must be at least 2")
    groups = [group for _, group in frame.groupby("y_true", sort=True)]
    if len(groups) != 2 or limit < 2:
        raise ValueError("A limited ML smoke fit requires both classes")
    counts = [max(1, round(limit * len(group) / len(frame))) for group in groups]
    while sum(counts) > limit:
        index = max(range(2), key=lambda i: counts[i])
        if counts[index] <= 1:
            raise ValueError("Cannot preserve both classes in requested ML subset")
        counts[index] -= 1
    while sum(counts) < limit:
        index = max(range(2), key=lambda i: len(groups[i]) - counts[i])
        counts[index] += 1
    rng = np.random.default_rng(seed)
    selected = [
        group.iloc[rng.choice(len(group), size=count, replace=False)]
        for group, count in zip(groups, counts, strict=True)
    ]
    return pd.concat(selected).sort_values("patient_id").reset_index(drop=True)


def train_clinical(
    choice_path: Path,
    *,
    model_names: list[str] | None = None,
    run_id: str | None = None,
    limit_train_patients: int | None = None,
    model_parameters: dict[str, Any] | None = None,
    reuse_saved: bool = True,
    model_version: int | None = None,
    output_directory: Path | None = None,
    mlflow_logging: bool = False,
) -> list[Path]:
    """Reuse configured saved runs, or fit YAML parameters; Optuna uses reuse_saved=False."""
    choice, basic, choice_path, basic_path = load_experiment_config(choice_path)
    selected_names = model_names or choice["models"]
    unknown = sorted(set(selected_names) - set(choice["model_configs"]))
    if unknown:
        raise ValueError(
            f"Unsupported ML models {unknown}; configured: {sorted(choice['model_configs'])}"
        )
    if not selected_names:
        raise ValueError("ML model_choice.yaml models must select at least one model")
    if model_parameters is not None and len(selected_names) != 1:
        raise ValueError("model_parameters requires exactly one selected ML model")
    if mlflow_logging:
        from ..tracking.check_connection import check_mlflow_connection

        check_mlflow_connection()
    selected_features = None if model_version is None else get_feature_columns(model_version)
    runs = {}
    pending = []
    for model_name in selected_names:
        parameter_path = resolve_config_path(choice["model_configs"][model_name], choice_path)
        if reuse_saved:
            saved_run = configured_model_run(
                parameter_path, model_name, model_version=model_version
            )
            if saved_run is not None:
                runs[model_name] = saved_run
                print(
                    json.dumps(
                        {
                            "model": model_name,
                            "source": "saved",
                            "run_dir": str(saved_run),
                        }
                    )
                )
                continue
        pending.append((model_name, parameter_path, load_yaml(parameter_path)))
    if not pending:
        return _finish_runs([runs[name] for name in selected_names], mlflow_logging)
    clinical, cohorts, audit, _ = load_cohorts(basic, basic_path)
    train = cohorts["ml"]["train"]
    validation = cohorts["ml"]["validation"]
    if train.empty or validation.empty:
        raise ValueError("ML requires non-empty train and validation intersections")
    available_features = clinical_feature_columns(
        clinical, "patient_id", basic["data"]["clinical"]["label_column"]
    )
    feature_names = available_features if selected_features is None else selected_features
    missing_features = sorted(set(feature_names) - set(available_features))
    if missing_features:
        raise ValueError(
            f"model_version={model_version}: missing clinical columns {missing_features}"
        )
    forbidden = {
        basic["data"]["clinical"]["id_column"],
        basic["data"]["clinical"]["label_column"],
        basic["data"]["split"]["id_column"],
        basic["data"]["split"]["split_column"],
    }
    if forbidden.intersection(feature_names):
        raise ValueError(
            f"Clinical CSV input columns contain IDs/target/split: {forbidden & set(feature_names)}"
        )
    categorical = [
        name for name in basic["data"]["clinical"]["categorical_features"] if name in feature_names
    ]
    output_root = (
        output_directory.resolve()
        if output_directory is not None
        else resolve_config_path(choice["model_output_dir"], choice_path)
    )
    for model_name, parameter_path, parameters in tqdm(pending, desc="ML models", unit="model"):
        for key in ("model", "classification"):
            if key not in parameters or not isinstance(parameters[key], dict):
                raise ValueError(f"{parameter_path}: selected model requires a {key} mapping")
        if model_parameters is not None:
            parameters["model"].update(model_parameters)
            if model_name == "CatBoost":
                # Remove incompatible YAML defaults when a trial changes the sampling mode.
                if model_parameters.get("bootstrap_type") == "Bayesian":
                    parameters["model"].pop("subsample", None)
                elif model_parameters.get("bootstrap_type") == "Bernoulli":
                    parameters["model"].pop("bagging_temperature", None)
                if "auto_class_weights" in model_parameters:
                    parameters["model"].pop("class_weights", None)
                    parameters["model"].pop("scale_pos_weight", None)
        training = parameters.get("training", {})
        if not isinstance(training, dict):
            raise ValueError(f"{parameter_path}: training must be a mapping when provided")
        threshold = validate_threshold(parameters["classification"].get("threshold"))
        model_params = dict(parameters["model"])
        fit_train = _limit_train(train, limit_train_patients, int(training.get("seed", 42)))
        uses_early_stopping = training.get(
            "early_stopping",
            model_name in {"XGBoost", "CatBoost"},
        )
        early_stopping = training.get("early_stopping_rounds") if uses_early_stopping else None
        if uses_early_stopping and (not isinstance(early_stopping, int) or early_stopping < 1):
            raise ValueError(
                f"{parameter_path}: training.early_stopping_rounds must be a positive integer"
            )
        if model_name in {"XGBoost", "CatBoost"} and not uses_early_stopping:
            raise ValueError(f"{parameter_path}: {model_name} requires validation early stopping")
        name_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.lower()).strip("_.-")
        if model_version is not None:
            name_slug = f"{name_slug}_v{model_version}"
        this_run_id = _run_id(name_slug, f"{run_id}_{name_slug}" if run_id else None)
        run_dir = output_root / this_run_id
        if run_dir.exists():
            raise FileExistsError(
                f"ML model run already exists and will not be overwritten: {run_dir}"
            )
        run_dir.mkdir(parents=True)
        (run_dir / "run_status.json").write_text(
            json.dumps(
                {"status": "RUNNING", "model": model_name, "run_id": this_run_id},
                indent=2,
            ),
            encoding="utf-8",
        )
        write_audit(audit, run_dir / "data_audit")

        x_train = fit_train[feature_names].copy()
        x_val = validation[feature_names].copy()
        x_train = prepare_categorical_frame(x_train, categorical)
        x_val = prepare_categorical_frame(x_val, categorical)
        for split_name, features in (("train", x_train), ("validation", x_val)):
            if features.isna().any().any():
                raise ValueError(f"ML {split_name} features contain missing values")
        y_train = fit_train.y_true.astype(int).to_numpy()
        y_val = validation.y_true.astype(int).to_numpy()
        if len(np.unique(y_val)) != 2:
            raise ValueError("Validation split must contain both classes for model selection")
        estimator = build_estimator(model_name, model_params)
        # Capture serializable settings before adding runtime metric/callback objects.
        effective_model_parameters = estimator.get_params(deep=False)
        if model_name in {"XGBoost", "Random Forest"}:
            preprocessor = make_preprocessor(feature_names, categorical)
            train_matrix = preprocessor.fit_transform(x_train)
            val_matrix = preprocessor.transform(x_val)
            joblib.dump(preprocessor, run_dir / "preprocessing.joblib")
            if model_name == "XGBoost":
                estimator.set_params(
                    eval_metric=_xgboost_average_precision,
                    early_stopping_rounds=early_stopping,
                    callbacks=[
                        callback.EarlyStopping(
                            rounds=early_stopping,
                            metric_name="_xgboost_average_precision",
                            data_name="validation_0",
                            maximize=True,
                            save_best=True,
                        )
                    ],
                )
                estimator.fit(train_matrix, y_train, eval_set=[(val_matrix, y_val)], verbose=False)
                best_iteration = int(estimator.best_iteration)
                estimator.save_model(run_dir / "best_model.json")
                model_file = "best_model.json"
                history = _history_frame(estimator, model_name)
            else:
                estimator.fit(train_matrix, y_train)
                best_iteration = None
                joblib.dump(estimator, run_dir / "best_model.joblib")
                model_file = "best_model.joblib"
                val_probability = _positive_probability(estimator, val_matrix)
                history = pd.DataFrame(
                    {
                        "iteration": [None],
                        "validation_metric": [
                            float(average_precision_score(y_val, val_probability))
                        ],
                        "metric_name": ["average_precision"],
                    }
                )
            if model_name == "XGBoost":
                val_probability = _positive_probability(estimator, val_matrix)
            feature_output_names = preprocessor.get_feature_names_out().tolist()
        elif model_name == "CatBoost":
            train_frame = x_train
            val_frame = x_val
            estimator.set_params(eval_metric=_CatBoostAveragePrecision())
            estimator.fit(
                train_frame,
                y_train,
                cat_features=categorical,
                eval_set=(val_frame, y_val),
                early_stopping_rounds=early_stopping,
                use_best_model=True,
                verbose=False,
            )
            best_iteration = int(estimator.get_best_iteration())
            estimator.save_model(run_dir / "best_model.cbm")
            model_file = "best_model.cbm"
            val_probability = _positive_probability(estimator, val_frame)
            history = _history_frame(estimator, model_name)
            feature_output_names = feature_names
        validation_metrics = binary_metrics(y_val, val_probability, threshold)
        validation_metrics["selection_metric"] = float(
            average_precision_score(y_val, val_probability)
        )
        validation_metrics["threshold_source"] = "selected model parameter YAML"
        history.to_csv(run_dir / "training_history.csv", index=False)
        pd.DataFrame(
            {
                "patient_id": validation.patient_id.astype(str),
                "split": "validation",
                "y_true": y_val,
                "p_positive": val_probability,
                "threshold": threshold,
                "y_pred": (val_probability >= threshold).astype(int),
                "model": model_name,
                "model_version": model_version,
                "run_id": this_run_id,
            }
        ).to_csv(run_dir / "validation_predictions.csv", index=False)
        resolved = {
            "model_choice": choice,
            "basic": basic,
            "resolved_data_paths": {
                name: str(resolve_data_path(section["path"], basic_path))
                for name, section in basic["data"].items()
                if name in {"clinical", "split"}
            },
            "ecg_directory": str(resolve_data_path(basic["data"]["ecg"]["directory"], basic_path)),
            "selected_model": model_name,
            "model_version": model_version,
            "parameter_file": str(parameter_path),
            "parameters": parameters,
            "effective_preprocessing": PREPROCESSING_DEFAULTS,
            "effective_model_parameters": effective_model_parameters,
            "effective_training_settings": {
                "seed": int(training.get("seed", 42)),
                "early_stopping": bool(uses_early_stopping),
                "early_stopping_rounds": early_stopping,
                "selection_metric": SELECTION_METRIC,
                "selection_direction": "maximize",
            },
            "smoke_train_patient_limit": limit_train_patients,
            "actual_train_patients_used": len(fit_train),
            "feature_names": feature_names,
            "categorical_features": categorical,
            "mlflow_logging": mlflow_logging,
        }
        _write_yaml(run_dir / "resolved_config.yaml", resolved)
        metadata = {
            "model": model_name,
            "model_version": model_version,
            "run_id": this_run_id,
            "target_column": basic["data"]["clinical"]["label_column"],
            "target_semantics": basic["data"]["clinical"].get("target_semantics"),
            "artifact_type": {
                "XGBoost": "xgboost",
                "CatBoost": "catboost",
                "Random Forest": "random_forest",
            }[model_name],
            "model_file": model_file,
            "feature_names": feature_names,
            "model_feature_names": feature_output_names,
            "categorical_features": categorical,
            "class_mapping": {"negative": 0, "positive": 1},
            "positive_class_index": int(np.flatnonzero(np.asarray(estimator.classes_) == 1)[0]),
            "classification_threshold": threshold,
            "best_iteration": best_iteration,
            "selection_metric": f"validation_{SELECTION_METRIC}",
            "preprocessing": PREPROCESSING_DEFAULTS,
            "data_audit_counts": audit["counts"],
            "package_versions": {
                package: importlib.metadata.version(package)
                for package in (
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "xgboost",
                    "catboost",
                )
            },
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps({"validation": validation_metrics}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "run_status.json").write_text(
            json.dumps(
                {"status": "COMPLETE", "model": model_name, "run_id": this_run_id},
                indent=2,
            ),
            encoding="utf-8",
        )
        if reuse_saved:
            save_model_path(parameter_path, run_dir)
        runs[model_name] = run_dir
        if mlflow_logging:
            # Persist this run remotely before starting the next selected algorithm.
            from ..tracking.log_best import finish_ml_training

            finish_ml_training(run_dir, mlflow_logging=True)
    # New runs were logged above; saved runs also need account-specific registration.
    saved = [runs[name] for name in selected_names if name not in {row[0] for row in pending}]
    _finish_runs(saved, mlflow_logging)
    return [runs[name] for name in selected_names]


def _finish_runs(runs: list[Path], mlflow_logging: bool) -> list[Path]:
    if mlflow_logging:
        from ..tracking.log_best import finish_ml_training

        for run in runs:
            finish_ml_training(run, mlflow_logging=True)
    return runs
