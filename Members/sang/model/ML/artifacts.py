"""Load native clinical model runs and resolve model paths stored in parameter YAML."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from ..config import load_yaml, resolve_config_path
from .feature_versions import get_feature_columns
from .models import prepare_categorical_frame


def configured_model_run(
    parameter_path: Path, model_name: str, *, model_version: int | None = None
) -> Path | None:
    """Return a complete, loadable run if the YAML model_path exists."""
    parameters = load_yaml(parameter_path)
    configured_path = parameters.get("model_path")
    if model_version is not None:
        get_feature_columns(model_version)
        version_paths = parameters.get("model_paths", {})
        if not isinstance(version_paths, dict):
            raise ValueError(f"{parameter_path}: model_paths must be a version-to-path mapping")
        configured_path = version_paths.get(
            model_version, version_paths.get(str(model_version), configured_path)
        )
    if configured_path is None or configured_path == "":
        return None
    if not isinstance(configured_path, str):
        raise ValueError(f"{parameter_path}: model_path must be a path string or null")
    model_path = resolve_config_path(configured_path, parameter_path)
    if not model_path.exists():
        return None
    run_dir = model_path if model_path.is_dir() else model_path.parent
    for filename in (
        "metadata.json",
        "resolved_config.yaml",
        "metrics.json",
        "validation_predictions.csv",
        "run_status.json",
    ):
        if not (run_dir / filename).is_file():
            raise FileNotFoundError(
                f"Existing model requires companion run file: {run_dir / filename}"
            )
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE":
        raise ValueError(f"Saved model run is not COMPLETE: {run_dir}")
    metadata = _metadata(run_dir)
    if metadata["model"] != model_name:
        raise ValueError(
            f"Expected {model_name}, but saved model is {metadata['model']}: {run_dir}"
        )
    if model_version is not None and (
        metadata.get("model_version") != model_version
        or metadata.get("feature_names") != get_feature_columns(model_version)
    ):
        return None
    load_ml_run(run_dir)
    native_model = (run_dir / metadata["model_file"]).resolve()
    if model_path.is_file() and model_path != native_model:
        raise ValueError(f"model_path must point to the native model file: {native_model}")
    return run_dir


def save_model_path(parameter_path: Path, run_dir: Path) -> Path:
    """Record a completed run's native model path while retaining YAML parameter values."""
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE":
        raise ValueError(f"Cannot register a model run that is not COMPLETE: {run_dir}")
    metadata = _metadata(run_dir)
    model_file = (run_dir / metadata["model_file"]).resolve()
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_file}")
    parameters = load_yaml(parameter_path)
    relative_path = Path(os.path.relpath(model_file, parameter_path.parent)).as_posix()
    model_version = metadata.get("model_version")
    if model_version is not None:
        if metadata.get("feature_names") != get_feature_columns(model_version):
            raise ValueError("Saved model input columns do not match its model_version")
        version_paths = parameters.setdefault("model_paths", {})
        if not isinstance(version_paths, dict):
            raise ValueError(f"{parameter_path}: model_paths must be a version-to-path mapping")
        version_paths.pop(str(model_version), None)
        version_paths[model_version] = relative_path
    parameters["model_path"] = relative_path
    temporary = parameter_path.with_name(f".{parameter_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(parameters, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        temporary.replace(parameter_path)
    finally:
        temporary.unlink(missing_ok=True)
    return model_file


def _metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model artifact metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_ml_run(artifact_dir: Path):
    artifact_dir = artifact_dir.resolve()
    metadata = _metadata(artifact_dir)
    if metadata.get("class_mapping") != {"negative": 0, "positive": 1}:
        raise ValueError(f"{artifact_dir}: unsupported ML class mapping")
    model_file = artifact_dir / metadata["model_file"]
    if metadata["artifact_type"] == "xgboost":
        model = XGBClassifier()
        model.load_model(model_file)
        preprocessing_path = artifact_dir / "preprocessing.joblib"
        if not preprocessing_path.is_file():
            raise FileNotFoundError(f"Missing saved XGBoost preprocessing: {preprocessing_path}")
        preprocessor = joblib.load(preprocessing_path)
    elif metadata["artifact_type"] == "random_forest":
        model = joblib.load(model_file)
        preprocessing_path = artifact_dir / "preprocessing.joblib"
        if not preprocessing_path.is_file():
            raise FileNotFoundError(
                f"Missing saved Random Forest preprocessing: {preprocessing_path}"
            )
        preprocessor = joblib.load(preprocessing_path)
    elif metadata["artifact_type"] == "catboost":
        model = CatBoostClassifier()
        model.load_model(model_file)
        preprocessor = None
    else:
        raise ValueError(f"Unsupported ML artifact_type {metadata.get('artifact_type')!r}")
    classes = np.asarray(model.classes_)
    if not np.array_equal(classes, [0, 1]):
        raise ValueError(f"{artifact_dir}: model class order must be [0,1], got {classes.tolist()}")
    return model, preprocessor, metadata


def predict_ml_run(artifact_dir: Path, patients: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    model, preprocessor, metadata = load_ml_run(artifact_dir)
    features = metadata["feature_names"]
    missing = sorted(set(features) - set(patients.columns))
    if missing:
        raise ValueError(f"ML inference input is missing saved features {missing}")
    x = patients[features].copy()
    missing_values = x.isna().sum()
    missing_values = missing_values[missing_values > 0].to_dict()
    if missing_values:
        raise ValueError(f"ML inference features contain missing values: {missing_values}")
    x = prepare_categorical_frame(x, metadata["categorical_features"])
    if metadata["artifact_type"] in {"xgboost", "random_forest"}:
        x = preprocessor.transform(x)
    positive_index = int(metadata["positive_class_index"])
    probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)[:, positive_index]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{artifact_dir}: invalid positive class probabilities")
    return probabilities, metadata
