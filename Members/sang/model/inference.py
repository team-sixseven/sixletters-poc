"""Load saved ML/DL runs and produce patient-aligned positive-class probabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from catboost import CatBoostClassifier
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from .data import load_cohorts
from .DL.composed import build_composed_model_from_specs
from .DL.dataloader.dataset import PatientWindowDataset
from .ML.models import prepare_categorical_frame


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
    if metadata["artifact_type"] in {"xgboost", "random_forest"}:
        x = preprocessor.transform(x)
    else:
        x = prepare_categorical_frame(
            x,
            metadata["categorical_features"],
            metadata["preprocessing"]["categorical_missing_value"],
        )
    positive_index = int(metadata["positive_class_index"])
    probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)[:, positive_index]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{artifact_dir}: invalid positive class probabilities")
    return probabilities, metadata


def load_dl_run(artifact_dir: Path, device: torch.device | None = None):
    artifact_dir = artifact_dir.resolve()
    metadata = _metadata(artifact_dir)
    resolved = yaml.safe_load((artifact_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    choice, basic = resolved["choice"], resolved["basic"]
    model = build_composed_model_from_specs(
        choice, basic, resolved["cnn_spec"], resolved["sequence_spec"]
    )
    checkpoint = torch.load(artifact_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    if int(checkpoint["best_epoch"]) != int(metadata["best_epoch"]):
        raise ValueError(f"{artifact_dir}: checkpoint and metadata best epoch disagree")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, metadata, resolved, device


@torch.inference_mode()
def predict_dl_run(
    artifact_dir: Path,
    patients: pd.DataFrame,
    window_indices: dict[str, np.ndarray],
    *,
    device: torch.device | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    model, metadata, resolved, device = load_dl_run(artifact_dir, device)
    dataset = PatientWindowDataset(patients, window_indices, resolved["basic"]["data"]["ecg"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probabilities, predicted_ids = [], []
    for patient_ids, waveforms, _ in loader:
        logits = model(waveforms.to(device))
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise ValueError(f"DL model must return (batch,1) logits, got {tuple(logits.shape)}")
        scores = torch.sigmoid(logits[:, 0])
        if not torch.isfinite(scores).all():
            raise ValueError(f"{artifact_dir}: invalid DL probabilities")
        probabilities.extend(scores.cpu().numpy().tolist())
        predicted_ids.extend(patient_ids)
    expected_ids = patients.patient_id.astype(str).tolist()
    if predicted_ids != expected_ids:
        raise ValueError("DL prediction order no longer matches patient ID input order")
    return np.asarray(probabilities), metadata


def load_split_data(basic: dict[str, Any], basic_path: Path, split_name: str):
    """Return one split and validated metadata-aligned ECG row indices."""
    _, cohorts, audit, indices = load_cohorts(basic, basic_path)
    if split_name not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split {split_name!r}")
    return cohorts, audit, indices
