"""Load saved ML/DL runs and produce patient-aligned positive-class probabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .data import load_cohorts
from .DL.composed import MODEL_FORMAT_VERSION, build_composed_model_from_specs
from .DL.dataloader.dataset import PatientWindowDataset, collate_patient_windows
from .ML.artifacts import load_ml_run as load_ml_run
from .ML.artifacts import predict_ml_run as predict_ml_run


def _metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model artifact metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_dl_run(artifact_dir: Path, device: torch.device | None = None):
    artifact_dir = artifact_dir.resolve()
    metadata = _metadata(artifact_dir)
    if metadata.get("model_format_version") != MODEL_FORMAT_VERSION:
        raise ValueError(
            "This fixed-lead DL checkpoint predates shared-lead format v2; use its original code or retrain. No automatic weight conversion is performed."
        )
    resolved = yaml.safe_load((artifact_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    choice, basic = resolved["choice"], resolved["basic"]
    model = build_composed_model_from_specs(
        choice,
        basic,
        resolved["cnn_spec"],
        resolved["sequence_spec"],
        resolved["training"],
    )
    checkpoint = torch.load(artifact_dir / "best_model.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Checkpoint format differs from shared-lead model format v2")
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
    dataset = PatientWindowDataset(
        patients,
        window_indices,
        resolved["basic"]["data"]["ecg"],
        int(resolved["training"]["max_segments_per_patient"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_patient_windows,
    )
    probabilities, predicted_ids = [], []
    for patient_ids, waveforms, _, mask, positions, lead_presence in loader:
        logits = model(
            waveforms.to(device),
            mask.to(device),
            positions.to(device),
            lead_presence.to(device),
        )
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
