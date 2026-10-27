"""Load patient-level ECG windows stored as one NPY array per patient."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch


def _read_split(path: Path) -> list[tuple[str, int]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"Patient ID", "SCD"}.issubset(rows[0]):
        raise ValueError(f"{path}: expected Patient ID and SCD columns")
    output = []
    seen = set()
    for row in rows:
        patient_id = row["Patient ID"].strip()
        if not patient_id or patient_id in seen:
            raise ValueError(f"{path}: patient IDs must be present and unique")
        seen.add(patient_id)
        try:
            label = int(row["SCD"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: SCD labels must be binary integers") from error
        if label not in (0, 1):
            raise ValueError(f"{path}: SCD labels must be binary integers")
        output.append((patient_id, label))
    if {label for _, label in output} != {0, 1}:
        raise ValueError(f"{path}: both SCD classes are required")
    return output


def load_patient_windows(
    path: Path,
    *,
    windows_per_patient: int,
    expected_leads: int,
    expected_samples: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3:
        raise ValueError(f"{path.name}: expected (window, lead, sample), got {array.shape}")
    if array.shape[1:] != (expected_leads, expected_samples):
        raise ValueError(
            f"{path.name}: expected (*,{expected_leads},{expected_samples}), got {array.shape}"
        )
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{path.name}: unsupported ECG dtype {array.dtype}")
    if array.shape[0] < windows_per_patient:
        raise ValueError(f"{path.name}: fewer windows than requested ({array.shape[0]})")

    indexes = np.linspace(0, array.shape[0] - 1, windows_per_patient, dtype=np.int64)
    raw = np.asarray(array[indexes], dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError(f"{path.name}: selected ECG windows contain NaN or Inf")

    means = raw.mean(axis=-1, keepdims=True)
    stds = raw.std(axis=-1, keepdims=True)
    constant_leads = stds.squeeze(-1) < 1e-8
    if np.any(constant_leads.all(axis=1)):
        raise ValueError(f"{path.name}: selected window has no varying ECG lead")
    safe_stds = np.where(stds < 1e-8, 1.0, stds)
    normalized = (raw - means) / safe_stds

    per_lead = np.stack(
        (
            raw.mean(axis=-1),
            raw.std(axis=-1),
            np.sqrt(np.mean(np.square(raw), axis=-1)),
            raw.max(axis=-1) - raw.min(axis=-1),
        ),
        axis=-1,
    ).reshape(windows_per_patient, -1)
    magnitude = np.linalg.norm(raw, axis=1)
    vector = np.stack(
        (
            magnitude.mean(axis=-1),
            magnitude.std(axis=-1),
            np.sqrt(np.mean(np.square(magnitude), axis=-1)),
            magnitude.max(axis=-1) - magnitude.min(axis=-1),
        ),
        axis=-1,
    )
    features = np.concatenate((per_lead, vector), axis=1).astype(np.float32)
    return (
        normalized.astype(np.float32, copy=False),
        features,
        int(array.shape[0]),
        int(constant_leads.sum()),
    )


def load_holter_windows(
    windows_dir: Path,
    splits_dir: Path,
    *,
    windows_per_patient: int = 8,
    leads: int = 3,
    samples_per_window: int = 2000,
    window_selection: str = "evenly_spaced_across_recording",
    normalization: str = "per_window_per_lead_zscore",
) -> tuple[dict, dict]:
    """Load selected windows for patients in the existing clinical splits.

    Window indices are sampled evenly from the full saved sequence to cover the
    recording rather than taking only its opening minutes. Missing ECG files are
    excluded and reported per split; patient-level labels come from split CSVs.
    """
    if windows_per_patient < 1:
        raise ValueError("windows_per_patient must be positive")
    if window_selection != "evenly_spaced_across_recording":
        raise ValueError(f"Unsupported window selection: {window_selection}")
    if normalization != "per_window_per_lead_zscore":
        raise ValueError(f"Unsupported waveform normalization: {normalization}")
    datasets = {}
    coverage = {}
    assigned_patients = set()
    for split in ("train", "val", "test"):
        rows = _read_split(splits_dir / f"{split}.csv")
        current_ids = {patient_id for patient_id, _ in rows}
        overlap = assigned_patients & current_ids
        if overlap:
            raise ValueError(f"Patient leakage between splits: {split} repeats an earlier patient")
        assigned_patients.update(current_ids)
        waveforms, features, labels, patient_ids, counts, missing = [], [], [], [], [], []
        constant_lead_count = 0
        for patient_id, label in rows:
            path = windows_dir / f"{patient_id}.npy"
            if not path.is_file():
                missing.append(patient_id)
                continue
            waveform, statistics, available_count, flat_leads = load_patient_windows(
                path,
                windows_per_patient=windows_per_patient,
                expected_leads=leads,
                expected_samples=samples_per_window,
            )
            waveforms.append(waveform)
            features.append(statistics)
            labels.append(label)
            patient_ids.append(patient_id)
            counts.append(available_count)
            constant_lead_count += flat_leads
        if not waveforms:
            raise RuntimeError(f"No ECG window files found for {split}")
        if len(set(labels)) != 2:
            raise RuntimeError(f"ECG-covered {split} must contain both SCD classes")
        datasets[split] = (
            torch.from_numpy(np.stack(waveforms)),
            torch.tensor(labels, dtype=torch.float32),
            patient_ids,
            torch.from_numpy(np.stack(features)),
        )
        coverage[split] = {
            "clinical_patients": len(rows),
            "ecg_patients": len(patient_ids),
            "missing_ecg": len(missing),
            "constant_selected_leads": constant_lead_count,
            "ecg_scd_positive": int(sum(labels)),
            "available_windows_min": int(min(counts)),
            "available_windows_median": int(np.median(counts)),
            "available_windows_max": int(max(counts)),
        }
    return datasets, coverage
