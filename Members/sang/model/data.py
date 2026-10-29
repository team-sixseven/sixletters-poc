"""Shared patient-ID based loading, validation, and cohort audit reporting."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_data_path


def clinical_feature_columns(frame: pd.DataFrame, id_column: str, label_column: str) -> list[str]:
    """Use CSV column order, excluding the patient identifier and target."""
    features = [name for name in frame.columns if name not in {id_column, label_column}]
    if not features:
        raise ValueError("Clinical CSV contains no input features after excluding ID and label")
    return features


def _read_clinical(basic: dict[str, Any], basic_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    config = basic["data"]["clinical"]
    path = resolve_data_path(config["path"], basic_path)
    if not path.is_file():
        raise FileNotFoundError(f"basic.yaml data.clinical.path -> {path} does not exist")
    frame = pd.read_csv(path, dtype={config["id_column"]: "string"})
    original_columns = list(frame.columns)
    if config.get("normalize_column_whitespace", False):
        frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"{path}: column-name normalization creates duplicates: {duplicated}")
    mapping = dict(zip(original_columns, frame.columns, strict=True))
    id_column, label_column = config["id_column"], config["label_column"]
    required = {id_column, label_column, *config["categorical_features"]}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing configured clinical columns {missing}")
    identifiers = frame[id_column].astype("string").str.strip()
    if identifiers.isna().any() or identifiers.eq("").any():
        raise ValueError(f"{path}: {id_column!r} contains an empty patient ID")
    frame[id_column] = identifiers
    duplicate_ids = frame.loc[frame[id_column].duplicated(keep=False), id_column].unique().tolist()
    if duplicate_ids:
        label_counts = frame.groupby(id_column, dropna=False)[label_column].nunique(dropna=False)
        collisions = label_counts[label_counts > 1].index.astype(str).tolist()
        raise ValueError(
            f"{path}: duplicate patient rows={duplicate_ids[:10]}; "
            f"label conflicts={collisions[:10]}"
        )
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{path}: {label_column!r} must contain only binary 0/1 labels")
    frame[label_column] = labels.astype("int64")
    features = clinical_feature_columns(frame, id_column, label_column)
    if not set(config["categorical_features"]).issubset(features):
        raise ValueError(f"{path}: categorical_features must contain only input feature columns")
    nonnumeric = [name for name in features if not pd.api.types.is_numeric_dtype(frame[name])]
    if nonnumeric:
        raise ValueError(f"{path}: configured features must be numeric: {nonnumeric}")
    missing_counts = frame[features].isna().sum()
    missing_counts = missing_counts[missing_counts > 0].to_dict()
    if missing_counts:
        raise ValueError(f"{path}: configured clinical features must be complete: {missing_counts}")
    return frame, mapping


def _read_splits(basic: dict[str, Any], basic_path: Path) -> pd.DataFrame:
    config = basic["data"]["split"]
    path = resolve_data_path(config["path"], basic_path)
    if not path.is_file():
        raise FileNotFoundError(f"basic.yaml data.split.path -> {path} does not exist")
    frame = pd.read_csv(path, dtype={config["id_column"]: "string"})
    required = {config["id_column"], config["split_column"]}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing split columns {missing}")
    frame = frame[[config["id_column"], config["split_column"]]].copy()
    frame[config["id_column"]] = frame[config["id_column"]].astype("string").str.strip()
    if frame[config["id_column"]].isna().any() or frame[config["id_column"]].eq("").any():
        raise ValueError(f"{path}: patient IDs must be non-empty")
    if frame[config["id_column"]].duplicated().any():
        duplicated = frame.loc[frame[config["id_column"]].duplicated(), config["id_column"]]
        raise ValueError(f"{path}: duplicate patient IDs {duplicated.head(10).tolist()}")
    values = config["values"]
    inverse = {value: name for name, value in values.items()}
    unknown = sorted(set(frame[config["split_column"]].dropna()) - set(inverse))
    if unknown or frame[config["split_column"]].isna().any():
        raise ValueError(f"{path}: unknown or empty split values {unknown}")
    frame["_split"] = frame[config["split_column"]].map(inverse)
    return frame.rename(columns={config["id_column"]: "patient_id"})[
        ["patient_id", "_split"]
    ].rename(columns={"_split": "split"})


def _ecg_inventory(
    basic: dict[str, Any], basic_path: Path
) -> tuple[dict[str, Path], dict[str, int]]:
    config = basic["data"]["ecg"]
    directory = resolve_data_path(config["directory"], basic_path)
    if not directory.is_dir():
        raise FileNotFoundError(f"basic.yaml data.ecg.directory -> {directory} does not exist")
    pattern = config["patient_id_regex"]
    files: dict[str, Path] = {}
    malformed = []
    for path in sorted(directory.glob("*.npy")):
        if path.name != config["filename_pattern"].format(pid=path.stem):
            malformed.append(path.name)
            continue
        match = re.fullmatch(pattern, path.stem)
        if not match:
            malformed.append(path.name)
            continue
        patient_id = match.group(1) if match.groups() else match.group(0)
        if patient_id in files:
            raise ValueError(f"{directory}: multiple NPY files map to patient {patient_id}")
        files[patient_id] = path
    if malformed:
        raise ValueError(
            f"{directory}: filenames do not match configured patient ID rule: {malformed[:10]}"
        )
    return files, {}


def _validate_ecg_headers(
    files: dict[str, Path], basic: dict[str, Any]
) -> tuple[dict[str, int], dict[str, str]]:
    counts: dict[str, int] = {}
    invalid: dict[str, str] = {}
    for patient_id, path in files.items():
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 3 or min(array.shape) < 1:
                raise ValueError(f"expected nonempty (windows,lead,sample), got {array.shape}")
            if array.dtype.kind not in "fiu":
                raise ValueError(f"unsupported dtype {array.dtype}")
            if array.shape[0] < 1:
                raise ValueError("no ECG windows available")
            counts[patient_id] = int(array.shape[0])
        except (OSError, ValueError, EOFError) as error:
            invalid[patient_id] = str(error)
    return counts, invalid


def _load_window_quality_index(
    basic: dict[str, Any], basic_path: Path, patient_ids: set[str]
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Read aligned sidecar rows in chunks and retain only SQI-passing NPY row indices."""
    if not patient_ids:
        return {}, {}
    config = basic["data"]["ecg"]
    metadata_path = resolve_data_path(config["metadata_path"], basic_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"basic.yaml data.ecg.metadata_path -> {metadata_path} does not exist"
        )
    stat = metadata_path.stat()
    directory = resolve_data_path(config["directory"], basic_path)
    durations = {}
    for pid in patient_ids:
        waveform_path = directory / config["filename_pattern"].format(pid=pid)
        if waveform_path.is_file():
            shape = np.load(waveform_path, mmap_mode="r", allow_pickle=False).shape
            if len(shape) == 3:
                durations[pid] = shape[-1] / config["sample_rate_hz"]
    # Revalidate timestamps if waveform sample lengths or the source rate change.
    signature = hashlib.sha256(json.dumps(durations, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = basic_path.parent / "outputs" / "cache"
    cache_path = (
        cache_dir / f"window_quality_index_{stat.st_size}_{stat.st_mtime_ns}_{signature}.npz"
    )
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            indices = {
                pid: cached[f"indices__{pid}"].copy()
                for pid in patient_ids
                if f"indices__{pid}" in cached
            }
            counts = {
                pid: int(cached[f"count__{pid}"])
                for pid in patient_ids
                if f"count__{pid}" in cached
            }
        for pid in patient_ids:
            indices.setdefault(pid, np.empty(0, dtype=np.int64))
            counts.setdefault(pid, 0)
        return indices, counts
    id_col = config["metadata_id_column"]
    row_col = config["metadata_row_column"]
    time_col = config["metadata_time_column"]
    quality_col = config["metadata_quality_column"]
    selected_rows: dict[str, list[np.ndarray]] = {}
    row_counts: dict[str, int] = {}
    last_times: dict[str, float] = {}
    for chunk in pd.read_csv(
        metadata_path,
        usecols=[id_col, row_col, time_col, quality_col],
        dtype={
            id_col: "string",
            row_col: "int64",
            time_col: "float64",
            quality_col: "int8",
        },
        chunksize=250_000,
    ):
        if chunk.empty:
            continue
        for patient_id, rows in chunk.groupby(id_col, sort=False):
            selected_rows.setdefault(patient_id, [])
            row_counts.setdefault(patient_id, 0)
            rows = rows.sort_values(row_col)
            row_numbers = rows[row_col].to_numpy(dtype=np.int64)
            expected_first = row_counts[patient_id]
            if row_numbers[0] != expected_first or not np.array_equal(
                row_numbers,
                np.arange(expected_first, expected_first + len(row_numbers)),
            ):
                raise ValueError(
                    f"{metadata_path}: {patient_id} row indices do not align with NPY order"
                )
            times = rows[time_col].to_numpy(dtype=np.float64)
            if patient_id in last_times:
                times = np.concatenate(([last_times[patient_id]], times))
            gaps = np.diff(times)
            if np.any(gaps < durations.get(patient_id, 0) - 1e-6) or np.any(gaps <= 0):
                raise ValueError(
                    f"{metadata_path}: {patient_id} windows overlap or timestamps are unordered"
                )
            last_times[patient_id] = float(times[-1])
            passed = rows[quality_col].to_numpy(dtype=np.int8)
            if not np.isin(passed, [0, 1]).all():
                raise ValueError(f"{metadata_path}: {quality_col} must contain only 0/1")
            good_rows = row_numbers[passed == 1]
            if len(good_rows):
                selected_rows[patient_id].append(good_rows)
            row_counts[patient_id] += len(rows)
    all_indices = {
        pid: np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        for pid, parts in selected_rows.items()
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_payload = {}
    for pid, positions in all_indices.items():
        cache_payload[f"indices__{pid}"] = positions
        cache_payload[f"count__{pid}"] = np.asarray(row_counts[pid], dtype=np.int64)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **cache_payload)
    temporary.replace(cache_path)
    return (
        {pid: all_indices.get(pid, np.empty(0, dtype=np.int64)) for pid in patient_ids},
        {pid: row_counts.get(pid, 0) for pid in patient_ids},
    )


def load_cohorts(
    basic: dict[str, Any], basic_path: Path
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, pd.DataFrame]],
    dict[str, Any],
    dict[str, np.ndarray],
]:
    """Build ML, DL, and fusion cohort frames without inventing split or label data."""
    clinical, column_mapping = _read_clinical(basic, basic_path)
    split = _read_splits(basic, basic_path)
    clinical_split_ids = set(clinical[basic["data"]["clinical"]["id_column"]].astype(str)) & set(
        split.patient_id
    )
    ecg_config = basic["data"].get("ecg", {})
    if ecg_config.get("directory"):
        ecg_files, _ = _ecg_inventory(basic, basic_path)
        window_counts, invalid_ecg = _validate_ecg_headers(ecg_files, basic)
        window_indices, indexed_window_counts = _load_window_quality_index(
            basic, basic_path, clinical_split_ids & set(window_counts)
        )
        for patient_id in window_indices:
            if indexed_window_counts[patient_id] != window_counts[patient_id]:
                invalid_ecg[patient_id] = (
                    f"NPY windows={window_counts[patient_id]} but metadata rows="
                    f"{indexed_window_counts[patient_id]}"
                )
            elif patient_id not in invalid_ecg:
                try:
                    load_patient_waveform(
                        patient_id,
                        ecg_files[patient_id],
                        window_indices[patient_id],
                        ecg_config,
                    )
                except (OSError, ValueError) as error:
                    invalid_ecg[patient_id] = f"selected_windows_invalid: {error}"
    else:
        ecg_files, window_counts, invalid_ecg, window_indices = {}, {}, {}, {}
    valid_ecg_ids = {
        patient_id
        for patient_id in window_indices
        if patient_id not in invalid_ecg and len(window_indices[patient_id]) >= 1
    }
    invalid_ecg.update(
        {
            patient_id: f"only {len(window_indices[patient_id])} SQI-passing windows"
            for patient_id in window_indices
            if patient_id not in invalid_ecg and patient_id not in valid_ecg_ids
        }
    )
    id_column = basic["data"]["clinical"]["id_column"]
    clinical = clinical.rename(columns={id_column: "patient_id"})
    label_column = basic["data"]["clinical"]["label_column"]
    clinical_ids, split_ids = set(clinical.patient_id), set(split.patient_id)
    ecg_ids = set(ecg_files)
    joined = clinical.merge(split, on="patient_id", how="inner", validate="one_to_one")
    joined = joined.rename(columns={label_column: "y_true"})
    joined["ecg_path"] = joined.patient_id.map(
        {patient_id: str(path) for patient_id, path in ecg_files.items()}
    )
    joined["ecg_windows"] = joined.patient_id.map(window_counts)
    joined["ecg_quality_windows"] = joined.patient_id.map(
        {pid: len(indices) for pid, indices in window_indices.items()}
    )
    ml = {split_name: frame.reset_index(drop=True) for split_name, frame in joined.groupby("split")}
    dl = {
        split_name: frame.loc[frame.patient_id.isin(valid_ecg_ids)].reset_index(drop=True)
        for split_name, frame in joined.groupby("split")
    }
    # Keep empty configured splits represented so audit and trainers fail explicitly.
    for split_name in ("train", "validation", "test"):
        ml.setdefault(split_name, joined.iloc[0:0].copy())
        dl.setdefault(split_name, joined.iloc[0:0].copy())

    exclusions = []
    for patient_id in sorted(clinical_ids - split_ids):
        exclusions.append({"patient_id": patient_id, "reason": "missing_split_assignment"})
    for patient_id in sorted(split_ids - clinical_ids):
        reason = "missing_clinical_label_and_features"
        if patient_id in ecg_ids:
            reason = "missing_clinical_label_and_features; ECG_available_but_unlabeled"
        exclusions.append({"patient_id": patient_id, "reason": reason})
    for patient_id, reason in sorted(invalid_ecg.items()):
        if patient_id in clinical_ids & split_ids:
            exclusions.append({"patient_id": patient_id, "reason": f"invalid_ecg: {reason}"})
    if ecg_config.get("directory"):
        for patient_id in sorted(clinical_split_ids - valid_ecg_ids - set(invalid_ecg)):
            exclusions.append({"patient_id": patient_id, "reason": "missing_ecg_file"})
    source_paths = {
        "clinical": str(resolve_data_path(basic["data"]["clinical"]["path"], basic_path)),
        "split": str(resolve_data_path(basic["data"]["split"]["path"], basic_path)),
    }
    if ecg_config.get("directory"):
        source_paths["ecg_directory"] = str(resolve_data_path(ecg_config["directory"], basic_path))
    audit = {
        "source_paths": source_paths,
        "counts": {
            "clinical_patients": len(clinical_ids),
            "split_patients": len(split_ids),
            "ecg_patients": len(ecg_ids),
            "clinical_split_intersection": len(joined),
            "clinical_only": len(clinical_ids - split_ids),
            "split_only": len(split_ids - clinical_ids),
            "valid_ecg_headers": len(window_counts),
            "labeled_ecg_eligible_patients": len(valid_ecg_ids),
            "invalid_ecg_or_quality": len(invalid_ecg),
        },
        "column_normalization": column_mapping,
        "split_summary": {},
        "available_patient_ids": {
            "ml": {name: frame.patient_id.astype(str).tolist() for name, frame in ml.items()},
            "dl": {name: frame.patient_id.astype(str).tolist() for name, frame in dl.items()},
            "fusion": {
                name: frame.loc[frame.patient_id.isin(valid_ecg_ids), "patient_id"]
                .astype(str)
                .tolist()
                for name, frame in ml.items()
            },
        },
        "invalid_ecg": invalid_ecg,
        "exclusions": exclusions,
        "label_policy": basic["data"]["clinical"].get("target_semantics"),
        "ecg_policy": (
            {
                "quality_filter": ecg_config["quality_filter"],
                "window_selection": ecg_config["window_selection"],
                "normalization": ecg_config["normalization"],
            }
            if ecg_config.get("directory")
            else None
        ),
    }
    for split_name in ("train", "validation", "test"):
        ml_frame, dl_frame = ml[split_name], dl[split_name]
        audit["split_summary"][split_name] = {
            "ml_patients": len(ml_frame),
            "ml_labels": ml_frame.y_true.value_counts().sort_index().astype(int).to_dict(),
            "dl_patients": len(dl_frame),
            "dl_labels": dl_frame.y_true.value_counts().sort_index().astype(int).to_dict(),
            "dl_window_count": int(dl_frame.ecg_windows.sum()) if not dl_frame.empty else 0,
            "dl_sqi_passing_window_count": (
                int(dl_frame.ecg_quality_windows.sum()) if not dl_frame.empty else 0
            ),
        }
    return clinical, {"ml": ml, "dl": dl}, audit, window_indices


def write_audit(audit: dict[str, Any], output_dir: Path) -> None:
    """Persist a machine-readable audit and exact patient exclusion list."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(audit["exclusions"], columns=["patient_id", "reason"]).to_csv(
        output_dir / "excluded_patients.csv", index=False
    )
    for modality in ("ml", "dl", "fusion"):
        rows = [
            {"patient_id": patient_id, "split": split_name}
            for split_name, patient_ids in audit["available_patient_ids"][modality].items()
            for patient_id in patient_ids
        ]
        pd.DataFrame(rows, columns=["patient_id", "split"]).to_csv(
            output_dir / f"{modality}_patients.csv", index=False
        )


def load_patient_waveform(
    patient_id: str,
    ecg_path: Path,
    valid_window_indices: np.ndarray,
    ecg_config: dict[str, Any],
    *,
    return_lead_mask: bool = False,
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SQI-passing windows, select them chronologically, and z-score each lead."""
    if ecg_config.get("quality_filter") != "sqi_pass_only":
        raise ValueError("Only ecg.quality_filter=sqi_pass_only is supported")
    if ecg_config.get("window_selection") != "evenly_spaced_chronological":
        raise ValueError(
            f"Unsupported ecg.window_selection: {ecg_config.get('window_selection')!r}"
        )
    if ecg_config.get("normalization") != "per_window_per_lead_zscore":
        raise ValueError(f"Unsupported ecg.normalization: {ecg_config.get('normalization')!r}")
    count = len(valid_window_indices) if max_windows is None else int(max_windows)
    if count < 1:
        raise ValueError("max_windows must be positive")
    if len(valid_window_indices) == 0:
        raise ValueError(f"{patient_id}: no SQI-passing windows are available")
    selected_count = min(count, len(valid_window_indices))
    selected = np.linspace(0, len(valid_window_indices) - 1, selected_count, dtype=np.int64)
    row_indices = valid_window_indices[selected]
    array = np.load(ecg_path, mmap_mode="r", allow_pickle=False)
    raw = np.asarray(array[row_indices], dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != selected_count or min(raw.shape[1:]) < 1:
        raise ValueError(f"{ecg_path}: unsupported selected ECG shape {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{ecg_path}: selected ECG windows contain NaN or Inf")
    mean = raw.mean(axis=-1, keepdims=True)
    std = raw.std(axis=-1, keepdims=True)
    flat = std < 1e-8
    if np.any(flat.all(axis=1)):
        raise ValueError(f"{ecg_path}: selected window has no varying lead")
    normalized = (raw - mean) / np.where(flat, 1.0, std)
    measured_leads = raw.shape[1]
    lead_mask = np.ones(measured_leads, dtype=bool)
    output = normalized.astype(np.float32, copy=False)
    if return_lead_mask:
        return output, row_indices, lead_mask
    return output, row_indices


def load_patient_quality_indices(
    patient_id: str, basic: dict[str, Any], basic_path: Path
) -> np.ndarray:
    """Load and validate one patient's SQI-passing row positions for unlabeled smoke input."""
    paths, _ = _ecg_inventory(basic, basic_path)
    if patient_id not in paths:
        raise FileNotFoundError(f"No ECG NPY found for patient ID {patient_id!r}")
    counts, invalid = _validate_ecg_headers({patient_id: paths[patient_id]}, basic)
    if invalid:
        raise ValueError(f"{patient_id}: invalid ECG array: {invalid[patient_id]}")
    indices, metadata_counts = _load_window_quality_index(basic, basic_path, {patient_id})
    if counts[patient_id] != metadata_counts[patient_id]:
        raise ValueError(
            f"{patient_id}: NPY has {counts[patient_id]} windows but sidecar has "
            f"{metadata_counts[patient_id]} rows"
        )
    if not len(indices[patient_id]):
        raise ValueError(f"{patient_id}: no SQI-passing windows for model input")
    return indices[patient_id]
