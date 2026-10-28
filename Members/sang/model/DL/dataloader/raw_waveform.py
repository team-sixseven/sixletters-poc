"""Memory-mapped 30-second patient sequences for the raw DL waveform cohort."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ...config import resolve_data_path
from .segmented_ecg import read_patient_split

QUALITY_COLUMNS = {"pid", "window_index", "usable"}


def load_raw_cohort(basic: dict, basic_path: Path, *, require_quality: bool = True):
    """Validate the raw split, inspect available files, and return frames plus usable indices."""
    config = basic["data"]["raw_ecg"]
    split_path = resolve_data_path(config["split_path"], basic_path)
    wave_dir = resolve_data_path(config["wave_directory"], basic_path)
    sample_rate = int(config["sample_rate_hz"])
    window_seconds = int(config["segment_seconds"])
    window_samples = sample_rate * window_seconds
    if sample_rate <= 0 or window_seconds <= 0:
        raise ValueError("raw_ecg sample_rate_hz and segment_seconds must be positive")
    rows = read_patient_split(split_path)
    duplicate_split = {}
    assigned = {}
    records = []
    all_usable: dict[str, np.ndarray] = {}
    for pid, split, label in rows:
        if pid in assigned and assigned[pid] != split:
            duplicate_split[pid] = [assigned[pid], split]
        assigned[pid] = split
        path = wave_dir / f"{pid}.npy"
        if not path.is_file():
            records.append(
                {
                    "patient_id": pid,
                    "split": split,
                    "y_true": label,
                    "ecg_path": None,
                    "reason": "missing_wave_file",
                }
            )
            continue
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 2 or array.shape[1] < 1:
                raise ValueError(
                    "expected sample-major (samples, measured_leads), with at least one lead; "
                    f"got {array.shape}"
                )
            if array.dtype.kind not in "fiu" or array.shape[0] < window_samples:
                raise ValueError(f"unsupported dtype or shorter than one {window_seconds}s window")
            count = int(array.shape[0] // window_samples)
            usable = np.ones(count, dtype=bool)
            all_usable[pid] = usable
            records.append(
                {
                    "patient_id": pid,
                    "split": split,
                    "y_true": label,
                    "ecg_path": str(path.resolve()),
                    "reason": None,
                    "sample_count": int(array.shape[0]),
                    "lead_count": int(array.shape[1]),
                    "window_count": count,
                    "discarded_tail_samples": int(array.shape[0] % window_samples),
                }
            )
        except (OSError, ValueError, EOFError) as error:
            records.append(
                {
                    "patient_id": pid,
                    "split": split,
                    "y_true": label,
                    "ecg_path": str(path.resolve()),
                    "reason": f"invalid_wave_file:{error}",
                }
            )
    if duplicate_split:
        raise ValueError(f"Patient leakage between splits: {duplicate_split}")

    quality_value = config.get("quality_index_path")
    quality_path = resolve_data_path(quality_value, basic_path) if quality_value else None
    if require_quality and config.get("source_metadata_confirmed") is not True:
        raise RuntimeError(
            "Raw ECG training requires source metadata confirmation in "
            "basic.yaml:data.raw_ecg.source_metadata_confirmed. Confirm the configured sample rate "
            "and sample/lead axes against the provenance of these NPY files first. "
            "Lead names/order are not assumed by the shared-lead encoder."
        )
    if require_quality and (quality_path is None or not quality_path.is_file()):
        raise RuntimeError(
            "Raw ECG training requires basic.yaml data.raw_ecg.quality_index_path. "
            "Expected CSV columns: pid, window_index (zero-based 30-second segment), usable (0/1)."
        )
    if quality_path is not None and quality_path.is_file():
        seen: set[tuple[str, int]] = set()
        validated_rows = 0
        with quality_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not QUALITY_COLUMNS.issubset(reader.fieldnames):
                raise ValueError(f"{quality_path}: expected columns {sorted(QUALITY_COLUMNS)}")
            for line, row in enumerate(reader, start=2):
                pid = (row.get("pid") or "").strip()
                if pid not in assigned:
                    raise ValueError(f"{quality_path}:{line}: pid {pid!r} is absent from the split")
                try:
                    index, usable = int(row["window_index"]), int(row["usable"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{quality_path}:{line}: index and usable must be integers"
                    ) from error
                if usable not in (0, 1) or index < 0:
                    raise ValueError(f"{quality_path}:{line}: invalid window_index or usable value")
                key = (pid, index)
                if key in seen:
                    raise ValueError(f"{quality_path}:{line}: duplicate quality row {key}")
                seen.add(key)
                # A complete upstream index may arrive before every waveform.
                # Missing/invalid files remain explicit patient exclusions.
                if pid not in all_usable:
                    continue
                if index >= len(all_usable[pid]):
                    raise ValueError(f"{quality_path}:{line}: window_index exceeds waveform length")
                all_usable[pid][index] = bool(usable)
                validated_rows += 1
        expected = sum(len(indexes) for indexes in all_usable.values())
        if validated_rows != expected:
            raise ValueError(
                f"{quality_path}: expected one quality row per full segment of available "
                f"waveforms ({expected}), got {validated_rows}"
            )

    frame = pd.DataFrame(records)
    valid = frame.loc[frame.reason.isna() & frame.patient_id.isin(all_usable)].copy()
    valid["usable_window_count"] = valid.patient_id.map(
        {pid: int(indices.sum()) for pid, indices in all_usable.items()}
    )
    valid = valid.loc[valid.usable_window_count > 0].reset_index(drop=True)
    exclusions = frame.loc[frame.reason.notna(), ["patient_id", "split", "reason"]].to_dict(
        "records"
    )
    exclusions.extend(
        {"patient_id": pid, "split": split, "reason": "no_usable_30_second_segments"}
        for pid, split, _ in rows
        if pid in all_usable and not all_usable[pid].any()
    )
    split_frames = {
        split: valid.loc[valid.split == split].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    available = {
        split: split_frames[split].patient_id.astype(str).tolist() for split in split_frames
    }
    audit = {
        "counts": {
            "split_patients": len(rows),
            "wave_patients": len(frame.loc[frame.reason.isna()]),
            "usable_wave_patients": len(valid),
            "quality_index": str(quality_path) if quality_path else None,
        },
        "exclusions": exclusions,
        "available_patient_ids": {
            "dl": available,
            "ml": {split: [] for split in available},
            "fusion": {split: [] for split in available},
        },
        "quality_window_counts": {
            pid: {"total": int(len(indices)), "usable": int(indices.sum())}
            for pid, indices in all_usable.items()
        },
        "split_summary": {
            split: {
                "patients": len(part),
                "positive": int(part.y_true.sum()) if len(part) else 0,
                "negative": int((part.y_true == 0).sum()) if len(part) else 0,
            }
            for split, part in split_frames.items()
        },
        "sample_rate_hz": sample_rate,
        "segment_seconds": window_seconds,
        "window_samples": window_samples,
        "quality_policy": "aligned usable flag; excluded segments retain original index positions",
    }
    return split_frames, all_usable, audit


class RawPatientSequenceDataset(Dataset):
    """Load selected valid segments from one patient's memory-mapped continuous array."""

    def __init__(
        self,
        frame: pd.DataFrame,
        usable_indices: dict[str, np.ndarray],
        config: dict,
        max_segments: int,
        normalization: str,
    ):
        self.frame = frame.reset_index(drop=True)
        self.usable_indices = usable_indices
        self.config = config
        self.max_segments = int(max_segments)
        self.window_samples = int(config["sample_rate_hz"]) * int(config["segment_seconds"])
        if self.max_segments < 1:
            raise ValueError("max_segments_per_patient must be positive")
        if normalization not in {"none", "per_window_per_lead_zscore"}:
            raise ValueError(f"Unsupported raw ECG normalization {normalization!r}")
        self.normalization = normalization

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        pid = str(row.patient_id)
        valid = np.flatnonzero(self.usable_indices[pid])
        if not len(valid):
            raise ValueError(f"{pid}: no usable 30-second segments")
        take = min(len(valid), self.max_segments)
        chosen = valid[np.linspace(0, len(valid) - 1, take, dtype=np.int64)]
        array = np.load(Path(row.ecg_path), mmap_mode="r", allow_pickle=False)
        lead_count = int(array.shape[1])
        lead_mask = np.ones(lead_count, dtype=bool)
        windows = []
        for window_index in chosen:
            start = int(window_index) * self.window_samples
            segment = np.asarray(array[start : start + self.window_samples].T, dtype=np.float32)
            if segment.shape != (lead_count, self.window_samples) or not np.isfinite(segment).all():
                raise ValueError(f"{pid}: invalid samples in segment {window_index}")
            if self.normalization == "per_window_per_lead_zscore":
                mean = segment.mean(axis=1, keepdims=True)
                std = segment.std(axis=1, keepdims=True)
                segment = (segment - mean) / np.where(std < 1e-8, 1.0, std)
            windows.append(segment)
        return (
            pid,
            torch.from_numpy(np.stack(windows)),
            torch.tensor(float(row.y_true), dtype=torch.float32),
            torch.from_numpy(chosen.astype(np.float32)),
            torch.from_numpy(lead_mask),
        )
