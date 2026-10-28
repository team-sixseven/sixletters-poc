"""Patient-level reader for the raw wave files in ``DL/data/wave``.

This reader deliberately refuses to call raw samples usable ECG windows unless
an aligned quality sidecar is supplied. The current CSV contains cohort labels
and splits only; it is not a waveform quality index.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...config import load_basic_config, resolve_config_path


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    split: str
    label: int
    path: Path | None
    reason: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None


def read_patient_split(path: Path) -> list[tuple[str, str, int]]:
    """Read and validate the patient-level CSV without numeric ID coercion."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"pid", "split", "SCD"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected columns pid, split, SCD")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no patient rows")
    seen: set[str] = set()
    output = []
    allowed_splits = {"train", "val", "validation", "test"}
    for line, row in enumerate(rows, start=2):
        pid = (row.get("pid") or "").strip()
        split = (row.get("split") or "").strip().lower()
        if not pid:
            raise ValueError(f"{path}:{line}: pid is empty")
        if pid in seen:
            raise ValueError(f"{path}:{line}: duplicate pid {pid!r}")
        seen.add(pid)
        if split not in allowed_splits:
            raise ValueError(f"{path}:{line}: unsupported split {split!r}")
        try:
            label = int(row["SCD"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line}: SCD must be binary 0/1") from error
        if label not in (0, 1):
            raise ValueError(f"{path}:{line}: SCD must be binary 0/1")
        output.append((pid, "validation" if split == "val" else split, label))
    return output


def inspect_raw_cohort(split_csv: Path, wave_dir: Path) -> tuple[list[PatientRecord], dict]:
    """Inventory files by exact PID and record missing/malformed waveforms."""
    records: list[PatientRecord] = []
    overlap: dict[str, set[str]] = {}
    split_rows = read_patient_split(split_csv)
    for pid, split, label in split_rows:
        path = wave_dir / f"{pid}.npy"
        if not path.is_file():
            records.append(PatientRecord(pid, split, label, None, "missing_wave_file"))
            continue
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            # Match the format observed in the supplied files and consumed by
            # raw_waveform.py; do not guess alternate segmented layouts.
            valid = array.ndim == 2 and array.shape[0] > 0 and array.shape[1] > 0
            if not valid:
                records.append(
                    PatientRecord(pid, split, label, path, f"unsupported_shape:{array.shape}")
                )
                continue
            if array.dtype.kind not in "fiu":
                records.append(
                    PatientRecord(pid, split, label, path, f"unsupported_dtype:{array.dtype}")
                )
                continue
            chunk_size = 500_000 if array.ndim == 2 else 128
            finite = True
            for start in range(0, array.shape[0], chunk_size):
                if not np.isfinite(array[start : start + chunk_size]).all():
                    finite = False
                    break
            if not finite:
                records.append(PatientRecord(pid, split, label, path, "contains_nan_or_inf"))
                continue
            records.append(
                PatientRecord(pid, split, label, path, None, tuple(array.shape), str(array.dtype))
            )
        except (OSError, ValueError, EOFError) as error:
            records.append(PatientRecord(pid, split, label, path, f"invalid_npy:{error}"))
        overlap.setdefault(pid, set()).add(split)
    leakage = {pid: sorted(splits) for pid, splits in overlap.items() if len(splits) > 1}
    if leakage:
        raise ValueError(f"Patient IDs occur in multiple splits: {leakage}")
    counts = {}
    for split in ("train", "validation", "test"):
        subset = [record for record in records if record.split == split]
        counts[split] = {
            "patients": len(subset),
            "positive": sum(record.label == 1 for record in subset),
            "negative": sum(record.label == 0 for record in subset),
            "files_present": sum(
                record.path is not None and record.reason is None for record in subset
            ),
            "missing_or_invalid": [
                {"pid": record.patient_id, "reason": record.reason}
                for record in subset
                if record.reason is not None
            ],
        }
    wave_ids = {path.stem for path in wave_dir.glob("*.npy")}
    split_ids = {pid for pid, _, _ in split_rows}
    report = {
        "split_csv": str(split_csv.resolve()),
        "wave_directory": str(wave_dir.resolve()),
        "split_counts": counts,
        "wave_files_without_split_row": sorted(wave_ids - split_ids),
        "label_semantics": "CSV SCD binary value; event horizon is not recorded here",
        "quality_status": "unverified_no_aligned_exclusion_mask",
    }
    return records, report


def require_quality_mask(mask_path: Path | None) -> None:
    """Prevent silently treating every raw sample as a valid ECG segment."""
    if mask_path is None or not mask_path.is_file():
        raise RuntimeError(
            "Training is disabled for DL/data/wave because no aligned noise/SQI mask was found. "
            "Provide a sidecar with patient ID, original sample/window position, and an explicit "
            "usable flag; zero-valued segments are not inferred to be noise."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the raw DL waveform cohort")
    parser.add_argument(
        "--basic",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "basic.yaml",
        help="Shared basic.yaml path",
    )
    args = parser.parse_args()
    basic_path = args.basic.resolve()
    basic = load_basic_config(basic_path)
    config = basic["data"]["raw_ecg"]
    split_csv = resolve_config_path(config["split_path"], basic_path)
    wave_dir = resolve_config_path(config["wave_directory"], basic_path)
    _, report = inspect_raw_cohort(split_csv, wave_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
