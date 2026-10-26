"""Run ECG CNN/Transformer smoke checks on the downloaded P0001 WFDB record."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from model.DL.models import (
    CNNLSTM,
    CNNTST,
    ECGCNN,
    MILTCN,
    ECGTransformer,
    RawECGMIL,
    SegmentAttentionMIL,
)  # noqa: E402
from torch import nn  # noqa: E402


def load_first_windows(record_base: Path, batch_size: int, window_samples: int):
    header_path = record_base.with_suffix(".hea")
    data_path = record_base.with_suffix(".dat")
    header_lines = header_path.read_text(encoding="ascii").splitlines()
    record = header_lines[0].split()
    channels, sample_rate, sample_count = map(int, record[1:4])
    signal_lines = [line.split() for line in header_lines[1:]]
    if channels != 3 or len(signal_lines) != channels:
        raise ValueError(
            f"Expected 3 signal lines; got channels={channels}, lines={len(signal_lines)}"
        )
    if sample_rate != 200:
        raise ValueError(f"Expected 200 Hz per P0001 header; got {sample_rate}")

    gains = np.asarray([float(line[2]) for line in signal_lines], dtype=np.float32)
    adc_zeros = np.asarray([float(line[4]) for line in signal_lines], dtype=np.float32)
    descriptions = [line[-1] for line in signal_lines]
    formats = [line[1] for line in signal_lines]
    if any(value.split("+")[0] != "16" for value in formats):
        raise ValueError(f"Expected WFDB format 16; got {formats}")
    if np.any(gains == 0):
        raise ValueError("WFDB ADC gain must be nonzero")

    expected_bytes = sample_count * channels * np.dtype("<i2").itemsize
    actual_bytes = data_path.stat().st_size
    if expected_bytes != actual_bytes:
        raise ValueError(
            f"Header/data size mismatch: expected {expected_bytes}, got {actual_bytes}"
        )

    digital = np.fromfile(data_path, dtype="<i2").reshape(sample_count, channels)
    invalid_value_count = int(np.count_nonzero(digital == np.iinfo(np.int16).min))
    needed = batch_size * window_samples
    if sample_count < needed:
        raise ValueError(f"Record has {sample_count} samples; need at least {needed}")
    selected = digital[:needed].reshape(batch_size, window_samples, channels).astype(np.float32)
    physical = (selected - adc_zeros[None, None, :]) / gains[None, None, :]
    # Per-window/per-lead standardization is a smoke-test transform, not a validated pipeline.
    mean = physical.mean(axis=1, keepdims=True)
    std = physical.std(axis=1, keepdims=True)
    if np.any(std < 1e-8):
        raise ValueError("At least one selected lead/window has near-zero variance")
    normalized = (physical - mean) / std
    # Model input convention: batch, channels/leads, samples.
    model_input = np.transpose(normalized, (0, 2, 1)).copy()
    summary = {
        "record": record_base.name,
        "header": str(header_path),
        "data": str(data_path),
        "channels": channels,
        "channel_names": descriptions,
        "sample_rate_hz": sample_rate,
        "samples_per_channel": sample_count,
        "duration_hours": sample_count / sample_rate / 3600,
        "format": formats[0],
        "gain_per_channel": gains.tolist(),
        "adc_zero_per_channel": adc_zeros.tolist(),
        "data_bytes": actual_bytes,
        "invalid_digital_samples_min_int16": invalid_value_count,
        "selected_window_count": batch_size,
        "window_samples": window_samples,
        "window_seconds": window_samples / sample_rate,
        "selected_digital_min": int(selected.min()),
        "selected_digital_max": int(selected.max()),
        "input_shape": list(model_input.shape),
        "preprocessing": "(digital - adc_zero) / gain, then per-window/per-lead z-score",
    }
    return torch.from_numpy(model_input), summary, physical


def extract_segment_features(physical: np.ndarray) -> np.ndarray:
    """Build 16 simple per-segment statistics for the feature-based MIL prototypes."""
    # physical: (segments, samples, leads)
    per_lead_mean = physical.mean(axis=1)
    per_lead_std = physical.std(axis=1)
    per_lead_rms = np.sqrt(np.mean(np.square(physical), axis=1))
    per_lead_range = np.ptp(physical, axis=1)
    magnitude = np.linalg.norm(physical, axis=2)
    global_features = np.column_stack(
        (
            magnitude.mean(axis=1),
            magnitude.std(axis=1),
            np.sqrt(np.mean(np.square(magnitude), axis=1)),
            np.ptp(magnitude, axis=1),
        )
    )
    features = np.concatenate(
        (per_lead_mean, per_lead_std, per_lead_rms, per_lead_range, global_features), axis=1
    )
    if features.shape[1] != 16 or not np.isfinite(features).all():
        raise ValueError(f"Expected 16 finite segment features; got shape={features.shape}")
    return features.astype(np.float32)


def load_patient_label(patient_id: str, splits_dir: Path) -> tuple[int, str]:
    matches = []
    for split_name in ("train", "val", "test"):
        split_path = splits_dir / f"{split_name}.csv"
        with split_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("Patient ID", "").strip() == patient_id:
                    matches.append((int(row["SCD"]), split_name))
    if len(matches) != 1:
        raise ValueError(f"Expected one patient-label row for {patient_id}; found {len(matches)}")
    label, split_name = matches[0]
    if label not in (0, 1):
        raise ValueError(f"Expected binary SCD label for {patient_id}; got {label}")
    return label, split_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record-base",
        type=Path,
        default=Path("Members/sang/datas/ECG_tem/ECG_tem_clean/P0001"),
    )
    parser.add_argument(
        "--splits-dir", type=Path, default=Path("Members/sang/datas/clincal/splits")
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-samples", type=int, default=2500)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, record_summary, physical = load_first_windows(
        args.record_base, args.batch_size, args.window_samples
    )
    patient_label, patient_split = load_patient_label(args.record_base.name, args.splits_dir)
    patient_target = torch.tensor([[patient_label]], dtype=torch.float32, device=device)
    inputs = inputs.to(device)
    results = []
    for name, model_type in (("1D-CNN", ECGCNN), ("ECG Transformer", ECGTransformer)):
        model = model_type().to(device)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        segment_logits = model(inputs)
        # The available outcome is patient-level; mean-logit pooling is used only for this smoke check.
        logits = segment_logits.mean(dim=0, keepdim=True)
        loss = nn.BCEWithLogitsLoss()(logits, patient_target)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        gradients_finite = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        probabilities = torch.sigmoid(logits.detach())
        passed = (
            tuple(segment_logits.shape) == (args.batch_size, 1)
            and tuple(logits.shape) == (1, 1)
            and bool(torch.isfinite(logits).all().item())
            and bool(torch.isfinite(loss).item())
            and gradients_finite
        )
        result = {
            "model": name,
            "input_shape": list(inputs.shape),
            "segment_logits_shape": list(segment_logits.shape),
            "output_shape": list(logits.shape),
            "device": str(device),
            "forward": True,
            "loss": float(loss.item()),
            "backward_finite_gradients": gradients_finite,
            "probability_min": float(probabilities.min().item()),
            "probability_max": float(probabilities.max().item()),
            "seconds_forward_loss_backward": elapsed,
            "pass": bool(passed),
        }
        if device.type == "cuda":
            result["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 2)
        results.append(result)
        if not passed:
            raise RuntimeError(f"Smoke check failed for {name}: {result}")

    feature_inputs = torch.from_numpy(extract_segment_features(physical)).unsqueeze(0).to(device)
    raw_mil_inputs = inputs.unsqueeze(0)
    mil_cases = (
        ("Segment Attention MIL", SegmentAttentionMIL(), feature_inputs),
        ("MIL–TCN", MILTCN(), feature_inputs),
        ("Raw ECG MIL", RawECGMIL(), raw_mil_inputs),
    )
    for name, model, model_input in mil_cases:
        model = model.to(device)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits = model(model_input)
        loss = nn.BCEWithLogitsLoss()(logits, patient_target)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        gradients_finite = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        probabilities = torch.sigmoid(logits.detach())
        passed = (
            tuple(logits.shape) == (1, 1)
            and bool(torch.isfinite(logits).all().item())
            and bool(torch.isfinite(loss).item())
            and gradients_finite
        )
        result = {
            "model": name,
            "input_shape": list(model_input.shape),
            "output_shape": list(logits.shape),
            "device": str(device),
            "forward": True,
            "loss": float(loss.item()),
            "backward_finite_gradients": gradients_finite,
            "probability_min": float(probabilities.min().item()),
            "probability_max": float(probabilities.max().item()),
            "seconds_forward_loss_backward": elapsed,
            "pass": bool(passed),
        }
        if device.type == "cuda":
            result["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 2)
        results.append(result)
        if not passed:
            raise RuntimeError(f"Smoke check failed for {name}: {result}")

    sequence_input = raw_mil_inputs
    for name, model_type in (("CNN–LSTM", CNNLSTM), ("CNN–TST", CNNTST)):
        model = model_type().to(device)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits = model(sequence_input)
        loss = nn.BCEWithLogitsLoss()(logits, patient_target)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        gradients_finite = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        probabilities = torch.sigmoid(logits.detach())
        passed = (
            tuple(logits.shape) == (1, 1)
            and bool(torch.isfinite(logits).all().item())
            and bool(torch.isfinite(loss).item())
            and gradients_finite
        )
        result = {
            "model": name,
            "input_shape": list(sequence_input.shape),
            "output_shape": list(logits.shape),
            "device": str(device),
            "forward": True,
            "loss": float(loss.item()),
            "backward_finite_gradients": gradients_finite,
            "probability_min": float(probabilities.min().item()),
            "probability_max": float(probabilities.max().item()),
            "seconds_forward_loss_backward": elapsed,
            "pass": bool(passed),
        }
        if device.type == "cuda":
            result["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 2)
        results.append(result)
        if not passed:
            raise RuntimeError(f"Smoke check failed for {name}: {result}")

    print(
        json.dumps(
            {
                "purpose": "single-record real-waveform functional smoke check; not clinical evaluation",
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "seed": 42,
                "patient_label": patient_label,
                "patient_label_split": patient_split,
                "backward_targets": "patient-level SCD label from the matching provided split",
                "record": record_summary,
                "feature_mil_input_shape": list(feature_inputs.shape),
                "feature_mil_features": [
                    "per-lead mean/std/RMS/range (12)",
                    "3-lead magnitude mean/std/RMS/range (4)",
                ],
                "sequence_model_input_shape": list(sequence_input.shape),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
