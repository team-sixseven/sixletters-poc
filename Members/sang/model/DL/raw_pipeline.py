"""Training and patient-level inference for continuous raw ECG NPY files."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import (
    load_experiment_config,
    load_yaml,
    resolve_data_path,
)
from ..evaluation.core import binary_metrics, validate_threshold
from .composed import (
    LEAD_STRATEGY,
    MODEL_FORMAT_VERSION,
    build_composed_model_from_specs,
    load_dl_specs,
    sequence_combination,
)
from .dataloader.dataset import collate_patient_windows
from .dataloader.raw_waveform import RawPatientSequenceDataset, load_raw_cohort
from .training import _make_optimizer, _seed_everything, _small_train_subset

RAW_USAGE_COLUMNS = (
    "windows_used",
    "first_window_index",
    "last_window_index",
    "used_window_indices",
)


def _make_loader(dataset, batch_size: int, shuffle: bool, workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        collate_fn=collate_patient_windows,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_evaluation_cohort(artifact_dir: Path, basic: dict, basic_path: Path):
    """Reject changed split/preprocessing before constructing evaluation segments."""
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    resolved = load_yaml(artifact_dir / "resolved_config.yaml")
    saved = resolved["basic"]["data"]["raw_ecg"]
    current = basic["data"]["raw_ecg"]
    for key in (
        "sample_rate_hz",
        "segment_seconds",
        "normalization",
    ):
        if current.get(key) != saved.get(key):
            raise ValueError(f"Raw DL evaluation {key} differs from the training configuration")
    if current.get("quality_index_format", "standard") != saved.get(
        "quality_index_format", "standard"
    ):
        raise ValueError("Raw DL evaluation quality index format differs from training")
    lead_value = current.get("lead_metadata_path")
    if bool(lead_value) != bool(saved.get("lead_metadata_path")):
        raise ValueError("Raw DL evaluation lead metadata configuration differs from training")
    if lead_value and _sha256(resolve_data_path(lead_value, basic_path)) != metadata.get(
        "lead_metadata_sha256"
    ):
        raise ValueError("Raw DL lead metadata changed since training")
    for path_key, hash_key in (
        ("split_path", "split_sha256"),
        ("quality_index_path", "quality_index_sha256"),
    ):
        value = current.get(path_key)
        if not value or not metadata.get(hash_key):
            raise ValueError(f"Raw DL evaluation requires recorded {path_key} and {hash_key}")
        if _sha256(resolve_data_path(value, basic_path)) != metadata[hash_key]:
            raise ValueError(f"Raw DL {path_key} changed since training")
    cohorts, indices, audit = load_raw_cohort(basic, basic_path, require_quality=True)
    for pid, counts in audit["quality_window_counts"].items():
        if counts != metadata["quality_window_counts"].get(pid):
            raise ValueError(f"Raw DL waveform segment counts changed since training: {pid}")
    return cohorts, indices, audit


@torch.inference_mode()
def _predict(model, loader, device, criterion=None):
    model.eval()
    ids, labels, probabilities, usage = [], [], [], []
    loss_sum, loss_count = 0.0, 0
    for patient_ids, waveforms, targets, mask, positions, lead_presence in tqdm(
        loader, desc="DL inference", unit="batch", leave=False
    ):
        logits = model(
            waveforms.to(device),
            mask.to(device),
            positions.to(device),
            lead_presence.to(device),
        )
        if criterion is not None:
            batch_targets = targets.to(device)
            batch_loss = criterion(logits[:, 0], batch_targets)
            if not torch.isfinite(batch_loss):
                raise RuntimeError("Non-finite validation loss")
            loss_sum += float(batch_loss.cpu()) * len(patient_ids)
            loss_count += len(patient_ids)
        scores = torch.sigmoid(logits[:, 0])
        if not torch.isfinite(scores).all():
            raise ValueError("Raw DL model returned NaN/Inf probabilities")
        ids.extend(patient_ids)
        labels.extend(targets.numpy().astype(int).tolist())
        probabilities.extend(scores.cpu().numpy().tolist())
        usage.extend(
            {
                "windows_used": int(mask[i].sum()),
                "first_window_index": int(positions[i, 0]),
                "last_window_index": int(positions[i, int(mask[i].sum()) - 1]),
                "used_window_indices": json.dumps(
                    positions[i, mask[i]].to(dtype=torch.int64).tolist()
                ),
            }
            for i in range(len(patient_ids))
        )
    mean_loss = loss_sum / loss_count if loss_count else None
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities), ids, usage, mean_loss


def train_raw_ecg(
    choice_path: Path,
    *,
    run_id: str | None = None,
    max_epochs: int | None = None,
    limit_train_patients: int | None = None,
    device_override: str | None = None,
    mlflow_logging: bool = False,
) -> Path:
    """Train CNN→sequence model→masked MIL→MLP from the raw sample-major NPY source."""
    choice, basic, choice_path, basic_path = load_experiment_config(choice_path)
    cnn_spec, head_spec, training = load_dl_specs(choice, choice_path)
    raw_training = training
    raw_config = basic["data"]["raw_ecg"]
    threshold = validate_threshold(training["classification_threshold"])
    epochs = int(training["epochs"] if max_epochs is None else max_epochs)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    cohorts, usable_indices, audit = load_raw_cohort(basic, basic_path, require_quality=True)
    train_frame = cohorts["train"]
    validation_frame = cohorts["validation"]
    if train_frame.empty or validation_frame.empty:
        raise ValueError("Raw DL requires waveform patients in both train and validation")
    available_train = train_frame
    train_frame = _small_train_subset(
        train_frame,
        limit_train_patients,
        int(training["seed"]),
    )
    if train_frame.y_true.nunique() != 2:
        raise ValueError("Raw DL train cohort requires both SCD classes")
    if validation_frame.y_true.nunique() != 2:
        raise ValueError(
            "Raw DL validation requires both SCD classes for AP-based checkpoint selection"
        )
    if mlflow_logging:
        from ..tracking.check_connection import check_mlflow_connection

        check_mlflow_connection()
    audit["actual_training_patient_ids"] = train_frame.patient_id.astype(str).tolist()
    audit["exclusions"].extend(
        {
            "patient_id": str(pid),
            "split": "train",
            "reason": "smoke_train_patient_limit",
        }
        for pid in available_train.patient_id
        if pid not in set(train_frame.patient_id)
    )

    seed = int(training["seed"])
    _seed_everything(seed)
    max_segments = int(raw_training["max_segments_per_patient"])
    model_basic = basic
    model = build_composed_model_from_specs(choice, model_basic, cnn_spec, head_spec, training)
    device_name = device_override or training["device"]
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_name == "cuda":
        raise RuntimeError("training.device is cuda but CUDA is unavailable")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("training.device must be auto, cpu, or cuda")
    model.to(device)

    normalization = raw_config.get("normalization", "none")
    train_data = RawPatientSequenceDataset(
        train_frame, usable_indices, raw_config, max_segments, normalization
    )
    validation_data = RawPatientSequenceDataset(
        validation_frame, usable_indices, raw_config, max_segments, normalization
    )
    workers = int(training["num_workers"])
    batch_size = int(raw_training["batch_size"])
    train_loader = _make_loader(train_data, batch_size, True, workers)
    validation_loader = _make_loader(validation_data, batch_size, False, workers)

    positives = float(train_frame.y_true.sum())
    negatives = float(len(train_frame) - positives)
    weight_mode = basic["dl"]["positive_class_weight"]
    if weight_mode == "train_negatives_divided_by_train_positives":
        positive_weight = negatives / positives
    elif weight_mode == "none":
        positive_weight = 1.0
    else:
        raise ValueError(f"Unsupported positive_class_weight {weight_mode!r}")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )
    optimizer = _make_optimizer(
        training["optimizer"],
        model.parameters(),
        float(training["learning_rate"]),
        float(training["weight_decay"]),
    )

    slug = f"raw_{choice['cnn_model'].lower()}_{choice['sequence_model'].lower().replace(' ', '_')}"
    generated_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{slug}_{uuid.uuid4().hex[:8]}"
    final_run_id = run_id or generated_id
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", final_run_id):
        raise ValueError(f"Invalid run_id {final_run_id!r}")
    run_dir = resolve_data_path(basic["outputs"]["dl_model_directory"], basic_path) / final_run_id
    if run_dir.exists():
        raise FileExistsError(f"Raw DL run exists and will not be overwritten: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "RUNNING", "run_id": final_run_id}), encoding="utf-8"
    )
    (run_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(audit["exclusions"], columns=["patient_id", "split", "reason"]).to_csv(
        run_dir / "excluded_patients.csv", index=False
    )
    train_frame[["patient_id", "split", "y_true"]].to_csv(
        run_dir / "train_patients.csv", index=False
    )

    history = []
    best_score, best_epoch, stale = float("-inf"), None, 0
    patience = int(training["early_stopping"]["patience"])
    min_delta = float(training["early_stopping"]["min_delta"])
    for epoch in tqdm(range(1, epochs + 1), desc="DL epochs", unit="epoch"):
        model.train()
        loss_sum, loss_count = 0.0, 0
        for _, waveforms, targets, mask, positions, lead_presence in tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False
        ):
            waveforms, targets = waveforms.to(device), targets.to(device)
            mask, positions, lead_presence = (
                mask.to(device),
                positions.to(device),
                lead_presence.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(waveforms, mask, positions, lead_presence).squeeze(1)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite raw ECG loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(targets)
            loss_count += len(targets)
        val_y, val_probability, val_ids, val_usage, val_loss = _predict(
            model, validation_loader, device, criterion
        )
        val_ap = float(average_precision_score(val_y, val_probability))
        record = {
            "epoch": epoch,
            "training_loss": loss_sum / loss_count,
            "validation_loss": val_loss,
            "validation_average_precision": val_ap,
            "validation_patients": len(val_ids),
        }
        history.append(record)
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
        print(json.dumps(record))
        if val_ap > best_score + min_delta:
            best_score, best_epoch, stale = val_ap, epoch, 0
            torch.save(
                {
                    "format_version": MODEL_FORMAT_VERSION,
                    "state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "best_epoch": best_epoch,
                    "best_validation_average_precision": best_score,
                },
                run_dir / "best_model.pt",
            )
        else:
            stale += 1
        if stale >= patience:
            break
    if best_epoch is None:
        raise RuntimeError("No validation checkpoint was selected")

    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    val_y, val_probability, val_ids, val_usage, _ = _predict(model, validation_loader, device)
    val_metrics = binary_metrics(val_y, val_probability, threshold)
    val_metrics["selection_metric"] = best_score
    val_metrics["threshold_source"] = "DL/training.yaml classification_threshold"
    pd.DataFrame(
        {
            "patient_id": val_ids,
            "split": "validation",
            "y_true": val_y,
            "p_positive": val_probability,
            "threshold": threshold,
            "y_pred": (val_probability >= threshold).astype(int),
            "model": f"{choice['cnn_model']}+{choice['sequence_model']}",
            "run_id": final_run_id,
            **{key: [item[key] for item in val_usage] for key in RAW_USAGE_COLUMNS},
        }
    ).to_csv(run_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    resolved = {
        "choice": choice,
        "basic": model_basic,
        "basic_path": str(basic_path),
        "training": training | {"epochs": epochs, "device": str(device)},
        "cnn_spec": cnn_spec,
        "sequence_spec": head_spec,
        "data_source": "raw_wave",
        "raw_normalization": normalization,
        "actual_train_patients_used": len(train_frame),
        "smoke_train_patient_limit": limit_train_patients,
        "mlflow_logging": mlflow_logging,
        "data_audit": audit,
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    metadata = {
        "model": f"{choice['cnn_model']}+{choice['sequence_model']}",
        "run_id": final_run_id,
        "data_source": "raw_wave",
        "model_format_version": MODEL_FORMAT_VERSION,
        "lead_strategy": LEAD_STRATEGY,
        "mil_pooling": training.get("mil", {}).get("pooling", "attention"),
        "target_column": "SCD",
        "target_semantics": raw_config.get("target_semantics")
        or basic["data"]["clinical"].get(
            "target_semantics",
            "Provided binary SCD label; prediction horizon is unspecified",
        ),
        "combination": sequence_combination(choice),
        "cnn_model": choice["cnn_model"],
        "sequence_model": choice["sequence_model"],
        "sample_rate_hz": raw_config["sample_rate_hz"],
        "segment_seconds": raw_config["segment_seconds"],
        "window_samples": audit["window_samples"],
        "split_sha256": _sha256(resolve_data_path(raw_config["split_path"], basic_path)),
        "quality_index_path": str(resolve_data_path(raw_config["quality_index_path"], basic_path)),
        "quality_index_sha256": _sha256(
            resolve_data_path(raw_config["quality_index_path"], basic_path)
        ),
        "lead_metadata_sha256": (
            _sha256(resolve_data_path(raw_config["lead_metadata_path"], basic_path))
            if raw_config.get("lead_metadata_path")
            else None
        ),
        "quality_window_counts": audit["quality_window_counts"],
        "normalization": normalization,
        "max_segments_per_patient": max_segments,
        "window_selection": "evenly_spaced_over_usable_segments_chronological",
        "class_mapping": {"negative": 0, "positive": 1},
        "classification_threshold": threshold,
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "device": str(device),
        "selection_metric": "validation_average_precision",
        "positive_class_weight": positive_weight,
        "actual_train_patients_used": len(train_frame),
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "torch", "scikit-learn")
        },
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"validation": val_metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "COMPLETE", "run_id": final_run_id}), encoding="utf-8"
    )
    from ..tracking.log_dl import finish_dl_training

    return finish_dl_training(run_dir, mlflow_logging=mlflow_logging)


@torch.inference_mode()
def predict_raw_dl_run(
    artifact_dir: Path,
    patients: pd.DataFrame,
    usable_indices: dict,
    *,
    device: torch.device | None = None,
):
    """Predict one probability per patient from a saved raw-wave model run."""
    from ..inference import load_dl_run

    model, metadata, resolved, device = load_dl_run(artifact_dir, device)
    basic = resolved["basic"]
    config = basic["data"]["raw_ecg"]
    quality_path = resolve_data_path(config["quality_index_path"], Path(resolved["basic_path"]))
    if _sha256(quality_path) != metadata.get("quality_index_sha256"):
        raise ValueError(
            "Raw DL quality index changed since training; refusing inconsistent evaluation"
        )
    if config.get("lead_metadata_path"):
        lead_path = resolve_data_path(config["lead_metadata_path"], Path(resolved["basic_path"]))
        if _sha256(lead_path) != metadata.get("lead_metadata_sha256"):
            raise ValueError("Raw DL lead metadata changed since training")
    raw_training = resolved["training"]
    dataset = RawPatientSequenceDataset(
        patients,
        usable_indices,
        config,
        int(raw_training["max_segments_per_patient"]),
        resolved["raw_normalization"],
    )
    loader = _make_loader(dataset, int(raw_training["batch_size"]), False, 0)
    labels, probabilities, ids, usage, _ = _predict(model, loader, device)
    if ids != patients.patient_id.astype(str).tolist():
        raise ValueError("Raw DL prediction IDs do not match the requested PID order")
    return probabilities, metadata, usage


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the raw continuous ECG DL pipeline")
    parser.add_argument(
        "--choice",
        type=Path,
        default=Path(__file__).resolve().with_name("model_choice.yaml"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--limit-train-patients", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--mlflow-logging", action="store_true")
    args = parser.parse_args()
    result = train_raw_ecg(
        args.choice,
        run_id=args.run_id,
        max_epochs=args.max_epochs,
        limit_train_patients=args.limit_train_patients,
        device_override=args.device,
        mlflow_logging=args.mlflow_logging,
    )
    print(json.dumps({"training_run": str(result.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
