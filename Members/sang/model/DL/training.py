"""Train a selected CNN+LSTM or CNN+Transformer on patient-labelled Holter data."""

from __future__ import annotations

import importlib.metadata
import json
import random
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

from ..config import load_experiment_config, load_yaml, resolve_config_path
from ..data import load_cohorts, write_audit
from ..evaluation.core import binary_metrics, validate_threshold
from .composed import build_composed_model
from .dataloader.dataset import PatientWindowDataset


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_optimizer(name: str, parameters, learning_rate: float, weight_decay: float):
    factories = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD,
    }
    if name not in factories:
        raise ValueError(f"Unsupported optimizer {name!r}; supported: {sorted(factories)}")
    kwargs = {"lr": learning_rate, "weight_decay": weight_decay}
    if name == "SGD":
        kwargs["momentum"] = 0.9
    return factories[name](parameters, **kwargs)


def _loader(dataset, *, batch_size: int, shuffle: bool, workers: int, pin_memory: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


@torch.inference_mode()
def _predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    probabilities, labels, patient_ids = [], [], []
    for ids, waveforms, targets in loader:
        logits = model(waveforms.to(device, non_blocking=device.type == "cuda"))
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise ValueError(f"DL model must return (batch,1) logits, got {tuple(logits.shape)}")
        probability = torch.sigmoid(logits[:, 0])
        if not torch.isfinite(probability).all():
            raise ValueError("DL model returned NaN/Inf probabilities")
        probabilities.extend(probability.cpu().numpy().tolist())
        labels.extend(targets.numpy().tolist())
        patient_ids.extend(list(ids))
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities), patient_ids


def _small_train_subset(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None:
        return frame
    if limit < 2:
        raise ValueError("--limit-train-patients must be at least 2")
    if limit >= len(frame):
        return frame
    rng = np.random.default_rng(seed)
    groups = [group for _, group in frame.groupby("y_true", sort=True)]
    if len(groups) != 2 or limit < len(groups):
        raise ValueError("Limited DL smoke training requires both classes and at least 2 patients")
    allocations = [max(1, round(limit * len(group) / len(frame))) for group in groups]
    while sum(allocations) > limit:
        largest = max(range(len(allocations)), key=lambda i: allocations[i])
        if allocations[largest] <= 1:
            raise ValueError("Cannot preserve both classes in the requested small training subset")
        allocations[largest] -= 1
    while sum(allocations) < limit:
        largest = max(range(len(groups)), key=lambda i: len(groups[i]) - allocations[i])
        allocations[largest] += 1
    parts = [
        group.iloc[rng.choice(len(group), size=count, replace=False)]
        for group, count in zip(groups, allocations, strict=True)
    ]
    return pd.concat(parts).sort_values("patient_id").reset_index(drop=True)


def train_holter(
    choice_path: Path,
    *,
    run_id: str | None = None,
    max_epochs: int | None = None,
    limit_train_patients: int | None = None,
    device_override: str | None = None,
) -> Path:
    """Train selected DL composition; test remains untouched for explicit evaluation."""
    choice, basic, choice_path, basic_path = load_experiment_config(choice_path)
    training_path = resolve_config_path(choice["training_config"], choice_path)
    training = load_yaml(training_path)
    if (
        training.get("selection_metric") != "average_precision"
        or training.get("selection_direction") != "maximize"
    ):
        raise ValueError("DL training must select best checkpoint by validation AP (maximize)")
    if training.get("loss") != "BCEWithLogitsLoss":
        raise ValueError("Only BCEWithLogitsLoss is supported")
    threshold = validate_threshold(choice.get("classification", {}).get("threshold"))
    epochs = int(training["epochs"] if max_epochs is None else max_epochs)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    seed = int(training["seed"])
    _seed_everything(seed)
    _, cohorts, audit, window_indices = load_cohorts(basic, basic_path)
    train_frame = cohorts["dl"]["train"]
    validation_frame = cohorts["dl"]["validation"]
    if train_frame.empty or validation_frame.empty:
        raise ValueError("DL requires labeled ECG patients in train and validation")
    train_frame = _small_train_subset(train_frame, limit_train_patients, seed)
    if train_frame.y_true.nunique() != 2 or validation_frame.y_true.nunique() != 2:
        raise ValueError("DL training and validation each require both classes")
    model = build_composed_model(choice, basic, choice_path=choice_path)
    device_name = device_override or training["device"]
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device is cuda but CUDA is unavailable")
        device = torch.device("cuda")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("training.device must be auto, cpu, or cuda")
    model.to(device)

    ecg_config = basic["data"]["ecg"]
    train_dataset = PatientWindowDataset(train_frame, window_indices, ecg_config)
    val_dataset = PatientWindowDataset(validation_frame, window_indices, ecg_config)
    workers = int(training["num_workers"])
    if workers < 0:
        raise ValueError("training.num_workers cannot be negative")
    pin_memory = bool(training["pin_memory"] and device.type == "cuda")
    train_loader = _loader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        workers=workers,
        pin_memory=pin_memory,
    )
    validation_loader = _loader(
        val_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        workers=workers,
        pin_memory=pin_memory,
    )
    positives = float(train_frame.y_true.sum())
    negatives = float(len(train_frame) - positives)
    weight_mode = training["positive_class_weight"]
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

    slug = f"{choice['cnn_model'].lower()}_{choice['sequence_model'].lower().replace(' ', '_')}"
    generated_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{slug}_{uuid.uuid4().hex[:8]}"
    final_run_id = run_id or generated_id
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", final_run_id):
        raise ValueError(f"Invalid run_id {final_run_id!r}")
    run_dir = resolve_config_path(choice["model_output_dir"], choice_path) / final_run_id
    if run_dir.exists():
        raise FileExistsError(f"DL model run already exists and will not be overwritten: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "RUNNING", "run_id": final_run_id}, indent=2), encoding="utf-8"
    )
    write_audit(audit, run_dir / "data_audit")

    history = []
    best_score = float("-inf")
    best_epoch = None
    stale_epochs = 0
    patience = int(training["early_stopping"]["patience"])
    min_delta = float(training["early_stopping"]["min_delta"])
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _, waveforms, targets in train_loader:
            waveforms = waveforms.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(waveforms).squeeze(1)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_y, val_probability, val_ids = _predict(model, validation_loader, device)
        val_ap = float(average_precision_score(val_y, val_probability))
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "validation_average_precision": val_ap,
            "validation_patients": len(val_ids),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_ap > best_score + min_delta:
            best_score = val_ap
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "format_version": 1,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "best_epoch": best_epoch,
                    "best_validation_average_precision": best_score,
                },
                run_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_epoch is None:
        raise RuntimeError("No finite validation AP was produced; no best checkpoint was saved")

    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    val_y, val_probability, val_ids = _predict(model, validation_loader, device)
    val_metrics = binary_metrics(val_y, val_probability, threshold)
    val_metrics["selection_metric"] = best_score
    val_metrics["selection_threshold"] = threshold
    val_metrics["threshold_source"] = "DL/model_choice.yaml classification.threshold"
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
        }
    ).to_csv(run_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    cnn_config = load_yaml(resolve_config_path(choice["cnn_config"], choice_path))
    cnn_spec = cnn_config["models"][choice["cnn_model"]]
    if choice["combination"] == "CNN+LSTM":
        head_config = load_yaml(resolve_config_path(choice["lstm_config"], choice_path))
    else:
        head_config = load_yaml(resolve_config_path(choice["transformer_config"], choice_path))
    head_spec = head_config["models"][choice["sequence_model"]]
    resolved = {
        "choice": choice,
        "basic": basic,
        "training": training | {"epochs": epochs},
        "cnn_spec": cnn_spec,
        "sequence_spec": head_spec,
        "data_audit": audit["counts"],
        "actual_train_patients_used": len(train_frame),
        "smoke_train_patient_limit": limit_train_patients,
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    metadata = {
        "model": f"{choice['cnn_model']}+{choice['sequence_model']}",
        "run_id": final_run_id,
        "target_column": basic["data"]["clinical"]["label_column"],
        "target_semantics": basic["data"]["clinical"].get("target_semantics"),
        "combination": choice["combination"],
        "cnn_model": choice["cnn_model"],
        "sequence_model": choice["sequence_model"],
        "input_shape_per_patient": [
            ecg_config["windows_per_patient"],
            ecg_config["leads"],
            ecg_config["samples_per_window"],
        ],
        "window_selection": ecg_config["window_selection"],
        "quality_filter": ecg_config["quality_filter"],
        "normalization": ecg_config["normalization"],
        "class_mapping": {"negative": 0, "positive": 1},
        "classification_threshold": threshold,
        "best_epoch": best_epoch,
        "selection_metric": "validation_average_precision",
        "positive_class_weight": positive_weight,
        "data_audit_counts": audit["counts"],
        "actual_train_patients_used": len(train_frame),
        "smoke_train_patient_limit": limit_train_patients,
        "package_versions": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "torch", "scikit-learn")
        },
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"validation": val_metrics}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "COMPLETE", "run_id": final_run_id}, indent=2), encoding="utf-8"
    )
    return run_dir
