"""Log completed DL weights and aggregate results using the checkout's account."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import warnings
from pathlib import Path

import yaml
from filelock import FileLock

from sixletters.local_settings import read_settings, validate_owner

from ..config import REPO_ROOT


def _record_status(run_dir: Path, status: dict) -> None:
    (run_dir / "mlflow_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    summary.update(model=metadata["model"], run_id=metadata["run_id"], mlflow=status)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def log_dl_training_to_mlflow(run_dir: Path) -> dict:
    """Upload weights/config and aggregate validation results; reuse the run on retry."""
    from sixletters.upload import upload

    run_dir = Path(run_dir).resolve()
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE":
        raise ValueError("DL MLflow upload requires a completed training run")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    validation = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["validation"]
    metrics = {
        f"validation.{key}": value
        for key, value in validation.items()
        if type(value) in (int, float) and math.isfinite(value)
    }
    matrix = validation.get("confusion_matrix")
    if matrix is not None:
        metrics.update(
            {
                f"validation.{key}": value
                for key, value in zip(
                    ("tn", "fp", "fn", "tp"), [*matrix[0], *matrix[1]], strict=True
                )
            }
        )
    params = {
        "stage": "training",
        "modality": "dl",
        "source_run_id": metadata["run_id"],
        "model_name": metadata["model"],
        "data_source": metadata.get("data_source", "sqi_window"),
        "selection_metric": metadata["selection_metric"],
        "best_epoch": metadata["best_epoch"],
        "classification_threshold": metadata["classification_threshold"],
        "actual_train_patients_used": metadata["actual_train_patients_used"],
        "smoke_limited": resolved.get("smoke_train_patient_limit") is not None,
    }
    if "mil_pooling" in metadata:
        params["mil_pooling"] = metadata["mil_pooling"]
    for section, values in (
        ("cnn", resolved["cnn_spec"]),
        ("sequence", resolved["sequence_spec"]),
        ("classifier", resolved["training"].get("classifier", {})),
        ("training", resolved["training"]),
    ):
        for key, value in values.items():
            params[f"{section}.{key}"] = value

    # Public companions restore the network without carrying patient lists,
    # quality counts by PID, waveform paths, predictions, or raw cohort audits.
    public_metadata = {
        key: metadata[key]
        for key in (
            "model",
            "run_id",
            "data_source",
            "target_column",
            "target_semantics",
            "class_mapping",
            "combination",
            "cnn_model",
            "sequence_model",
            "classification_threshold",
            "best_epoch",
            "sample_rate_hz",
            "segment_seconds",
            "window_samples",
            "normalization",
            "max_segments_per_patient",
            "window_selection",
            "package_versions",
            "model_format_version",
            "lead_strategy",
            "mil_pooling",
        )
        if key in metadata
    }
    architecture = {
        "choice": {key: resolved["choice"][key] for key in ("cnn_model", "sequence_model")},
        "basic": {},
        "cnn_spec": resolved["cnn_spec"],
        "sequence_spec": resolved["sequence_spec"],
        "training": resolved["training"],
    }
    owner = validate_owner(read_settings(REPO_ROOT).get("owner", ""))
    identity = hashlib.sha256(json.dumps([str(run_dir), metadata["run_id"]]).encode()).hexdigest()
    cache = REPO_ROOT / ".poc-cache" / "mlflow-dl"
    cache.mkdir(parents=True, exist_ok=True)
    staging = REPO_ROOT / "Members" / owner / "results" / "mlflow" / f"dl_training_{identity}"
    with FileLock(str(cache / f"{identity}.lock"), timeout=0):
        marker = staging / ".dl-upload-source.json"
        if staging.exists() or staging.is_symlink():
            if staging.is_symlink() or not marker.is_file():
                raise FileExistsError(f"Unmanaged DL staging directory: {staging}")
            if json.loads(marker.read_text())["source"] != str(run_dir):
                raise ValueError("DL upload staging source mismatch")
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        marker.write_text(json.dumps({"source": str(run_dir)}), encoding="utf-8")
        try:
            for filename, values in (
                ("metrics.json", metrics),
                ("params.json", params),
            ):
                (staging / filename).write_text(
                    json.dumps(values, sort_keys=True, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            artifacts = staging / "artifacts"
            artifacts.mkdir()
            for filename in ("training_history.csv", "metrics.json"):
                shutil.copy2(run_dir / filename, artifacts / filename)
            model_dir = staging / "model"
            model_dir.mkdir()
            checkpoint = run_dir / "best_model.pt"
            if checkpoint.is_symlink() or not checkpoint.is_file():
                raise ValueError("DL checkpoint must be a regular file")
            shutil.copy2(checkpoint, model_dir / checkpoint.name)
            (model_dir / "metadata.json").write_text(
                json.dumps(public_metadata, sort_keys=True, indent=2), encoding="utf-8"
            )
            (model_dir / "model_config.yaml").write_text(
                yaml.safe_dump(architecture, sort_keys=True), encoding="utf-8"
            )
            result = upload(REPO_ROOT, staging, model=model_dir)
        finally:
            shutil.rmtree(staging)
    _record_status(run_dir, {"status": "FINISHED", **result})
    return result


def finish_dl_training(run_dir: Path, *, mlflow_logging: bool) -> Path:
    """Preserve a complete local model when tracking is unavailable."""
    if mlflow_logging:
        try:
            result = log_dl_training_to_mlflow(run_dir)
            print("MLflow DL 학습 결과:", result["url"])
        except Exception as error:
            _record_status(run_dir, {"status": "PENDING", "error_type": type(error).__name__})
            warnings.warn(
                f"MLflow DL 전송 미완료 ({type(error).__name__}). 로컬 모델은 저장됐습니다. "
                "log_dl_training_to_mlflow(run_dir)로 재학습 없이 재전송하세요.",
                RuntimeWarning,
                stacklevel=2,
            )
    return run_dir
