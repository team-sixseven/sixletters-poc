"""Upload one completed Optuna winner through the repository's authenticated uploader."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import yaml
from filelock import FileLock

from sixletters.local_settings import read_settings, validate_owner
from sixletters.upload import upload

from ..config import REPO_ROOT


def log_best_to_mlflow(best_run_dir: Path) -> dict:
    """Send aggregate validation results and model files, then remove local staging copies."""
    best_run_dir = Path(best_run_dir).resolve()
    metadata = json.loads((best_run_dir / "metadata.json").read_text(encoding="utf-8"))
    status = json.loads((best_run_dir / "run_status.json").read_text(encoding="utf-8"))
    summary = json.loads((best_run_dir / "summary.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((best_run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE" or summary.get("storage_policy") != "best_only":
        raise ValueError("MLflow upload requires a completed best-only Optuna result")
    if resolved.get("smoke_train_patient_limit") is not None:
        raise ValueError("Smoke-limited models are not uploaded as completed experiment results")
    if summary.get("model") != metadata["model"] or summary.get("model_version") != metadata.get(
        "model_version"
    ):
        raise ValueError("Best result summary and model metadata do not match")
    source_run_id = metadata["run_id"]
    if source_run_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", source_run_id):
        raise ValueError("Unexpected model run ID")
    settings = read_settings(REPO_ROOT)
    owner = validate_owner(settings.get("owner", ""))
    source = json.loads((best_run_dir / "metrics.json").read_text(encoding="utf-8"))["validation"]
    metrics = {
        f"validation.{key}": value
        for key, value in source.items()
        if type(value) in (int, float) and math.isfinite(value)
    }
    matrix = source.get("confusion_matrix")
    if matrix is not None:
        for name, value in zip(("tn", "fp", "fn", "tp"), [*matrix[0], *matrix[1]], strict=True):
            metrics[f"validation.{name}"] = value
    params = {
        "model_name": metadata["model"],
        "model_version": metadata.get("model_version"),
        "source_run_id": source_run_id,
        "feature_names": metadata["feature_names"],
        "n_features": len(metadata["feature_names"]),
        "classification_threshold": metadata["classification_threshold"],
        "selection_metric": "validation_average_precision",
        "study_name": summary["study_name"],
        "n_trials": summary["n_trials"],
        "best_trial": summary["best_trial"],
    }
    for key, value in resolved["parameters"]["model"].items():
        params[f"model.{key}"] = value
    for key, value in resolved.get("effective_training_settings", {}).items():
        params[f"training.{key}"] = value

    identity = hashlib.sha256(source_run_id.encode()).hexdigest()
    cache = REPO_ROOT / ".poc-cache" / "mlflow-best"
    cache.mkdir(parents=True, exist_ok=True)
    staging = REPO_ROOT / "Members" / owner / "results" / "mlflow" / f"optuna_best_{source_run_id}"
    marker_name = ".best-upload-source.json"
    with FileLock(str(cache / f"{identity}.lock"), timeout=0):
        if staging.exists() or staging.is_symlink():
            marker = staging / marker_name
            if staging.is_symlink() or not marker.is_file():
                raise FileExistsError(
                    f"Refusing to replace an unrelated staging directory: {staging}"
                )
            if json.loads(marker.read_text()).get("source_run_id") != source_run_id:
                raise FileExistsError("Upload staging belongs to another source run")
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        (staging / marker_name).write_text(json.dumps({"source_run_id": source_run_id}))
        try:
            for filename, values in (("metrics.json", metrics), ("params.json", params)):
                (staging / filename).write_text(
                    json.dumps(values, sort_keys=True, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            artifacts = staging / "artifacts"
            artifacts.mkdir()
            for filename in ("best_parameters.yaml", "trials.csv"):
                shutil.copy2(best_run_dir / filename, artifacts / filename)
            public_summary = {key: value for key, value in summary.items() if key != "mlflow"}
            (artifacts / "summary.json").write_text(
                json.dumps(public_summary, sort_keys=True, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            model_directory = staging / "model"
            model_directory.mkdir()
            filenames = [
                metadata["model_file"],
                "metadata.json",
                "resolved_config.yaml",
                "run_status.json",
                "metrics.json",
            ]
            if (best_run_dir / "preprocessing.joblib").is_file():
                filenames.append("preprocessing.joblib")
            for filename in filenames:
                source_file = best_run_dir / filename
                if (
                    Path(filename).name != filename
                    or source_file.is_symlink()
                    or not source_file.is_file()
                ):
                    raise ValueError(f"Expected a regular model companion file: {filename}")
                shutil.copy2(source_file, model_directory / filename)
            current = json.loads((best_run_dir / "metadata.json").read_text(encoding="utf-8"))
            if current["run_id"] != source_run_id:
                raise ValueError(
                    "Best model changed during upload preparation; retry its new result"
                )
            result = upload(REPO_ROOT, staging, model=model_directory)
        finally:
            shutil.rmtree(staging)
    summary["mlflow"] = {"status": "FINISHED", **result}
    (best_run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
