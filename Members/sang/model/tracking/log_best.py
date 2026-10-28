"""Upload one completed Optuna winner through the repository's authenticated uploader."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import warnings
from pathlib import Path

import yaml
from filelock import FileLock

from sixletters.local_settings import read_settings, validate_owner
from sixletters.upload import upload

from ..config import REPO_ROOT


def log_best_to_mlflow(best_run_dir: Path) -> dict:
    """Send aggregate validation results and model files, then remove local staging copies."""
    return _log_ml_run(best_run_dir, require_best=True)


def log_ml_training_to_mlflow(run_dir: Path) -> dict:
    """Log an ordinary clinical run or a completed Optuna winner using local account settings."""
    summary_path = Path(run_dir) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return _log_ml_run(run_dir, require_best=summary.get("storage_policy") == "best_only")


def _log_ml_run(best_run_dir: Path, *, require_best: bool) -> dict:
    best_run_dir = Path(best_run_dir).resolve()
    metadata = json.loads((best_run_dir / "metadata.json").read_text(encoding="utf-8"))
    status = json.loads((best_run_dir / "run_status.json").read_text(encoding="utf-8"))
    summary_path = best_run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    resolved = yaml.safe_load((best_run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE" or (
        require_best and summary.get("storage_policy") != "best_only"
    ):
        raise ValueError("MLflow upload requires a completed best-only Optuna result")
    if require_best and resolved.get("smoke_train_patient_limit") is not None:
        raise ValueError("Smoke-limited models are not uploaded as completed experiment results")
    if not require_best:
        summary.update(
            model=metadata["model"],
            model_version=metadata.get("model_version"),
            run_id=metadata["run_id"],
            storage_policy="run",
        )
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
    }
    if require_best:
        params.update({key: summary[key] for key in ("study_name", "n_trials", "best_trial")})
    else:
        params.update(
            stage="training",
            modality="ml",
            smoke_limited=resolved.get("smoke_train_patient_limit") is not None,
            actual_train_patients_used=resolved["actual_train_patients_used"],
        )
    for key, value in resolved["parameters"]["model"].items():
        params[f"model.{key}"] = value
    for key, value in resolved.get("effective_training_settings", {}).items():
        params[f"training.{key}"] = value

    identity = hashlib.sha256(source_run_id.encode()).hexdigest()
    cache = REPO_ROOT / ".poc-cache" / "mlflow-best"
    cache.mkdir(parents=True, exist_ok=True)
    prefix = "optuna_best" if require_best else "ml_training"
    staging = REPO_ROOT / "Members" / owner / "results" / "mlflow" / f"{prefix}_{source_run_id}"
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
            extra_artifacts = (
                ("best_parameters.yaml", "trials.csv")
                if require_best
                else ("training_history.csv",)
            )
            for filename in extra_artifacts:
                if require_best or (best_run_dir / filename).is_file():
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
    (best_run_dir / "mlflow_status.json").write_text(
        json.dumps(summary["mlflow"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def finish_ml_training(run_dir: Path, *, mlflow_logging: bool) -> Path:
    """Preserve the completed local model if automatic tracking cannot finish."""
    if mlflow_logging:
        try:
            result = log_ml_training_to_mlflow(run_dir)
            print("MLflow ML 학습 결과:", result["url"])
        except Exception as error:
            status = {"status": "PENDING", "error_type": type(error).__name__}
            summary_path = run_dir / "summary.json"
            summary = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.exists()
                else {}
            )
            summary["mlflow"] = status
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (run_dir / "mlflow_status.json").write_text(json.dumps(status), encoding="utf-8")
            warnings.warn(
                f"MLflow ML 전송 미완료 ({type(error).__name__}). 로컬 모델은 보존했습니다. "
                "log_ml_training_to_mlflow(run_dir)로 재학습 없이 재전송하세요.",
                RuntimeWarning,
                stacklevel=2,
            )
    return run_dir
