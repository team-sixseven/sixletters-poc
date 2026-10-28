"""Record aggregate saved-model evaluation results without patient-level files."""

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


def log_evaluation_to_mlflow(evaluation_dir: Path) -> dict:
    """Upload an existing evaluation; retries reuse the same remote run."""
    from sixletters.upload import upload

    evaluation_dir = Path(evaluation_dir).resolve()
    source = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((evaluation_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    split = source["split"]
    if split not in {"validation", "test"}:
        raise ValueError("Only validation/test evaluation results can be uploaded")
    metrics = {
        f"{split}.{key}": value
        for key, value in source.items()
        if type(value) in (int, float) and math.isfinite(value)
    }
    matrix = source.get("confusion_matrix")
    if matrix is not None:
        for name, value in zip(("tn", "fp", "fn", "tp"), [*matrix[0], *matrix[1]], strict=True):
            metrics[f"{split}.{name}"] = value
    params = {
        "stage": "evaluation",
        "split": split,
        "evaluation_id": evaluation_dir.name,
        "classification_threshold": source["threshold_applied"],
        "threshold_original": source["threshold_original"],
    }
    model = resolved.get("model")
    if model is not None:
        params.update(
            model_name=model["model"],
            model_version=model.get("model_version"),
            source_run_id=model["run_id"],
            feature_names=model.get("feature_names", []),
        )
        summary_path = Path(resolved["artifact_dir"]) / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            # Link only if the directory still holds the model actually evaluated.
            metadata_path = summary_path.parent / "metadata.json"
            current = json.loads(metadata_path.read_text(encoding="utf-8"))
            if current["run_id"] == model["run_id"]:
                training = summary.get("mlflow", {})
                if training.get("status") == "FINISHED":
                    params["source_mlflow_run_id"] = training["run_id"]
                    params["source_mlflow_url"] = training["url"]
    else:
        params["model_name"] = "Fusion"
        for modality, metadata in resolved["selected_models"].items():
            params[f"{modality}_source_run_id"] = metadata["run_id"]
        params["fusion_weights"] = resolved["learned_or_manual_weights"]

    owner = validate_owner(read_settings(REPO_ROOT).get("owner", ""))
    identity = hashlib.sha256(str(evaluation_dir).encode()).hexdigest()
    cache = REPO_ROOT / ".poc-cache" / "mlflow-evaluation"
    cache.mkdir(parents=True, exist_ok=True)
    staging = (
        REPO_ROOT / "Members" / owner / "results" / "mlflow" / f"evaluation_{split}_{identity}"
    )
    with FileLock(str(cache / f"{identity}.lock"), timeout=0):
        # Only this private staging directory is removed; evaluation files stay local.
        marker = staging / ".evaluation-source.json"
        if staging.exists() or staging.is_symlink():
            if staging.is_symlink() or not marker.is_file():
                raise FileExistsError(f"Unmanaged evaluation staging directory: {staging}")
            if json.loads(marker.read_text())["source"] != str(evaluation_dir):
                raise ValueError("Evaluation staging source mismatch")
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        marker.write_text(json.dumps({"source": str(evaluation_dir)}))
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
            # Explicit allowlist excludes predictions.csv and patient-level audits.
            for filename in (
                "metrics.json",
                "resolved_config.yaml",
                "fusion_weights.json",
            ):
                path = evaluation_dir / filename
                if path.is_file():
                    shutil.copy2(path, artifacts / filename)
            result = upload(REPO_ROOT, staging)
        finally:
            shutil.rmtree(staging)
    (evaluation_dir / "mlflow_status.json").write_text(
        json.dumps({"status": "FINISHED", **result}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def finish_evaluation(evaluation_dir: Path, *, mlflow_logging: bool) -> Path:
    """Keep successful local evaluation even if the tracking server is unavailable."""
    if mlflow_logging:
        try:
            result = log_evaluation_to_mlflow(evaluation_dir)
            print("MLflow 평가 결과:", result["url"])
        except Exception as error:
            (evaluation_dir / "mlflow_status.json").write_text(
                json.dumps({"status": "PENDING", "error_type": type(error).__name__}),
                encoding="utf-8",
            )
            warnings.warn(
                f"MLflow 평가 전송 미완료 ({type(error).__name__}). 로컬 결과는 저장됐습니다. "
                "log_evaluation_to_mlflow(evaluation_dir)로 재전송하세요.",
                RuntimeWarning,
                stacklevel=2,
            )
    return evaluation_dir
