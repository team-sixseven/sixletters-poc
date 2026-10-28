"""Evaluate saved ML, DL, or fused models on an explicitly configured patient split."""

from __future__ import annotations

import argparse
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import load_experiment_config, load_yaml, resolve_config_path
from ..data import load_cohorts, write_audit
from ..fusion import run_fusion
from ..inference import predict_dl_run, predict_ml_run
from ..ML.artifacts import configured_model_run
from ..tracking.log_evaluation import finish_evaluation
from .core import binary_metrics, validate_threshold


def _dl_artifact_directory(value, config_path: Path) -> Path:
    path = resolve_config_path(value, config_path)
    if not path.exists():
        raise FileNotFoundError(f"Saved DL model path does not exist: {path}")
    if path.is_file():
        if path.name != "best_model.pt":
            raise ValueError(
                "DL model_path must select the saved run's best_model.pt or its run directory"
            )
        path = path.parent
    required = ("best_model.pt", "metadata.json", "resolved_config.yaml")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Saved DL run {path} is missing companion files: {missing}")
    return path


def evaluate_saved_model(
    config_path: Path,
    *,
    split_override: str | None = None,
    threshold_override: float | None = None,
    run_id: str | None = None,
) -> Path:
    """Run one evaluation without fitting model weights or preprocessing."""
    config, basic, config_path, basic_path = load_experiment_config(config_path)
    evaluation = config["evaluation"]
    split_name = split_override or evaluation["split"]
    if split_name not in {"validation", "test"}:
        raise ValueError("evaluation.split must be validation or test")
    configured_override = evaluation.get("threshold_override")
    explicit_override = (
        threshold_override if threshold_override is not None else configured_override
    )
    if explicit_override is not None:
        explicit_override = validate_threshold(explicit_override)
    target = evaluation["target"]
    output_base = resolve_config_path(evaluation["results_directory"], config_path)
    if target == "fusion":
        fusion_path = resolve_config_path(evaluation["fusion_config"], config_path)
        fusion_output = run_fusion(
            fusion_path,
            split_name=split_name,
            run_id=run_id,
            threshold_override=explicit_override,
            output_root=output_base / "fusion",
        )
        return finish_evaluation(
            fusion_output, mlflow_logging=evaluation.get("mlflow_logging", False)
        )
    if target not in {"ml", "dl"}:
        raise ValueError("evaluation.target must be ml, dl, or fusion")
    artifact_path = evaluation.get("artifact_dir")
    if target == "dl":
        model_path = evaluation.get("model_path")
        if not model_path and not artifact_path:
            raise ValueError(
                "DL evaluation requires evaluation.model_path (or legacy evaluation.artifact_dir)"
            )
        artifact_dir = _dl_artifact_directory(model_path or artifact_path, config_path)
        if model_path and artifact_path:
            if _dl_artifact_directory(artifact_path, config_path) != artifact_dir:
                raise ValueError(
                    "DL model_path and artifact_dir point to different runs; select one model"
                )
    elif artifact_path:
        artifact_dir = resolve_config_path(artifact_path, config_path)
    elif target == "ml":
        model_name = evaluation.get("model_name")
        choice_path = Path(__file__).resolve().parents[1] / "ML/model_choice.yaml"
        if evaluation.get("model_choice"):
            choice_path = resolve_config_path(evaluation["model_choice"], config_path)
        choice = load_yaml(choice_path)
        if model_name not in choice["model_configs"]:
            raise ValueError("evaluation.model_name must name an ML model in model_choice.yaml")
        parameter_path = resolve_config_path(choice["model_configs"][model_name], choice_path)
        artifact_dir = configured_model_run(
            parameter_path, model_name, model_version=evaluation.get("model_version")
        )
        if artifact_dir is None:
            raise FileNotFoundError(
                f"{parameter_path}: no saved model at model_path for the requested version. "
                "Train a model or save the best Optuna result before evaluation."
            )
    raw_dl = False
    if target == "dl":
        metadata_path = artifact_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"DL model artifact metadata not found: {metadata_path}")
        raw_dl = (
            json.loads(metadata_path.read_text(encoding="utf-8")).get("data_source") == "raw_wave"
        )
    raw_usage = None
    if raw_dl:
        from ..DL.raw_pipeline import (
            RAW_USAGE_COLUMNS,
            load_raw_evaluation_cohort,
            predict_raw_dl_run,
        )

        cohorts, window_indices, audit = load_raw_evaluation_cohort(artifact_dir, basic, basic_path)
        patients = cohorts[split_name]
    else:
        _, cohorts, audit, window_indices = load_cohorts(basic, basic_path)
        patients = cohorts[target][split_name]
    if patients.empty:
        raise ValueError(f"No {target.upper()} patients are available in {split_name}")
    if target == "ml":
        probabilities, metadata = predict_ml_run(artifact_dir, patients)
    elif raw_dl:
        probabilities, metadata, raw_usage = predict_raw_dl_run(
            artifact_dir, patients, window_indices
        )
    else:
        probabilities, metadata = predict_dl_run(artifact_dir, patients, window_indices)
    original_threshold = validate_threshold(metadata.get("classification_threshold"))
    applied_threshold = original_threshold if explicit_override is None else explicit_override
    metrics = binary_metrics(
        patients.y_true.to_numpy(dtype=np.int64), probabilities, applied_threshold
    )
    metrics["split"] = split_name
    metrics["threshold_original"] = original_threshold
    metrics["threshold_applied"] = applied_threshold
    metrics["threshold_overridden"] = explicit_override is not None
    metrics["average_precision_definition"] = "sklearn.metrics.average_precision_score"
    if target == "ml":
        metrics["model_version"] = metadata.get("model_version")
    predictions = pd.DataFrame(
        {
            "patient_id": patients.patient_id.astype(str),
            "split": split_name,
            "y_true": patients.y_true.astype(int).to_numpy(),
            "p_positive": probabilities,
            "threshold": applied_threshold,
            "y_pred": (probabilities >= applied_threshold).astype(int),
            "model": metadata["model"],
            "run_id": metadata["run_id"],
        }
    )
    if target == "ml":
        predictions["model_version"] = metadata.get("model_version")
    if raw_usage is not None:
        for column in RAW_USAGE_COLUMNS:
            predictions[column] = [row[column] for row in raw_usage]
    run_name = run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError(f"Invalid evaluation run ID {run_name!r}")
    run_dir = output_base / target / run_name
    if run_dir.exists():
        raise FileExistsError(f"Evaluation result exists and will not be overwritten: {run_dir}")
    run_dir.mkdir(parents=True)
    write_audit(audit, run_dir / "data_audit")
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    resolved = {
        "evaluation_config": config,
        "basic": basic,
        "artifact_dir": str(artifact_dir),
        "model": metadata,
        "split": split_name,
        "threshold_original": original_threshold,
        "threshold_applied": applied_threshold,
        "evaluation_metrics": evaluation["metrics"],
        "validation_metrics_independent_of_selection": not (
            split_name == "validation" and metadata.get("selection_metric") is not None
        ),
    }
    if target == "dl":
        resolved["model_path"] = str(artifact_dir / "best_model.pt")
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return finish_evaluation(run_dir, mlflow_logging=evaluation.get("mlflow_logging", False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "eval.yaml",
    )
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--threshold-override", type=float)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    output = evaluate_saved_model(
        args.config,
        split_override=args.split,
        threshold_override=args.threshold_override,
        run_id=args.run_id,
    )
    print(json.dumps({"evaluation_output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
