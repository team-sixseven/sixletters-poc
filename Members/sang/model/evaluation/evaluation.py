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

from ..config import load_experiment_config, resolve_config_path
from ..data import load_cohorts, write_audit
from ..fusion import run_fusion
from ..inference import predict_dl_run, predict_ml_run
from .core import binary_metrics, validate_threshold


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
        return run_fusion(
            fusion_path,
            split_name=split_name,
            run_id=run_id,
            threshold_override=explicit_override,
            output_root=output_base / "fusion",
        )
    if target not in {"ml", "dl"}:
        raise ValueError("evaluation.target must be ml, dl, or fusion")
    artifact_dir = resolve_config_path(evaluation["artifact_dir"], config_path)
    _, cohorts, audit, window_indices = load_cohorts(basic, basic_path)
    patients = cohorts[target][split_name]
    if patients.empty:
        raise ValueError(f"No {target.upper()} patients are available in {split_name}")
    if target == "ml":
        probabilities, metadata = predict_ml_run(artifact_dir, patients)
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
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return run_dir


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
