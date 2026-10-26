"""Write a portable, local evaluation bundle for the MLflow integration owner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _json_value(value):
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        label = "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
        return {"__sixletters_float__": label}
    raise TypeError(f"Unsupported tracking metadata value: {type(value).__name__}")


def _safe_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not slug:
        raise ValueError("Run names must contain at least one portable character")
    return slug


def _write_predictions(path: Path, frame: pd.DataFrame) -> None:
    required = {"patient_id", "target", "score"}
    if not isinstance(frame, pd.DataFrame) or set(frame.columns) != required:
        raise ValueError("Prediction frames must have exactly patient_id, target and score")
    if frame["patient_id"].isna().any() or frame["patient_id"].duplicated().any():
        raise ValueError("Prediction patient IDs must be present and unique")
    if not set(frame["target"].unique()).issubset({0, 1}):
        raise ValueError("Prediction targets must be binary")
    scores = pd.to_numeric(frame["score"], errors="raise").astype(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Prediction scores must be finite probabilities in [0, 1]")
    output = frame.copy()
    output["patient_id"] = output["patient_id"].astype(str)
    output["target"] = output["target"].astype(int)
    output["score"] = scores
    output.sort_values("patient_id").to_csv(path, index=False)


def _prediction_identity(frame: pd.DataFrame) -> dict[str, str]:
    ordered = frame.assign(patient_id=frame["patient_id"].astype(str)).sort_values("patient_id")
    patient_ids = ordered["patient_id"].tolist()
    labeled_ids = list(zip(patient_ids, ordered["target"].astype(int).tolist(), strict=True))

    def encode(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    return {
        "patient_hash": hashlib.sha256(encode(patient_ids)).hexdigest(),
        "target_hash": hashlib.sha256(encode(labeled_ids)).hexdigest(),
    }


def write_bundle(
    output_dir: Path,
    *,
    experiment: str,
    dataset: str,
    input_group: str,
    split_summary: dict,
    selected_model: str,
    environment: dict,
    runs: list[dict],
    source_files: list[Path] | None = None,
) -> Path:
    """Write one manifest and one folder per candidate without uploading data."""
    output_dir = Path(output_dir).absolute()
    if output_dir.is_symlink():
        raise ValueError("Tracking output must not be a symbolic link")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"Tracking output must be a new or empty directory: {output_dir}")
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)

    names = [run.get("model") for run in runs]
    if not names or len(set(names)) != len(names) or selected_model not in names:
        raise ValueError("Runs must have unique model names and include selected_model")

    staging_parent = output_dir.parent
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=staging_parent) as temp:
        stage = Path(temp) / "bundle"
        stage.mkdir()
        manifest_runs = []
        seen_slugs = set()
        for run in runs:
            model_name = str(run["model"])
            slug = _safe_name(model_name)
            if slug in seen_slugs:
                raise ValueError(f"Model names collide after filename normalization: {model_name}")
            seen_slugs.add(slug)
            run_dir = stage / "runs" / slug
            model_dir = run_dir / "models"
            model_dir.mkdir(parents=True)
            validation_predictions = run_dir / "validation_predictions.csv"
            _write_predictions(validation_predictions, run["validation_predictions"])
            test_prediction_path = None
            if run.get("test_predictions") is not None:
                test_prediction_path = run_dir / "test_predictions.csv"
                _write_predictions(test_prediction_path, run["test_predictions"])

            artifacts = {}
            for artifact_name, source in run.get("artifacts", {}).items():
                source = Path(source)
                if not source.is_file():
                    raise FileNotFoundError(f"Model artifact does not exist: {source}")
                name = Path(artifact_name).name
                if not name or name in artifacts:
                    raise ValueError(f"Invalid or duplicate model artifact name: {artifact_name}")
                destination = model_dir / name
                shutil.copyfile(source, destination)
                artifacts[name] = {
                    "path": destination.relative_to(stage).as_posix(),
                    "bytes": destination.stat().st_size,
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                }

            detail = {
                "model": model_name,
                "input_group": str(run.get("input_group", input_group)),
                "dataset": dataset,
                "selected_by_validation_average_precision": model_name == selected_model,
                "parameters": _json_value(run.get("parameters", {})),
                "validation_metrics": _json_value(run["validation_metrics"]),
                "validation_threshold_metrics": _json_value(
                    run.get("validation_threshold_metrics", {})
                ),
                "test_metrics": _json_value(run["test_metrics"])
                if run.get("test_metrics") is not None
                else None,
                "training_metrics": _json_value(run.get("training_metrics", {})),
                "prediction_files": {
                    "validation": validation_predictions.relative_to(stage).as_posix(),
                    "test": test_prediction_path.relative_to(stage).as_posix()
                    if test_prediction_path
                    else None,
                },
                "prediction_sha256": {
                    "validation": hashlib.sha256(validation_predictions.read_bytes()).hexdigest(),
                    "test": hashlib.sha256(test_prediction_path.read_bytes()).hexdigest()
                    if test_prediction_path
                    else None,
                },
                "prediction_identity": {
                    "validation": _prediction_identity(run["validation_predictions"]),
                    "test": _prediction_identity(run["test_predictions"])
                    if test_prediction_path
                    else None,
                },
                "model_artifacts": artifacts,
                "warnings": _json_value(run.get("warnings", [])),
            }
            (run_dir / "run.json").write_text(
                json.dumps(detail, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            manifest_runs.append({"model": model_name, "path": f"runs/{slug}/run.json"})

        manifest = {
            "contract": "sixletters-model-handoff",
            "contract_version": 2,
            "experiment": experiment,
            "dataset": dataset,
            "input_group": input_group,
            "split_summary": _json_value(split_summary),
            "selection": {"metric": "validation_average_precision", "model": selected_model},
            "metric_name_mapping": {
                "auroc": "roc_auc",
                "auprc_average_precision": "average_precision",
                "brier": "brier_score",
            },
            "environment": _json_value(environment),
            "source_files": [
                {
                    "path": source.resolve()
                    .relative_to(Path(__file__).resolve().parents[4])
                    .as_posix(),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
                for source in (source_files or [])
            ],
            "contains_patient_level_predictions": True,
            "contains_raw_training_data": False,
            "runs": manifest_runs,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(stage, output_dir)
    return output_dir
