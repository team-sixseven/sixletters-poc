"""Patient-ID aligned probability fusion for already-trained ML and DL artifacts."""

from __future__ import annotations

import json
import math
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .config import load_experiment_config, resolve_config_path
from .data import load_cohorts, write_audit
from .evaluation.core import binary_metrics, validate_threshold
from .inference import predict_dl_run, predict_ml_run


def _validate_manual_weights(weights: dict[str, Any]) -> dict[str, float]:
    if set(weights) != {"ml", "dl"}:
        raise ValueError("fusion.weights must specify exactly ml and dl")
    result = {name: float(weights[name]) for name in ("ml", "dl")}
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError("Fusion weights must be finite and non-negative")
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"Fusion weights must sum to 1, got {result}")
    return result


def _learn_weights(
    p_ml: np.ndarray, p_dl: np.ndarray, y: np.ndarray, config: dict[str, Any]
) -> tuple[dict[str, float], dict[str, Any]]:
    if config.get("objective") != "binary_cross_entropy":
        raise ValueError("fusion.learning.objective must be binary_cross_entropy")
    if config.get("parameterization") != "two_class_softmax":
        raise ValueError("fusion.learning.parameterization must be two_class_softmax")
    if config.get("optimizer") != "LBFGS":
        raise ValueError("Only fusion.learning.optimizer=LBFGS is supported")
    if len(np.unique(y)) != 2:
        raise ValueError("Validation fusion weight learning requires both target classes")
    logits = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [logits],
        max_iter=int(config["max_iterations"]),
        tolerance_grad=float(config["tolerance_grad"]),
        line_search_fn="strong_wolfe",
    )
    ml_tensor = torch.as_tensor(p_ml, dtype=torch.float64)
    dl_tensor = torch.as_tensor(p_dl, dtype=torch.float64)
    target = torch.as_tensor(y, dtype=torch.float64)

    def closure():
        optimizer.zero_grad()
        weights = torch.softmax(logits, dim=0)
        probability = weights[0] * ml_tensor + weights[1] * dl_tensor
        loss = torch.nn.functional.binary_cross_entropy(probability.clamp(1e-12, 1 - 1e-12), target)
        loss.backward()
        return loss

    loss = float(optimizer.step(closure).detach().cpu())
    weights = torch.softmax(logits.detach(), dim=0).cpu().numpy()
    return {"ml": float(weights[0]), "dl": float(weights[1])}, {
        "optimizer": "LBFGS",
        "objective": "binary_cross_entropy_on_validation_probabilities",
        "iterations_max": int(config["max_iterations"]),
        "final_objective": loss,
        "n_validation_patients": int(len(y)),
    }


def _aligned_predictions(
    patients_ml: pd.DataFrame,
    p_ml: np.ndarray,
    patients_dl: pd.DataFrame,
    p_dl: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(p_ml) != len(patients_ml) or len(p_dl) != len(patients_dl):
        raise ValueError("Prediction count does not match patient count")
    ml_frame = patients_ml[["patient_id", "split", "y_true"]].copy()
    ml_frame["p_ml"] = p_ml
    dl_frame = patients_dl[["patient_id", "split", "y_true"]].copy()
    dl_frame["p_dl"] = p_dl
    if ml_frame.patient_id.duplicated().any() or dl_frame.patient_id.duplicated().any():
        raise ValueError("Model predictions contain duplicate patient IDs")
    merged = ml_frame.merge(
        dl_frame,
        on="patient_id",
        how="outer",
        suffixes=("_ml", "_dl"),
        indicator=True,
        validate="one_to_one",
    )
    mismatch = merged.loc[
        merged["_merge"].eq("both") & merged.y_true_ml.ne(merged.y_true_dl),
        "patient_id",
    ].tolist()
    if mismatch:
        raise ValueError(f"ML/DL label disagreement for patient IDs {mismatch[:10]}")
    split_mismatch = merged.loc[
        merged["_merge"].eq("both") & merged.split_ml.ne(merged.split_dl), "patient_id"
    ].tolist()
    if split_mismatch:
        raise ValueError(f"ML/DL split disagreement for patient IDs {split_mismatch[:10]}")
    excluded = []
    for row in merged.loc[~merged["_merge"].eq("both")].to_dict("records"):
        reason = (
            "missing_dl_prediction" if row["_merge"] == "left_only" else "missing_ml_prediction"
        )
        excluded.append({"patient_id": row["patient_id"], "reason": reason})
    common = merged.loc[merged["_merge"].eq("both")].copy()
    common["split"] = common.split_ml
    common["y_true"] = common.y_true_ml.astype(int)
    common["p_ml"] = common.p_ml
    common["p_dl"] = common.p_dl
    return common[["patient_id", "split", "y_true", "p_ml", "p_dl"]].sort_values(
        "patient_id"
    ).reset_index(drop=True), pd.DataFrame(excluded, columns=["patient_id", "reason"])


def run_fusion(
    fusion_config_path: Path,
    *,
    split_name: str,
    run_id: str | None = None,
    threshold_override: float | None = None,
    output_root: Path | None = None,
) -> Path:
    """Fuse two saved models; weight fitting (if selected) always uses validation only."""
    config, basic, config_path, basic_path = load_experiment_config(fusion_config_path)
    if config.get("missing_prediction_policy") != "intersection_and_report":
        raise ValueError("Only missing_prediction_policy=intersection_and_report is supported")
    if split_name not in {"validation", "test"}:
        raise ValueError("Fusion evaluation supports validation or test only")
    mode = config["fusion"]["mode"]
    saved_weights = None
    learning_details = None
    if mode == "saved":
        saved_dir_value = config["fusion"].get("saved_run_dir")
        if not saved_dir_value:
            raise ValueError("fusion.mode=saved requires fusion.saved_run_dir")
        saved_dir = resolve_config_path(saved_dir_value, config_path)
        weights_path = saved_dir / "fusion_weights.json"
        resolved_path = saved_dir / "resolved_config.yaml"
        if not weights_path.is_file() or not resolved_path.is_file():
            raise FileNotFoundError(
                f"Saved fusion run must contain fusion_weights.json and resolved_config.yaml: {saved_dir}"
            )
        saved_weights = json.loads(weights_path.read_text(encoding="utf-8"))
        saved_config = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        saved_models = saved_config["selected_models"]
        ml_path = Path(saved_models["ml"]["artifact_dir"]).resolve()
        dl_path = Path(saved_models["dl"]["artifact_dir"]).resolve()
        learning_details = saved_weights.get("learning")
    else:
        ml_path = resolve_config_path(config["ml"]["artifact_dir"], config_path)
        dl_path = resolve_config_path(config["dl"]["artifact_dir"], config_path)
    ml_metadata = json.loads((ml_path / "metadata.json").read_text(encoding="utf-8"))
    dl_metadata = json.loads((dl_path / "metadata.json").read_text(encoding="utf-8"))
    _, cohorts, audit, window_indices = load_cohorts(basic, basic_path)
    raw_dl = dl_metadata.get("data_source") == "raw_wave"
    raw_usage = None
    if raw_dl:
        from .DL.raw_pipeline import (
            RAW_USAGE_COLUMNS,
            load_raw_evaluation_cohort,
            predict_raw_dl_run,
        )

        raw_cohorts, raw_indices, raw_audit = load_raw_evaluation_cohort(dl_path, basic, basic_path)
        ml_all = pd.concat(cohorts["ml"].values(), ignore_index=True)
        raw_all = pd.concat(raw_cohorts.values(), ignore_index=True)
        comparison = ml_all[["patient_id", "split", "y_true"]].merge(
            raw_all[["patient_id", "split", "y_true"]],
            on="patient_id",
            how="inner",
            suffixes=("_ml", "_raw"),
            validate="one_to_one",
        )
        split_mismatch = comparison.loc[
            comparison.split_ml.ne(comparison.split_raw), "patient_id"
        ].tolist()
        label_mismatch = comparison.loc[
            comparison.y_true_ml.ne(comparison.y_true_raw), "patient_id"
        ].tolist()
        if split_mismatch or label_mismatch:
            raise ValueError(
                "ML/raw cohort assignment mismatch: "
                f"split={split_mismatch[:10]}, labels={label_mismatch[:10]}"
            )
        cohorts["dl"] = raw_cohorts
        window_indices = raw_indices
        audit["available_patient_ids"]["dl"] = raw_audit["available_patient_ids"]["dl"]
        audit["available_patient_ids"]["fusion"] = {
            name: sorted(
                set(audit["available_patient_ids"]["ml"][name])
                & set(raw_audit["available_patient_ids"]["dl"][name])
            )
            for name in ("train", "validation", "test")
        }
        audit["exclusions"].extend(raw_audit["exclusions"])
        audit["raw_wave_audit"] = raw_audit["counts"]
    for metadata, label in ((ml_metadata, "ML"), (dl_metadata, "DL")):
        if metadata.get("class_mapping") != {"negative": 0, "positive": 1}:
            raise ValueError(f"{label} artifact has incompatible positive class mapping")
    if ml_metadata.get("target_column", "SCD") != dl_metadata.get("target_column", "SCD"):
        raise ValueError("ML and DL artifacts predict different target columns")
    if ml_metadata.get("target_semantics") != dl_metadata.get("target_semantics"):
        raise ValueError("ML and DL artifacts disagree on target semantics")

    patients_ml = cohorts["ml"][split_name]
    patients_dl = cohorts["dl"][split_name]
    p_ml, _ = predict_ml_run(ml_path, patients_ml)
    if raw_dl:
        p_dl, _, raw_usage = predict_raw_dl_run(dl_path, patients_dl, window_indices)
    else:
        p_dl, _ = predict_dl_run(dl_path, patients_dl, window_indices)
    predictions, excluded = _aligned_predictions(patients_ml, p_ml, patients_dl, p_dl)
    if raw_usage is not None:
        raw_usage_by_pid = dict(zip(patients_dl.patient_id.astype(str), raw_usage, strict=True))
        for column in RAW_USAGE_COLUMNS:
            predictions[column] = predictions.patient_id.map(
                {pid: values[column] for pid, values in raw_usage_by_pid.items()}
            )
    if predictions.empty:
        raise ValueError(f"No common {split_name} patients have both ML and DL predictions")

    if mode == "manual":
        weights = _validate_manual_weights(config["fusion"]["weights"])
    elif mode == "learn":
        val_ml = cohorts["ml"]["validation"]
        val_dl = cohorts["dl"]["validation"]
        val_p_ml, _ = predict_ml_run(ml_path, val_ml)
        if raw_dl:
            val_p_dl, _, _ = predict_raw_dl_run(dl_path, val_dl, window_indices)
        else:
            val_p_dl, _ = predict_dl_run(dl_path, val_dl, window_indices)
        val_predictions, _ = _aligned_predictions(val_ml, val_p_ml, val_dl, val_p_dl)
        weights, learning_details = _learn_weights(
            val_predictions.p_ml.to_numpy(),
            val_predictions.p_dl.to_numpy(),
            val_predictions.y_true.to_numpy(dtype=np.int64),
            config["fusion"]["learning"],
        )
    elif mode == "saved":
        weights = _validate_manual_weights(saved_weights.get("weights", {}))
    else:
        raise ValueError("fusion.mode must be manual, learn, or saved")

    configured_threshold = validate_threshold(config.get("classification", {}).get("threshold"))
    actual_threshold = (
        configured_threshold
        if threshold_override is None
        else validate_threshold(threshold_override)
    )
    predictions["p_final"] = weights["ml"] * predictions.p_ml + weights["dl"] * predictions.p_dl
    predictions["weight_ml"] = weights["ml"]
    predictions["weight_dl"] = weights["dl"]
    predictions["threshold"] = actual_threshold
    predictions["y_pred"] = (predictions.p_final >= actual_threshold).astype(int)
    predictions["ml_model"] = ml_metadata["model"]
    predictions["ml_run_id"] = ml_metadata["run_id"]
    predictions["dl_model"] = dl_metadata["model"]
    predictions["dl_run_id"] = dl_metadata["run_id"]
    metrics = binary_metrics(
        predictions.y_true.to_numpy(dtype=np.int64),
        predictions.p_final.to_numpy(dtype=np.float64),
        actual_threshold,
    )
    metrics["threshold_original"] = configured_threshold
    metrics["threshold_applied"] = actual_threshold
    metrics["split"] = split_name
    metrics["n_excluded_no_pair"] = len(excluded)
    independent = not (learning_details is not None and split_name == "validation")
    metrics["validation_metrics_independent_of_weight_fit"] = independent

    generated_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    final_run_id = run_id or generated_id
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", final_run_id):
        raise ValueError(f"Invalid fusion run_id {final_run_id!r}")
    base_output = output_root or resolve_config_path(config["results_directory"], config_path)
    run_dir = base_output / final_run_id
    if run_dir.exists():
        raise FileExistsError(f"Fusion output exists and will not be overwritten: {run_dir}")
    run_dir.mkdir(parents=True)
    write_audit(audit, run_dir / "data_audit")
    predictions.to_csv(run_dir / f"{split_name}_predictions.csv", index=False)
    excluded.to_csv(run_dir / "excluded_patients.csv", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    resolved = {
        "fusion_config": config,
        "basic": basic,
        "selected_models": {
            "ml": {"artifact_dir": str(ml_path), **ml_metadata},
            "dl": {"artifact_dir": str(dl_path), **dl_metadata},
        },
        "learned_or_manual_weights": weights,
        "weight_learning": learning_details,
        "threshold_original": configured_threshold,
        "threshold_applied": actual_threshold,
        "split_evaluated": split_name,
        "validation_independent": independent,
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (run_dir / "fusion_weights.json").write_text(
        json.dumps(
            {
                "mode": config["fusion"]["mode"],
                "weights": weights,
                "learning": learning_details,
                "ml_run_id": ml_metadata["run_id"],
                "dl_run_id": dl_metadata["run_id"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir
