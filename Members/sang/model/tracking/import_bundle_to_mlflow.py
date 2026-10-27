"""Validate and import a local SCD handoff bundle into MLflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

CONTRACT = "sixletters-model-handoff"
MAX_PARAM_LENGTH = 500
METRIC_NAMES = {
    "auroc": "roc_auc",
    "auprc_average_precision": "average_precision",
    "brier": "brier_score",
}


def _contained_file(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"Bundle files must not be symlinks: {relative}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"Invalid or missing bundle file: {relative}")
    return resolved


def _contained_path(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"Bundle paths must not be symlinks: {relative}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
        raise ValueError(f"Invalid or missing bundle path: {relative}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_identity(rows: list[dict[str, str]]) -> dict[str, str]:
    ordered = sorted(rows, key=lambda row: row["patient_id"])

    def encode(value: Any) -> bytes:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    patient_ids = [row["patient_id"] for row in ordered]
    labeled_ids = [[row["patient_id"], int(row["target"])] for row in ordered]
    return {
        "patient_hash": hashlib.sha256(encode(patient_ids)).hexdigest(),
        "target_hash": hashlib.sha256(encode(labeled_ids)).hexdigest(),
    }


def _validate_predictions(root: Path, relative: str, expected_identity: dict) -> Path:
    path = _contained_file(root, relative)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["patient_id", "target", "score"]:
            raise ValueError(f"Unexpected prediction columns in {relative}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Prediction file is empty: {relative}")
    seen = set()
    for row in rows:
        patient_id = row["patient_id"]
        if not patient_id or patient_id in seen:
            raise ValueError(f"Prediction patient IDs must be present and unique: {relative}")
        seen.add(patient_id)
        if row["target"] not in {"0", "1"}:
            raise ValueError(f"Prediction target must be binary: {relative}")
        try:
            score = float(row["score"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Prediction score must be numeric: {relative}") from error
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Prediction score must be a probability: {relative}")
    if _prediction_identity(rows) != expected_identity:
        raise ValueError(f"Prediction patient/target identity mismatch: {relative}")
    return path


def _content_hash(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        if path.is_symlink():
            raise ValueError(f"Bundle must not contain symlinks: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def validate_bundle(bundle_path: Path) -> tuple[dict, list[dict], str]:
    if bundle_path.is_symlink():
        raise ValueError("--bundle must be a real directory, not a symlink")
    root = bundle_path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--bundle must be a real directory, not a symlink")
    manifest_path = _contained_file(root, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != CONTRACT or manifest.get("contract_version") not in {1, 2, 3}:
        raise ValueError("Unsupported handoff bundle contract or version")
    if not isinstance(manifest.get("runs"), list) or not manifest["runs"]:
        raise ValueError("Manifest must contain at least one run")

    runs = []
    names = set()
    for entry in manifest["runs"]:
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative.startswith("runs/"):
            raise ValueError("Each manifest run path must be under runs/")
        run_path = _contained_file(root, relative)
        run = json.loads(run_path.read_text(encoding="utf-8"))
        model = run.get("model")
        if not isinstance(model, str) or not model or model in names:
            raise ValueError("Run model names must be present and unique")
        names.add(model)
        if model != entry.get("model"):
            raise ValueError(f"Manifest/run model name mismatch: {relative}")

        prediction_files = run.get("prediction_files", {})
        identity = run.get("prediction_identity", {})
        validation_path = _validate_predictions(
            root, prediction_files.get("validation", ""), identity.get("validation", {})
        )
        expected_hashes = run.get("prediction_sha256", {})
        if (
            expected_hashes.get("validation")
            and _sha256(validation_path) != expected_hashes["validation"]
        ):
            raise ValueError(f"Validation prediction checksum mismatch: {relative}")
        test_path = None
        test_relative = prediction_files.get("test")
        if test_relative:
            test_path = _validate_predictions(root, test_relative, identity.get("test", {}))
            if expected_hashes.get("test") and _sha256(test_path) != expected_hashes["test"]:
                raise ValueError(f"Test prediction checksum mismatch: {relative}")
        elif run.get("test_metrics") is not None:
            raise ValueError(f"Test metrics have no test predictions: {relative}")

        model_artifacts = run.get("model_artifacts", {})
        if not model_artifacts:
            raise ValueError(f"Run has no model artifacts: {relative}")
        for artifact in model_artifacts.values():
            artifact_path = _contained_path(root, artifact.get("path", ""))
            model_root = (run_path.parent / "models").resolve()
            if not artifact_path.is_relative_to(model_root):
                raise ValueError("Model artifacts must stay under the run's models/ directory")
            if artifact_path.is_file():
                if artifact_path.stat().st_size != artifact.get("bytes"):
                    raise ValueError(f"Model artifact size mismatch: {artifact_path}")
                if _sha256(artifact_path) != artifact.get("sha256"):
                    raise ValueError(f"Model artifact checksum mismatch: {artifact_path}")
            elif artifact_path.is_dir():
                if manifest.get("contract_version") < 3:
                    raise ValueError("Directory model artifacts require handoff contract version 3")
                entries = list(artifact_path.rglob("*"))
                if any(path.is_symlink() for path in entries):
                    raise ValueError(
                        f"Model artifact directories must not contain symlinks: {artifact_path}"
                    )
                expected_files = artifact.get("files")
                if not isinstance(expected_files, list) or not expected_files:
                    raise ValueError(
                        f"Directory model artifact has no file checksums: {artifact_path}"
                    )
                expected_by_path = {item.get("path"): item for item in expected_files}
                if len(expected_by_path) != len(expected_files):
                    raise ValueError(
                        f"Duplicate file paths in model artifact manifest: {artifact_path}"
                    )
                actual_files = {
                    path.relative_to(artifact_path).as_posix() for path in entries if path.is_file()
                }
                if actual_files != set(expected_by_path) or "MLmodel" not in actual_files:
                    raise ValueError(
                        f"MLflow model directory contents do not match the manifest: {artifact_path}"
                    )
                for file_relative, file_metadata in expected_by_path.items():
                    file_path = _contained_file(root, f"{artifact['path']}/{file_relative}")
                    if file_path.stat().st_size != file_metadata.get("bytes"):
                        raise ValueError(f"Model artifact size mismatch: {file_path}")
                    if _sha256(file_path) != file_metadata.get("sha256"):
                        raise ValueError(f"Model artifact checksum mismatch: {file_path}")
            else:
                raise ValueError(
                    f"Model artifact is not a regular file or directory: {artifact_path}"
                )

        runs.append(
            {
                "data": run,
                "manifest_run_path": run_path,
                "validation_prediction": validation_path,
                "test_prediction": test_path,
            }
        )

    if manifest.get("selection", {}).get("model") not in names:
        raise ValueError("Selected model is missing from manifest runs")
    selected_model = manifest["selection"]["model"]
    for item in runs:
        is_selected = item["data"]["model"] == selected_model
        if bool(item["data"].get("selected_by_validation_average_precision")) != is_selected:
            raise ValueError(f"Selection marker mismatch for {item['data']['model']}")
        if (item["data"].get("test_metrics") is not None) != is_selected:
            raise ValueError("Only the validation-selected model may contain test metrics")
    return manifest, runs, _content_hash(root)


def _flatten_parameters(value: Any, prefix: str = "") -> dict[str, str]:
    flattened = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_parameters(item, name))
    elif isinstance(value, list):
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if len(encoded) <= MAX_PARAM_LENGTH:
            flattened[prefix] = encoded
    elif value is None:
        flattened[prefix] = "null"
    elif isinstance(value, (str, int, float, bool)):
        encoded = str(value)
        if len(encoded) <= MAX_PARAM_LENGTH:
            flattened[prefix] = encoded
    return flattened


def _numeric_metrics(prefix: str, values: dict) -> dict[str, float]:
    metrics = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            metrics[f"{prefix}_{METRIC_NAMES.get(key, key)}"] = number
    return metrics


def _get_or_create_experiment(client, name: str) -> str:
    experiment = client.get_experiment_by_name(name)
    return experiment.experiment_id if experiment else client.create_experiment(name)


def import_bundle(
    bundle_path: Path,
    *,
    experiment_name: str,
    tracking_uri: str | None,
    owner: str | None,
) -> dict[str, list[str]]:
    if not tracking_uri:
        raise ValueError("Set MLFLOW_TRACKING_URI or pass --tracking-uri before uploading")
    try:
        from mlflow import MlflowClient
    except ImportError as error:
        raise RuntimeError(
            "MLflow client is unavailable. Run this command in the Global project environment."
        ) from error

    manifest, runs, bundle_hash = validate_bundle(bundle_path)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _get_or_create_experiment(client, experiment_name)
    existing = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.handoff_bundle_sha256 = '{bundle_hash}'",
        max_results=10000,
    )
    finished_models = {}
    attempts_by_model: dict[str, int] = {}
    for existing_run in existing:
        existing_model = existing_run.data.tags.get("sang_model")
        if not existing_model:
            continue
        attempts_by_model[existing_model] = attempts_by_model.get(existing_model, 0) + 1
        if existing_run.info.status == "RUNNING":
            raise ValueError(
                f"Bundle import is already running as MLflow run {existing_run.info.run_id}"
            )
        if existing_run.info.status == "FINISHED":
            finished_models[existing_model] = existing_run.info.run_id

    selected_model = manifest["selection"]["model"]
    imported_ids = []
    skipped_models = []
    for item in runs:
        run_data = item["data"]
        model_name = run_data["model"]
        if model_name in finished_models:
            skipped_models.append(model_name)
            continue
        is_selected = model_name == selected_model
        tags = {
            "handoff_contract": CONTRACT,
            "handoff_contract_version": str(manifest["contract_version"]),
            "handoff_bundle_sha256": bundle_hash,
            "sang_dataset": str(manifest.get("dataset", "unknown")),
            "sang_input_group": str(
                run_data.get("input_group", manifest.get("input_group", "unknown"))
            ),
            "sang_model": model_name,
            "sang_selected_by_validation_average_precision": str(is_selected).lower(),
            "handoff_attempt": str(attempts_by_model.get(model_name, 0) + 1),
        }
        if owner:
            tags["sang_owner"] = owner
        tags["mlflow.runName"] = f"{model_name} | {manifest.get('dataset', 'SCD')}"
        run = client.create_run(experiment_id, tags=tags)
        run_id = run.info.run_id
        imported_ids.append(run_id)
        try:
            parameters = {
                "model": model_name,
                "dataset": str(manifest.get("dataset", "unknown")),
                "input_group": str(
                    run_data.get("input_group", manifest.get("input_group", "unknown"))
                ),
                "selected_by_validation_average_precision": str(is_selected).lower(),
                **_flatten_parameters(run_data.get("parameters", {})),
            }
            for key, value in parameters.items():
                client.log_param(run_id, key[:250], value[:MAX_PARAM_LENGTH])

            metrics = {}
            metrics.update(_numeric_metrics("validation", run_data.get("validation_metrics", {})))
            metrics.update(
                _numeric_metrics(
                    "validation_threshold", run_data.get("validation_threshold_metrics", {})
                )
            )
            metrics.update(_numeric_metrics("training", run_data.get("training_metrics", {})))
            if run_data.get("test_metrics") is not None:
                metrics.update(_numeric_metrics("test", run_data["test_metrics"]))
            for key, value in metrics.items():
                client.log_metric(run_id, key[:250], value)

            client.log_artifact(
                run_id, str(bundle_path.resolve() / "manifest.json"), artifact_path="handoff"
            )
            client.log_artifact(run_id, str(item["manifest_run_path"]), artifact_path="handoff")
            client.log_artifact(run_id, str(item["validation_prediction"]), artifact_path="handoff")
            if item["test_prediction"] is not None:
                client.log_artifact(run_id, str(item["test_prediction"]), artifact_path="handoff")
            for artifact in run_data.get("model_artifacts", {}).values():
                artifact_path = (bundle_path.resolve() / artifact["path"]).resolve()
                if artifact_path.is_dir():
                    client.log_artifacts(
                        run_id,
                        str(artifact_path),
                        artifact_path=f"handoff/models/{artifact_path.name}",
                    )
                else:
                    client.log_artifact(run_id, str(artifact_path), artifact_path="handoff/models")
            client.set_terminated(run_id, status="FINISHED")
        except Exception:
            client.set_terminated(run_id, status="FAILED")
            raise
    return {"created_run_ids": imported_ids, "already_imported_models": skipped_models}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, required=True, help="Local bundle created by tracking/export_runs.py"
    )
    parser.add_argument("--experiment", default="sang-scd-model-comparison")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--owner", default="sang")
    parser.add_argument(
        "--upload", action="store_true", help="Create MLflow runs and upload bundle artifacts"
    )
    args = parser.parse_args()

    manifest, runs, bundle_hash = validate_bundle(args.bundle)
    if not args.upload:
        print(
            json.dumps(
                {
                    "status": "VALIDATED_ONLY",
                    "bundle_sha256": bundle_hash,
                    "dataset": manifest.get("dataset"),
                    "selected_model": manifest["selection"]["model"],
                    "runs": [run["data"]["model"] for run in runs],
                    "tracking_uri_configured": bool(args.tracking_uri),
                    "upload_required_flag": "--upload",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    imported = import_bundle(
        args.bundle,
        experiment_name=args.experiment,
        tracking_uri=args.tracking_uri,
        owner=args.owner,
    )
    print(json.dumps({"status": "IMPORTED", **imported}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
