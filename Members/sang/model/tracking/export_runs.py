"""Export completed ML/DL training runs into a local, validated MLflow handoff bundle."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Members.sang.model.config import load_basic_config  # noqa: E402
from Members.sang.model.data import load_cohorts  # noqa: E402
from Members.sang.model.DL.dataloader.dataset import PatientWindowDataset  # noqa: E402
from Members.sang.model.evaluation.core import binary_metrics  # noqa: E402
from Members.sang.model.inference import (  # noqa: E402
    load_dl_run,
    load_ml_run,
    predict_dl_run,
    predict_ml_run,
)
from Members.sang.model.tracking.bundle import write_bundle  # noqa: E402
from Members.sang.model.tracking.import_bundle_to_mlflow import validate_bundle  # noqa: E402
from Members.sang.model.tracking.model_flavors import (  # noqa: E402
    ClinicalRunModel,
    save_pytorch_model,
    save_sklearn_model,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_completed_run(run_dir: Path, modality: str) -> tuple[dict, dict, dict]:
    run_dir = run_dir.resolve(strict=True)
    if not run_dir.is_dir():
        raise ValueError(f"Not a run directory: {run_dir}")
    status = _read_json(run_dir / "run_status.json")
    if status.get("status") != "COMPLETE":
        raise ValueError(f"Run is not COMPLETE: {run_dir}")
    metadata = _read_json(run_dir / "metadata.json")
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    limit = resolved.get("smoke_train_patient_limit")
    if limit is not None:
        raise ValueError(
            f"Smoke-limited training run cannot be exported: {run_dir} (limit={limit})"
        )
    if modality == "dl" and metadata.get("smoke_train_patient_limit") is not None:
        raise ValueError(f"Smoke-limited training run cannot be exported: {run_dir}")
    return metadata, resolved, _read_json(run_dir / "metrics.json")


def _predictions(frame: pd.DataFrame, patients: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    if len(frame) != len(patients) or len(scores) != len(patients):
        raise ValueError("Prediction count does not match the configured evaluation cohort")
    actual_ids = frame["patient_id"].astype(str).tolist()
    expected_ids = patients["patient_id"].astype(str).tolist()
    if actual_ids != expected_ids:
        raise ValueError("Saved predictions do not match the current split patient order")
    if not np.array_equal(
        frame["y_true"].to_numpy(dtype=int), patients["y_true"].to_numpy(dtype=int)
    ):
        raise ValueError("Saved prediction labels do not match the current configured split")
    return pd.DataFrame(
        {
            "patient_id": expected_ids,
            "target": patients["y_true"].to_numpy(dtype=int),
            "score": np.asarray(scores, dtype=float),
        }
    )


def _metrics(predictions: pd.DataFrame, threshold: float) -> dict:
    return binary_metrics(predictions["target"], predictions["score"], threshold)


def _versions() -> dict[str, str]:
    names = ("python", "mlflow", "numpy", "pandas", "scikit-learn", "torch")
    result = {"python": sys.version.split()[0]}
    for name in names[1:]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def _parameters(metadata: dict, resolved: dict, modality: str) -> dict:
    if modality == "ml":
        return {
            "artifact_type": metadata["artifact_type"],
            "classification_threshold": metadata["classification_threshold"],
            "best_iteration": metadata.get("best_iteration"),
            "parameters": resolved.get("parameters", {}),
            "training": resolved.get("training", {}),
        }
    return {
        "combination": metadata["combination"],
        "cnn_model": metadata["cnn_model"],
        "sequence_model": metadata["sequence_model"],
        "classification_threshold": metadata["classification_threshold"],
        "best_epoch": metadata["best_epoch"],
        "training": resolved.get("training", {}),
    }


def _ml_runs(run_dirs: list[Path], cohorts: dict, flavor_root: Path) -> tuple[list[dict], str]:
    validation = cohorts["ml"]["validation"]
    test = cohorts["ml"]["test"]
    if validation.empty or test.empty:
        raise ValueError("ML handoff requires non-empty validation and test cohorts")
    prepared = []
    names = set()
    for run_dir in run_dirs:
        metadata, resolved, metrics = _check_completed_run(run_dir, "ml")
        name = metadata["model"]
        if name in names:
            raise ValueError(f"Only one trained run per ML model is allowed: {name}")
        names.add(name)
        saved_validation = pd.read_csv(
            run_dir / "validation_predictions.csv", dtype={"patient_id": str}
        )
        validation_scores, _ = predict_ml_run(run_dir, validation)
        validation_predictions = _predictions(saved_validation, validation, validation_scores)
        threshold = float(metadata["classification_threshold"])
        computed_validation = _metrics(validation_predictions, threshold)
        saved_ap = metrics["validation"].get("average_precision")
        if saved_ap is None or not np.isclose(
            computed_validation["average_precision"], saved_ap, rtol=1e-6, atol=1e-8
        ):
            raise ValueError(f"{run_dir}: reconstructed validation AP differs from saved metrics")

        estimator, preprocessor, _ = load_ml_run(run_dir)
        wrapper = ClinicalRunModel(estimator, preprocessor, metadata)
        feature_names = metadata["feature_names"]
        flavor_dir = flavor_root / f"{len(prepared)}_{name.replace(' ', '_')}"
        save_sklearn_model(
            wrapper,
            flavor_dir,
            feature_names=feature_names,
            categorical_features=(),
            feature_dtypes={name: str(validation[name].dtype) for name in feature_names},
            model_name=name,
        )
        import mlflow.pyfunc

        exported_model = mlflow.pyfunc.load_model(str(flavor_dir))
        exported_output = np.asarray(exported_model.predict(validation[feature_names]))
        if exported_output.shape != (len(validation), 2) or not np.allclose(
            exported_output[:, 1], validation_scores, rtol=1e-6, atol=1e-7
        ):
            raise ValueError(f"{run_dir}: MLflow flavor predictions do not match the trained model")
        artifact_paths = {
            "mlflow_model": flavor_dir,
            "native_model": run_dir / metadata["model_file"],
            "metadata": run_dir / "metadata.json",
        }
        preprocessing = run_dir / "preprocessing.joblib"
        if preprocessing.exists():
            artifact_paths["preprocessing"] = preprocessing
        entry = {
            "model": name,
            "input_group": "clinical",
            "validation_predictions": validation_predictions,
            "validation_metrics": computed_validation,
            "validation_threshold_metrics": {
                key: computed_validation[key] for key in ("threshold", "sensitivity", "specificity")
            },
            "training_metrics": metrics.get("training", {}),
            "parameters": _parameters(metadata, resolved, "ml"),
            "artifacts": artifact_paths,
            "_run_dir": run_dir,
            "_metadata": metadata,
        }
        prepared.append(entry)

    selected = max(prepared, key=lambda item: item["validation_metrics"]["average_precision"])
    test_scores, _ = predict_ml_run(selected["_run_dir"], test)
    test_predictions = _predictions(
        pd.DataFrame({"patient_id": test.patient_id, "y_true": test.y_true}), test, test_scores
    )
    selected["test_predictions"] = test_predictions
    selected["test_metrics"] = _metrics(
        test_predictions, float(selected["_metadata"]["classification_threshold"])
    )
    return prepared, selected["model"]


def _dl_run(
    run_dir: Path, cohorts: dict, indices: dict, source_root: Path, flavor_root: Path
) -> tuple[list[dict], str]:
    if (
        len(
            [
                cohorts["dl"][split]
                for split in ("validation", "test")
                if not cohorts["dl"][split].empty
            ]
        )
        != 2
    ):
        raise ValueError("DL handoff requires non-empty validation and test ECG cohorts")
    metadata, resolved, metrics = _check_completed_run(run_dir, "dl")
    name = metadata["model"]
    threshold = float(metadata["classification_threshold"])
    validation, test = cohorts["dl"]["validation"], cohorts["dl"]["test"]
    saved_validation = pd.read_csv(
        run_dir / "validation_predictions.csv", dtype={"patient_id": str}
    )
    validation_scores, _ = predict_dl_run(run_dir, validation, indices, device=torch.device("cpu"))
    validation_predictions = _predictions(saved_validation, validation, validation_scores)
    validation_metrics = _metrics(validation_predictions, threshold)
    saved_ap = metrics["validation"].get("average_precision")
    if saved_ap is None or not np.isclose(
        validation_metrics["average_precision"], saved_ap, rtol=1e-6, atol=1e-8
    ):
        raise ValueError(f"{run_dir}: reconstructed validation AP differs from saved metrics")
    model, _, _, _ = load_dl_run(run_dir, device=torch.device("cpu"))
    flavor_dir = flavor_root / "mlflow_model"
    save_pytorch_model(
        model,
        flavor_dir,
        input_shape=tuple(metadata["input_shape_per_patient"]),
        source_model_dir=source_root,
        model_name=name,
    )
    import mlflow.pytorch

    exported_model = mlflow.pytorch.load_model(str(flavor_dir), map_location="cpu").eval()
    sample_dataset = PatientWindowDataset(
        validation.iloc[:1], indices, resolved["basic"]["data"]["ecg"]
    )
    _, sample_waveform, _ = sample_dataset[0]
    model.eval()
    with torch.inference_mode():
        expected_logit = model(sample_waveform.unsqueeze(0))
        exported_logit = exported_model(sample_waveform.unsqueeze(0))
    if not torch.allclose(expected_logit, exported_logit, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{run_dir}: MLflow PyTorch flavor output differs from the trained model")
    test_scores, _ = predict_dl_run(run_dir, test, indices, device=torch.device("cpu"))
    test_predictions = _predictions(
        pd.DataFrame({"patient_id": test.patient_id, "y_true": test.y_true}), test, test_scores
    )
    entry = {
        "model": name,
        "input_group": "holter_raw_waveform",
        "validation_predictions": validation_predictions,
        "validation_metrics": validation_metrics,
        "validation_threshold_metrics": {
            key: validation_metrics[key] for key in ("threshold", "sensitivity", "specificity")
        },
        "test_predictions": test_predictions,
        "test_metrics": _metrics(test_predictions, threshold),
        "training_metrics": metrics.get("training", {}),
        "parameters": _parameters(metadata, resolved, "dl"),
        "artifacts": {
            "mlflow_model": flavor_dir,
            "checkpoint": run_dir / "best_model.pt",
            "metadata": run_dir / "metadata.json",
        },
    }
    return [entry], name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("ml", "dl"), required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Completed model run directory; repeat for ML candidate models",
    )
    parser.add_argument("--basic-config", type=Path, default=Path("Members/sang/model/basic.yaml"))
    parser.add_argument(
        "--output", type=Path, required=True, help="New or empty local handoff bundle directory"
    )
    args = parser.parse_args()
    if args.modality == "dl" and len(args.run_dir) != 1:
        parser.error("DL export accepts exactly one --run-dir")
    if args.modality == "ml" and not args.run_dir:
        parser.error("ML export requires at least one --run-dir")

    basic_path = (
        args.basic_config.resolve()
        if args.basic_config.is_absolute()
        else (REPO_ROOT / args.basic_config).resolve()
    )
    basic = load_basic_config(basic_path)
    _, cohorts, audit, indices = load_cohorts(basic, basic_path)
    with tempfile.TemporaryDirectory(prefix="scd-mlflow-flavors-") as flavor_temp:
        flavor_root = Path(flavor_temp)
        if args.modality == "ml":
            runs, selected = _ml_runs(args.run_dir, cohorts, flavor_root)
        else:
            runs, selected = _dl_run(
                args.run_dir[0], cohorts, indices, REPO_ROOT / "Members/sang/model", flavor_root
            )
        modality = "clinical" if args.modality == "ml" else "holter_raw_waveform"
        split_name = "ml_patients" if args.modality == "ml" else "dl_patients"
        split_summary = {
            name: audit["split_summary"][name][split_name]
            for name in ("train", "validation", "test")
        }
        if args.modality == "ml":
            source_files = [
                REPO_ROOT / "Members/sang/model/ML/training.py",
                REPO_ROOT / "Members/sang/model/inference.py",
                REPO_ROOT / "Members/sang/model/data.py",
                REPO_ROOT / "Members/sang/model/basic.yaml",
                REPO_ROOT / "Members/sang/model/tracking/model_flavors.py",
            ]
            dataset = "sang-scd-clinical-v1"
        else:
            source_files = [
                REPO_ROOT / "Members/sang/model/DL/training.py",
                REPO_ROOT / "Members/sang/model/DL/composed.py",
                REPO_ROOT / "Members/sang/model/inference.py",
                REPO_ROOT / "Members/sang/model/data.py",
                REPO_ROOT / "Members/sang/model/basic.yaml",
                REPO_ROOT / "Members/sang/model/tracking/model_flavors.py",
            ]
            dataset = "sang-scd-holter-v1"

        for run in runs:
            run.pop("_run_dir", None)
            run.pop("_metadata", None)
        output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        bundle_path = write_bundle(
            output,
            experiment=f"sang-scd-{args.modality}-comparison",
            dataset=dataset,
            input_group=modality,
            split_summary=split_summary,
            selected_model=selected,
            environment=_versions(),
            runs=runs,
            source_files=source_files,
        )
        manifest, validated_runs, bundle_hash = validate_bundle(bundle_path)
        print(
            json.dumps(
                {
                    "status": "VALIDATED_LOCAL_BUNDLE",
                    "bundle": str(bundle_path),
                    "bundle_sha256": bundle_hash,
                    "selected_model": manifest["selection"]["model"],
                    "runs": [item["data"]["model"] for item in validated_runs],
                    "upload_performed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
