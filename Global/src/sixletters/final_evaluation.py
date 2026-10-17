"""Freeze validation-selected candidates and evaluate held-out predictions once.

Personal code owns model loading and inference. This module never imports or
deserializes a model and never fits preprocessing or selects a threshold on test.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from .contracts import _hash_json
from .data.prepared import BUNDLE_FILENAME, load_prepared_dataset
from .final_metrics import build_final_report, select_threshold
from .provenance import file_hash
from .tracking import _full_manifest


def _replay_tolerance(plan: dict) -> dict[str, float]:
    # Version 1 retains its original float64-oriented policy. Version 2 allows
    # ordinary float32 batch-size rounding, not a user-adjustable model change.
    if plan["schema_version"] == 1:
        if "validation_replay_tolerance" in plan:
            raise ValueError("Legacy evaluation plans cannot change their replay tolerance")
        return {"atol": 1e-10, "rtol": 1e-8}
    tolerance = {"atol": 1e-7, "rtol": 1e-5}
    if plan.get("validation_replay_tolerance") != tolerance:
        raise ValueError("Frozen validation replay tolerance is missing or modified")
    return tolerance


def _engine_identity() -> dict:
    import importlib.metadata

    package = Path(__file__).parent
    return {
        "sources": {
            name: file_hash(package / name) for name in ("final_evaluation.py", "final_metrics.py")
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "matplotlib")
        },
    }


def _safe_path(path: Path) -> Path:
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Final evaluation paths must not contain symbolic links")
    return path.resolve()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Final evaluation metadata must be an object")
    return value


def _data(path: Path):
    path = _safe_path(path)
    path = path / BUNDLE_FILENAME if path.is_dir() else path
    return load_prepared_dataset(path)


def _predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, dtype={"patient_id": str}, keep_default_na=False, float_precision="round_trip"
    )


def _plan(directory: Path) -> dict:
    path = _safe_path(directory) / "plan.json"
    plan = _read_json(path)
    digest = plan.pop("plan_hash", None)
    if plan.get("schema_version") not in (1, 2) or digest != _hash_json(plan):
        raise ValueError("Frozen evaluation plan is incomplete or modified")
    _replay_tolerance(plan)
    if plan.get("engine") != _engine_identity():
        raise ValueError(
            "Final evaluation code or packages changed after freezing; use the frozen evaluation environment"
        )
    plan["plan_hash"] = digest
    return plan


def freeze_candidates(
    run_dirs: list[Path],
    data: Path,
    destination: Path,
    *,
    threshold: float | None = None,
    reference: str | None = None,
) -> Path:
    """Snapshot selected models and validation-derived thresholds before test access."""
    destination = _safe_path(destination)
    if not run_dirs:
        raise ValueError("Select at least one completed validation run")
    dataset, synthetic = _data(data)
    fingerprint = dataset.fingerprint()
    selected = []
    for run_dir in run_dirs:
        run_dir = _safe_path(run_dir)
        if destination.is_relative_to(run_dir) or run_dir.is_relative_to(destination):
            raise ValueError("Final plan and source runs must not overlap")
        result, manifest = _full_manifest(run_dir)
        if (
            result["comparison_context"]["dataset"] != fingerprint
            or result["synthetic"] != synthetic
        ):
            raise ValueError("Every candidate must use the selected common data and patient split")
        if selected and result["comparison_context"] != selected[0][1]["comparison_context"]:
            raise ValueError("Candidates must share the same validation evaluation version")
        validation = _predictions(run_dir / "predictions.csv")
        expected = dataset.frame.loc[dataset.frame["split"].eq("validation")].set_index(
            "patient_id"
        )
        if set(validation["patient_id"]) != set(expected.index) or not np.array_equal(
            validation["target"].to_numpy(),
            expected.loc[validation["patient_id"], "target"].to_numpy(),
        ):
            raise ValueError("Candidate validation patients or targets differ from common data")
        choice = select_threshold(validation, fixed=threshold)
        selected.append((run_dir, result, manifest, choice))
    ids = [result["run_id"] for _, result, _, _ in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Choose distinct validation runs")
    if reference is not None and reference not in ids:
        raise ValueError("The reference must be a selected local run ID")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination.parent / f".{destination.name}.lock")):
        if destination.exists():
            raise FileExistsError(
                "This final plan already exists; frozen candidates cannot be replaced"
            )
        with tempfile.TemporaryDirectory(prefix=".freeze-", dir=destination.parent) as temporary:
            stage = Path(temporary) / "plan"
            stage.mkdir()
            plan = {
                "schema_version": 2,
                "created_at": datetime.now(UTC).isoformat(),
                "dataset": fingerprint,
                "synthetic": synthetic,
                "reference": reference or ids[0],
                "engine": _engine_identity(),
                "validation_replay_tolerance": {"atol": 1e-7, "rtol": 1e-5},
                "candidates": {},
                "protocol": "Frozen validation-selected models; no refitting on validation or test",
            }
            for run_dir, result, manifest, choice in selected:
                relative = Path("runs") / result["owner"] / result["experiment"] / result["run_id"]
                copied = stage / relative
                shutil.copytree(run_dir, copied)
                if _full_manifest(copied)[1] != manifest:
                    raise ValueError("Candidate files changed while freezing")
                plan["candidates"][result["run_id"]] = {
                    "directory": relative.as_posix(),
                    "manifest": manifest,
                    "threshold": choice,
                    "owner": result["owner"],
                    "experiment": result["experiment"],
                    "feature_set": result["feature_set"],
                    "features": result["features"],
                    "model": result["training"]["model"],
                }
            plan["plan_hash"] = _hash_json(plan)
            _write_json(stage / "plan.json", plan)
            stage.rename(destination)
    return destination / "plan.json"


class FinalEvaluation:
    """Provide frozen-model inputs and accept checked predictions, never fit a model."""

    def __init__(self, plan: Path, run_id: str, data: Path):
        self.directory = _safe_path(plan)
        self.plan = _plan(self.directory)
        if run_id not in self.plan["candidates"]:
            raise ValueError("Run ID is not in this frozen evaluation plan")
        self.run_id = run_id
        self.candidate = self.plan["candidates"][run_id]
        self._dataset, synthetic = _data(data)
        if (
            self._dataset.fingerprint() != self.plan["dataset"]
            or synthetic != self.plan["synthetic"]
        ):
            raise ValueError("Final evaluation requires the exact frozen data and patient split")
        relative = Path(self.candidate["directory"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid frozen model path")
        self.run_directory = _safe_path(self.directory / relative)
        self._verify()

    def _verify(self) -> None:
        if _plan(self.directory) != self.plan:
            raise ValueError("Frozen plan changed during inference")
        if _full_manifest(self.run_directory)[1] != self.candidate["manifest"]:
            raise ValueError("Frozen model or reproduction files were modified")

    def _inputs(self, split: str) -> pd.DataFrame:
        self._verify()
        frame = self._dataset.frame
        inputs = (
            frame.loc[frame["split"].eq(split)]
            .set_index("patient_id")
            .sort_index()[self.candidate["features"]]
            .copy()
        )
        for column in self._dataset.categorical_features:
            if column in inputs:
                inputs[column] = inputs[column].astype(object).where(inputs[column].notna(), np.nan)
        return inputs

    def validation_data(self) -> pd.DataFrame:
        """Replay validation inference to confirm the selected model before test evaluation."""
        return self._inputs("validation")

    def test_data(self) -> pd.DataFrame:
        """Return held-out features without labels for the already frozen candidate."""
        return self._inputs("test")

    def _checked(self, predictions: pd.DataFrame, split: str) -> pd.DataFrame:
        if not isinstance(predictions, pd.DataFrame) or list(predictions.columns) != [
            "patient_id",
            "score",
        ]:
            raise ValueError(
                "Submit patient_id and score columns, in that order; labels come from common data"
            )
        ids = predictions["patient_id"]
        rows = self._dataset.frame.loc[self._dataset.frame["split"].eq(split)]
        if ids.isna().any() or ids.duplicated().any() or set(ids) != set(rows["patient_id"]):
            raise ValueError(f"Predictions must contain exactly the common {split} patients")
        if (
            not pd.api.types.is_numeric_dtype(predictions["score"])
            or pd.api.types.is_bool_dtype(predictions["score"])
            or pd.api.types.is_complex_dtype(predictions["score"])
        ):
            raise ValueError("Scores must be numeric probabilities")
        scores = predictions["score"].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Scores must be finite probabilities between 0 and 1")
        checked = predictions.copy()
        checked["score"] = scores
        return (
            rows[["patient_id", "target"]]
            .merge(checked, on="patient_id", validate="one_to_one")
            .sort_values("patient_id")
            .reset_index(drop=True)
        )

    def save(self, validation: pd.DataFrame, test: pd.DataFrame) -> Path:
        """Persist one immutable test prediction set after matching saved validation scores."""
        self._verify()
        replay = self._checked(validation, "validation")
        expected = _predictions(self.run_directory / "predictions.csv").sort_values("patient_id")
        tolerance = _replay_tolerance(self.plan)
        # The saved probability is the reference for the relative tolerance.
        actual_scores = replay["score"].to_numpy(dtype=float)
        expected_scores = expected["score"].to_numpy(dtype=float)
        matched = np.isclose(actual_scores, expected_scores, **tolerance)
        diagnostics = {
            "max_absolute_delta": float(np.max(np.abs(actual_scores - expected_scores))),
            "mismatch_count": int(np.count_nonzero(~matched)),
            "prediction_count": len(actual_scores),
        }
        if not matched.all():
            raise ValueError(
                "Validation replay differs from the selected run; load its frozen model without refitting. "
                f"Outside tolerance: {diagnostics['mismatch_count']}/{diagnostics['prediction_count']}; "
                f"max absolute delta={diagnostics['max_absolute_delta']:.9g}; "
                f"allowed difference <= atol={tolerance['atol']:g} + "
                f"rtol={tolerance['rtol']:g} * abs(saved probability)"
            )
        checked = self._checked(test, "test")
        destination = _safe_path(self.directory / "predictions" / self.run_id)
        with FileLock(str(self.directory / ".predictions.lock")):
            _safe_path(destination)
            if (self.directory / "report").exists():
                raise ValueError(
                    "The final report is already complete; test predictions are closed"
                )
            if destination.exists():
                raise FileExistsError(
                    "Test predictions already exist for this frozen candidate; they cannot be replaced"
                )
            destination.parent.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".predict-", dir=destination.parent
            ) as temporary:
                stage = Path(temporary) / "result"
                stage.mkdir()
                checked.to_csv(stage / "predictions.csv", index=False, float_format="%.17g")
                _write_json(
                    stage / "receipt.json",
                    {
                        "plan_hash": self.plan["plan_hash"],
                        "run_id": self.run_id,
                        "predictions_sha256": file_hash(stage / "predictions.csv"),
                        "created_at": datetime.now(UTC).isoformat(),
                        "validation_replay": "matched",
                        "validation_replay_tolerance": tolerance,
                        "validation_replay_diagnostics": diagnostics,
                    },
                )
                self._verify()
                _safe_path(destination)
                stage.rename(destination)
        return destination


def generate_final_report(
    plan_directory: Path, data: Path, *, repeats: int = 2000, seed: int = 42
) -> Path:
    """Generate aggregate final metrics once, after every frozen candidate has predictions."""
    directory = _safe_path(plan_directory)
    plan = _plan(directory)
    with FileLock(str(directory / ".predictions.lock")):
        destination = directory / "report"
        if destination.exists():
            raise FileExistsError(
                "The final report already exists; use its saved metrics and figures"
            )
        predictions, thresholds, labels = {}, {}, {}
        for run_id, candidate in plan["candidates"].items():
            session = FinalEvaluation(directory, run_id, data)
            stored = _safe_path(directory / "predictions" / run_id)
            if not (stored / "receipt.json").is_file():
                raise ValueError(
                    f"Missing test predictions for {candidate['owner']}/{candidate['experiment']} ({run_id})"
                )
            receipt = _read_json(stored / "receipt.json")
            if (
                receipt.get("plan_hash") != plan["plan_hash"]
                or receipt.get("run_id") != run_id
                or receipt.get("predictions_sha256") != file_hash(stored / "predictions.csv")
                or receipt.get("validation_replay") != "matched"
                or (
                    plan["schema_version"] >= 2
                    and receipt.get("validation_replay_tolerance") != _replay_tolerance(plan)
                )
            ):
                raise ValueError("Test prediction files are incomplete or modified")
            saved = _predictions(stored / "predictions.csv")
            checked = session._checked(saved[["patient_id", "score"]], "test")
            ordered = saved.sort_values("patient_id").reset_index(drop=True)
            if list(saved.columns) != ["patient_id", "target", "score"] or not np.array_equal(
                ordered["target"], checked["target"]
            ):
                raise ValueError("Stored test labels differ from common data")
            predictions[run_id] = checked
            thresholds[run_id] = candidate["threshold"]["threshold"]
            model_label = " ".join(str(candidate["model"]).split())
            model_label = "".join(char for char in model_label if char.isprintable())[:55]
            labels[run_id] = f"{candidate['feature_set']} / {model_label} / {run_id[:8]}"
        with tempfile.TemporaryDirectory(prefix=".report-", dir=directory) as temporary:
            stage = Path(temporary) / "report"
            stage.mkdir()
            build_final_report(
                predictions,
                thresholds,
                stage,
                reference=plan["reference"],
                repeats=repeats,
                seed=seed,
                labels=labels,
            )
            _write_json(
                stage / "protocol.json",
                {
                    "plan_hash": plan["plan_hash"],
                    "dataset": plan["dataset"],
                    "synthetic": plan["synthetic"],
                    "reference": plan["reference"],
                    "engine": plan["engine"],
                    "validation_replay_tolerance": _replay_tolerance(plan),
                    "candidates": {
                        key: {
                            field: value
                            for field, value in candidate.items()
                            if field not in {"directory", "manifest"}
                        }
                        for key, candidate in plan["candidates"].items()
                    },
                    "warning": "Research/education only; test results must not be used for further candidate selection",
                },
            )
            if _plan(directory) != plan:
                raise ValueError("Frozen plan changed during report generation")
            stage.rename(destination)
        return destination
