"""Keep one completed Optuna winner per clinical algorithm and input version."""

from __future__ import annotations

import json
import math
import re
import shutil
import uuid
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import optuna
import yaml

from ..config import load_experiment_config, resolve_config_path
from .artifacts import save_model_path
from .feature_versions import get_feature_columns
from .training import train_clinical


class ClinicalOptunaExperiment:
    """Trial artifacts are temporary; a successful study replaces its managed best directory."""

    def __init__(
        self,
        choice_path: Path,
        model_name: str,
        model_version: int,
        *,
        mlflow_logging: bool = False,
    ):
        get_feature_columns(model_version)
        choice, _, choice_path, _ = load_experiment_config(choice_path)
        if model_name not in choice["model_configs"]:
            raise ValueError(f"Unsupported ML model: {model_name}")
        if mlflow_logging:
            from ..tracking.check_connection import check_mlflow_connection

            check_mlflow_connection()
        self.choice_path = choice_path
        self.model_name = model_name
        self.model_version = model_version
        self.mlflow_logging = mlflow_logging
        self.parameter_path = resolve_config_path(choice["model_configs"][model_name], choice_path)
        output_root = resolve_config_path(choice["model_output_dir"], choice_path)
        output_root.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
        self.best_directory = output_root / "best" / f"{slug}_v{model_version}"
        self._workspace = TemporaryDirectory(prefix=".optuna_", dir=output_root)
        self._candidate = Path(self._workspace.name) / "best_candidate"
        self._best_score = -math.inf
        self._best_trial = None
        self._closed = False

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Create a new experiment by rerunning the Optuna preparation cell")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self._closed:
            self._workspace.cleanup()
            self._closed = True

    def run_trial(self, trial: optuna.Trial, parameters: dict) -> float:
        if self._closed:
            raise RuntimeError("Create a new experiment by rerunning the Optuna preparation cell")
        with TemporaryDirectory(prefix="trial_", dir=self._workspace.name) as temporary:
            run = train_clinical(
                self.choice_path,
                model_names=[self.model_name],
                model_parameters=parameters,
                model_version=self.model_version,
                reuse_saved=False,
                output_directory=Path(temporary),
                mlflow_logging=False,
            )[0]
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            score = float(metrics["validation"]["average_precision"])
            if not math.isfinite(score):
                raise ValueError("Optuna objective requires a finite validation AP")
            if score > self._best_score:
                metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
                if self._candidate.exists():
                    shutil.rmtree(self._candidate)
                self._candidate.mkdir()
                filenames = [
                    metadata["model_file"],
                    "metadata.json",
                    "metrics.json",
                    "resolved_config.yaml",
                    "run_status.json",
                    "validation_predictions.csv",
                ]
                if (run / "preprocessing.joblib").is_file():
                    filenames.append("preprocessing.joblib")
                for filename in filenames:
                    shutil.copy2(run / filename, self._candidate / filename)
                self._best_score = score
                self._best_trial = trial.number
        trial.set_user_attr("model_version", self.model_version)
        return score

    def save_best(self, study: optuna.Study) -> Path:
        if self._closed or not self._candidate.is_dir():
            raise RuntimeError("No completed trial candidate is available")
        if study.best_trial.number != self._best_trial or study.best_value != self._best_score:
            raise ValueError("Study and retained best model do not match")
        target = self.best_directory
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not (target / "summary.json").is_file():
                raise FileExistsError(f"Refusing to replace an unmanaged model directory: {target}")
            previous = json.loads((target / "summary.json").read_text(encoding="utf-8"))
            if (
                previous.get("storage_policy") != "best_only"
                or previous.get("model") != self.model_name
                or previous.get("model_version") != self.model_version
            ):
                raise FileExistsError(f"Refusing to replace an unrelated model directory: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.loads((self._candidate / "metadata.json").read_text(encoding="utf-8"))
        resolved = yaml.safe_load(
            (self._candidate / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        best_parameters = dict(resolved["parameters"])
        best_parameters.pop("model_paths", None)
        model_file = target / metadata["model_file"]
        best_parameters["model_path"] = str(model_file)
        summary = {
            "storage_policy": "best_only",
            "study_name": study.study_name,
            "model": self.model_name,
            "model_version": self.model_version,
            "feature_names": metadata["feature_names"],
            "best_validation_ap": study.best_value,
            "best_trial": study.best_trial.number,
            "best_model_directory": str(target),
            "model_path": str(model_file),
            "parameter_file": str(self.parameter_path),
            "best_parameters": study.best_params,
            "n_trials": len(study.trials),
        }
        (self._candidate / "best_parameters.yaml").write_text(
            yaml.safe_dump(best_parameters, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        study.trials_dataframe(attrs=("number", "value", "params", "state")).to_csv(
            self._candidate / "trials.csv", index=False
        )
        (self._candidate / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Stage and replace on the same filesystem; keep the previous model until YAML is saved.
        with TemporaryDirectory(prefix=f".{target.name}_new_", dir=target.parent) as temporary:
            staging = Path(temporary) / "run"
            shutil.copytree(self._candidate, staging)
            backup = target.parent / f".{target.name}_old_{uuid.uuid4().hex}"
            if target.exists():
                target.rename(backup)
            try:
                staging.rename(target)
                save_model_path(self.parameter_path, target)
            except Exception:
                if target.exists():
                    shutil.rmtree(target)
                if backup.exists():
                    backup.rename(target)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        if self.mlflow_logging:
            try:
                from ..tracking.log_best import log_best_to_mlflow

                result = log_best_to_mlflow(target)
                print("MLflow 결과:", result["url"])
            except Exception as error:
                summary["mlflow"] = {
                    "status": "PENDING",
                    "error_type": type(error).__name__,
                }
                (target / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (target / "mlflow_status.json").write_text(
                    json.dumps(summary["mlflow"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                warnings.warn(
                    f"MLflow 전송 미완료 ({type(error).__name__}). 로컬 best는 저장됐습니다. "
                    "서버 연결을 확인한 뒤 log_best_to_mlflow(best_run_dir)로 재전송하세요.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return target
