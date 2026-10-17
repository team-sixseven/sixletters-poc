"""Checked common inputs and evaluation/storage for independently trained models.

This module never imports a personal training script, fits a model or deserializes
model files. Callers own training, inference and model export.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import shutil
import sys
import tomllib
import unicodedata
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from . import evaluation
from .contracts import _hash_json
from .data.prepared import BUNDLE_FILENAME, load_prepared_dataset
from .features import select_features
from .provenance import (
    PACKAGE,
    experiment_resources,
    file_hash,
    git_state,
    python_sources,
    resource_metadata,
    scientific_sources,
    shared_code_scope,
    source_hash,
)
from .workspace import validate_component, validate_experiment_identity

METRICS = ("roc_auc", "average_precision", "brier_score")
RESULT_SCHEMA_VERSION = 3
_FLOAT_TAG = "__sixletters_float__"
_FLOAT_VALUES = {"nan", "+inf", "-inf"}


def expected_features(groups: dict, feature_set: str) -> tuple[str, ...]:
    if feature_set == "combined":
        return (*groups["clinical"], *groups["ecg"])
    return tuple(groups[feature_set])


def prediction_identity(predictions: pd.DataFrame) -> dict[str, str]:
    ordered = predictions.sort_values("patient_id")
    ids = ordered["patient_id"].tolist()
    labels = ordered["target"].astype(int).tolist()
    return {
        "patient_hash": _hash_json(ids),
        "target_hash": _hash_json(list(zip(ids, labels, strict=True))),
    }


def _validate_training_settings(seed, parameters) -> None:
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a non-negative integer or None")
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a dictionary")
    try:
        json.dumps(parameters, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Training settings must be finite JSON-compatible values") from error


def _normalize_parameters(parameters: dict, *, stored: bool) -> dict:
    """Record scalar settings without confusing non-finite numbers with strings.

    The reserved tag is only accepted when validating an already saved result.
    Model objects, callbacks and arrays must be described explicitly by callers;
    this function never executes them or silently replaces them with repr().
    """
    active = set()

    def visit(value, path):
        if isinstance(value, np.generic) and not stored:
            converted = value.item()
            if isinstance(converted, np.generic):
                raise ValueError(f"{path}: unsupported NumPy scalar {type(value).__name__}")
            value = converted
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float:
            if math.isfinite(value):
                return value
            if stored:
                raise ValueError(f"{path}: stored numbers must be finite JSON values")
            return {_FLOAT_TAG: "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
        if type(value) not in (dict, list, tuple):
            raise ValueError(
                f"{path}: unsupported {type(value).__name__}; "
                "record a scalar value, list or dictionary instead"
            )
        if id(value) in active:
            raise ValueError(f"{path}: circular reference in training parameters")
        active.add(id(value))
        try:
            if isinstance(value, dict):
                if _FLOAT_TAG in value:
                    if (
                        stored
                        and path != "training.parameters"
                        and set(value) == {_FLOAT_TAG}
                        and type(value[_FLOAT_TAG]) is str
                        and value[_FLOAT_TAG] in _FLOAT_VALUES
                    ):
                        return dict(value)
                    raise ValueError(f"{path}: {_FLOAT_TAG} is reserved for saved float metadata")
                result = {}
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise ValueError(f"{path}: parameter dictionary keys must be strings")
                    result[key] = visit(item, f"{path}[{json.dumps(key, ensure_ascii=False)}]")
                return result
            return [visit(item, f"{path}[{index}]") for index, item in enumerate(value)]
        finally:
            active.remove(id(value))

    try:
        return visit(parameters, "training.parameters")
    except RecursionError as error:
        raise ValueError("training.parameters: nesting is too deep") from error


def validate_training(training: dict, *, stored: bool = False) -> dict:
    if not isinstance(training, dict) or set(training) != {"model", "seed", "parameters"}:
        raise ValueError("Training metadata requires model, seed and parameters")
    if not isinstance(training["model"], str) or not training["model"].strip():
        raise ValueError("model must describe the training method")
    seed = training["seed"]
    if isinstance(seed, np.integer) and not stored:
        seed = int(seed)
    parameters = training["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a dictionary")
    parameters = _normalize_parameters(parameters, stored=stored)
    _validate_training_settings(seed, parameters)
    # Detach metadata and retain strict JSON output; config.toml stays finite-only.
    return json.loads(
        json.dumps(
            {"model": training["model"], "seed": seed, "parameters": parameters}, allow_nan=False
        )
    )


def _load_config(path: Path) -> tuple[bytes, dict, dict]:
    raw = path.read_bytes()
    config = tomllib.loads(raw.decode("utf-8"))
    if set(config) - {"experiment", "parameters"}:
        raise ValueError("Config supports [experiment] and [parameters] only")
    settings = config.get("experiment")
    required = {"owner", "name", "feature_set"}
    if (
        not isinstance(settings, dict)
        or not required.issubset(settings)
        or set(settings) - required - {"seed", "features", "model"}
    ):
        raise ValueError("[experiment] requires owner, name and feature_set")
    for key in ("owner", "name"):
        settings[key] = validate_component(settings[key], key)
    if settings["feature_set"] not in {"clinical", "ecg", "combined"}:
        raise ValueError("feature_set must be clinical, ecg or combined")
    if "features" in settings:
        features = settings["features"]
        if (
            not isinstance(features, list)
            or not features
            or any(not isinstance(value, str) or not value for value in features)
            or len(set(features)) != len(features)
        ):
            raise ValueError("features must be a non-empty list of unique column names")
    settings.setdefault("seed", None)
    parameters = config.get("parameters", {})
    _validate_training_settings(settings["seed"], parameters)
    # Older scripts may still use the configuration as their metadata fallback.
    if "model" in settings and (
        not isinstance(settings["model"], str) or not settings["model"].strip()
    ):
        raise ValueError("model must describe the training method")
    return raw, settings, parameters


def _model_paths(model_files: dict[str, Path]) -> dict[str, Path]:
    if not isinstance(model_files, dict) or not model_files:
        raise ValueError("model_files must contain at least one exported model file")
    paths = {}
    portable = []
    for name, source in model_files.items():
        if not isinstance(name, str):
            raise ValueError("Model filenames must be relative paths")
        relative = PurePosixPath(name)
        if (
            not relative.parts
            or relative.is_absolute()
            or relative.as_posix() != name
            or any(validate_component(part, "model filename") != part for part in relative.parts)
        ):
            raise ValueError("Model filenames must be portable relative paths")
        parts = tuple(unicodedata.normalize("NFC", part).casefold() for part in relative.parts)
        if any(parts[: len(other)] == other or other[: len(parts)] == parts for other in portable):
            raise ValueError("Model filenames collide across operating systems")
        portable.append(parts)
        path = Path(source)
        if not path.is_file():
            raise ValueError("Every model_files value must name an existing exported file")
        paths[name] = path
    return paths


def _experiment_resources(config: Path, *, exclude: set[str] | None = None) -> dict[str, bytes]:
    """Flatten only the active package manifest alongside experiment-local settings."""
    from .personal_env import LOCK_NAMES, environment_resources

    # Inactive experiment-local package files must not shadow a selected member
    # environment. Never walk the member directory: it contains unrelated runs.
    resources = experiment_resources(
        config.parent, config_name=config.name, exclude=set(LOCK_NAMES) | (exclude or set())
    )
    resources.update(environment_resources(config.parent))
    resource_metadata(resources)
    return resources


def _validate_environment_runtime(resources: dict[str, bytes], software: dict) -> None:
    """Do not describe an unused managed package lock as the training environment."""
    from .personal_env import _digest, _validate_lock

    managed_fields = {
        "schema_version",
        "uv_version",
        "python",
        "base_packages",
        "base_fingerprint",
        "requirements_sha256",
        "lock_sha256",
        "torch_backend",
    }
    try:
        manifest = json.loads(resources.get("environment.json", b"null"))
    except (ValueError, UnicodeDecodeError):
        return  # Older handwritten manifests remain opaque experiment resources.
    if not isinstance(manifest, dict) or not managed_fields.issubset(manifest):
        return
    if manifest["schema_version"] != 1:
        return
    try:
        content = resources.get("requirements.in", b"").decode("utf-8")
        lock = resources["requirements.lock"].decode("utf-8")
        if (
            _digest(content) != manifest["requirements_sha256"]
            or _digest(lock) != manifest["lock_sha256"]
        ):
            raise ValueError("Managed package records do not match their manifest")
        _validate_lock(lock, {"packages": manifest["base_packages"]})
        expected = {}
        for line in lock.splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                requirement = Requirement(line)
                expected[canonicalize_name(requirement.name)] = str(
                    Version(next(iter(requirement.specifier)).version)
                )
        actual = {
            canonicalize_name(name): str(Version(version))
            for name, version in software["packages"].items()
            if canonicalize_name(name) != "sixletters-poc"
        }
        if expected != actual:
            raise ValueError("Installed package versions differ from the active package lock")
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise ValueError(
            "The active package lock does not match this Python environment. "
            "Run the script with ./poc exec, or select its registered SixSeven notebook "
            "kernel; synchronize changed package files with ./poc env sync first."
        ) from error


class Experiment:
    """Capture provenance before personal training, then validate and save its results.

    Input helpers return copies, with patient IDs as their index. They omit held-out
    labels/test rows by default, but are not a security boundary for arbitrary code.
    """

    def __init__(self, data: Path, config: Path, output_root: Path | None = None):
        if Path(config).is_symlink():
            raise ValueError("The experiment configuration must not be a symbolic link")
        self._config = Path(config).resolve()
        self._raw_config, self._settings, self._parameters = _load_config(self._config)
        validate_experiment_identity(self._config, settings=self._settings)
        from .local_settings import project_root, resolve_workspace_path

        root = project_root(self._config.parent)
        output = (
            Path(output_root).absolute() if output_root is not None else root / "Global/artifacts"
        )
        self._output_root = resolve_workspace_path(root, output).resolve()
        result_parent = self._output_root / self._settings["owner"] / self._settings["name"]
        if result_parent.resolve().is_relative_to(self._config.parent):
            raise ValueError("output_root must keep results outside the experiment source folder")
        data = resolve_workspace_path(root, Path(data).absolute())
        self._data_path = (data / BUNDLE_FILENAME if data.is_dir() else data).resolve()
        self._data_hash = file_hash(self._data_path)
        self._dataset, self._synthetic = load_prepared_dataset(self._data_path)
        if file_hash(self._data_path) != self._data_hash:
            raise ValueError("Common data changed while loading")
        self._fingerprint = self._dataset.fingerprint()
        available = tuple(select_features(self._dataset, self._settings["feature_set"]))
        if available != expected_features(
            self._dataset.feature_groups, self._settings["feature_set"]
        ):
            raise ValueError("Selected features differ from the declared feature group")
        self._columns = list(self._settings.get("features", available))
        if not set(self._columns).issubset(available):
            raise ValueError("features must stay within the permitted input group")
        self._sources = python_sources(self._config.parent)
        if not self._sources:
            raise ValueError("The experiment folder must contain at least one Python source file")
        from .personal_env import environment_directory

        self._environment_directory = environment_directory(self._config.parent)
        self._resources = _experiment_resources(self._config)
        self._shared_sources = python_sources(PACKAGE)
        self._software = {
            "python": sys.version,
            "packages": dict(
                sorted(
                    (dist.metadata["Name"], dist.version)
                    for dist in importlib.metadata.distributions()
                    if dist.metadata["Name"]
                )
            ),
        }
        _validate_environment_runtime(self._resources, self._software)
        self._provenance = git_state()
        from .execution import current_invocation

        self._execution = current_invocation(self._config.parent)
        train = self._dataset.frame.loc[self._dataset.frame["split"].eq("train")]
        if set(train["target"]) != {0, 1}:
            raise ValueError("Training data must contain both target classes")

    @property
    def settings(self) -> dict:
        return copy.deepcopy(self._settings)

    @property
    def parameters(self) -> dict:
        return copy.deepcopy(self._parameters)

    @property
    def categorical_features(self) -> tuple[str, ...]:
        return tuple(name for name in self._dataset.categorical_features if name in self._columns)

    def _inputs(self, split: str) -> pd.DataFrame:
        rows = self._dataset.frame.loc[self._dataset.frame["split"].eq(split)]
        frame = rows.set_index("patient_id").sort_index()[self._columns].copy()
        for column in self.categorical_features:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
        return frame

    def training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return training features/labels; fit preprocessing on these patients only."""
        inputs = self._inputs("train")
        target = self._dataset.frame.set_index("patient_id").loc[inputs.index, "target"].copy()
        return inputs, target

    def validation_data(self) -> pd.DataFrame:
        """Return common validation inputs without their labels or any test rows."""
        return self._inputs("validation")

    def _checked_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if (
            not isinstance(predictions, pd.DataFrame)
            or predictions.columns.has_duplicates
            or set(predictions.columns) != {"patient_id", "score"}
        ):
            raise ValueError("Submit exactly patient_id and score; targets come from common data")
        ids = predictions["patient_id"]
        if (
            ids.isna().any()
            or ids.duplicated().any()
            or not ids.map(lambda value: isinstance(value, str)).all()
        ):
            raise ValueError("Predictions require unique string patient IDs")
        validation = self._dataset.frame.loc[self._dataset.frame["split"].eq("validation")]
        if set(ids) != set(validation["patient_id"]):
            raise ValueError("Predictions must contain exactly the common validation patients")
        scores = predictions["score"]
        if (
            not is_numeric_dtype(scores.dtype)
            or is_complex_dtype(scores.dtype)
            or is_bool_dtype(scores.dtype)
        ):
            raise ValueError("Scores must be numeric probabilities")
        # Use one precision for evaluation and CSV persistence. A float32 column's
        # abbreviated CSV representation would otherwise change its Brier score.
        predictions = predictions.copy()
        predictions["score"] = scores.astype(float)
        result = (
            validation[["patient_id", "target"]]
            .merge(predictions, on="patient_id", how="left", validate="one_to_one")
            .sort_values("patient_id")
            .reset_index(drop=True)
        )
        return result

    def _check_unchanged(self, model_resources: set[str]) -> None:
        from .personal_env import environment_directory

        if (
            file_hash(self._data_path) != self._data_hash
            or self._dataset.fingerprint() != self._fingerprint
        ):
            raise ValueError("Common data changed during the experiment")
        if self._config.is_symlink() or self._config.read_bytes() != self._raw_config:
            raise ValueError("Configuration changed during training; start a new experiment")
        if python_sources(self._config.parent) != self._sources:
            raise ValueError("Personal source code changed during training; run again")
        if environment_directory(self._config.parent) != self._environment_directory:
            raise ValueError("Package environment changed during training; run again")
        if _experiment_resources(self._config, exclude=model_resources) != {
            name: raw for name, raw in self._resources.items() if name not in model_resources
        }:
            raise ValueError("Experiment resources changed during training; run again")
        if scientific_sources(python_sources(PACKAGE)) != scientific_sources(self._shared_sources):
            raise ValueError("Shared scientific code changed during training; run again")

    def save(
        self,
        predictions: pd.DataFrame,
        *,
        model_files: dict[str, Path],
        training: dict | None = None,
    ) -> Path:
        """Evaluate validation predictions and copy exported models into a fresh run.

        Models are opaque files: the caller defines their format and how to reload
        them. Training metadata records the actual model and fitted settings.
        Older configurations with a model field remain supported as a fallback
        when training metadata is omitted.
        Saving is local only. The workspace worker or explicit synchronization
        discovers the completed result and uploads it separately.
        """
        from .personal_env import LOCK_NAMES, environment_directory

        files = _model_paths(model_files)
        active_environment = environment_directory(self._config.parent)
        environment_files = {(active_environment / name).resolve() for name in LOCK_NAMES}
        model_resources = set()
        for source in files.values():
            resolved = source.resolve()
            if resolved in environment_files:
                raise ValueError(
                    "Model exports must not overwrite active package environment files"
                )
            if resolved.is_relative_to(self._config.parent):
                model_resources.add(resolved.relative_to(self._config.parent).as_posix())
        # Explicit exports are outputs, even when their format also resembles a
        # configuration file. Active config and Python sources are checked separately.
        self._check_unchanged(model_resources)
        resources = {
            name: raw for name, raw in self._resources.items() if name not in model_resources
        }
        checked = self._checked_predictions(predictions)
        metrics = evaluation.evaluate_predictions(checked)
        if training is None:
            if "model" not in self._settings:
                raise ValueError(
                    "Pass training={'model': ..., 'seed': ..., 'parameters': ...} "
                    "from the trained model to save(); config.toml does not need a model name."
                )
            training = {
                "model": self._settings["model"],
                "seed": self._settings["seed"],
                "parameters": self._parameters,
            }
        method = validate_training(training)
        method.update(
            {
                "source_hash": source_hash(self._sources),
                "shared_code_hash": source_hash(self._shared_sources),
            }
        )
        run_id = uuid4().hex
        run_dir = self._output_root / self._settings["owner"] / self._settings["name"] / run_id
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "owner": self._settings["owner"],
            "experiment": self._settings["name"],
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "feature_set": self._settings["feature_set"],
            "feature_count": len(self._columns),
            "features": self._columns,
            "feature_groups": self._dataset.feature_groups,
            "synthetic": self._synthetic,
            "comparison_context": {
                "dataset": self._fingerprint,
                "evaluation": {
                    "split": "validation",
                    "metrics": list(METRICS),
                    "code_hash": source_hash(
                        {"evaluation.py": self._shared_sources["evaluation.py"]}
                    ),
                    "versions": {
                        "numpy": np.__version__,
                        "pandas": pd.__version__,
                        "scikit_learn": sklearn.__version__,
                    },
                    **prediction_identity(checked),
                },
            },
            "training": method,
            "shared_code_scope": shared_code_scope(self._shared_sources),
            "software": self._software,
            "metrics": metrics,
            "provenance": self._provenance,
            "model_files": {},
            "resources": resource_metadata(resources),
            "config_filename": self._config.name,
        }
        if self._execution is not None:
            result["execution"] = self._execution
        # Exclusive creation prevents a UUID collision from overwriting an existing run.
        run_dir.mkdir(parents=True, exist_ok=False)
        try:
            (run_dir / "config.toml").write_bytes(self._raw_config)
            checked.to_csv(run_dir / "predictions.csv", index=False)
            for folder, sources in (
                ("source", self._sources),
                ("shared_source", self._shared_sources),
                ("resources", resources),
            ):
                for name, content in sources.items():
                    destination = run_dir / folder / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content)
            for name, source in files.items():
                destination = run_dir / "models" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                before = file_hash(source)
                shutil.copyfile(source, destination)
                if file_hash(destination) != before or file_hash(source) != before:
                    raise ValueError("Exported model file changed while saving")
                result["model_files"][name] = {
                    "sha256": before,
                    "bytes": destination.stat().st_size,
                }
            self._check_unchanged(model_resources)
            # Publish completion atomically: a concurrent worker must never see
            # partially written JSON or upload a run that is still being saved.
            marker = run_dir / ".result.json.pending"
            with marker.open("x", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                )
            marker.replace(run_dir / "result.json")
        except BaseException:
            shutil.rmtree(run_dir)
            raise
        return run_dir
