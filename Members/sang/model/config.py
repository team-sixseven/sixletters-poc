"""Shared YAML configuration loading for the sang model experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

_BASIC_DEFAULTS = {
    "data": {
        "clinical": {
            "categorical_features": [
                "NYHA class",
                "Right ventricle contractility (altered=1)",
                "Mitral valve insufficiency",
                "Diabetes (yes=1)",
                "Prior Myocardial Infarction (yes=1)",
                "Gender (male=1)",
                "Ischemic etiology",
            ],
            "normalize_column_whitespace": True,
            "target_semantics": (
                "Provided binary SCD label; prediction time and horizon are not specified "
                "in the local data dictionary."
            ),
        },
        "split": {
            "id_column": "pid",
            "split_column": "split",
            "values": {"train": "train", "validation": "val", "test": "test"},
        },
        "ecg": {
            "filename_pattern": "{pid}.npy",
            "patient_id_regex": "^(P[0-9]{4})$",
            "metadata_id_column": "pid",
            "metadata_row_column": "row",
            "metadata_time_column": "start_sec",
            "metadata_quality_column": "sqi_pass",
            "sample_rate_status": "assumed_from_source_recording_spec",
            "window_selection": "evenly_spaced_chronological",
            "normalization": "per_window_per_lead_zscore",
            "quality_filter": "sqi_pass_only",
        },
    },
    "outputs": {"audit_directory": "reports/data_audit"},
}


def resolve_config_path(value: str | Path, owner: Path) -> Path:
    """Resolve a path written in a YAML file relative to that YAML file."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (owner.resolve().parent / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping; a relative entry point is resolved from the repository root."""
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    with resolved.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {resolved}")
    return value


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_basic_config(path: Path) -> dict[str, Any]:
    """Resolve shared defaults without mutating defaults or the loaded YAML mapping."""

    def merge(defaults: dict, overrides: dict) -> dict:
        result = deepcopy(defaults)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = value
        return result

    config = merge(_BASIC_DEFAULTS, load_yaml(path))
    root_value = config.get("drive_root")
    path_fields = {
        "clinical": ("path",),
        "split": ("path",),
        "raw_ecg": ("split_path", "wave_directory", "quality_index_path", "lead_metadata_path"),
    }
    for section, fields in path_fields.items():
        values = config.get("data", {}).get(section, {})
        for field in fields:
            value = values.get(field)
            if isinstance(value, str) and value.startswith("drive://"):
                relative = Path(value.removeprefix("drive://"))
                if not root_value or not Path(root_value).is_absolute():
                    raise ValueError("drive:// paths require an absolute drive_root")
                if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Invalid drive-relative path: {value}")
                values[field] = str(Path(root_value) / relative)
    return config


def load_experiment_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    """Load an experiment YAML and its shared basic.yaml reference."""
    experiment_path = path.resolve()
    experiment = load_yaml(experiment_path)
    if "cnn_model" in experiment:
        if set(experiment) != {"cnn_model", "sequence_model"}:
            raise ValueError("DL/model_choice.yaml must contain only cnn_model and sequence_model")
        basic_path = dl_config_paths(experiment_path)["basic"]
        return experiment, load_basic_config(basic_path), experiment_path, basic_path
    if "basic_config" not in experiment:
        raise ValueError(f"{experiment_path}: missing required basic_config")
    basic_path = resolve_config_path(experiment["basic_config"], experiment_path)
    basic = load_basic_config(basic_path)
    return experiment, basic, experiment_path, basic_path


def resolve_data_path(value: str | Path, basic_path: Path) -> Path:
    """Resolve a data/output path relative to the basic config that declares it."""
    return resolve_config_path(value, basic_path)


def dl_config_paths(choice_path: Path) -> dict[str, Path]:
    """Repository layout, not experiment YAML, owns configuration-file routing."""
    directory = choice_path.resolve().parent
    return {
        "basic": directory.parent / "basic.yaml",
        "training": directory / "training.yaml",
        "cnn": directory / "CNN/model_set.yaml",
        "lstm": directory / "LSTM/model_set.yaml",
        "transformer": directory / "Transformer/transformer.yaml",
    }
