"""Shared YAML configuration loading for the sang model experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


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


def load_experiment_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    """Load an experiment YAML and its shared basic.yaml reference."""
    experiment_path = path.resolve()
    experiment = load_yaml(experiment_path)
    if "basic_config" not in experiment:
        raise ValueError(f"{experiment_path}: missing required basic_config")
    basic_path = resolve_config_path(experiment["basic_config"], experiment_path)
    basic = load_yaml(basic_path)
    return experiment, basic, experiment_path, basic_path


def resolve_data_path(value: str | Path, basic_path: Path) -> Path:
    """Resolve a data/output path relative to the basic config that declares it."""
    return resolve_config_path(value, basic_path)
