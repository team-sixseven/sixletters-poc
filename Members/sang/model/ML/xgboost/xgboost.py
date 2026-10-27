"""XGBoost estimator factory backed by the adjacent parameter YAML."""

from pathlib import Path

from ...config import load_yaml
from ..models import build_estimator


def build_model(parameters: dict | None = None):
    """Build the configured XGBoost estimator, allowing explicit overrides."""
    settings = parameters
    if settings is None:
        settings = load_yaml(Path(__file__).with_name("parameter.yaml"))["model"]
    return build_estimator("XGBoost", settings)
