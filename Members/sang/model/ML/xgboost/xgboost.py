"""XGBoost candidate factory for the common clinical comparison."""

from Members.sang.model.ML.models import build_model as _build_model


def build_model(features: list[str]):
    return _build_model("XGBoost", features)
