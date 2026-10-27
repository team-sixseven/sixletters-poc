"""Save trained estimators as portable MLflow model flavor directories."""

from __future__ import annotations

import importlib.metadata
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


class ClinicalRunModel:
    """MLflow sklearn wrapper accepting the original configured clinical columns."""

    def __init__(self, estimator, preprocessor, metadata: dict) -> None:
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.metadata = metadata
        self.classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        if not isinstance(inputs, pd.DataFrame):
            raise TypeError("Clinical model input must be a pandas DataFrame")
        feature_names = self.metadata["feature_names"]
        missing = sorted(set(feature_names) - set(inputs.columns))
        if missing:
            raise ValueError(f"Clinical input is missing saved features {missing}")
        frame = inputs.loc[:, feature_names].copy()
        artifact_type = self.metadata["artifact_type"]
        if artifact_type in {"xgboost", "random_forest"}:
            frame = self.preprocessor.transform(frame)
        elif artifact_type == "catboost":
            missing_value = self.metadata["preprocessing"]["categorical_missing_value"]
            for name in self.metadata["categorical_features"]:
                frame[name] = frame[name].astype("string").fillna(missing_value).astype(str)
        else:
            raise ValueError(f"Unsupported clinical artifact type {artifact_type!r}")
        probability = np.asarray(self.estimator.predict_proba(frame), dtype=np.float64)
        if probability.ndim != 2 or probability.shape[1] != 2:
            raise ValueError(f"Expected binary predict_proba output, got {probability.shape}")
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("Clinical estimator returned invalid probabilities")
        return probability


def _requirement(distribution: str) -> str:
    return f"{distribution}=={importlib.metadata.version(distribution)}"


def save_sklearn_model(
    model,
    output_dir: Path,
    *,
    feature_names: list[str],
    categorical_features: tuple[str, ...],
    feature_dtypes: dict[str, str],
    model_name: str,
) -> Path:
    """Save a sklearn flavor whose generic predict API returns class probabilities."""
    import mlflow.sklearn
    from mlflow.models import ModelSignature
    from mlflow.types.schema import ColSpec, Schema, TensorSpec

    categorical = set(categorical_features)
    inputs = Schema(
        [
            ColSpec(
                "string" if name in categorical else feature_dtypes[name],
                name=name,
            )
            for name in feature_names
        ]
    )
    signature = ModelSignature(
        inputs=inputs,
        outputs=Schema([TensorSpec(np.dtype("float64"), (-1, 2))]),
    )
    requirements = [
        _requirement("mlflow"),
        _requirement("numpy"),
        _requirement("pandas"),
        _requirement("scikit-learn"),
        _requirement("cloudpickle"),
    ]
    if model_name == "XGBoost":
        requirements.append(_requirement("xgboost"))
    elif model_name == "CatBoost":
        requirements.append(_requirement("catboost"))

    output_dir = Path(output_dir)
    with tempfile.TemporaryDirectory(prefix="scd-mlflow-clinical-code-") as temporary:
        code_root = Path(temporary)
        package_path = code_root / "Members" / "sang" / "model" / "tracking"
        package_path.mkdir(parents=True)
        for directory in (
            code_root / "Members",
            code_root / "Members" / "sang",
            code_root / "Members" / "sang" / "model",
            package_path,
        ):
            (directory / "__init__.py").write_text(
                '"""MLflow packaged model code."""\n', encoding="utf-8"
            )
        shutil.copyfile(Path(__file__).resolve(), package_path / "model_flavors.py")
        mlflow.sklearn.save_model(
            sk_model=model,
            path=str(output_dir),
            signature=signature,
            code_paths=[str(code_root)],
            pyfunc_predict_fn="predict_proba",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            pip_requirements=requirements,
            metadata={"model_name": model_name, "positive_class": 1},
        )
    return output_dir


def save_pytorch_model(
    model,
    output_dir: Path,
    *,
    input_shape: tuple[int, ...],
    source_model_dir: Path,
    model_name: str,
) -> Path:
    """Save a PyTorch flavor with its model code and tensor input/output schema."""
    import mlflow.pytorch
    import torch
    from mlflow.models import ModelSignature
    from mlflow.types.schema import Schema, TensorSpec

    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float32"), (-1, *input_shape))]),
        outputs=Schema([TensorSpec(np.dtype("float32"), (-1, 1))]),
    )
    output_dir = Path(output_dir)
    source_model_dir = Path(source_model_dir)
    with tempfile.TemporaryDirectory(prefix="scd-mlflow-model-code-") as temporary:
        code_root = Path(temporary)
        package_dir = code_root / "Members" / "sang" / "model"
        dl_dir = package_dir / "DL"
        for directory in (
            code_root / "Members",
            code_root / "Members" / "sang",
            package_dir,
            dl_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "__init__.py").write_text(
                '"""Portable ECG model package."""\n', encoding="utf-8"
            )
        shutil.copyfile(source_model_dir / "config.py", package_dir / "config.py")
        shutil.copyfile(source_model_dir / "DL/models.py", dl_dir / "models.py")
        shutil.copyfile(source_model_dir / "DL/composed.py", dl_dir / "composed.py")
        mlflow.pytorch.save_model(
            pytorch_model=model.to(torch.device("cpu")).eval(),
            path=str(output_dir),
            code_paths=[str(code_root)],
            signature=signature,
            serialization_format="pickle",
            pip_requirements=[
                _requirement("mlflow"),
                _requirement("numpy"),
                _requirement("PyYAML"),
                _requirement("torch"),
            ],
            metadata={"model_name": model_name, "output": "patient-level SCD logit"},
        )
    return output_dir
