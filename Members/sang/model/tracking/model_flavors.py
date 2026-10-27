"""Save trained estimators as portable MLflow model flavor directories."""

from __future__ import annotations

import importlib.metadata
import shutil
import tempfile
from pathlib import Path

import numpy as np


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
        _requirement("mlflow-skinny"),
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
    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(output_dir),
        signature=signature,
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
        package_dir = Path(temporary) / "model"
        dl_dir = package_dir / "DL"
        dl_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text(
            '"""Portable model package."""\n', encoding="utf-8"
        )
        (dl_dir / "__init__.py").write_text('"""Portable ECG model package."""\n', encoding="utf-8")
        shutil.copyfile(source_model_dir / "config.py", package_dir / "config.py")
        shutil.copyfile(source_model_dir / "DL/models.py", dl_dir / "models.py")
        mlflow.pytorch.save_model(
            pytorch_model=model.to(torch.device("cpu")).eval(),
            path=str(output_dir),
            code_paths=[str(package_dir)],
            signature=signature,
            serialization_format="pickle",
            pip_requirements=[
                _requirement("mlflow-skinny"),
                _requirement("numpy"),
                _requirement("PyYAML"),
                _requirement("torch"),
            ],
            metadata={"model_name": model_name, "output": "patient-level SCD logit"},
        )
    return output_dir
