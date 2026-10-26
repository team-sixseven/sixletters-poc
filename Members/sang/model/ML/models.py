"""Clinical tabular model factories and their shared preprocessing schema."""

from __future__ import annotations

from pathlib import Path

from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from ..config import load_yaml

CLINICAL_CHOICE_PATH = Path(__file__).parent / "model_choice.yaml"


def clinical_choice() -> dict:
    return load_yaml(CLINICAL_CHOICE_PATH)


CATEGORICAL_FEATURES = tuple(clinical_choice()["categorical_features"])


def make_preprocessor(features: list[str], config: dict | None = None) -> ColumnTransformer:
    config = config or clinical_choice()["preprocessing"]
    categorical = [name for name in CATEGORICAL_FEATURES if name in features]
    numeric = [name for name in features if name not in categorical]
    if config["categorical_encoding"] != "one_hot":
        raise ValueError("Only one_hot categorical encoding is currently supported")
    if config["numeric_scaling"] != "standard":
        raise ValueError("Only standard numeric scaling is currently supported")
    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy=config["numeric_imputation"]),
                    StandardScaler(with_mean=config["numeric_scaling"] == "standard"),
                ),
                numeric,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(
                        strategy=config["categorical_imputation"],
                        fill_value=config["categorical_missing_value"],
                    ),
                    OneHotEncoder(handle_unknown=config["unknown_category"], sparse_output=False),
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


MODEL_CONFIGS = {
    "Logistic Regression": Path(__file__).parent / "Logistic_Regression/parameter.yaml",
    "XGBoost": Path(__file__).parent / "xgboost/parameter.yaml",
    "CatBoost": Path(__file__).parent / "catboost/parameter.yaml",
}


def build_models(
    features: list[str],
    model_names: list[str] | None = None,
    preprocessing: dict | None = None,
) -> dict[str, object]:
    """Create configured clinical candidates; preprocessors fit only on training data."""
    names = model_names or list(MODEL_CONFIGS)
    unknown = set(names) - set(MODEL_CONFIGS)
    if unknown:
        raise ValueError(f"Unknown clinical models in config: {sorted(unknown)}")
    models = {}
    for name in names:
        params = load_yaml(MODEL_CONFIGS[name])
        preprocessor = make_preprocessor(features, preprocessing)
        if name == "Logistic Regression":
            estimator = LogisticRegression(
                max_iter=params["max_iter"], random_state=params["random_state"]
            )
            models[name] = make_pipeline(preprocessor, estimator)
        elif name == "XGBoost":
            estimator = XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                n_jobs=params["n_jobs"],
                tree_method=params["tree_method"],
                eval_metric=params["eval_metric"],
                random_state=params["random_state"],
                verbosity=params["verbosity"],
            )
            models[name] = make_pipeline(preprocessor, estimator)
        else:
            models[name] = CatBoostClassifier(
                iterations=params["iterations"],
                depth=params["depth"],
                learning_rate=params["learning_rate"],
                random_seed=params["random_seed"],
                verbose=params["verbose"],
                thread_count=params["thread_count"],
                allow_writing_files=params["allow_writing_files"],
            )
    return models


def build_model(name: str, features: list[str]):
    try:
        return build_models(features, [name])[name]
    except KeyError as error:
        raise ValueError(f"Unknown clinical model: {name}") from error
