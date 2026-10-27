"""Configured clinical estimator and train-fitted preprocessing factories."""

from __future__ import annotations

import inspect
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def make_preprocessor(
    features: list[str], categorical_features: list[str], config: dict[str, Any]
) -> ColumnTransformer:
    categorical = [name for name in categorical_features if name in features]
    numeric = [name for name in features if name not in categorical]
    if config["categorical_encoding"] != "one_hot":
        raise ValueError("Only preprocessing.categorical_encoding=one_hot is supported")
    if config["numeric_scaling"] != "standard":
        raise ValueError("Only preprocessing.numeric_scaling=standard is supported")
    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy=config["numeric_imputation"], keep_empty_features=True),
                    StandardScaler(),
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
        verbose_feature_names_out=True,
    )


def prepare_categorical_frame(
    frame: pd.DataFrame, categorical_features: list[str], missing_value: str
) -> pd.DataFrame:
    prepared = frame.copy()
    for name in categorical_features:
        if name in prepared:
            prepared[name] = prepared[name].astype("string").fillna(missing_value).astype(str)
    return prepared


def build_estimator(name: str, parameters: dict[str, Any]):
    if name == "XGBoost":
        estimator = XGBClassifier()
        supported = set(estimator.get_params(deep=False))
    elif name == "CatBoost":
        estimator = CatBoostClassifier()
        supported = set(inspect.signature(CatBoostClassifier.__init__).parameters) - {"self"}
    elif name == "Random Forest":
        estimator = RandomForestClassifier()
        supported = set(estimator.get_params(deep=False))
    else:
        raise ValueError(
            f"Unsupported clinical model {name!r}; supported: XGBoost, CatBoost, Random Forest"
        )
    unknown = sorted(set(parameters) - supported)
    if unknown:
        raise ValueError(f"{name} parameter YAML has unsupported model keys: {unknown}")
    estimator.set_params(**parameters)
    return estimator
