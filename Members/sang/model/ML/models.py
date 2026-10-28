"""Configured clinical estimator and train-fitted preprocessing factories."""

from __future__ import annotations

import inspect
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

_ESTIMATOR_DEFAULTS = {
    "XGBoost": {
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": 0,
    },
    "CatBoost": {
        "random_seed": 42,
        "verbose": False,
        "thread_count": 1,
        "allow_writing_files": False,
    },
    "Random Forest": {
        "max_depth": None,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": 1,
    },
}

PREPROCESSING_DEFAULTS = {
    "numeric_scaling": "standard",
    "categorical_encoding": "one_hot",
    "unknown_category": "ignore",
}


def make_preprocessor(features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    categorical = [name for name in categorical_features if name in features]
    numeric = [name for name in features if name not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                StandardScaler(),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown=PREPROCESSING_DEFAULTS["unknown_category"],
                    sparse_output=False,
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def prepare_categorical_frame(frame: pd.DataFrame, categorical_features: list[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for name in categorical_features:
        if name in prepared:
            if prepared[name].isna().any():
                raise ValueError(f"Categorical feature {name!r} contains missing values")
            prepared[name] = prepared[name].astype(str)
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
    estimator.set_params(**{**_ESTIMATOR_DEFAULTS[name], **parameters})
    return estimator
