"""Train clinical candidate models on the shared MUSIC split and score validation only."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sixletters.data.prepared import load_prepared_dataset


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    )
    categorical_pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="__MISSING__"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "Global/data/prepared/music-scd-basic-v1",
    )
    args = parser.parse_args()

    dataset, synthetic = load_prepared_dataset(args.data_dir)
    if synthetic:
        raise ValueError("Refusing to report synthetic data as actual MUSIC results")
    frame = dataset.frame
    features = list(dataset.feature_groups["clinical"])
    categorical = [name for name in dataset.categorical_features if name in features]
    numeric = [name for name in features if name not in categorical]
    train = frame.loc[frame["split"].eq("train")]
    validation = frame.loc[frame["split"].eq("validation")]
    test = frame.loc[frame["split"].eq("test")]
    if train["patient_id"].isin(validation["patient_id"]).any():
        raise ValueError("patient leakage between train and validation")
    if (
        train["patient_id"].isin(test["patient_id"]).any()
        or validation["patient_id"].isin(test["patient_id"]).any()
    ):
        raise ValueError("patient leakage with the held-out test split")

    x_train, y_train = train[features], train["target"].astype(int)
    x_validation, y_validation = validation[features], validation["target"].astype(int)
    preprocessor = make_preprocessor(numeric, categorical)

    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier

    models = {
        "Logistic Regression": make_pipeline(
            clone(preprocessor),
            LogisticRegression(max_iter=1000, random_state=42),
        ),
        "XGBoost": make_pipeline(
            clone(preprocessor),
            XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                n_jobs=1,
                tree_method="hist",
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            ),
        ),
        "CatBoost": CatBoostClassifier(
            iterations=100,
            depth=4,
            learning_rate=0.05,
            random_seed=42,
            verbose=False,
            thread_count=1,
            allow_writing_files=False,
        ),
    }

    rows = []
    for name, model in models.items():
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if name == "CatBoost":
                fit_x = x_train.copy()
                for column in categorical:
                    fit_x[column] = fit_x[column].astype("string").fillna("__MISSING__").astype(str)
                model.set_params(cat_features=categorical)
                model.fit(fit_x, y_train)
            else:
                model.fit(x_train, y_train)
        elapsed = time.perf_counter() - started
        if name == "CatBoost":
            score_x = x_validation.copy()
            for column in categorical:
                score_x[column] = score_x[column].astype("string").fillna("__MISSING__").astype(str)
        else:
            score_x = x_validation
        probabilities = model.predict_proba(score_x)[:, 1]
        if probabilities.shape != (len(validation),) or not np.isfinite(probabilities).all():
            raise ValueError(f"{name} produced invalid validation probabilities")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(f"{name} probability outside [0, 1]")
        rows.append(
            {
                "model": name,
                "input": "Clinical (64 features)",
                "n": len(validation),
                "scd": int(y_validation.sum()),
                "auroc": roc_auc_score(y_validation, probabilities),
                "auprc_average_precision": average_precision_score(y_validation, probabilities),
                "brier": brier_score_loss(y_validation, probabilities),
                "fit_seconds": elapsed,
                "warnings": [f"{w.category.__name__}: {w.message}" for w in caught],
            }
        )

    print(
        json.dumps(
            {
                "dataset_id": dataset.dataset_id,
                "target_definition": dataset.target_definition,
                "split": {
                    "train_n": len(train),
                    "train_scd": int(train["target"].sum()),
                    "validation_n": len(validation),
                    "validation_scd": int(validation["target"].sum()),
                    "test_n": len(test),
                    "test_used": False,
                },
                "feature_count": len(features),
                "seed": 42,
                "versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__,
                    "xgboost": importlib.metadata.version("xgboost"),
                    "catboost": importlib.metadata.version("catboost"),
                },
                "validation_results": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
