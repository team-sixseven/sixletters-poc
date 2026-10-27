"""Compare XGBoost on declared clinical, ECG, and combined feature sets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from Members.sang.model.ML.models import make_preprocessor
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from sixletters.data.prepared import load_prepared_dataset


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
    train = frame.loc[frame["split"].eq("train")]
    validation = frame.loc[frame["split"].eq("validation")]
    test = frame.loc[frame["split"].eq("test")]
    for other in (validation, test):
        if train["patient_id"].isin(other["patient_id"]).any():
            raise ValueError("patient leakage across train and evaluation split")
    if validation["patient_id"].isin(test["patient_id"]).any():
        raise ValueError("patient leakage between validation and test")

    groups = dataset.feature_groups
    feature_sets = {
        "Clinical Baseline": list(groups["clinical"]),
        "ECG Features Baseline": list(groups["ecg"]),
        "Clinical + ECG Fusion": [*groups["clinical"], *groups["ecg"]],
    }
    y_train = train["target"].astype(int)
    y_validation = validation["target"].astype(int)
    rows = []
    for name, features in feature_sets.items():
        categorical = [c for c in dataset.categorical_features if c in features]
        model = make_pipeline(
            make_preprocessor(
                features,
                categorical,
                {
                    "categorical_encoding": "one_hot",
                    "numeric_scaling": "standard",
                    "numeric_imputation": "median",
                    "categorical_imputation": "constant",
                    "categorical_missing_value": "__MISSING__",
                    "unknown_category": "ignore",
                },
            ),
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
        )
        started = time.perf_counter()
        model.fit(train[features], y_train)
        fit_seconds = time.perf_counter() - started
        probabilities = model.predict_proba(validation[features])[:, 1]
        if probabilities.shape != (len(validation),) or not np.isfinite(probabilities).all():
            raise ValueError(f"{name} produced invalid validation probabilities")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(f"{name} probability outside [0, 1]")
        rows.append(
            {
                "experiment": name,
                "model": "XGBoost",
                "feature_count": len(features),
                "categorical_feature_count": len(categorical),
                "n": len(validation),
                "scd": int(y_validation.sum()),
                "auroc": float(roc_auc_score(y_validation, probabilities)),
                "auprc_average_precision": float(
                    average_precision_score(y_validation, probabilities)
                ),
                "brier": float(brier_score_loss(y_validation, probabilities)),
                "fit_seconds": fit_seconds,
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
                    "validation_scd": int(y_validation.sum()),
                    "test_n": len(test),
                    "test_used": False,
                },
                "seed": 42,
                "device": "CPU",
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
