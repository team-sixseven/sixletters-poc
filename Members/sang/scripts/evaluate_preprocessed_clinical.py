"""Compare tabular models on the user's preprocessed, fixed clinical splits."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pickle
import platform
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.config import load_yaml, resolve_repo_path
from model.evaluation.DL.holter import choose_youden_threshold
from model.evaluation.ML.clinical import evaluate_clinical_predictions
from model.evaluation.reporting import create_report_dir, metric, write_report
from model.ML.dataloader.clinical import load_clinical_splits
from model.ML.models import CATEGORICAL_FEATURES as CATEGORICAL
from model.ML.models import build_models
from model.tracking.bundle import write_bundle


def _classifier(estimator):
    return estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator


def _model_seed(estimator):
    params = _classifier(estimator).get_params(deep=False)
    return params.get("random_state", params.get("random_seed"))


def main() -> int:
    config_path = Path(__file__).resolve().parents[1] / "model/ML/model_choice.yaml"
    choice = load_yaml(config_path)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clinical-csv",
        type=Path,
        default=None,
        help="Clean clinical feature source CSV",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Saved patient ID assignments for train/val/test",
    )
    parser.add_argument(
        "--tracking-output",
        type=Path,
        help="Optional local MLflow handoff bundle directory; does not upload to a server",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "model/outputs/evaluation",
        help="Parent directory for this run's mandatory report.md and results.json",
    )
    args = parser.parse_args()
    args.clinical_csv = args.clinical_csv or resolve_repo_path(choice["clinical_csv"])
    args.splits_dir = args.splits_dir or resolve_repo_path(choice["splits_dir"])
    model_names = choice["models"]
    if choice["feature_selection"] != "all_columns_except_patient_id_and_scd":
        raise ValueError("Unsupported feature_selection in ML model_choice.yaml")
    if choice["selection_metric"] != "validation_average_precision":
        raise ValueError("Only validation_average_precision selection is currently supported")
    if choice["threshold_strategy"] != "validation_youden_j":
        raise ValueError("Unsupported threshold_strategy in ML model_choice.yaml")
    report_dir = create_report_dir(args.report_root, "clinical")
    frames = load_clinical_splits(args.clinical_csv, args.splits_dir)
    features = [name for name in frames["train"].columns if name not in {"Patient ID", "SCD"}]
    if not set(CATEGORICAL).issubset(features):
        missing = sorted(set(CATEGORICAL) - set(features))
        raise ValueError(
            f"Configured categorical features are absent from clinical_clean.csv: {missing}"
        )
    train, validation, test = frames["train"], frames["val"], frames["test"]
    x_train, y_train = train[features].copy(), train["SCD"].astype(int)
    x_val, y_val = validation[features].copy(), validation["SCD"].astype(int)
    # These columns are numeric codes in the CSV but categorical in the model schema.
    for column in CATEGORICAL:
        x_train[column] = x_train[column].astype(str)
        x_val[column] = x_val[column].astype(str)
    models = build_models(features, model_names, choice["preprocessing"])

    rows = []
    validation_predictions = {}
    fitted_models = {}
    for name, model in models.items():
        fit_x = x_train
        score_x = x_val
        if name == "CatBoost":
            fit_x = x_train.copy()
            score_x = x_val.copy()
            for column in CATEGORICAL:
                fit_x[column] = fit_x[column].astype(str)
                score_x[column] = score_x[column].astype(str)
            model.set_params(cat_features=CATEGORICAL)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(fit_x, y_train)
        elapsed = time.perf_counter() - started
        probabilities = model.predict_proba(score_x)[:, 1]
        if probabilities.shape != (len(validation),) or not np.isfinite(probabilities).all():
            raise ValueError(f"{name} returned invalid validation probabilities")
        selected_threshold = choose_youden_threshold(y_val, probabilities)
        score_metrics = evaluate_clinical_predictions(
            y_val, probabilities, selected_threshold["threshold"]
        )
        rows.append(
            {
                "model": name,
                "input": f"Clinical ({len(features)} features)",
                **score_metrics,
                "validation_threshold_metrics": selected_threshold,
                "fit_seconds": elapsed,
                "warnings": [f"{w.category.__name__}: {w.message}" for w in caught],
            }
        )
        validation_predictions[name] = pd.DataFrame(
            {
                "patient_id": validation["Patient ID"].astype(str),
                "target": y_val.astype(int).to_numpy(),
                "score": probabilities.astype(float),
            }
        )
        fitted_models[name] = model

    # Select on validation AP only, then score that frozen choice once on held-out test.
    selected = max(rows, key=lambda row: row["auprc_average_precision"])
    selected_name = selected["model"]
    final_model = build_models(features, model_names, choice["preprocessing"])[selected_name]
    x_train_val = pd.concat([x_train, x_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    x_test = test[features].copy()
    for column in CATEGORICAL:
        x_test[column] = x_test[column].astype(str)
    if selected_name == "CatBoost":
        final_model.set_params(cat_features=CATEGORICAL)
    started = time.perf_counter()
    final_model.fit(x_train_val, y_train_val)
    final_fit_seconds = time.perf_counter() - started
    test_probabilities = final_model.predict_proba(x_test)[:, 1]
    if not np.isfinite(test_probabilities).all():
        raise ValueError(f"{selected_name} returned invalid test probabilities")
    test_result = {
        "model_selected_by_validation_ap": selected_name,
        "threshold_selected_on_validation": selected["validation_threshold_metrics"]["threshold"],
        **evaluate_clinical_predictions(
            test["SCD"],
            test_probabilities,
            selected["validation_threshold_metrics"]["threshold"],
        ),
        "fit_seconds_train_plus_validation": final_fit_seconds,
        "test_used_for_model_selection": False,
    }

    if args.tracking_output is not None:
        test_targets = test["SCD"].to_numpy(dtype=int)
        test_prediction_frame = pd.DataFrame(
            {
                "patient_id": test["Patient ID"].astype(str),
                "target": test_targets,
                "score": test_probabilities.astype(float),
            }
        )
        run_records = []
        with tempfile.TemporaryDirectory(prefix="scd-clinical-models-") as temporary:
            temporary_dir = Path(temporary)
            for row in rows:
                name = row["model"]
                estimator = fitted_models[name]
                classifier = _classifier(estimator)
                classifier_params = classifier.get_params(deep=False)
                seed = _model_seed(estimator)
                model_path = temporary_dir / f"{name.lower().replace(' ', '_')}.pkl"
                with model_path.open("wb") as stream:
                    pickle.dump(estimator, stream)
                run_record = {
                    "model": name,
                    "parameters": {
                        "classifier": classifier_params,
                        "seed": seed,
                        "feature_names": features,
                        "preprocessing": choice["preprocessing"],
                    },
                    "validation_metrics": {
                        key: row[key]
                        for key in (
                            "n",
                            "scd",
                            "auroc",
                            "auprc_average_precision",
                            "brier",
                            "fit_seconds",
                        )
                    },
                    "validation_threshold_metrics": {
                        **row["validation_threshold_metrics"],
                    },
                    "validation_predictions": validation_predictions[name],
                    "artifacts": {"validation_model.pkl": model_path},
                    "warnings": row["warnings"],
                }
                if name == selected_name:
                    final_model_path = temporary_dir / "selected_test_model.pkl"
                    with final_model_path.open("wb") as stream:
                        pickle.dump(final_model, stream)
                    run_record["artifacts"]["selected_test_model.pkl"] = final_model_path
                    run_record["test_metrics"] = {
                        key: value
                        for key, value in test_result.items()
                        if key
                        not in {"model_selected_by_validation_ap", "test_used_for_model_selection"}
                    }
                    run_record["test_predictions"] = test_prediction_frame
                run_records.append(run_record)

            saved_bundle = write_bundle(
                args.tracking_output,
                experiment="sang-clinical-model-comparison",
                dataset="preprocessed-clinical-552-v1",
                input_group="clinical",
                split_summary={
                    name: {"n": len(frame), "scd": int(frame["SCD"].sum())}
                    for name, frame in frames.items()
                },
                selected_model=selected_name,
                environment={
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__,
                    "xgboost": importlib.metadata.version("xgboost"),
                    "catboost": importlib.metadata.version("catboost"),
                },
                runs=run_records,
                source_files=[
                    Path(__file__),
                    config_path,
                    *[
                        Path(__file__).resolve().parents[1] / f"model/ML/{subdir}/parameter.yaml"
                        for subdir in ("xgboost", "catboost")
                    ],
                    Path(__file__).resolve().parents[1] / "model/ML/models.py",
                    Path(__file__).resolve().parents[1] / "model/ML/dataloader/clinical.py",
                    Path(__file__).resolve().parents[1] / "model/evaluation/DL/holter.py",
                    Path(__file__).resolve().parents[1] / "model/evaluation/core.py",
                    Path(__file__).resolve().parents[1] / "model/config.py",
                    Path(__file__).resolve().parents[1] / "model/tracking/bundle.py",
                ],
            )

    payload = {
        "clinical_csv": str(args.clinical_csv),
        "splits_dir": str(args.splits_dir),
        "input_features": len(features),
        "categorical_features": CATEGORICAL,
        "split": {
            name: {"n": len(frame), "scd": int(frame["SCD"].sum())}
            for name, frame in frames.items()
        },
        "patient_overlap": False,
        "test_used_for_model_selection": False,
        "selection_rule": "highest validation Average Precision; test scored once after selection",
        "selected_model_test_result": test_result,
        "model_seeds": {name: _model_seed(estimator) for name, estimator in fitted_models.items()},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": importlib.metadata.version("xgboost"),
            "catboost": importlib.metadata.version("catboost"),
        },
        "validation_results": rows,
        "tracking_handoff_bundle": str(saved_bundle) if args.tracking_output is not None else None,
        "report_dir": str(report_dir),
    }
    table = [
        "| Model | Val AUROC | Val AP | Val Brier | Sens. | Spec. | Fit (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            "| {model} | {auroc} | {ap} | {brier} | {sensitivity} | {specificity} | {fit:.2f} |".format(
                model=row["model"],
                auroc=metric(row["auroc"]),
                ap=metric(row["auprc_average_precision"]),
                brier=metric(row["brier"]),
                sensitivity=metric(row["sensitivity"]),
                specificity=metric(row["specificity"]),
                fit=row["fit_seconds"],
            )
        )
    split_lines = [
        f"| {name} | {len(frame)} | {int(frame['SCD'].sum())} |" for name, frame in frames.items()
    ]
    report = "\n".join(
        [
            "# Clinical Model Evaluation",
            "",
            "- Status: COMPLETE",
            f"- Feature source: `{args.clinical_csv}`",
            f"- Patient splits: `{args.splits_dir}`",
            f"- Input: {len(features)} clinical features",
            "- Training: each candidate fitted independently on the training split; validation AP selects the candidate.",
            "- Test: evaluated once for the validation-selected model after refitting that model on train + validation.",
            "- Sensitivity and specificity thresholds are selected independently per model on validation by Youden J.",
            f"- Selected model: **{selected_name}**",
            "",
            "## Split summary",
            "",
            "| Split | N | SCD |",
            "|---|---:|---:|",
            *split_lines,
            "",
            "## Validation comparison",
            "",
            *table,
            "",
            "## Selected model test result",
            "",
            f"- AUROC: {metric(test_result['auroc'])}",
            f"- Average Precision: {metric(test_result['auprc_average_precision'])}",
            f"- Brier: {metric(test_result['brier'])}",
            f"- Validation-selected threshold: {metric(test_result['threshold_selected_on_validation'])}",
            f"- Sensitivity / Specificity: {metric(test_result['sensitivity'])} / {metric(test_result['specificity'])}",
            f"- Train + validation fit time: {test_result['fit_seconds_train_plus_validation']:.2f} s",
            "",
            "## Reproducibility",
            "",
            f"- Model random seeds: {payload['model_seeds']}",
            f"- Versions: Python {payload['versions']['python']}, scikit-learn {payload['versions']['scikit_learn']}, XGBoost {payload['versions']['xgboost']}, CatBoost {payload['versions']['catboost']}",
            f"- Warnings captured during fitting: {sum(len(row['warnings']) for row in rows)}",
            "- Test data was not used for model selection.",
        ]
    )
    write_report(report_dir, payload, report)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
