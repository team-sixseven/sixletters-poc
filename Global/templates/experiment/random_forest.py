"""Editable random forest; preprocessing, fitting, and prediction belong here."""

import argparse
import pickle
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sixletters.experiment import Experiment
from sixletters.local_settings import configured_data, project_root


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Train a personal model and save validation results"
    )
    root = project_root(Path(__file__).resolve().parent)
    parser.add_argument(
        "--data", type=Path, help="Override the data location registered by poc init"
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.toml"))
    parser.add_argument("--output-root", type=Path, default=root / "Global/artifacts")
    args = parser.parse_args()
    experiment_directory = Path(__file__).resolve().parent
    if not args.config.is_absolute() and len(args.config.parts) == 1:
        args.config = experiment_directory / args.config
    if args.config.resolve().parent != experiment_directory:
        parser.error("--config must be in the same experiment folder as the training script")
    if args.data is None:
        try:
            args.data = configured_data(root)
        except (ValueError, OSError) as error:
            parser.error(str(error))

    experiment = Experiment(data=args.data, config=args.config, output_root=args.output_root)
    X_train, y_train = experiment.training_data()
    X_validation = experiment.validation_data()

    numeric_columns = [
        column for column in X_train.columns if column not in experiment.categorical_features
    ]
    categorical_columns = list(experiment.categorical_features)
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ]
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    **{
                        "random_state": experiment.settings.get("seed"),
                        "n_estimators": 300,
                        "min_samples_leaf": 2,
                        "class_weight": "balanced",
                        **experiment.parameters,
                    }
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_validation)[:, list(model.classes_).index(1)]
    predictions = pd.DataFrame({"patient_id": X_validation.index, "score": scores})
    classifier = model.named_steps["classifier"]
    training = {
        "model": type(classifier).__name__,
        "seed": classifier.random_state,
        "parameters": classifier.get_params(deep=False),
    }

    with TemporaryDirectory() as temporary:
        model_path = Path(temporary) / "model.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)
        run = experiment.save(predictions, model_files={"model.pkl": model_path}, training=training)
    print(f"Validation results saved: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
