"""The model choice creates runnable independent baselines on the same prepared data."""

import json
import pickle
import runpy
import sys
import tomllib

import numpy as np
import pandas as pd
import pytest

from sixletters.contracts import Dataset
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.workspace import init_experiment


@pytest.mark.parametrize(
    "choice, expected_model",
    [("logistic", "LogisticRegression"), ("random-forest", "RandomForestClassifier")],
)
@pytest.mark.parametrize("feature_set", ["clinical", "ecg", "combined"])
def test_each_starter_trains_selected_model_without_fitting_held_out_data(
    tmp_path, monkeypatch, choice, expected_model, feature_set
):
    frame = pd.DataFrame(
        {
            "patient_id": [f"patient-{index}" for index in range(12)],
            "split": ["train"] * 8 + ["validation"] * 2 + ["test"] * 2,
            "target": [0] * 6 + [1, 1, 0, 1, 0, 1],
            "numeric": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 1000, np.nan, 9999, 9999],
            "category": ["A", "A", "B", "B", "B", None, "A", "B", "UNSEEN", "A", "TEST", "TEST"],
            "wave_feature": [0.1] * 6 + [0.8, 0.9, 1000, np.nan, 9999, 9999],
        }
    )
    dataset = Dataset(
        frame=frame,
        feature_groups={"clinical": ("numeric", "category"), "ecg": ("wave_feature",)},
        categorical_features=("category",),
        dataset_id="starter-model-fixture",
        target_definition="Synthetic binary target for baseline regression tests.",
    )
    prepared = save_prepared_dataset(dataset, tmp_path / "data", synthetic=True)
    config = init_experiment(
        "Alice",
        "baseline",
        feature_set=feature_set,
        model=choice,
        root=tmp_path / "Members",
        script="my_fit.py",
    )
    assert "model" not in tomllib.loads(config.read_text())["experiment"]
    output = tmp_path / "artifacts"
    script = config.with_name("my_fit.py")
    monkeypatch.setattr(
        sys, "argv", [str(script), "--data", str(prepared), "--output-root", str(output)]
    )
    # Invoke the generated editable file, rather than a shared training implementation.
    assert runpy.run_path(str(script))["main"]() == 0
    run = next(output.rglob("result.json")).parent
    result = json.loads((run / "result.json").read_text())
    assert result["training"]["model"] == expected_model
    assert result["training"]["parameters"]["class_weight"] == "balanced"
    with (run / "models/model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    assert type(model.named_steps["classifier"]).__name__ == expected_model
    assert model.named_steps["classifier"].class_weight == "balanced"
    preprocessing = model.named_steps["preprocessing"]
    numeric_columns = next(
        columns for name, _, columns in preprocessing.transformers_ if name == "numeric"
    )
    expected_medians = frame.loc[frame["split"].eq("train"), numeric_columns].median()
    np.testing.assert_allclose(
        preprocessing.named_transformers_["numeric"].named_steps["imputer"].statistics_,
        expected_medians,
    )
    if feature_set != "ecg":
        categories = (
            preprocessing.named_transformers_["categorical"].named_steps["encoder"].categories_[0]
        )
        assert "UNSEEN" not in categories and "TEST" not in categories
    validation = frame.loc[frame["split"].eq("validation")].copy()
    probabilities = model.predict_proba(validation[list(model.feature_names_in_)])
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_unknown_model_does_not_create_an_experiment(tmp_path):
    with pytest.raises(ValueError, match="model must be logistic or random-forest"):
        init_experiment("Alice", "exp001", model="misspelled", root=tmp_path / "Members")
    assert not (tmp_path / "Members").exists()


@pytest.mark.parametrize(
    "choice, old_parameter, new_parameter, key, value",
    [
        ("logistic", "C = 1.0", "C = 0.3", "C", 0.3),
        ("random-forest", "n_estimators = 300", "n_estimators = 7", "n_estimators", 7),
    ],
)
def test_bare_config_name_loads_personal_file_from_workspace_cwd(
    tmp_path, monkeypatch, choice, old_parameter, new_parameter, key, value
):
    root = tmp_path.resolve()
    (root / "Global").mkdir()
    (root / "Global/pyproject.toml").write_text('[project]\nname = "sixletters-poc"\n')
    prepared = save_prepared_dataset(make_demo_dataset(), root / "Global/data", synthetic=True)
    config = init_experiment("Alice", "exp001", model=choice, root=root / "Members")
    alternate = config.with_name("balanced.toml")
    alternate.write_text(config.read_text().replace(old_parameter, new_parameter))
    # A same-named file in the workspace must not shadow the personal configuration.
    (root / "balanced.toml").write_text("This is not the experiment configuration")
    script = config.with_name("train.py")
    output = root / "Global/artifacts"
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "--config", "balanced.toml", "--data", str(prepared)],
    )
    assert runpy.run_path(str(script))["main"]() == 0
    result_path = next(output.glob("Alice/exp001/*/result.json"))
    result = json.loads(result_path.read_text())
    assert result["config_filename"] == "balanced.toml"
    assert result["training"]["parameters"][key] == value
    assert result_path.with_name("config.toml").read_bytes() == alternate.read_bytes()
