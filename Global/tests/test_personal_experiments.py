import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sixletters.comparison import compare_runs
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment
from sixletters.workspace import init_experiment


@pytest.fixture
def setup_experiment(tmp_path):
    dataset = make_demo_dataset()
    data = save_prepared_dataset(dataset, tmp_path / "data", synthetic=True)
    config = init_experiment("팀원가", "exp001", root=tmp_path / "experiments")
    config.write_text(
        '[experiment]\nowner = "팀원가"\nname = "exp001"\n'
        'feature_set = "clinical"\nmodel = "FreeMethod"\nseed = 42\n'
        "\n[parameters]\nlearning_rate = 0.1\n",
        encoding="utf-8",
    )
    (config.parent / "train.py").write_text(
        'raise RuntimeError("Shared evaluation must not execute training")\n', encoding="utf-8"
    )
    model = tmp_path / "weights.npz"
    np.savez(model, weights=[1.0])
    return dataset, data, config, model


def test_shared_inputs_expose_training_labels_and_validation_features_only(setup_experiment):
    dataset, data, config, _ = setup_experiment
    experiment = Experiment(data, config)
    train, labels = experiment.training_data()
    validation = experiment.validation_data()
    assert set(train.index) == set(dataset.frame.loc[dataset.frame.split.eq("train"), "patient_id"])
    assert set(validation.index) == set(
        dataset.frame.loc[dataset.frame.split.eq("validation"), "patient_id"]
    )
    assert train.index.equals(labels.index)
    assert set(labels) == {0, 1}
    assert train.columns.tolist() == list(dataset.feature_groups["clinical"])
    assert validation.columns.tolist() == train.columns.tolist()
    assert {"patient_id", "target", "split"}.isdisjoint(validation.columns)
    # Returned inputs are private working copies, not mutable shared data.
    original = train.iloc[0, 0]
    train.iloc[0, 0] = 123456.0
    assert experiment.training_data()[0].iloc[0, 0] == original


def test_save_aligns_shuffled_predictions_by_id_and_uses_shared_truth(setup_experiment, tmp_path):
    dataset, data, config, model = setup_experiment
    experiment = Experiment(data, config, output_root=tmp_path / "artifacts")
    validation = experiment.validation_data()
    predictions = pd.DataFrame(
        {"patient_id": validation.index, "score": np.linspace(0.1, 0.9, len(validation))}
    ).sample(frac=1, random_state=7)
    run = experiment.save(predictions, model_files={"weights.npz": model})
    saved = pd.read_csv(run / "predictions.csv", dtype={"patient_id": str}).set_index("patient_id")
    truth = dataset.frame.set_index("patient_id")["target"]
    assert saved["target"].equals(truth.loc[saved.index].astype(saved["target"].dtype))
    pd.testing.assert_series_equal(
        saved["score"].sort_index(),
        predictions.set_index("patient_id")["score"].sort_index(),
        check_exact=False,
    )
    assert (run / "models/weights.npz").read_bytes() == model.read_bytes()
    assert json.loads((run / "result.json").read_text(encoding="utf-8"))["schema_version"] == 3


@pytest.mark.parametrize("score_dtype", [np.float16, np.float32])
def test_lower_precision_predictions_roundtrip_and_compare_without_metric_drift(
    setup_experiment, tmp_path, score_dtype
):
    _, data, config, model = setup_experiment
    runs = []
    for _ in range(2):
        experiment = Experiment(data, config, output_root=tmp_path / "artifacts")
        validation = experiment.validation_data()
        scores = np.linspace(0.1, 0.9, len(validation), dtype=score_dtype)
        predictions = pd.DataFrame({"patient_id": validation.index, "score": scores})
        run = experiment.save(predictions, model_files={"weights.npz": model})
        reloaded = pd.read_csv(run / "predictions.csv", float_precision="round_trip")
        np.testing.assert_array_equal(reloaded["score"].to_numpy(), scores.astype(float))
        assert predictions["score"].dtype == score_dtype
        runs.append(run)
    summary = compare_runs(runs, tmp_path / "comparison.csv")
    assert len(summary) == 2
    for metric in ("roc_auc", "average_precision", "brier_score"):
        assert summary[metric].iloc[0] == summary[metric].iloc[1]


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "extra",
        "duplicate",
        "train",
        "test",
        "target",
        "nan",
        "infinity",
        "negative",
        "above_one",
    ],
)
def test_save_rejects_invalid_or_incomplete_validation_predictions(
    setup_experiment, tmp_path, problem
):
    dataset, data, config, model = setup_experiment
    experiment = Experiment(data, config, output_root=tmp_path / "artifacts")
    predictions = pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5})
    if problem == "missing":
        predictions = predictions.iloc[1:]
    elif problem == "extra":
        predictions.loc[len(predictions)] = ["UNEXPECTED", 0.5]
    elif problem == "duplicate":
        predictions.loc[1, "patient_id"] = predictions.loc[0, "patient_id"]
    elif problem in {"train", "test"}:
        predictions.loc[0, "patient_id"] = dataset.frame.loc[
            dataset.frame.split.eq(problem), "patient_id"
        ].iloc[0]
    elif problem == "target":
        predictions["target"] = 1
    else:
        predictions.loc[0, "score"] = {
            "nan": np.nan,
            "infinity": np.inf,
            "negative": -0.1,
            "above_one": 1.1,
        }[problem]
    with pytest.raises(ValueError):
        experiment.save(predictions, model_files={"weights.npz": model})
    assert not list((tmp_path / "artifacts").rglob("result.json"))


@pytest.mark.parametrize("feature", ["target", "patient_id", "split", "ecg_1", "unknown"])
def test_config_cannot_include_features_outside_declared_group(setup_experiment, feature):
    _, data, config, _ = setup_experiment
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "[parameters]", f'features = ["{feature}"]\n[parameters]'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        Experiment(data, config)


def test_free_training_script_uses_helpers_and_numpy_model_without_estimator_contract(tmp_path):
    root = tmp_path / "한글 개인 작업"
    data = save_prepared_dataset(make_demo_dataset(), root / "data", synthetic=True)
    config = init_experiment("팀원가", "exp001", root=root / "experiments")
    config.write_text(
        '[experiment]\nowner = "팀원가"\nname = "exp001"\n'
        'feature_set = "clinical"\nseed = 7\n'
        "\n[parameters]\nsteps = 3\n",
        encoding="utf-8",
    )
    (config.parent / "helpers.py").write_text(
        "import numpy as np\n"
        "def predict(values, offset):\n"
        "    logits = np.nan_to_num(values, nan=0).sum(axis=1) * 0.01 + offset\n"
        "    return 1 / (1 + np.exp(-logits))\n",
        encoding="utf-8",
    )
    script = config.parent / "train.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\n"
        "from helpers import predict\nfrom sixletters.experiment import Experiment\n"
        "experiment = Experiment(Path(sys.argv[1]), Path(__file__).with_name('config.toml'), "
        "output_root=Path(sys.argv[2]))\n"
        "train, target = experiment.training_data()\n"
        "offset = 0.0\n"
        "for _ in range(experiment.parameters['steps']):\n"
        "    offset += float(target.mean()) * 0.01\n"
        "validation = experiment.validation_data()\n"
        "weights = Path(sys.argv[3])\nnp.savez(weights, offset=offset)\n"
        "predictions = pd.DataFrame({'patient_id': validation.index, "
        "'score': predict(validation.to_numpy(), offset)})\n"
        "training = {'model': 'NumpyTrainingLoop', 'seed': None, "
        "'parameters': {'steps': experiment.parameters['steps']}}\n"
        "print(experiment.save(predictions, model_files={'weights.npz': weights}, "
        "training=training))\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(data),
            str(root / "artifacts"),
            str(root / "weights.npz"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    run = Path(result.stdout.strip().splitlines()[-1])
    saved = json.loads((run / "result.json").read_text(encoding="utf-8"))
    assert saved["owner"] == "팀원가"
    assert saved["training"]["model"] == "NumpyTrainingLoop"
    assert (run / "source/helpers.py").read_bytes() == (config.parent / "helpers.py").read_bytes()
    with np.load(run / "models/weights.npz") as model:
        assert model["offset"] > 0
