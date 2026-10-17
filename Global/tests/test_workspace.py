import json
import os
import pickle
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sixletters.contracts import Dataset
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.workspace import init_experiment, validate_component

REPO_ROOT = Path(__file__).resolve().parents[2]


def python(*args, check=True):
    return subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=check,
    )


def cli(*args, check=True):
    return python("-m", "sixletters", *args, check=check)


def test_personal_scaffold_preserves_existing_experiments(tmp_path):
    root = tmp_path / "개인 실험"
    first = init_experiment("팀원 가", "exp001", feature_set="combined", root=root)
    second = init_experiment("팀원 나", "exp001", root=root)
    first_config = tomllib.loads(first.read_text(encoding="utf-8"))
    assert first_config["experiment"]["owner"] == "팀원 가"
    assert first_config["experiment"]["feature_set"] == "combined"
    assert "model" not in first_config["experiment"]
    first_train = first.with_name("train.py")
    first_train.write_text("# Existing individual work\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        init_experiment("팀원 가", "exp001", root=root)
    assert before == {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert (
        second.with_name("train.py").read_text(encoding="utf-8") != "# Existing individual work\n"
    )


@pytest.mark.parametrize(
    "name",
    [
        "",
        "..",
        "../외부",
        "a/b",
        "a\\b",
        "C:drive",
        "CON",
        "nul.txt",
        "LPT1",
        "COM¹.txt",
        "이름.",
        " 이름",
        "이름 ",
        "foo\x00bar",
        "foo\nbar",
        "_template",
        ".hidden",
        "a?b",
    ],
)
def test_invalid_names_cannot_create_paths(tmp_path, name):
    root = tmp_path / "experiments"
    with pytest.raises(ValueError):
        init_experiment(name, "exp001", root=root)
    with pytest.raises(ValueError):
        init_experiment("팀원", name, root=root)
    assert not root.exists()


def test_cross_platform_names_cannot_create_case_variants(tmp_path):
    root = tmp_path / "experiments"
    init_experiment("Alice", "exp001", root=root)
    with pytest.raises(ValueError, match="collides"):
        init_experiment("alice", "exp002", root=root)
    assert validate_component("조성률", "owner") == "조성률"


def test_starter_fits_numeric_and_categorical_preprocessing_on_training_inputs(tmp_path):
    config = init_experiment("팀원", "exp001", root=tmp_path)
    prepared = tmp_path / "공통 자료"
    output_root = tmp_path / "개인 결과"
    frame = pd.DataFrame(
        {
            "patient_id": [f"patient-{index}" for index in range(10)],
            "split": ["train"] * 6 + ["validation"] * 2 + ["test"] * 2,
            "target": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
            "numeric": [1.0, 2.0, np.nan, 8.0, 9.0, 10.0, 1000.0, np.nan, 0.0, 1.0],
            "category": ["A", "A", "B", "B", "B", np.nan, "UNSEEN", "A", "A", "B"],
            "wave_feature": [0.0] * 10,
        }
    )
    dataset = Dataset(
        frame=frame,
        feature_groups={"clinical": ("numeric", "category"), "ecg": ("wave_feature",)},
        categorical_features=("category",),
        dataset_id="mixed-starter-fixture",
        target_definition="Synthetic binary target for starter validation.",
    )
    save_prepared_dataset(dataset, prepared, synthetic=True)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "C = 1.0", 'C = 0.5\nclass_weight = "balanced"\nrandom_state = 123'
        ),
        encoding="utf-8",
    )
    completed = python(
        config.with_name("train.py"), "--data", prepared, "--output-root", output_root
    )
    assert "Validation results saved" in completed.stdout
    run = next(output_root.rglob("result.json")).parent
    training = json.loads((run / "result.json").read_text(encoding="utf-8"))["training"]
    assert training["model"] == "LogisticRegression"
    assert training["seed"] == 123
    assert training["parameters"]["random_state"] == 123
    assert training["parameters"]["max_iter"] == 1000
    with (run / "models" / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    held_out = pd.DataFrame({"numeric": [1000.0, np.nan], "category": ["UNSEEN", "A"]})
    probabilities = model.predict_proba(held_out)
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    preprocessing = model.named_steps["preprocessing"]
    assert preprocessing.named_transformers_["numeric"].named_steps["imputer"].statistics_[0] == 8.0
    assert (
        "UNSEEN"
        not in preprocessing.named_transformers_["categorical"]
        .named_steps["encoder"]
        .categories_[0]
    )


def test_personal_script_rejects_another_experiments_config_before_saving(tmp_path):
    data = save_prepared_dataset(make_demo_dataset(), tmp_path / "data", synthetic=True)
    first = init_experiment("팀원 가", "exp001", root=tmp_path / "experiments")
    second = init_experiment("팀원 나", "exp001", root=tmp_path / "experiments")
    second.with_name("train.py").write_text(
        'raise RuntimeError("This is not the training code being executed")\n',
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"
    rejected = python(
        first.with_name("train.py"),
        "--data",
        data,
        "--config",
        second,
        "--output-root",
        output_root,
        check=False,
    )
    assert rejected.returncode == 2
    assert "same experiment folder" in rejected.stderr
    assert "Traceback" not in rejected.stderr
    assert not output_root.exists()


def test_personal_script_accepts_alternate_config_in_its_own_folder(tmp_path):
    data = save_prepared_dataset(make_demo_dataset(), tmp_path / "data", synthetic=True)
    default = init_experiment("팀원", "exp001", root=tmp_path / "experiments")
    alternate = default.with_name("balanced.toml")
    alternate.write_text(
        default.read_text(encoding="utf-8").replace("C = 1.0", "C = 0.25"),
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"
    python(
        default.with_name("train.py"),
        "--data",
        data,
        "--config",
        alternate,
        "--output-root",
        output_root,
    )
    run = next(output_root.rglob("result.json")).parent
    assert (run / "config.toml").read_bytes() == alternate.read_bytes()
    assert (run / "source" / "train.py").read_bytes() == default.with_name("train.py").read_bytes()
    training = json.loads((run / "result.json").read_text(encoding="utf-8"))["training"]
    assert training["parameters"]["C"] == 0.25


@pytest.mark.parametrize(
    "arguments, expected",
    [
        (["prepare", "--csv", "patient.csv", "--output", "unused"], "requires --schema"),
        (["prepare", "--demo", "--schema", "schema.json", "--output", "unused"], "only valid"),
        (["run", "--config", "config.toml"], "invalid choice"),
        (["init-experiment", "--owner", "이름/하위", "--name", "exp001"], "portable folder name"),
    ],
)
def test_cli_reports_actionable_input_errors_with_legacy_encoding(arguments, expected):
    result = cli(*arguments, check=False)
    assert result.returncode == 2
    assert expected in result.stderr
    assert "Traceback" not in result.stderr
