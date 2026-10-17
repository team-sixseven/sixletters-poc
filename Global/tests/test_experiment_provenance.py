import json
import shutil
from datetime import datetime

import pandas as pd
import pytest

from sixletters import experiment as experiment_module
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment
from sixletters.provenance import OPERATIONAL_SOURCE_PATHS, shared_code_scope, source_hash
from sixletters.workspace import init_experiment


@pytest.fixture
def shared_package(tmp_path, monkeypatch):
    package = tmp_path / "shared_package"
    shutil.copytree(experiment_module.PACKAGE, package)
    monkeypatch.setattr(experiment_module, "PACKAGE", package)
    return package


@pytest.fixture
def pending_run(tmp_path, shared_package):
    data = save_prepared_dataset(make_demo_dataset(), tmp_path / "data", synthetic=True)
    config = init_experiment("팀원가", "exp001", root=tmp_path / "experiments")
    config.write_text(
        '[experiment]\nowner = "팀원가"\nname = "exp001"\n'
        'feature_set = "clinical"\nmodel = "OwnModel"\nseed = 42\n'
        "\n[parameters]\nsteps = 10\n",
        encoding="utf-8",
    )
    (config.parent / "train.py").write_text(
        'raise RuntimeError("Source capture must not execute this file")\n', encoding="utf-8"
    )
    helper = config.parent / "preprocessing" / "helper.py"
    helper.parent.mkdir()
    helper.write_text("OFFSET = 1\n", encoding="utf-8")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"opaque model bytes; common code must not deserialize")
    experiment = Experiment(data, config, output_root=tmp_path / "artifacts")
    predictions = pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5})
    return experiment, predictions, data, config, model


@pytest.fixture
def runtime_run(pending_run, tmp_path):
    _, predictions, data, config, model = pending_run
    config.write_text(
        config.read_text(encoding="utf-8").replace('model = "OwnModel"\n', ""),
        encoding="utf-8",
    )
    experiment = Experiment(data, config, output_root=tmp_path / "artifacts")
    return experiment, predictions, model


def test_legacy_config_save_records_source_config_environment_and_arbitrary_model(pending_run):
    experiment, predictions, _, config, model = pending_run
    run = experiment.save(predictions, model_files={"model.onnx": model})
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    assert result["schema_version"] == 3
    assert result["shared_code_scope"]["version"] == 4
    assert "personal_env.py" in result["shared_code_scope"]["excluded"]
    assert datetime.fromisoformat(result["created_at"]).utcoffset().total_seconds() == 0
    assert result["training"]["model"] == "OwnModel"
    assert result["training"]["source_hash"]
    assert result["software"]["python"]
    assert result["software"]["packages"]["numpy"]
    assert (run / "config.toml").read_bytes() == config.read_bytes()
    assert (run / "source/train.py").read_bytes() == (config.parent / "train.py").read_bytes()
    assert (run / "source/preprocessing/helper.py").read_bytes() == (
        config.parent / "preprocessing/helper.py"
    ).read_bytes()
    assert (run / "shared_source/evaluation.py").is_file()
    assert (run / "models/model.onnx").read_bytes() == model.read_bytes()


@pytest.mark.parametrize("name", OPERATIONAL_SOURCE_PATHS)
def test_operational_source_changes_allow_save_and_preserve_original_snapshot(
    pending_run, shared_package, name
):
    experiment, predictions, _, _, model = pending_run
    path = shared_package / name
    original = experiment._shared_sources.get(name)
    path.write_bytes((original or b"") + b"\n# updated operational behavior\n")

    run = experiment.save(predictions, model_files={"model.onnx": model})
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    assert result["training"]["shared_code_hash"] == source_hash(experiment._shared_sources)
    assert result["shared_code_scope"] == shared_code_scope(experiment._shared_sources)
    snapshot = run / "shared_source" / name
    if original is None:
        assert not snapshot.exists()
    else:
        assert snapshot.read_bytes() == original


@pytest.mark.parametrize(
    "name",
    [
        "evaluation.py",
        "contracts.py",
        "comparison.py",
        "provenance.py",
        "features/clinical.py",
        "new_training_helper.py",
        "custom/tracking.py",
    ],
)
def test_scientific_and_unknown_shared_source_changes_still_block_save(
    pending_run, shared_package, tmp_path, name
):
    experiment, predictions, _, _, model = pending_run
    path = shared_package / name
    path.parent.mkdir(parents=True, exist_ok=True)
    original = path.read_bytes() if path.is_file() else b""
    path.write_bytes(original + b"\n# changed scientific behavior\n")
    with pytest.raises(ValueError, match="Shared scientific code changed"):
        experiment.save(predictions, model_files={"model.onnx": model})
    assert not list((tmp_path / "artifacts").rglob("result.json"))


def test_scientific_source_deletion_still_blocks_save(pending_run, shared_package, tmp_path):
    experiment, predictions, _, _, model = pending_run
    (shared_package / "features/clinical.py").unlink()
    with pytest.raises(ValueError, match="Shared scientific code changed"):
        experiment.save(predictions, model_files={"model.onnx": model})
    assert not list((tmp_path / "artifacts").rglob("result.json"))


@pytest.mark.parametrize("changed", ["config", "train", "helper", "new_helper", "data"])
def test_save_rejects_changes_since_experiment_start(pending_run, tmp_path, changed):
    experiment, predictions, data, config, model = pending_run
    if changed == "config":
        config.write_text(config.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    elif changed == "train":
        (config.parent / "train.py").write_text("print('changed')\n", encoding="utf-8")
    elif changed == "helper":
        (config.parent / "preprocessing/helper.py").write_text("OFFSET = 2\n", encoding="utf-8")
    elif changed == "new_helper":
        (config.parent / "new_helper.py").write_text("VALUE = 3\n", encoding="utf-8")
    else:
        payload = json.loads(data.read_text(encoding="utf-8"))
        payload["rows"][0][3] = 12345.0
        data.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        experiment.save(predictions, model_files={"model.onnx": model})
    assert not list((tmp_path / "artifacts").rglob("result.json"))


def test_independent_sessions_never_overwrite_completed_results(pending_run, tmp_path):
    first, predictions, data, config, model = pending_run
    run1 = first.save(predictions, model_files={"model.onnx": model})
    before = {
        path.relative_to(run1): path.read_bytes() for path in run1.rglob("*") if path.is_file()
    }
    second = Experiment(data, config, output_root=tmp_path / "artifacts")
    run2 = second.save(predictions, model_files={"model.onnx": model})
    assert run1 != run2
    assert before == {
        path.relative_to(run1): path.read_bytes() for path in run1.rglob("*") if path.is_file()
    }


def test_failed_artifact_save_does_not_publish_completed_result(pending_run, tmp_path):
    experiment, predictions, _, _, _ = pending_run
    with pytest.raises((ValueError, OSError)):
        experiment.save(predictions, model_files={"model.onnx": tmp_path / "missing.onnx"})
    assert not list((tmp_path / "artifacts").rglob("result.json"))


def test_artifact_copy_failure_removes_incomplete_run(pending_run, tmp_path, monkeypatch):
    experiment, predictions, _, _, model = pending_run
    destinations = []

    def fail_copy(source, destination):
        assert source == model
        assert destination.parent.is_dir()
        destinations.append(destination)
        raise OSError("Simulated model copy failure")

    monkeypatch.setattr("sixletters.experiment.shutil.copyfile", fail_copy)
    with pytest.raises(OSError, match="Simulated model copy failure"):
        experiment.save(predictions, model_files={"model.onnx": model})
    assert len(destinations) == 1
    assert not destinations[0].parent.parent.exists()
    assert not list((tmp_path / "artifacts").rglob("result.json"))


@pytest.mark.parametrize("location", ["inside_source", "source_directory"])
def test_result_paths_cannot_overlap_personal_source(pending_run, location):
    _, _, data, config, _ = pending_run
    output_root = config.parent / "artifacts" if location == "inside_source" else config.parent
    with pytest.raises(ValueError, match="outside.*source|source.*outside"):
        Experiment(data, config, output_root=output_root)


def test_runtime_training_metadata_records_actual_selected_settings(pending_run):
    experiment, predictions, _, _, model = pending_run
    assert experiment.settings["model"] == "OwnModel"
    training = {"model": "SelectedModel", "seed": None, "parameters": {"depth": 4, "epochs": 9}}
    run = experiment.save(predictions, model_files={"model.onnx": model}, training=training)
    saved = json.loads((run / "result.json").read_text(encoding="utf-8"))["training"]
    assert {key: saved[key] for key in training} == training


def test_config_without_model_saves_runtime_metadata(runtime_run):
    experiment, predictions, model = runtime_run
    assert "model" not in experiment.settings
    training = {"model": "SelectedModel", "seed": 17, "parameters": {"depth": 4}}
    run = experiment.save(predictions, model_files={"model.onnx": model}, training=training)
    saved = json.loads((run / "result.json").read_text(encoding="utf-8"))["training"]
    assert {key: saved[key] for key in training} == training


def test_missing_runtime_metadata_without_legacy_model_cannot_publish_result(runtime_run, tmp_path):
    experiment, predictions, model = runtime_run
    with pytest.raises(ValueError, match="Pass training=.*trained model"):
        experiment.save(predictions, model_files={"model.onnx": model})
    assert not list((tmp_path / "artifacts").rglob("result.json"))


@pytest.mark.parametrize(
    "training",
    [
        {},
        {"model": "", "seed": None, "parameters": {}},
        {"model": "SelectedModel", "seed": -1, "parameters": {}},
        {"model": "SelectedModel", "seed": None, "parameters": []},
        {"model": "SelectedModel", "seed": None, "parameters": {"callback": object()}},
    ],
)
def test_invalid_runtime_metadata_cannot_publish_result(runtime_run, tmp_path, training):
    experiment, predictions, model = runtime_run
    with pytest.raises(ValueError):
        experiment.save(predictions, model_files={"model.onnx": model}, training=training)
    assert not list((tmp_path / "artifacts").rglob("result.json"))


@pytest.mark.parametrize("model_value", ['""', "7", '["OwnModel"]'])
def test_optional_legacy_model_still_requires_a_nonempty_name(pending_run, model_value):
    _, _, data, config, _ = pending_run
    config.write_text(
        config.read_text(encoding="utf-8").replace('model = "OwnModel"', f"model = {model_value}"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model must describe"):
        Experiment(data, config)


@pytest.mark.parametrize("name", ["../outside.onnx", "/absolute.onnx", "nested/../../outside.onnx"])
def test_model_artifact_names_cannot_escape_run_directory(pending_run, tmp_path, name):
    experiment, predictions, _, _, model = pending_run
    with pytest.raises(ValueError):
        experiment.save(predictions, model_files={name: model})
    assert not list((tmp_path / "artifacts").rglob("result.json"))
