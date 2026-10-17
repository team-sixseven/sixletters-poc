"""Declared model exports are outputs without weakening config/source checks."""

import json

import pandas as pd
import pytest

from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment
from sixletters.provenance import MAX_RESOURCE_BYTES


@pytest.fixture
def workspace(tmp_path):
    data = save_prepared_dataset(make_demo_dataset(), tmp_path / "data", synthetic=True)
    folder = tmp_path / "personal"
    folder.mkdir()
    config = folder / "config.toml"
    config.write_text(
        '[experiment]\nowner = "member"\nname = "json-model"\nfeature_set = "clinical"\n',
        encoding="utf-8",
    )
    (folder / "train.py").write_text("# personal training\n", encoding="utf-8")
    (folder / "settings.json").write_text('{"depth": 3}\n', encoding="utf-8")
    return data, config, tmp_path / "artifacts"


def save(experiment, model):
    return experiment.save(
        pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5}),
        model_files={"export.json": model},
        training={"model": "JSONModel", "seed": None, "parameters": {}},
    )


@pytest.mark.parametrize("large", [False, True])
def test_new_json_model_is_saved_only_as_model(workspace, large):
    data, config, output = workspace
    experiment = Experiment(data, config, output_root=output)
    model = config.with_name("model.json")
    raw = b'{"weights": [1, 2, 3]}' + (b" " * MAX_RESOURCE_BYTES if large else b"")
    model.write_bytes(raw)

    run = save(experiment, model)

    result = json.loads((run / "result.json").read_text())
    assert set(result["resources"]) == {"settings.json"}
    assert set(result["model_files"]) == {"export.json"}
    assert not (run / "resources/model.json").exists()
    assert (run / "resources/settings.json").read_bytes() == config.with_name(
        "settings.json"
    ).read_bytes()
    assert (run / "models/export.json").read_bytes() == raw


def test_existing_json_model_can_be_overwritten_on_next_run(workspace):
    data, config, output = workspace
    model = config.with_name("model.json")
    model.write_text('{"weights": [0]}', encoding="utf-8")
    runs = []
    for value in (1, 2):
        experiment = Experiment(data, config, output_root=output)
        raw = json.dumps({"weights": [value]}).encode()
        model.write_bytes(raw)
        run = save(experiment, model)
        runs.append(run)
        assert (run / "models/export.json").read_bytes() == raw
        assert set(json.loads((run / "result.json").read_text())["resources"]) == {"settings.json"}
    assert runs[0] != runs[1]
    assert json.loads((runs[0] / "models/export.json").read_text()) == {"weights": [1]}


@pytest.mark.parametrize("directory", ["models", "outputs"])
def test_existing_large_json_model_in_output_folder_can_be_reused(workspace, directory):
    data, config, output = workspace
    model = config.parent / directory / "model.json"
    model.parent.mkdir()
    model.write_bytes(b"{}" + b" " * MAX_RESOURCE_BYTES)
    experiment = Experiment(data, config, output_root=output)
    model.write_bytes(b"{}" + b" " * (MAX_RESOURCE_BYTES + 1))
    run = save(experiment, model)
    assert (run / "models/export.json").read_bytes() == model.read_bytes()


def test_json_model_export_does_not_hide_changed_settings(workspace):
    data, config, output = workspace
    experiment = Experiment(data, config, output_root=output)
    model = config.with_name("model.json")
    model.write_text('{"weights": [1]}', encoding="utf-8")
    config.with_name("settings.json").write_text('{"depth": 4}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="resources changed"):
        save(experiment, model)
    assert not list(output.rglob("result.json"))


@pytest.mark.parametrize(
    "filename,message",
    [("config.toml", "Configuration changed"), ("train.py", "source code changed")],
)
def test_declaring_active_config_or_source_as_model_cannot_hide_changes(
    workspace, filename, message
):
    data, config, output = workspace
    experiment = Experiment(data, config, output_root=output)
    changed = config.with_name(filename)
    changed.write_bytes(changed.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match=message):
        save(experiment, changed)
    assert not list(output.rglob("result.json"))
