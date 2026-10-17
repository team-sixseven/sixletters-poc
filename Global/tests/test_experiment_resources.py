"""Personal source/config snapshots stay reproducible without copying environments."""

import hashlib
import json

import pandas as pd
import pytest

from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment
from sixletters.provenance import (
    MAX_RESOURCE_BYTES,
    experiment_resources,
    python_sources,
    resource_metadata,
    validate_resource_metadata,
)


@pytest.fixture
def personal_run(tmp_path):
    data = save_prepared_dataset(make_demo_dataset(), tmp_path / "data", synthetic=True)
    folder = tmp_path / "personal"
    folder.mkdir()
    config = folder / "config.toml"
    config.write_text(
        '[experiment]\nowner = "member"\nname = "experiment"\nfeature_set = "clinical"\n',
        encoding="utf-8",
    )
    (folder / "fit_and_evaluate.py").write_text(
        "# Personal learning entry point\n", encoding="utf-8"
    )
    model = tmp_path / "model.bin"
    model.write_bytes(b"opaque exported model")
    return data, config, model, tmp_path / "artifacts"


def save(personal_run, experiment=None):
    data, config, model, output = personal_run
    experiment = experiment or Experiment(data, config, output_root=output)
    return experiment.save(
        pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5}),
        model_files={"model.bin": model},
        training={"model": "PersonalMethod", "seed": None, "parameters": {}},
    )


def write(root, relative, content=b"value = 1\n"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def symlink(path, target):
    try:
        path.symlink_to(target, target_is_directory=target.is_dir())
    except OSError:
        pytest.skip("Creating symbolic links is unavailable on this platform")


def test_arbitrary_entry_point_and_nested_models_are_saved(personal_run):
    _, config, _, _ = personal_run
    write(config.parent, "models/network.py", b"# An actual model module\n")
    run = save(personal_run)
    assert not (run / "source/train.py").exists()
    assert (run / "source/fit_and_evaluate.py").is_file()
    assert (run / "source/models/network.py").is_file()


def test_at_least_one_python_source_is_required(personal_run):
    data, config, _, output = personal_run
    config.with_name("fit_and_evaluate.py").unlink()
    write(config.parent, ".venv/dependencies/fake.py")
    with pytest.raises(ValueError, match="at least one Python source"):
        Experiment(data, config, output_root=output)


def test_nested_settings_and_environment_manifests_preserve_exact_bytes(personal_run):
    _, config, _, _ = personal_run
    expected = {
        "settings/features.yaml": "features: [나이, 성별]\r\n".encode(),
        "settings/preprocessing.yml": b"imputer: median\n",
        "settings/model.json": b'{"max_depth": 5}\n',
        "settings/model.toml": b"max_depth = 5\n",
        "training.ini": b"[training]\nseed = 42\n",
        "parameters.cfg": b"[training]\nseed = 42\n",
        "requirements.in": b"xgboost==3.0.0\n",
        "requirements.txt": b"optuna==4.0.0\n",
        "requirements.lock": b"optuna==4.0.0\n",
        "environment.json": b'{"python": "3.12"}\n',
        "tokenizer.json": b'{"setting": true}\n',
    }
    for name, content in expected.items():
        write(config.parent, name, content)
    run = save(personal_run)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    assert result["schema_version"] == 3
    assert result["resources"] == {
        name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        for name, raw in expected.items()
    }
    assert (run / "config.toml").read_bytes() == config.read_bytes()
    assert not (run / "resources/config.toml").exists()
    for name, raw in expected.items():
        assert (run / "resources" / name).read_bytes() == raw


def test_custom_active_config_name_is_not_saved_twice(personal_run):
    data, config, model, output = personal_run
    renamed = config.with_name("balanced.toml")
    renamed.write_bytes(config.read_bytes())
    run = save((data, renamed, model, output))
    result = json.loads((run / "result.json").read_text())
    assert set(result["resources"]) == {"config.toml"}
    assert not (run / "resources/balanced.toml").exists()
    assert (run / "config.toml").read_bytes() == renamed.read_bytes()


@pytest.mark.parametrize(
    "directory",
    [
        ".venv",
        ".cache",
        ".git",
        ".poc-env-tmp",
        "__pycache__",
        "venv",
        "venvs",
        "env",
        "envs",
        "virtualenv",
        "site-packages",
        "__pypackages__",
        "node_modules",
        "build",
        "dist",
        "package.egg-info",
        "package.dist-info",
    ],
)
def test_environment_and_build_folders_never_enter_snapshots(personal_run, directory):
    data, config, _, output = personal_run
    write(config.parent, f"{directory}/third_party.py")
    ignored = write(config.parent, f"{directory}/settings.json", b'{"old": true}')
    experiment = Experiment(data, config, output_root=output)
    ignored.write_bytes(b'{"changed": true}')
    run = save(personal_run, experiment)
    assert set(python_sources(run / "source")) == {"fit_and_evaluate.py"}
    assert json.loads((run / "result.json").read_text())["resources"] == {}


def test_raw_data_secrets_and_unrelated_files_are_not_resources(personal_run):
    _, config, _, _ = personal_run
    excluded = [
        "data/input.json",
        "artifacts/run.json",
        "results/result.json",
        "outputs/metrics.json",
        "reports/report.json",
        "checkpoints/model.json",
        "models/model.json",
        "model/network.json",
        ".env",
        ".env.json",
        ".poc-env.lock",
        ".poc-env-tmp.json",
        "secret.yaml",
        "credentials.json",
        "service-account.json",
        "api-key.ini",
        "training_password.cfg",
        "secret/config.toml",
        "README.md",
        "input.csv",
        "weights.pkl",
    ]
    for name in excluded:
        write(config.parent, name)
    assert experiment_resources(config.parent) == {}
    assert json.loads((save(personal_run) / "result.json").read_text())["resources"] == {}


@pytest.mark.parametrize("change", ["modify", "add", "remove", "rename"])
def test_config_resource_changes_during_training_prevent_completed_result(personal_run, change):
    data, config, _, output = personal_run
    resource = write(config.parent, "settings.yaml", b"depth: 3\n")
    experiment = Experiment(data, config, output_root=output)
    if change == "modify":
        resource.write_bytes(b"depth: 4\n")
    elif change == "add":
        write(config.parent, "extra.json", b"{}")
    elif change == "remove":
        resource.unlink()
    else:
        resource.rename(resource.with_name("renamed.yaml"))
    with pytest.raises(ValueError, match="resources changed"):
        save(personal_run, experiment)
    assert not list(output.rglob("result.json"))


def test_config_resource_change_while_copying_model_removes_partial_run(personal_run, monkeypatch):
    import sixletters.experiment as module

    data, config, _, output = personal_run
    resource = write(config.parent, "settings.yaml", b"depth: 3\n")
    experiment = Experiment(data, config, output_root=output)
    original = module.shutil.copyfile

    def copy_and_modify(source, destination):
        original(source, destination)
        resource.write_bytes(b"depth: 9\n")

    monkeypatch.setattr(module.shutil, "copyfile", copy_and_modify)
    with pytest.raises(ValueError, match="resources changed"):
        save(personal_run, experiment)
    assert not list(output.rglob("result.json"))
    assert not list(output.rglob("model.bin"))


@pytest.mark.parametrize("name", ["settings.yaml", "fit.py"])
@pytest.mark.parametrize("outside", [False, True])
def test_snapshot_file_symlinks_are_rejected(personal_run, name, outside):
    _, config, _, output = personal_run
    target = write(output if outside else config.parent, "target.txt", b"text")
    symlink(config.parent / name, target)
    with pytest.raises(ValueError, match="regular files"):
        save(personal_run)


def test_snapshot_directory_symlink_is_rejected(personal_run):
    _, config, _, output = personal_run
    output.mkdir()
    write(output, "source.py")
    symlink(config.parent / "helpers", output)
    with pytest.raises(ValueError, match="symbolic links"):
        save(personal_run)


def test_ignored_environment_symlink_is_not_followed(personal_run):
    _, config, _, output = personal_run
    output.mkdir()
    write(output, "source.py")
    write(output, "secret.json")
    symlink(config.parent / ".venv", output)
    run = save(personal_run)
    assert set(python_sources(run / "source")) == {"fit_and_evaluate.py"}


def test_active_config_symlink_is_rejected(personal_run):
    data, config, model, output = personal_run
    linked = config.with_name("linked.toml")
    symlink(linked, config)
    with pytest.raises(ValueError, match="configuration.*symbolic link"):
        save((data, linked, model, output))


def test_active_config_cannot_be_replaced_by_a_symlink_during_training(personal_run):
    data, config, _, output = personal_run
    experiment = Experiment(data, config, output_root=output)
    copy = write(config.parent.parent, "config-copy.toml", config.read_bytes())
    config.unlink()
    symlink(config, copy)
    with pytest.raises(ValueError, match="Configuration changed"):
        save(personal_run, experiment)


def test_unreadable_snapshot_directory_is_not_silently_omitted(tmp_path, monkeypatch):
    import sixletters.provenance as module

    def unreadable_walk(directory, *, followlinks, onerror):
        onerror(PermissionError("unreadable directory"))
        yield  # Make this a walker whose error occurs when consumed.

    monkeypatch.setattr(module.os, "walk", unreadable_walk)
    with pytest.raises(ValueError, match="could not be read"):
        python_sources(tmp_path)


@pytest.mark.parametrize("content", [b"\xff\xfe", b'{"value": "\x00"}', b"\x1btest"])
def test_binary_resource_is_rejected(personal_run, content):
    _, config, _, _ = personal_run
    write(config.parent, "settings.json", content)
    with pytest.raises(ValueError, match="UTF-8 text|not binary"):
        save(personal_run)


def test_large_configuration_fails_explicitly(personal_run):
    _, config, _, _ = personal_run
    write(config.parent, "settings.json", b" " * (MAX_RESOURCE_BYTES + 1))
    with pytest.raises(ValueError, match="1 MiB"):
        save(personal_run)


def test_combined_resource_size_is_bounded(personal_run):
    _, config, _, _ = personal_run
    for index in range(9):
        write(config.parent, f"settings{index}.yaml", b" " * MAX_RESOURCE_BYTES)
    with pytest.raises(ValueError, match="8 MiB"):
        save(personal_run)


def test_resource_count_is_bounded(personal_run):
    _, config, _, _ = personal_run
    for index in range(257):
        write(config.parent, f"settings{index}.yaml", b"")
    with pytest.raises(ValueError, match="256 file"):
        save(personal_run)


@pytest.mark.parametrize(
    "name",
    [
        "../settings.yaml",
        "/settings.json",
        "settings//file.json",
        "settings\\file.json",
        ".env.json",
        "data/patients.json",
        "credentials.json",
        "settings.txt",
        "C:/settings.json",
        "settings/../file.json",
        "settings/./file.json",
        "NUL.json",
    ],
)
def test_untrusted_resource_metadata_rejects_unsafe_paths(name):
    with pytest.raises(ValueError):
        validate_resource_metadata({name: {"sha256": "a" * 64, "bytes": 0}})


@pytest.mark.parametrize(
    "info",
    [
        None,
        {},
        {"sha256": "a" * 64, "bytes": -1},
        {"sha256": "a" * 64, "bytes": True},
        {"sha256": "a" * 64, "bytes": MAX_RESOURCE_BYTES + 1},
        {"sha256": "invalid", "bytes": 1},
        {"sha256": "a" * 64, "bytes": 1, "extra": "field"},
    ],
)
def test_untrusted_resource_metadata_rejects_invalid_identity(info):
    with pytest.raises(ValueError):
        validate_resource_metadata({"settings.json": info})


def test_resource_manifest_detects_cross_platform_name_collisions():
    with pytest.raises(ValueError, match="collide"):
        resource_metadata({"Settings.yaml": b"1", "settings.yaml": b"2"})


@pytest.mark.parametrize(
    "names",
    [
        ["Settings.yaml", "settings.yaml/extra.json"],
        ["settings.yaml/extra.json", "Settings.yaml"],
        ["configuration/Nested.cfg", "Configuration/nested.cfg/extra.json"],
    ],
)
def test_resource_manifest_detects_portable_file_directory_collisions(names):
    with pytest.raises(ValueError, match="collide"):
        validate_resource_metadata({name: {"sha256": "a" * 64, "bytes": 0} for name in names})
