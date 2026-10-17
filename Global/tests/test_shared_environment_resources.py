"""Shared member dependencies stay exact in results and isolated on reproduction."""

import json
import shutil

import pandas as pd
import pytest

from sixletters import experiment as experiment_module
from sixletters import personal_env
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment
from sixletters.workspace import init_experiment


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    package = tmp_path / "shared-package"
    shutil.copytree(experiment_module.PACKAGE, package)
    monkeypatch.setattr(experiment_module, "PACKAGE", package)
    config = init_experiment("Alice", "first", root=tmp_path / "Members")
    data = save_prepared_dataset(make_demo_dataset(), tmp_path / "data", synthetic=True)
    model = tmp_path / "model.bin"
    model.write_bytes(b"opaque fitted model")
    return tmp_path, config, data, model


def package_files(directory, version="1.0.0"):
    files = {
        "requirements.in": f"personal-package=={version}\n".encode(),
        "requirements.lock": f"personal-package=={version}\n".encode(),
        "environment.json": json.dumps({"version": version}).encode(),
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name, raw in files.items():
        (directory / name).write_bytes(raw)
    return files


def experiment(workspace):
    root, config, data, _ = workspace
    return Experiment(data, config, output_root=root / "artifacts")


def save(workspace, active=None, model=None):
    active = active or experiment(workspace)
    return active.save(
        pd.DataFrame({"patient_id": active.validation_data().index, "score": 0.5}),
        model_files={"model.bin": model or workspace[3]},
        training={"model": "Constant", "seed": None, "parameters": {}},
    )


def snapshot(directory):
    return {
        name: (directory / name).read_bytes()
        for name in personal_env.LOCK_NAMES
        if (directory / name).is_file()
    }


@pytest.mark.parametrize("change", ["modify", "remove", "add"])
def test_shared_environment_change_during_training_blocks_save(workspace, change):
    root, config, _, _ = workspace
    owner = config.parent.parent.parent
    package_files(owner)
    if change == "add":
        (owner / "environment.json").unlink()
    active = experiment(workspace)
    if change == "modify":
        (owner / "requirements.lock").write_text("personal-package==2.0.0\n")
    elif change == "remove":
        (owner / "requirements.lock").unlink()
    else:
        (owner / "environment.json").write_text("{}\n")
    with pytest.raises(ValueError, match="resources changed"):
        save(workspace, active)
    assert not list((root / "artifacts").rglob("result.json"))


def test_other_member_packages_and_inactive_experiment_packages_do_not_invalidate_save(workspace):
    _, config, _, _ = workspace
    owner = config.parent.parent.parent
    expected = package_files(owner)
    (config.parent / "environment.toml").write_text('scope = "shared"\n')
    active = experiment(workspace)
    package_files(owner.parent / "Bob", version="8.0.0")
    package_files(config.parent, version="9.0.0")
    run = save(workspace, active)
    assert snapshot(run / "resources") == expected


@pytest.mark.parametrize("explicit", [True, False])
def test_changing_scope_with_identical_package_files_still_requires_new_experiment(
    workspace, explicit
):
    root, config, _, _ = workspace
    package_files(config.parent.parent.parent)
    active = experiment(workspace)
    package_files(config.parent)
    if explicit:
        (config.parent / "environment.toml").write_text('scope = "isolated"\n')
    with pytest.raises(ValueError, match="Package environment changed"):
        save(workspace, active)
    assert not list((root / "artifacts").rglob("result.json"))


def test_active_package_metadata_cannot_be_used_as_model_export(workspace):
    root, config, _, _ = workspace
    owner = config.parent.parent.parent
    package_files(owner)
    with pytest.raises(ValueError, match="active package environment files"):
        save(workspace, model=owner / "environment.json")
    assert not list((root / "artifacts").rglob("result.json"))


def test_equivalent_runtime_may_use_managed_lock_without_matching_venv_path(workspace):
    _, config, _, _ = workspace
    owner = config.parent.parent.parent
    baseline = personal_env._baseline()
    lock = "".join(f"{name}=={version}\n" for name, version in baseline["packages"].items())
    (owner / "requirements.in").write_text("")
    (owner / "requirements.lock").write_text(lock)
    (owner / "environment.json").write_text(json.dumps(personal_env._manifest("", lock, baseline)))
    assert not (owner / ".venv").exists()
    run = save(workspace)
    assert snapshot(run / "resources") == snapshot(owner)


def test_direct_common_python_cannot_claim_unused_personal_package_lock(workspace):
    root, config, _, _ = workspace
    owner = config.parent.parent.parent
    baseline = personal_env._baseline()
    lock = "".join(f"{name}=={version}\n" for name, version in baseline["packages"].items())
    content = "sixletters-not-installed-package==1.0.0\n"
    lock += content
    (owner / "requirements.in").write_text(content)
    (owner / "requirements.lock").write_text(lock)
    (owner / "environment.json").write_text(
        json.dumps(personal_env._manifest(content, lock, baseline))
    )
    with pytest.raises(ValueError, match="active package lock.*Python environment"):
        experiment(workspace)
    assert not list((root / "artifacts").rglob("result.json"))
