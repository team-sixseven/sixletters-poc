"""New experiments inherit editable inputs without copying old training outputs."""

import json
import tomllib

import pytest

from sixletters.personal_env import environment_directory
from sixletters.workspace import clone_experiment, init_experiment


@pytest.fixture
def source(tmp_path):
    root = tmp_path.resolve() / "Members"
    config = init_experiment("연구자", "first", root=root, script="fit_free.py")
    return root, config.parent


def write(path, name, raw):
    target = path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)


def test_clone_keeps_custom_code_settings_and_clears_notebooks(source):
    root, folder = source
    write(folder, "models/network.py", b"class Network: pass\n")
    write(folder, "helpers/preprocess.py", b"def prepare(x): return x\n")
    write(folder, "configs/tuning.yaml", b"steps: 10\n")
    write(folder, "README.md", "모델 실행 설명\n".encode())
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "widgets": {"generated": "content"},
            "kernelspec": {"name": "old-experiment", "display_name": "first"},
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "code",
                "metadata": {"execution": {"time": "yesterday"}},
                "execution_count": 10,
                "outputs": [{"text": "patient output"}],
                "source": ["print('experiment')"],
            }
        ],
    }
    write(folder, "analysis.ipynb", json.dumps(notebook).encode())
    for name in (
        "outputs/model.json",
        "outputs/generated.py",
        "data/patients.csv",
        "data/load.py",
        "models/weights.pkl",
        ".venv/site-packages/package.py",
        "__pycache__/compiled.pyc",
        ".env",
        "result.npz",
        "credentials.md",
    ):
        write(folder, name, b"never copy this")
    cloned = clone_experiment(folder, "연구자", "second", root=root)
    settings = tomllib.loads(cloned.read_text(encoding="utf-8"))
    assert settings["experiment"]["name"] == "second"
    assert settings["experiment"]["owner"] == "연구자"
    assert (cloned.parent / "fit_free.py").read_bytes() == (folder / "fit_free.py").read_bytes()
    assert (cloned.parent / "models/network.py").is_file()
    assert (cloned.parent / "helpers/preprocess.py").is_file()
    assert (cloned.parent / "configs/tuning.yaml").read_bytes() == b"steps: 10\n"
    assert (cloned.parent / "README.md").is_file()
    assert not (cloned.parent / "outputs").exists()
    assert not (cloned.parent / "data").exists()
    assert not (cloned.parent / "models/weights.pkl").exists()
    assert not (cloned.parent / ".venv").exists()
    assert not (cloned.parent / ".env").exists()
    assert not (cloned.parent / "credentials.md").exists()
    saved = json.loads((cloned.parent / "analysis.ipynb").read_text())
    assert saved["cells"][0]["outputs"] == []
    assert saved["cells"][0]["execution_count"] is None
    assert "widgets" not in saved["metadata"]
    assert "kernelspec" not in saved["metadata"]
    assert saved["metadata"]["language_info"] == {"name": "python"}
    assert json.loads((folder / "analysis.ipynb").read_text()) == notebook
    assert environment_directory(cloned.parent) == root / "연구자"


def package_files(path):
    for name, raw in {
        "requirements.in": b"xgboost==3.0.0\n",
        "requirements.lock": b"xgboost==3.0.0\n",
        "environment.json": b'{"version": "fixture"}\n',
    }.items():
        write(path, name, raw)


def test_shared_clone_ignores_inactive_package_records(source):
    root, folder = source
    owner = root / "연구자"
    package_files(owner)
    package_files(folder)
    write(folder, "environment.toml", b'scope = "shared"\n')
    cloned = clone_experiment(folder, "연구자", "second", root=root)
    assert environment_directory(cloned.parent) == owner
    assert not (cloned.parent / "requirements.lock").exists()
    assert (owner / "requirements.lock").read_bytes() == b"xgboost==3.0.0\n"


@pytest.mark.parametrize("scope", [None, "isolated"])
def test_isolated_and_legacy_environment_records_are_preserved(source, scope):
    root, folder = source
    package_files(folder)
    if scope:
        write(folder, "environment.toml", f'scope = "{scope}"\n'.encode())
    cloned = clone_experiment(folder, "연구자", "second", root=root)
    assert environment_directory(cloned.parent) == cloned.parent
    for name in ("requirements.in", "requirements.lock", "environment.json"):
        assert (cloned.parent / name).read_bytes() == (folder / name).read_bytes()
    assert not (cloned.parent / ".venv").exists()


def test_other_member_clone_snapshots_packages_without_changing_receiver(source):
    root, folder = source
    package_files(root / "연구자")
    write(root / "수신자", "requirements.in", b"receiver-package\n")
    cloned = clone_experiment(folder, "수신자", "second", root=root)
    assert environment_directory(cloned.parent) == cloned.parent
    assert (cloned.parent / "requirements.in").read_bytes() == b"xgboost==3.0.0\n"
    assert (root / "수신자/requirements.in").read_bytes() == b"receiver-package\n"


def test_alternative_experiment_config_updates_identity_only(source):
    root, folder = source
    raw = (folder / "config.toml").read_bytes()
    write(folder, "config-variant.toml", raw.replace(b"C = 1.0", b"C = 2.0"))
    cloned = clone_experiment(folder, "연구자", "second", root=root)
    alternative = tomllib.loads((cloned.parent / "config-variant.toml").read_text(encoding="utf-8"))
    assert alternative["experiment"]["name"] == "second"
    assert alternative["parameters"]["C"] == 2.0


def test_clone_never_overwrites_existing_experiment(source):
    root, folder = source
    with pytest.raises(FileExistsError):
        clone_experiment(folder, "연구자", "first", root=root)
    assert (
        tomllib.loads((folder / "config.toml").read_text(encoding="utf-8"))["experiment"]["name"]
        == "first"
    )


@pytest.mark.parametrize("name", ["helper.py", "configs/settings.yaml"])
def test_symlinked_source_is_rejected_without_creating_destination(source, tmp_path, name):
    root, folder = source
    target = folder / name
    target.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("do not follow")
    try:
        target.symlink_to(outside)
    except OSError:
        pytest.skip("Symbolic links unavailable")
    with pytest.raises(ValueError, match="regular files"):
        clone_experiment(folder, "연구자", "second", root=root)
    assert not (folder.parent / "second").exists()


@pytest.mark.parametrize(
    "notebook",
    [[], {"cells": [], "metadata": []}, {"cells": [[]]}, {"cells": [{"metadata": []}]}],
)
def test_invalid_notebook_fails_before_destination_is_created(source, notebook):
    root, folder = source
    write(folder, "analysis.ipynb", json.dumps(notebook).encode())
    with pytest.raises(ValueError, match="Invalid notebook"):
        clone_experiment(folder, "연구자", "second", root=root)
    assert not (folder.parent / "second").exists()
