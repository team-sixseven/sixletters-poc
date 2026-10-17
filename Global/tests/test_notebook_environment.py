"""Run real isolated Jupyter kernels without downloading packages during tests."""

import importlib.metadata
import json
import os
import shutil
import sys
import venv
from pathlib import Path

import pytest
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from packaging.requirements import Requirement

from sixletters import personal_env as pe
from sixletters.execution import ENVIRONMENT_KEY


def _copy_kernel_dependencies(site: Path) -> None:
    """Copy only installed kernel dependencies into a genuinely isolated interpreter."""
    pending = ["ipykernel", "filelock", "packaging"]
    copied = set()
    while pending:
        name = pending.pop().lower().replace("_", "-")
        if name in copied:
            continue
        distribution = importlib.metadata.distribution(name)
        copied.add(name)
        for value in distribution.requires or ():
            requirement = Requirement(value)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
        for relative in distribution.files or ():
            if ".." in relative.parts or relative.is_absolute():
                continue
            source = Path(distribution.locate_file(relative))
            if not source.is_file():
                continue
            destination = site / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


@pytest.fixture
def kernels(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    (root / "Global").mkdir()
    (root / "Global/pyproject.toml").touch()
    source = Path(pe.__file__).parent
    shutil.copytree(
        source, root / "Global/src/sixletters", ignore=shutil.ignore_patterns("__pycache__")
    )
    baseline = pe._baseline()
    monkeypatch.setattr(pe, "_baseline", lambda: baseline)
    experiments = []
    for name, version in (("실험 A", "one"), ("실험 B", "two")):
        path = root / "Members/팀원 가/experiments" / name
        path.mkdir(parents=True)
        (path / "config.toml").write_text('[experiment]\nname = "test"\n')
        (path / "helper.py").write_text(f"VALUE = {version!r}\n")
        venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(path / pe.ENV_DIR)
        if os.name == "nt":
            site = path / pe.ENV_DIR / "Lib/site-packages"
        else:
            site = (
                path
                / pe.ENV_DIR
                / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
            )
        _copy_kernel_dependencies(site)
        (site / "personal_kernel_extra.py").write_text(f"VALUE = {version!r}\n")
        (path / pe.INPUT_FILE).write_text("")
        (path / pe.LOCK_FILE).write_text("")
        manifest = pe._manifest("", "", baseline)
        (path / pe.MANIFEST_FILE).write_text(json.dumps(manifest))
        (path / pe.STATE_FILE).write_text(pe._state(manifest))
        experiments.append(path)
    monkeypatch.setattr(pe, "_sync", lambda path, *args, **kwargs: pe._python(path / pe.ENV_DIR))
    monkeypatch.setattr(pe, "_prepare", lambda path, **kwargs: pe._python(path / pe.ENV_DIR))
    return root, experiments, baseline


def _execute(client, code):
    message = client.execute(code)
    streams = []
    while True:
        response = client.get_iopub_msg(timeout=30)
        if response["parent_header"].get("msg_id") != message:
            continue
        if response["msg_type"] == "error":
            pytest.fail("\n".join(response["content"]["traceback"]))
        if response["msg_type"] == "stream":
            streams.append(response["content"]["text"])
        if response["msg_type"] == "status" and response["content"]["execution_state"] == "idle":
            return "".join(streams)


def test_registered_kernels_execute_personal_packages_and_hold_environment_lock(
    kernels, monkeypatch
):
    root, experiments, _ = kernels
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "must-not-be-persisted")
    monkeypatch.setenv(ENVIRONMENT_KEY, '{"script": "old_training.py"}')
    names = []
    for index, path in enumerate(experiments):
        display, location = pe.env_kernel(path)
        names.append(location.parent.name)
        spec = json.loads(location.read_text())
        assert display == f"SixSeven · 팀원 가 / {path.name}"
        assert spec["argv"][0] == str(pe._python(path / pe.ENV_DIR))
        assert spec["env"] == {"PYTHONPATH": str(root / "Global/src")}
        assert "must-not-be-persisted" not in location.read_text()
        assert "old_training" not in location.read_text()
        # A fresh manager discovers the persistent registry, including after recreation.
        registry = KernelSpecManager(kernel_dirs=[str(pe.jupyter_data_path(root) / "kernels")])
        manager = KernelManager(kernel_name=location.parent.name, kernel_spec_manager=registry)
        manager.start_kernel(cwd=root, env=os.environ.copy())
        client = manager.client()
        client.start_channels()
        try:
            client.wait_for_ready(timeout=30)
            observed = json.loads(
                _execute(
                    client,
                    """
import json, os, sys, helper, personal_kernel_extra
print(json.dumps({'extra': personal_kernel_extra.VALUE, 'helper': helper.VALUE,
                  'prefix': sys.prefix, 'cwd': os.getcwd(),
                  'invocation': os.getenv('SIXLETTERS_EXECUTION_JSON'),
                  'tracking': os.getenv('MLFLOW_TRACKING_PASSWORD')}))
""",
                )
            )
            assert observed == {
                "extra": ("one", "two")[index],
                "helper": ("one", "two")[index],
                "prefix": str(path / pe.ENV_DIR),
                "cwd": str(root),
                "invocation": None,
                "tracking": "must-not-be-persisted",
            }
            before = (path / pe.MANIFEST_FILE).read_bytes()
            for action in (
                lambda path=path: pe.env_sync(path),
                lambda path=path: pe.env_sync(path, relock=True),
                lambda path=path: pe.env_add(path, ["optuna"]),
                lambda path=path: pe.env_remove(path, ["optuna"]),
                lambda path=path: pe.env_kernel(path),
            ):
                with pytest.raises(ValueError, match="Shut Down Kernel"):
                    action()
            assert (path / pe.MANIFEST_FILE).read_bytes() == before
        finally:
            manager.shutdown_kernel(now=True)
            client.stop_channels()
        assert pe.env_sync(path) == pe._python(path / pe.ENV_DIR)
    assert names[0] != names[1]
    assert importlib.util.find_spec("personal_kernel_extra") is None


def test_registry_identity_changes_for_runtime_and_checkout(kernels, monkeypatch):
    root, experiments, baseline = kernels
    _, first = pe.env_kernel(experiments[0])
    _, again = pe.env_kernel(experiments[0])
    assert first == again
    current = pe.jupyter_data_path(root)
    assert pe.jupyter_data_path(root / "another-checkout") != current
    monkeypatch.setattr(sys, "executable", str(Path(sys.executable).with_name("python3")))
    assert pe.jupyter_data_path(root) == current
    monkeypatch.setattr(sys, "prefix", "/different-runtime")
    assert pe.jupyter_data_path(root) != current
    monkeypatch.undo()
    changed = {**baseline, "fingerprint": "new-shared-image"}
    monkeypatch.setattr(pe, "_baseline", lambda: changed)
    assert pe.jupyter_data_path(root) != current


def test_kernel_launcher_rejects_shared_interpreter(kernels):
    _, experiments, _ = kernels
    with pytest.raises(ValueError, match="등록된 SixSeven"):
        pe.launch_kernel(experiments[0], Path("connection.json"))


def test_shared_member_kernels_run_concurrently_with_distinct_helpers(kernels):
    root, experiments, _ = kernels
    owner = experiments[0].parent.parent
    (experiments[0] / pe.ENV_DIR).rename(owner / pe.ENV_DIR)
    for name in (*pe.LOCK_NAMES, pe.STATE_FILE):
        shutil.copyfile(experiments[0] / name, owner / name)
    for path in experiments:
        (path / pe.SCOPE_FILE).write_text('scope = "shared"\n')
    managers = []
    try:
        for index, path in enumerate(experiments):
            _, location = pe.env_kernel(path)
            spec = json.loads(location.read_text())
            assert spec["argv"][0] == str(pe._python(owner / pe.ENV_DIR))
            registry = KernelSpecManager(kernel_dirs=[str(pe.jupyter_data_path(root) / "kernels")])
            manager = KernelManager(kernel_name=location.parent.name, kernel_spec_manager=registry)
            manager.start_kernel(cwd=root)
            client = manager.client()
            client.start_channels()
            managers.append((manager, client))
            client.wait_for_ready(timeout=30)
            observed = json.loads(
                _execute(
                    client,
                    "import helper, personal_kernel_extra, sys, json\nprint(json.dumps([helper.VALUE, personal_kernel_extra.VALUE, sys.prefix]))",
                )
            )
            assert observed == [("one", "two")[index], "one", str(owner / pe.ENV_DIR)]
        for path in experiments:
            for action in (
                lambda path=path: pe.env_sync(path),
                lambda path=path: pe.env_add(path, ["optuna"]),
                lambda path=path: pe.env_isolate(path),
            ):
                with pytest.raises(ValueError, match="Shut Down Kernel"):
                    action()
        manager, client = managers.pop(0)
        manager.shutdown_kernel(now=True)
        client.stop_channels()
        with pytest.raises(ValueError, match="Shut Down Kernel"):
            pe.env_sync(experiments[0], repair=True)
    finally:
        for manager, client in managers:
            manager.shutdown_kernel(now=True)
            client.stop_channels()
    assert pe.env_sync(experiments[0]) == pe._python(owner / pe.ENV_DIR)
