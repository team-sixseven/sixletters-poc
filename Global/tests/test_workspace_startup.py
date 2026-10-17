"""Startup must check prerequisites and preserve safe host and Git behavior."""

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(os.name == "nt", reason="Container identity is managed on Linux")
def test_entrypoint_uses_checkout_identity_before_executing_even_dependency_repairs(monkeypatch):
    entrypoint = load_script("container_entrypoint")
    order = []
    monkeypatch.setenv("POC_SKIP_DEPENDENCY_CHECK", "1")
    monkeypatch.setattr(entrypoint, "use_checkout_owner", lambda: order.append("drop privileges"))
    monkeypatch.setattr(entrypoint.os, "chdir", lambda _: order.append("enter checkout"))
    monkeypatch.setattr(entrypoint, "check_dependencies", lambda: pytest.fail("must skip hash"))
    monkeypatch.setattr(entrypoint.os, "execvp", lambda *args: order.append(args))
    monkeypatch.setattr(sys, "argv", ["poc-container", "uv", "lock"])
    assert entrypoint.main() == 0
    assert order == ["drop privileges", "enter checkout", ("uv", ["uv", "lock"])]


@pytest.mark.skipif(os.name == "nt", reason="Container identity is managed on Linux")
@pytest.mark.parametrize(
    ("owner", "expected"), [((1234, 5678), (1234, 5678)), ((0, 0), (1000, 1000))]
)
def test_checkout_uid_is_derived_and_supplementary_root_groups_are_removed(
    monkeypatch, owner, expected
):
    entrypoint = load_script("container_entrypoint")
    workspace = Mock()
    workspace.stat.return_value = SimpleNamespace(st_uid=owner[0], st_gid=owner[1])
    operations = []
    monkeypatch.delenv("POC_UID", raising=False)
    monkeypatch.delenv("POC_GID", raising=False)
    monkeypatch.setenv("USER", "original")
    monkeypatch.setenv("LOGNAME", "original")
    monkeypatch.setattr(entrypoint.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        entrypoint.os, "setgroups", lambda value: operations.append(("groups", value))
    )
    monkeypatch.setattr(entrypoint.os, "setgid", lambda value: operations.append(("gid", value)))
    monkeypatch.setattr(entrypoint.os, "setuid", lambda value: operations.append(("uid", value)))
    monkeypatch.setattr(entrypoint.os, "access", lambda *args: True)
    monkeypatch.setattr(entrypoint.pwd, "getpwuid", Mock(side_effect=KeyError))
    entrypoint.use_checkout_owner(workspace)
    assert operations == [("groups", []), ("gid", expected[1]), ("uid", expected[0])]
    assert os.environ["USER"] == f"poc-{expected[0]}"
    assert os.environ["LOGNAME"] == f"poc-{expected[0]}"


@pytest.mark.skipif(os.name == "nt", reason="Container identity is managed on Linux")
def test_explicit_root_identity_is_not_used_for_training(monkeypatch):
    entrypoint = load_script("container_entrypoint")
    workspace = Mock()
    workspace.stat.return_value = SimpleNamespace(st_uid=0, st_gid=0)
    monkeypatch.setenv("POC_UID", "0")
    monkeypatch.delenv("POC_GID", raising=False)
    with pytest.raises(RuntimeError, match="POC_UID=0"):
        entrypoint.use_checkout_owner(workspace)


@pytest.mark.skipif(os.name == "nt", reason="Container entrypoint is Linux-only")
def test_dependency_guard_detects_change_before_command_execution(tmp_path, monkeypatch):
    entrypoint = load_script("container_entrypoint")
    dependency = tmp_path / "uv.lock"
    dependency.write_bytes(b"original locked versions")
    manifest = tmp_path / "hashes"
    manifest.write_text(f"{hashlib.sha256(dependency.read_bytes()).hexdigest()}  uv.lock\n")
    entrypoint.check_dependencies(tmp_path, manifest)
    dependency.write_bytes(b"different locked versions")
    monkeypatch.delenv("POC_SKIP_DEPENDENCY_CHECK", raising=False)
    monkeypatch.setattr(entrypoint, "use_checkout_owner", lambda: None)
    monkeypatch.setattr(entrypoint.os, "chdir", lambda _: None)
    check = entrypoint.check_dependencies
    monkeypatch.setattr(entrypoint, "check_dependencies", lambda: check(tmp_path, manifest))
    monkeypatch.setattr(entrypoint.os, "execvp", lambda *args: pytest.fail("stale image executed"))
    monkeypatch.setattr(sys, "argv", ["poc-container", "python", "train.py"])
    assert entrypoint.main() == 2


@pytest.mark.parametrize("gpu", [False, True])
def test_failed_environment_check_never_starts_workspace(monkeypatch, gpu):
    startup = load_script("start_workspace")
    monkeypatch.setenv("POC_REQUIRE_GPU", "1" if gpu else "0")
    checked = []

    def reject(command, **kwargs):
        if command[1] == "Global/scripts/migrate_layout.py":
            return
        checked.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(startup.subprocess, "run", reject)
    monkeypatch.setattr(
        startup.os, "execvp", lambda *args: pytest.fail("started after failed check")
    )
    assert startup.main() == 1
    assert ("--gpu" in checked[0]) is gpu


@pytest.mark.parametrize("configured", [False, True])
def test_tracking_worker_starts_only_for_configured_server(monkeypatch, configured):
    startup = load_script("start_workspace")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking:5000" if configured else "")
    monkeypatch.setenv("POC_TRACKING_SYNC_INTERVAL", "60")
    monkeypatch.setattr(startup.subprocess, "run", Mock())
    worker = Mock()
    monkeypatch.setattr(startup.subprocess, "Popen", worker)
    monkeypatch.setattr(startup.os, "chdir", Mock())
    notebook = Mock()
    monkeypatch.setattr(startup.os, "execvp", notebook)
    assert startup.main() == 0
    assert worker.call_count == int(configured)
    assert notebook.call_args.args[0] == "jupyter"
    if configured:
        assert worker.call_args.args[0][-1] == "Global/scripts/tracking_worker.py"


def test_invalid_retry_interval_does_not_start_workspace(monkeypatch):
    startup = load_script("start_workspace")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking:5000")
    monkeypatch.setenv("POC_TRACKING_SYNC_INTERVAL", "0")
    monkeypatch.setattr(startup.subprocess, "Popen", lambda *a, **k: pytest.fail("started worker"))
    monkeypatch.setattr(startup.os, "execvp", lambda *a: pytest.fail("started notebook"))
    assert startup.main() == 1


@pytest.mark.skipif(os.name == "nt", reason="Container identity is managed on Linux")
def test_unwritable_checkout_stays_blocked_after_nonroot_fallback(monkeypatch):
    entrypoint = load_script("container_entrypoint")
    workspace = Mock()
    workspace.stat.return_value = SimpleNamespace(st_uid=0, st_gid=0)
    monkeypatch.delenv("POC_UID", raising=False)
    monkeypatch.delenv("POC_GID", raising=False)
    monkeypatch.setattr(entrypoint.os, "geteuid", lambda: 0)
    monkeypatch.setattr(entrypoint.os, "setgroups", lambda _: None)
    monkeypatch.setattr(entrypoint.os, "setgid", lambda _: None)
    monkeypatch.setattr(entrypoint.os, "setuid", lambda _: None)
    monkeypatch.setattr(entrypoint.os, "access", lambda *args: False)
    with pytest.raises(RuntimeError, match="쓸 수 없습니다"):
        entrypoint.use_checkout_owner(workspace)


def test_layout_migration_completes_before_environment_check_and_worker(monkeypatch):
    startup = load_script("start_workspace")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking:5000")
    monkeypatch.setenv("POC_TRACKING_SYNC_INTERVAL", "60")
    operations = []
    monkeypatch.setattr(
        startup.subprocess, "run", lambda command, **kwargs: operations.append(command[1])
    )
    monkeypatch.setattr(
        startup.subprocess, "Popen", lambda command, **kwargs: operations.append(command[1])
    )
    monkeypatch.setattr(startup.os, "chdir", lambda _: None)
    monkeypatch.setattr(startup.os, "execvp", lambda *args: operations.append("jupyter"))
    assert startup.main() == 0
    assert operations == [
        "Global/scripts/migrate_layout.py",
        "Global/scripts/check_environment.py",
        "Global/scripts/tracking_worker.py",
        "jupyter",
    ]


def test_failed_migration_stops_before_check_worker_or_notebook(monkeypatch):
    startup = load_script("start_workspace")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking:5000")
    monkeypatch.setenv("POC_TRACKING_SYNC_INTERVAL", "60")
    commands = []

    def fail(command, **kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(startup.subprocess, "run", fail)
    monkeypatch.setattr(startup.subprocess, "Popen", lambda *a, **k: pytest.fail("worker started"))
    monkeypatch.setattr(startup.os, "execvp", lambda *a: pytest.fail("notebook started"))
    assert startup.main() == 1
    assert len(commands) == 1
    assert commands[0][1] == "Global/scripts/migrate_layout.py"


def test_workspace_reuses_persistent_runtime_kernel_registry(monkeypatch):
    from sixletters.personal_env import jupyter_data_path

    startup = load_script("start_workspace")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    monkeypatch.setenv("JUPYTER_PATH", "/custom/jupyter")
    monkeypatch.setattr(startup.subprocess, "run", Mock())
    monkeypatch.setattr(startup.os, "chdir", Mock())
    notebook = Mock()
    monkeypatch.setattr(startup.os, "execvp", notebook)
    expected = str(jupyter_data_path(Path("/workspace")))
    assert startup.main() == 0
    assert os.environ["JUPYTER_PATH"].split(os.pathsep) == [expected, "/custom/jupyter"]
    assert startup.main() == 0
    assert os.environ["JUPYTER_PATH"].split(os.pathsep) == [expected, "/custom/jupyter"]
