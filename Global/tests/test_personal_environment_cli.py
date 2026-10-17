"""CLI routing preserves ownership, arbitrary filenames, and personal script arguments."""

import sys
from pathlib import Path

import pytest

from sixletters import __main__ as cli
from sixletters import personal_env
from sixletters.local_settings import write_settings
from sixletters.workspace import init_experiment


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    (root / "Global").mkdir()
    (root / "Global/pyproject.toml").write_text('[project]\nname = "sixletters-poc"\n')
    write_settings(root, "팀원", Path("Global/data/prepared/demo-v1"))
    monkeypatch.setattr(cli, "project_root", lambda: root)
    return root


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["poc", *args])
    return cli.main()


def test_new_custom_filename_can_be_run_with_forwarded_arguments(workspace, monkeypatch):
    assert invoke(monkeypatch, "new", "exp001", "--script", "fit_rf.py") == 0
    experiment = workspace / "Members/팀원/experiments/exp001"
    assert (experiment / "fit_rf.py").is_file()
    assert not (experiment / "requirements.in").exists()
    assert not (experiment / "train.py").exists()
    calls = []
    monkeypatch.setattr(personal_env, "execute_script", lambda *args: calls.append(args) or 7)
    assert (
        invoke(monkeypatch, "exec", "exp001", "fit_rf.py", "--seed", "42", "--owner", "model") == 7
    )
    assert calls == [(experiment, Path("fit_rf.py"), ["--seed", "42", "--owner", "model"])]


def test_exec_owner_override_is_parsed_before_script(workspace, monkeypatch):
    config = init_experiment("다른 팀원", "exp002", root=workspace / "Members")
    calls = []
    monkeypatch.setattr(personal_env, "execute_script", lambda *args: calls.append(args) or 0)
    assert invoke(monkeypatch, "exec", "exp002", "--owner", "다른 팀원", "scripts/fit.py") == 0
    assert calls == [(config.parent, Path("scripts/fit.py"), [])]


@pytest.mark.parametrize(
    "arguments, function, kwargs, extra",
    [
        (
            ["env", "add", "exp001", "optuna==4.0", "xgboost"],
            "env_add",
            {},
            [["optuna==4.0", "xgboost"]],
        ),
        (["env", "remove", "exp001", "optuna"], "env_remove", {}, [["optuna"]]),
        (["env", "sync", "exp001"], "env_sync", {"relock": False}, []),
        (["env", "sync", "exp001", "--relock"], "env_sync", {"relock": True}, []),
        (["env", "isolate", "exp001"], "env_isolate", {}, []),
        (["env", "share", "exp001"], "env_share", {}, []),
    ],
)
def test_environment_actions_dispatch_to_one_experiment(
    workspace, monkeypatch, arguments, function, kwargs, extra
):
    config = init_experiment("팀원", "exp001", root=workspace / "Members")
    calls = []

    def record(*args, **options):
        calls.append((args, options))
        return config.parent / ".venv/bin/python"

    monkeypatch.setattr(personal_env, function, record)
    assert invoke(monkeypatch, *arguments) == 0
    assert calls == [((config.parent, *extra), kwargs)]


@pytest.mark.parametrize("script", ["../fit.py", "nested/fit.py", "script.sh", ".hidden.py"])
def test_scaffold_invalid_script_does_not_create_partial_experiment(workspace, script):
    with pytest.raises(ValueError):
        init_experiment("팀원", "exp001", root=workspace / "Members", script=script)
    assert not (workspace / "Members/팀원/experiments/exp001").exists()


def test_kernel_command_preserves_owner_and_relock(workspace, monkeypatch, capsys):
    config = init_experiment("다른 팀원", "실험 1", root=workspace / "Members")
    calls = []

    def register(directory, **options):
        calls.append((directory, options))
        return "SixSeven · 다른 팀원 / 실험 1", workspace / "kernel.json"

    monkeypatch.setattr(personal_env, "env_kernel", register)
    assert invoke(monkeypatch, "env", "kernel", "실험 1", "--owner", "다른 팀원", "--relock") == 0
    assert calls == [(config.parent, {"relock": True})]
    output = capsys.readouterr().out
    assert "SixSeven · 다른 팀원 / 실험 1" in output
    assert "모든 학습과 노트북 커널을 종료" in output


def test_environment_status_reports_member_default_without_installing(
    workspace, monkeypatch, capsys
):
    config = init_experiment("팀원", "exp001", root=workspace / "Members")
    assert invoke(monkeypatch, "env", "status", "exp001") == 0
    output = capsys.readouterr().out
    assert "개인 공통 · Members/팀원" in output
    assert f"Python: {sys.executable}" in output
    assert "공통 환경" in output
    assert not (workspace / "Members/팀원/.venv").exists()
    (config.parent / "environment.toml").write_text('scope = "isolated"\n')
    assert invoke(monkeypatch, "env", "status", "exp001") == 0
    assert "실험 전용 · Members/팀원/experiments/exp001" in capsys.readouterr().out


@pytest.mark.parametrize(
    "context, container, gpu, prefix",
    [
        (None, None, None, ["uv", "run", "--project", "Global", "poc"]),
        (None, "1", None, ["uv", "run", "--project", "Global", "poc"]),
        ("container", "1", "gpu", ["uv", "run", "--project", "Global", "poc"]),
        ("host", "1", "cpu", ["uv", "run", "--project", "Global", "poc"]),
        ("host", "1", "gpu", ["uv", "run", "--project", "Global", "poc"]),
    ],
)
def test_exec_and_status_share_refresh_reason_and_complete_command(
    workspace, monkeypatch, capsys, context, container, gpu, prefix
):
    import shlex

    owner, name = "다른 팀원's", "실험 1"
    config = init_experiment(owner, name, root=workspace / "Members")
    active = personal_env.environment_directory(config.parent)
    (active / "requirements.in").write_text("tabulate==0.9.0\n")
    (active / "requirements.lock").write_text("tabulate==0.9.0\n")
    (active / "environment.json").write_text("{}\n")
    for key, value in {
        "SIXLETTERS_COMMAND_CONTEXT": context,
        "SIXLETTERS_CONTAINER": container,
        "SIXLETTERS_COMMAND_GPU": gpu,
    }.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        personal_env, "_sync", lambda *a, **kw: pytest.fail("must not relock automatically")
    )
    assert invoke(monkeypatch, "env", "status", name, "--owner", owner) == 0
    status_output = capsys.readouterr().out
    command = status_output.split("다음: ")[-1].strip()
    assert shlex.split(command) == [*prefix, "env", "sync", name, "--owner", owner, "--relock"]
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, "exec", name, "--owner", owner, "train.py")
    assert error.value.code == 2
    exec_output = capsys.readouterr().err
    assert personal_env.LOCK_REFRESH_REASON in status_output
    assert personal_env.LOCK_REFRESH_REASON in exec_output
    assert f"다음: {command}" in exec_output
    assert (active / "environment.json").read_text() == "{}\n"


def test_notebook_refresh_error_has_actual_experiment_command(workspace, monkeypatch, capsys):
    config = init_experiment("팀원", "실험 1", root=workspace / "Members")
    active = personal_env.environment_directory(config.parent)
    (active / "requirements.in").write_text("tabulate==0.9.0\n")
    (active / "requirements.lock").write_text("tabulate==0.9.0\n")
    (active / "environment.json").write_text("{}\n")
    monkeypatch.setenv("SIXLETTERS_CONTAINER", "1")
    monkeypatch.setenv("SIXLETTERS_COMMAND_CONTEXT", "container")
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, "notebook", "실험 1")
    assert error.value.code == 2
    import shlex

    command = capsys.readouterr().err.split("다음: ")[-1].strip()
    assert shlex.split(command) == [
        "uv",
        "run",
        "--project",
        "Global",
        "poc",
        "env",
        "sync",
        "실험 1",
        "--owner",
        "팀원",
        "--relock",
    ]
    assert not (config.parent / "analysis.ipynb").exists()
