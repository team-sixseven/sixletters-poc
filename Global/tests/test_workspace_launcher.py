"""Tools must reuse a GPU-only workspace without requiring a CPU image."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="The host launcher runs in WSL/Linux")


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            ["check"],
            [
                "exec",
                "-T",
                "gpu",
                "poc-container",
                "python",
                "Global/scripts/check_environment.py",
                "--gpu",
            ],
        ),
        (["shell", "-c", "true"], ["exec", "-T", "gpu", "poc-container", "bash", "-c", "true"]),
        (["compare"], ["exec", "-T", "gpu", "poc-container", "dev", "compare"]),
        (
            ["status"],
            ["exec", "-T", "gpu", "poc-container", "python", "-m", "sixletters", "status"],
        ),
        (
            ["exec", "exp001", "fit.py", "--epochs", "2"],
            [
                "exec",
                "-T",
                "gpu",
                "poc-container",
                "dev",
                "exec",
                "exp001",
                "fit.py",
                "--epochs",
                "2",
            ],
        ),
        (
            ["env", "add", "exp001", "optuna"],
            ["exec", "-T", "gpu", "poc-container", "dev", "env", "add", "exp001", "optuna"],
        ),
        (["sync"], ["exec", "-T", "gpu", "poc-container", "dev", "sync"]),
        (["archive"], ["exec", "-T", "gpu", "poc-container", "dev", "archive"]),
        (
            ["compare", "--remote"],
            ["exec", "-T", "gpu", "poc-container", "dev", "compare", "--remote"],
        ),
        (
            ["music", "--cohort", "hrv", "--offline"],
            [
                "exec",
                "-T",
                "gpu",
                "poc-container",
                "dev",
                "music",
                "--cohort",
                "hrv",
                "--offline",
            ],
        ),
        (["lint", "--staged"], ["exec", "-T", "gpu", "poc-container", "pre-commit", "run"]),
        (
            ["test"],
            ["exec", "-T", "gpu", "poc-container", "pytest", "-c", "Global/pyproject.toml", "-q"],
        ),
        (
            ["deps", "lock"],
            [
                "run",
                "--rm",
                "--no-deps",
                "-T",
                "-e",
                "POC_SKIP_DEPENDENCY_CHECK=1",
                "gpu",
                "uv",
                "lock",
                "--project",
                "Global",
            ],
        ),
    ],
)
def test_tools_select_running_gpu_service(tmp_path, arguments, expected):
    root = tmp_path / "checkout"
    root.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[2] / "dev", root / "dev")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['DOCKER_CALL_LOG'], 'a') as log:\n"
        "    log.write(json.dumps(args) + '\\n')\n"
        "if 'ps' in args and args[-1] == 'gpu':\n"
        "    print('gpu-container')\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    completed = subprocess.run(
        ["sh", str(root / "dev"), *arguments],
        cwd=root,
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "DOCKER_CALL_LOG": str(log),
            "POC_NO_TTY": "1",
        },
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    operation = calls[-1]
    if expected[0] == "exec":
        context = "host" if arguments[0] == "status" or expected[4] == "dev" else "container"
        skip_dependency_check = (
            ["-e", "POC_SKIP_DEPENDENCY_CHECK=1"] if arguments[0] == "status" else []
        )
        expected = [
            *expected[:2],
            *skip_dependency_check,
            "-e",
            f"SIXLETTERS_COMMAND_CONTEXT={context}",
            "-e",
            "SIXLETTERS_COMMAND_GPU=gpu",
            *expected[2:],
        ]
    assert operation[operation.index(expected[0]) :] == expected
    assert not any("run" in call and "cpu" in call for call in calls)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["up"], ["up", "--build", "-d", "--wait"]),
        (["down"], ["down"]),
        (["status"], ["ps"]),
        (["logs"], ["logs"]),
        (
            ["user", "member01"],
            [
                "exec",
                "tracking",
                "python",
                "/opt/tracking/start_tracking_server.py",
                "user",
                "member01",
            ],
        ),
    ],
)
def test_tracking_commands_use_only_server_compose(tmp_path, arguments, expected):
    root = tmp_path / "checkout"
    root.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[2] / "dev", root / "dev")
    (root / ".env.tracking").write_text("", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['DOCKER_CALL_LOG'], 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    completed = subprocess.run(
        ["sh", str(root / "dev"), "tracking", *arguments],
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "DOCKER_CALL_LOG": str(log),
        },
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2
    assert calls[-1] == [
        "compose",
        "--project-directory",
        str(root),
        "--env-file",
        str(root / ".env.tracking"),
        "-f",
        str(root / "compose.tracking.yaml"),
        *expected,
    ]
