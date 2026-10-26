import asyncio
import base64
import json
import os
import secrets
import shlex
import shutil
import socket
import subprocess
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

import pytest
from tornado.httpclient import HTTPRequest
from tornado.websocket import websocket_connect

from scripts.ci_docker_cache import prepare_prebuilt_image


def copy_workspace(source: Path, workspace: Path) -> None:
    """Copy executable project files without a developer's data, results or experiments."""

    def ignore(directory, names):
        excluded = {
            ".git",
            ".venv",
            ".pytest_cache",
            ".ruff_cache",
            ".poc-cache",
            "__pycache__",
            ".poc.local.toml",
        }
        if Path(directory) == source:
            excluded.update({"data", "artifacts", "reports", "experiments", "notebooks"})
        if Path(directory) == source / "Global":
            excluded.update({"data", "artifacts", "reports"})
        if Path(directory) == source / "Members":
            return set(names) - {"README.md"}
        return {name for name in names if name in excluded or name.startswith(".env")}

    shutil.copytree(source, workspace, ignore=ignore)


@pytest.fixture
def docker_workspace(tmp_path_factory):
    source = Path(__file__).resolve().parents[2]
    workspace = tmp_path_factory.mktemp("docker") / "한글 공동 작업"
    copy_workspace(source, workspace)
    namespace = f"sixletters-test-{uuid.uuid4().hex[:12]}"
    compose_path = workspace / "compose.yaml"
    compose = compose_path.read_text(encoding="utf-8")
    assert compose.count("image: sixseven-poc:") == 2
    compose_path.write_text(
        compose.replace("image: sixseven-poc:", f"image: {namespace}:"), encoding="utf-8"
    )
    prebuilt_start = prepare_prebuilt_image(source, "workspace", f"{namespace}:cpu")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        notebook_port = listener.getsockname()[1]
    environment = {
        **{
            key: value
            for key, value in os.environ.items()
            if not key.startswith("MLFLOW_") and key != "POC_TRACKING_SYNC_INTERVAL"
        },
        "COMPOSE_PROJECT_NAME": namespace,
        "POC_NO_TTY": "1",
        "POC_NOTEBOOK_PORT": str(notebook_port),
    }
    environment.pop("RUN_DOCKER_TESTS", None)
    # Raw Compose must work without the launcher providing the host identity.
    environment.pop("POC_UID", None)
    environment.pop("POC_GID", None)
    environment.pop("COMPOSE_PROFILES", None)
    environment.pop("COMPOSE_FILE", None)

    def run(*arguments, timeout=900, expected_returncode=0, error_output=False):
        nonlocal prebuilt_start
        if prebuilt_start and arguments[:3] == ("docker", "compose", "up"):
            # Rebuild tests later in the suite must still exercise a real Compose build.
            arguments = tuple("--no-build" if item == "--build" else item for item in arguments)
            prebuilt_start = False
        completed = subprocess.run(
            arguments,
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        assert completed.returncode == expected_returncode, (
            f"{arguments!r} failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
        return completed.stderr if error_output else completed.stdout

    yield workspace, run, notebook_port
    subprocess.run(
        ["docker", "compose", "--profile", "gpu", "down", "--remove-orphans"],
        cwd=workspace,
        env=environment,
        capture_output=True,
        timeout=60,
        check=False,
    )
    subprocess.run(
        ["docker", "image", "rm", f"{namespace}:cpu"],
        capture_output=True,
        timeout=60,
        check=False,
    )


def assert_gpu_configuration(docker_workspace):
    _, run, _ = docker_workspace
    config = json.loads(run("docker", "compose", "--profile", "gpu", "config", "--format", "json"))
    cpu = config["services"]["cpu"]
    gpu = config["services"]["gpu"]
    assert not cpu.get("gpus")
    assert cpu["environment"]["SIXLETTERS_CONTAINER"] == "1"
    assert not cpu.get("devices")
    assert cpu.get("runtime") != "nvidia"
    assert not cpu.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices")
    assert "gpu" in gpu["profiles"]
    assert gpu["environment"]["POC_REQUIRE_GPU"] == "1"
    default_config = json.loads(run("docker", "compose", "config", "--format", "json"))
    assert set(default_config["services"]) == {"cpu"}
    reservations = gpu["deploy"]["resources"]["reservations"]["devices"]
    assert any(
        item["driver"] == "nvidia" and "gpu" in item["capabilities"] for item in reservations
    )


def assert_new_terminal_uses_project_environment(run, notebook_port):
    servers = run("./dev", "shell", "-c", "jupyter server list --json")
    server = next(json.loads(line) for line in servers.splitlines() if line.startswith("{"))
    headers = {"Authorization": f"token {server['token']}"}
    base_url = f"http://127.0.0.1:{notebook_port}"
    with urlopen(
        Request(f"{base_url}/api/terminals", method="POST", headers=headers), timeout=15
    ) as response:
        terminal = json.load(response)["name"]
    nonce = uuid.uuid4().hex
    marker = f"JUPYTER_ENV_{nonce}"
    # The complete marker is absent from the typed command, so terminal echo cannot pass it.
    python_probe = (
        "import sys; "
        "assert sys.executable == '/opt/venv/bin/python', sys.executable; "
        f"print('JUPYTER_ENV_' + {nonce!r})"
    )
    command = f"dev --help && python -c {shlex.quote(python_probe)}\n"

    async def check_terminal():
        connection = await websocket_connect(
            HTTPRequest(
                f"ws://127.0.0.1:{notebook_port}/terminals/websocket/{terminal}",
                headers=headers,
                connect_timeout=10,
            ),
        )
        output = ""
        try:
            await connection.write_message(json.dumps(["stdin", command]))
            async with asyncio.timeout(30):
                while marker not in output:
                    message = await connection.read_message()
                    assert message is not None, f"Jupyter terminal closed unexpectedly:\n{output}"
                    event = json.loads(message)
                    if event[0] == "stdout":
                        output += event[1]
            assert "usage: dev " in output
        except TimeoutError:
            pytest.fail(f"A fresh Jupyter terminal could not run dev and project Python:\n{output}")
        finally:
            connection.close()

    try:
        asyncio.run(check_terminal())
    finally:
        with urlopen(
            Request(f"{base_url}/api/terminals/{terminal}", method="DELETE", headers=headers),
            timeout=15,
        ):
            pass


@pytest.fixture(scope="module")
def tracking_server(tmp_path_factory):
    source = Path(__file__).resolve().parents[2]
    workspace = tmp_path_factory.mktemp("tracking-docker")
    (workspace / "Global/scripts").mkdir(parents=True)
    for filename in (
        "Global/Dockerfile.tracking",
        "compose.tracking.yaml",
        "Global/pyproject.toml",
        "Global/uv.lock",
        "Global/.python-version",
        ".dockerignore",
        "Global/scripts/start_tracking_server.py",
    ):
        shutil.copy2(source / filename, workspace / filename)
    namespace = f"sixletters-tracking-test-{uuid.uuid4().hex[:12]}"
    compose_path = workspace / "compose.tracking.yaml"
    compose_path.write_text(
        compose_path.read_text().replace("sixseven-poc:tracking", f"{namespace}:tracking")
    )
    prebuilt_start = prepare_prebuilt_image(source, "tracking", f"{namespace}:tracking")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    admin = "test-admin"
    password = secrets.token_urlsafe(24)
    environment = {
        **os.environ,
        "COMPOSE_PROJECT_NAME": namespace,
        "POC_MLFLOW_PORT": str(port),
        "POC_MLFLOW_BIND_HOST": "127.0.0.1",
        "MLFLOW_AUTH_ADMIN_USERNAME": admin,
        "MLFLOW_AUTH_ADMIN_PASSWORD": password,
        "MLFLOW_FLASK_SERVER_SECRET_KEY": secrets.token_urlsafe(48),
        "MLFLOW_SERVER_ALLOWED_HOSTS": "localhost:*,127.0.0.1:*,tracking:5000",
    }
    for name in ("COMPOSE_PROFILES", "COMPOSE_FILE", "MLFLOW_TRACKING_TOKEN"):
        environment.pop(name, None)
    compose = ["docker", "compose", "-f", "compose.tracking.yaml"]

    def run(*arguments, timeout=180):
        nonlocal prebuilt_start
        if prebuilt_start and arguments[:1] == ("up",):
            arguments = tuple("--no-build" if item == "--build" else item for item in arguments)
            prebuilt_start = False
        completed = subprocess.run(
            [*compose, *arguments],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        assert completed.returncode == 0, (
            f"Compose {arguments!r} failed\n{completed.stdout}\n{completed.stderr}"
        )
        return completed.stdout

    uri = f"http://127.0.0.1:{port}"

    def request(path, data=None, *, credentials=(admin, password), method=None, raw=None):
        headers = {}
        if credentials:
            token = base64.b64encode(":".join(credentials).encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        if data is not None:
            raw = json.dumps(data).encode()
            headers["Content-Type"] = "application/json"
        with urlopen(
            Request(uri + path, data=raw, headers=headers, method=method), timeout=30
        ) as response:
            body = response.read()
            return (
                json.loads(body)
                if body and response.headers.get_content_type() == "application/json"
                else body
            )

    try:
        run("up", "--build", "-d", "--wait", timeout=900)
        yield {
            "workspace": workspace,
            "run": run,
            "uri": uri,
            "request": request,
            "admin": admin,
            "password": password,
            "environment": environment,
        }
    finally:
        # Only this isolated test project's volume is deleted.
        subprocess.run(
            [*compose, "down", "--volumes", "--remove-orphans"],
            cwd=workspace,
            env=environment,
            capture_output=True,
            timeout=90,
            check=False,
        )
        subprocess.run(
            ["docker", "image", "rm", f"{namespace}:tracking"],
            capture_output=True,
            timeout=60,
            check=False,
        )
