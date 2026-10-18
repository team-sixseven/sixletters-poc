"""Opt-in real Docker checks; no GPU hardware or GPU image downloads are required."""

import asyncio
import csv
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request, urlopen

import pytest
from test_tracking_docker import tracking_server  # noqa: F401
from tornado.httpclient import HTTPRequest
from tornado.websocket import websocket_connect

from scripts.ci_docker_cache import prepare_prebuilt_image

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1",
    reason="Set RUN_DOCKER_TESTS=1 on a host with Docker to run container integration checks",
)


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

    def run(*arguments, timeout=180, expected_returncode=0, error_output=False):
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


def test_gpu_configuration_is_opt_in_and_cpu_needs_no_nvidia_device(docker_workspace):
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


def assert_installed_project_commands(run):
    # Read the current project declaration, not a hard-coded command list. This
    # catches renamed or newly added console scripts in the actual built image.
    run(
        "./dev",
        "shell",
        "-c",
        "python -m pytest Global/tests/test_cli_registration.py -q",
        timeout=90,
    )


def test_workspace_trains_saves_and_compares_results(docker_workspace):
    workspace, run, _ = docker_workspace
    run("docker", "compose", "up", "--build", "-d", "--wait", timeout=900)
    container_id = run("docker", "compose", "ps", "-q", "cpu").strip()
    environment_check = workspace / "Global/artifacts/environment-check/cpu/environment.json"
    run("./dev", "demo", "--output", "Global/data/prepared/demo-v1")
    run("./dev", "init", "--owner", "팀원 가", "--data", "Global/data/prepared/demo-v1")
    owners = ("팀원 가", "팀원 나")
    run("./dev", "new", "exp001", "--script", "fit_rf.py")
    run("./dev", "new", "exp001", "--owner", owners[1])
    runs = []
    for owner in owners:
        if owner == owners[0]:
            # Only the additional package is installed; numerical packages stay in /opt/venv.
            run("./dev", "env", "add", "exp001", "tabulate==0.9.0", timeout=300)
            experiment_dir = workspace / "Members" / owner / "experiments" / "exp001"
            (experiment_dir / "package_probe.py").write_text(
                "import importlib.metadata as m, json, sysconfig, numpy, sklearn, tabulate\n"
                "print(json.dumps({'local': sorted(d.metadata['Name'] for d in "
                "m.distributions(path=[sysconfig.get_path('purelib')])), "
                "'numpy': numpy.__file__, 'sklearn': sklearn.__file__, "
                "'tabulate': tabulate.__file__}))\n"
            )
            packages = json.loads(run("./dev", "exec", "exp001", "package_probe.py"))
            assert packages["local"] == ["tabulate"]
            assert packages["numpy"].startswith("/opt/venv/")
            assert packages["sklearn"].startswith("/opt/venv/")
            assert packages["tabulate"].startswith(f"/workspace/Members/{owner}/.venv/")
            (experiment_dir / "preprocessing.yaml").write_text("imputer: median\n")
            run("./dev", "exec", "exp001", "fit_rf.py", timeout=300)
            assert (workspace / "Members" / owner / ".venv/pyvenv.cfg").is_file()
        else:
            run("./dev", "exec", "exp001", "--owner", owner, "train.py")
            member = workspace / "Members" / owner
            assert not any(
                (member / name).exists()
                for name in (".venv", "requirements.lock", "environment.json")
            )
        # The running workspace is reused; training is not replaced with a common runner.
        assert run("docker", "compose", "ps", "-q", "cpu").strip() == container_id
        saved = list((workspace / "Global/artifacts" / owner / "exp001").glob("*/result.json"))
        assert len(saved) == 1
        result = json.loads(saved[0].read_text(encoding="utf-8"))
        assert result["owner"] == owner
        if owner == owners[0]:
            assert result["software"]["packages"]["tabulate"] == "0.9.0"
            assert "environment.json" in result["resources"]
            assert (
                saved[0].parent / "resources/preprocessing.yaml"
            ).read_text() == "imputer: median\n"
            assert (saved[0].parent / "source/fit_rf.py").is_file()
            assert not (saved[0].parent / "source/.venv").exists()
        else:
            assert "environment.json" not in result["resources"]
            assert "scikit-learn" in result["software"]["packages"]
        assert result["synthetic"] is True
        assert result["comparison_context"]["evaluation"]["split"] == "validation"
        assert (saved[0].parent / "predictions.csv").stat().st_size > 0
        for filename in result["model_files"]:
            assert (saved[0].parent / "models" / filename).stat().st_size > 0
        runs.append(str(saved[0].parent.relative_to(workspace)))

    # Recovery commands must match the terminal where the failed command was entered.
    requirements = workspace / "Members" / owners[0] / "requirements.in"
    original_requirements = requirements.read_text(encoding="utf-8")
    requirements.write_text(original_requirements.replace("==0.9.0", ">=0.9.0"), encoding="utf-8")
    try:
        host_error = run(
            "./dev",
            "exec",
            "exp001",
            "package_probe.py",
            expected_returncode=2,
            error_output=True,
        )
        inner_error = run(
            "./dev",
            "shell",
            "-c",
            "dev exec exp001 package_probe.py",
            expected_returncode=2,
            error_output=True,
        )
        status = run("./dev", "env", "status", "exp001")
        assert "./dev env sync exp001 --owner '팀원 가' --relock" in host_error
        assert "dev env sync exp001 --owner '팀원 가' --relock" in inner_error
        assert "./dev" not in inner_error
        assert "./dev env sync exp001 --owner '팀원 가' --relock" in status
    finally:
        requirements.write_text(original_requirements, encoding="utf-8")

    comparison = "Global/artifacts/비교 결과/validation.csv"
    reference_id = Path(runs[1]).name
    output = run(
        "./dev",
        "compare",
        "--runs",
        *runs,
        "--reference",
        reference_id,
        "--output",
        comparison,
    )
    assert "기준 실행:" in output and reference_id in output
    with (workspace / comparison).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["owner"] for row in rows} == set(owners)
    assert len(rows) == 2
    assert all(row["evaluation_split"] == "validation" for row in rows)
    assert all(0 <= float(row["roc_auc"]) <= 1 for row in rows)
    assert all(row["reference_run_id"] == reference_id for row in rows)
    assert rows[1]["is_reference"] == "True"
    assert float(rows[1]["roc_auc_delta"]) == 0
    assert float(rows[1]["roc_auc_delta_ci_lower"]) == 0
    assert float(rows[1]["roc_auc_delta_ci_upper"]) == 0

    # Direct Compose startup must derive the mounted checkout owner, including non-1000 IDs.
    host_uid = workspace.stat().st_uid
    host_gid = workspace.stat().st_gid
    for file in [environment_check, workspace / comparison, workspace / ".poc.local.toml"]:
        assert file.stat().st_uid == host_uid
        assert file.stat().st_gid == host_gid


@contextmanager
def connected_tracking_workspace(docker_workspace, server, *, purpose):
    workspace, run, _ = docker_workspace
    suffix = uuid.uuid4().hex[:8]
    username, password = f"retry-member-{suffix}", secrets.token_urlsafe(24)
    server["request"]("/api/2.0/mlflow/users/create", {"username": username, "password": password})
    experiment_name = f"{purpose}-{suffix}"
    experiment_id = server["request"](
        "/api/2.0/mlflow/experiments/create", {"name": experiment_name}
    )["experiment_id"]
    settings = workspace / ".env"
    settings.write_text(
        "MLFLOW_TRACKING_URI=http://tracking:5000\n"
        f"MLFLOW_TRACKING_USERNAME={username}\n"
        f"MLFLOW_TRACKING_PASSWORD={password}\n"
        f"MLFLOW_EXPERIMENT_NAME={experiment_name}\n"
        "POC_TRACKING_SYNC_INTERVAL=5\n",
        encoding="utf-8",
    )
    settings.chmod(0o600)
    connected = False
    try:
        run("docker", "compose", "up", "--build", "-d", "--wait", timeout=900)
        workspace_id = run("docker", "compose", "ps", "-q", "cpu").strip()
        server_id = server["run"]("ps", "-q", "tracking").strip()
        networks = json.loads(
            run("docker", "inspect", "--format", "{{json .NetworkSettings.Networks}}", server_id)
        )
        assert len(networks) == 1
        network = next(iter(networks))
        # Keep the shared server on its original network across restarts. Moving
        # it between temporary networks can invalidate Docker's published port.
        run("docker", "network", "connect", network, workspace_id)
        connected = True
        yield workspace_id, experiment_id, (username, password)
    finally:
        settings.unlink(missing_ok=True)
        if connected:
            run("docker", "network", "disconnect", network, workspace_id)


def train_retry_runs(workspace, run, *, owner_prefix, personal_packages=False):
    """Make real common and optional overlay runs for independent recovery scenarios."""
    run("./dev", "demo", "--output", "Global/data/prepared/demo-v1")
    owners = (f"{owner_prefix}-a", f"{owner_prefix}-b")
    run("./dev", "init", "--owner", owners[0], "--data", "Global/data/prepared/demo-v1")
    local_runs = {}
    for owner in owners:
        run("./dev", "new", "retry001", "--owner", owner, "--script", "fit_model.py")
        if personal_packages and owner == owners[0]:
            run("./dev", "env", "add", "retry001", "--owner", owner, "tabulate==0.9.0", timeout=300)
        resource = workspace / "Members" / owner / "experiments/retry001/settings.yaml"
        resource.write_text("preprocessing: median\n", encoding="utf-8")
        run("./dev", "exec", "retry001", "--owner", owner, "fit_model.py", timeout=300)
        saved = list((workspace / "Global/artifacts" / owner / "retry001").glob("*/result.json"))
        assert len(saved) == 1
        result = json.loads(saved[0].read_text(encoding="utf-8"))
        assert result["synthetic"] is True
        if personal_packages and owner == owners[0]:
            assert result["software"]["packages"]["tabulate"] == "0.9.0"
            assert "environment.json" in result["resources"]
        local_runs[result["run_id"]] = {
            "owner": owner,
            "directory": saved[0].parent,
            "hashes": {
                path.relative_to(saved[0].parent).as_posix(): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in saved[0].parent.rglob("*")
                if path.is_file()
            },
        }
    return local_runs


def wait_for_published_runs(server, experiment_id, credentials, local_runs):
    # The retry interval is not an upload deadline: authenticated file uploads
    # and an existing queue can take longer on shared CI machines.
    deadline = time.monotonic() + 180
    published = {}
    while time.monotonic() < deadline:
        response = server["request"](
            "/api/2.0/mlflow/runs/search",
            {
                "experiment_ids": [experiment_id],
                "filter": "attributes.status = 'FINISHED' and tags.`sixletters.complete` = 'true'",
                "max_results": 100,
            },
            credentials=credentials,
        )
        published = {
            tags["sixletters.local_run_id"]: remote
            for remote in response.get("runs", [])
            if (tags := {tag["key"]: tag["value"] for tag in remote["data"]["tags"]})
            and tags.get("sixletters.local_run_id") in local_runs
        }
        if set(published) == set(local_runs):
            break
        time.sleep(5)
    if set(published) != set(local_runs):
        remaining = server["request"](
            "/api/2.0/mlflow/runs/search",
            {"experiment_ids": [experiment_id], "max_results": 100},
            credentials=credentials,
        )
        states = [
            {"status": remote["info"]["status"], "tags": remote["data"].get("tags", [])}
            for remote in remaining.get("runs", [])
        ]
        pytest.fail(f"Background retry did not finish both saved runs; server states: {states}")
    for local_id, saved in local_runs.items():
        assert saved["hashes"] == {
            path.relative_to(saved["directory"]).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in saved["directory"].rglob("*")
            if path.is_file()
        }
        assert {metric["key"] for metric in published[local_id]["data"]["metrics"]} == {
            "roc_auc",
            "average_precision",
            "brier_score",
        }
    return published


def test_workspace_worker_retries_saved_runs_when_tracking_returns(
    docker_workspace,
    tracking_server,  # noqa: F811
):
    workspace, run, _ = docker_workspace
    server = tracking_server
    with connected_tracking_workspace(docker_workspace, server, purpose="result-retry") as context:
        workspace_id, experiment_id, credentials = context
        server["run"]("stop", "tracking")
        local_runs = train_retry_runs(
            workspace, run, owner_prefix="result-retry", personal_packages=True
        )
        # Both personal script processes have exited. Only the live worker can retry.
        server["run"]("up", "-d", "--wait")
        published = wait_for_published_runs(server, experiment_id, credentials, local_runs)
        assert run("docker", "compose", "ps", "-q", "cpu").strip() == workspace_id
        remote_comparison = "Global/artifacts/retry-remote-comparison.csv"
        run(
            "./dev",
            "compare",
            "--remote",
            "--run-ids",
            *(item["info"]["run_id"] for item in published.values()),
            "--output",
            remote_comparison,
        )
        with (workspace / remote_comparison).open(encoding="utf-8", newline="") as handle:
            compared = list(csv.DictReader(handle))
        assert {item["owner"] for item in compared} == {"result-retry-a", "result-retry-b"}
        for remote in published.values():
            tags = {tag["key"]: tag["value"] for tag in remote["data"]["tags"]}
            assert tags["sixletters.archive_status"] == "none"


def test_workspace_worker_retries_selected_archive_when_tracking_returns(
    docker_workspace,
    tracking_server,  # noqa: F811
):
    workspace, run, _ = docker_workspace
    server = tracking_server
    with connected_tracking_workspace(docker_workspace, server, purpose="archive-retry") as context:
        workspace_id, experiment_id, credentials = context
        local_runs = train_retry_runs(
            workspace, run, owner_prefix="archive-retry", personal_packages=True
        )
        published = wait_for_published_runs(server, experiment_id, credentials, local_runs)
        # Selection made offline must survive the command exiting and be retried
        # by the workspace worker without promoting other local runs.
        selected_id, selected_run = next(iter(local_runs.items()))
        server["run"]("stop", "tracking")
        response = run(
            "./dev",
            "archive",
            "--run",
            selected_run["directory"].relative_to(workspace).as_posix(),
            expected_returncode=1,
        )
        assert "pending" in response
        server["run"]("up", "-d", "--wait")
        remote_id = published[selected_id]["info"]["run_id"]
        deadline = time.monotonic() + 180
        archive_status = None
        while time.monotonic() < deadline:
            remote = server["request"](
                f"/api/2.0/mlflow/runs/get?run_id={remote_id}", credentials=credentials
            )["run"]
            tags = {tag["key"]: tag["value"] for tag in remote["data"]["tags"]}
            archive_status = tags.get("sixletters.archive_status")
            if archive_status == "complete":
                break
            time.sleep(5)
        assert archive_status == "complete"
        assert run("docker", "compose", "ps", "-q", "cpu").strip() == workspace_id
        for local_id, base in published.items():
            remote = server["request"](
                f"/api/2.0/mlflow/runs/get?run_id={base['info']['run_id']}",
                credentials=credentials,
            )["run"]
            tags = {tag["key"]: tag["value"] for tag in remote["data"]["tags"]}
            assert remote["info"]["status"] == "FINISHED"
            assert tags["sixletters.complete"] == "true"
            assert tags["sixletters.archive_status"] == (
                "complete" if local_id == selected_id else "none"
            )
            if local_id == selected_id:
                root = remote["info"]["artifact_uri"].removeprefix("mlflow-artifacts:")
                for filename in ("resources/environment.json", "models/model.pkl"):
                    uploaded = server["request"](
                        f"/api/2.0/mlflow-artifacts/artifacts{root}/reproduction/{filename}",
                        credentials=credentials,
                    )
                    expected = (selected_run["directory"] / filename).read_bytes()
                    if isinstance(uploaded, dict):
                        assert uploaded == json.loads(expected)
                    else:
                        assert uploaded == expected


def test_personal_notebook_kernel_runs_packages_and_survives_workspace_restart(docker_workspace):
    workspace, run, notebook_port = docker_workspace
    run("docker", "compose", "up", "--build", "-d", "--wait", timeout=900)
    experiment = "notebook001"
    probes = []
    # Two members share the running Jupyter service, but retain independent environments.
    for personal_packages in (False, True):
        owner = "개인 패키지 팀원" if personal_packages else "공통 패키지 팀원"
        expected_prefix = f"/workspace/Members/{owner}/.venv" if personal_packages else "/opt/venv"
        run("./dev", "new", experiment, "--owner", owner)
        if personal_packages:
            run("./dev", "env", "add", experiment, "--owner", owner, "tabulate==0.9.0", timeout=300)
        run("./dev", "env", "kernel", experiment, "--owner", owner, timeout=300)
        display = f"SixSeven · {owner} / {experiment}"
        kernel_code = (
            "import sys, os\n"
            "from pathlib import Path\n"
            "import sixletters, sklearn\n"
            f"assert str(Path(sys.prefix)) == {expected_prefix!r}\n"
            'assert Path.cwd() == Path("/workspace")\n'
            'assert not os.environ.get("SIXLETTERS_EXECUTION_JSON")\n'
        )
        if personal_packages:
            kernel_code += (
                'import tabulate; assert tabulate.__version__ == "0.9.0"\n'
                'assert Path(sklearn.__file__).is_relative_to("/opt/venv")\n'
                "assert Path(tabulate.__file__).is_relative_to(sys.prefix)\n"
            )
        kind = "overlay" if personal_packages else "common"
        probe = workspace / f"Global/artifacts/notebook-kernel-{kind}-check.py"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "from jupyter_client import KernelManager\n"
            "from jupyter_client.kernelspec import KernelSpecManager\n"
            "from sixletters.personal_env import jupyter_data_path, env_add\n"
            "root = Path('/workspace')\n"
            f"experiment = root / 'Members' / {owner!r} / 'experiments' / {experiment!r}\n"
            "specs = KernelSpecManager(kernel_dirs=[str(jupyter_data_path(root) / 'kernels')])\n"
            f"name = next(name for name, spec in specs.get_all_specs().items() if spec['spec']['display_name'] == {display!r})\n"
            "manager = KernelManager(kernel_name=name, kernel_spec_manager=specs)\n"
            "manager.start_kernel(cwd=str(root))\n"
            "client = manager.blocking_client()\n"
            "client.start_channels()\n"
            "try:\n"
            "    client.wait_for_ready(timeout=60)\n"
            f"    code = {kernel_code!r}\n"
            "    reply = client.execute_interactive(code, timeout=60)\n"
            "    assert reply['content']['status'] == 'ok', reply['content']\n"
            "    try:\n"
            "        env_add(experiment, ['tabulate==0.8.10'])\n"
            "    except ValueError as error:\n"
            "        assert 'kernel' in str(error).lower()\n"
            "    else:\n"
            "        raise AssertionError('An open notebook must protect its environment')\n"
            "finally:\n"
            "    client.stop_channels()\n"
            "    manager.shutdown_kernel(now=True)\n"
            "print('Personal notebook kernel verified')\n",
            encoding="utf-8",
        )
        probes.append((owner, display, probe.relative_to(workspace).as_posix()))
    for restart in (False, True):
        if restart:
            run("docker", "compose", "down")
            run("docker", "compose", "up", "-d", "--wait", timeout=120)
        # Check discovery by the actual running Jupyter service, not just a local spec reader.
        servers = run("./dev", "shell", "-c", "jupyter server list --json")
        server = next(json.loads(line) for line in servers.splitlines() if line.startswith("{"))
        request = Request(
            f"http://127.0.0.1:{notebook_port}/api/kernelspecs",
            headers={"Authorization": f"token {server['token']}"},
        )
        with urlopen(request, timeout=15) as response:
            specs = json.load(response)["kernelspecs"]
        displays = {item["spec"]["display_name"] for item in specs.values()}
        for _, display, probe_path in probes:
            assert display in displays
            assert "Personal notebook kernel verified" in run(
                "./dev", "shell", "-c", f"python {shlex.quote(probe_path)}", timeout=120
            )
    run("./dev", "env", "remove", experiment, "--owner", probes[1][0], "tabulate", timeout=300)


def test_member_environment_reuse_isolation_and_reproduction(docker_workspace):
    workspace, run, _ = docker_workspace
    run("docker", "compose", "up", "--build", "-d", "--wait", timeout=900)
    run("./dev", "demo", "--output", "Global/data/prepared/demo-v1")
    owner = "공유 환경 팀원"
    member = workspace / "Members" / owner
    run("./dev", "init", "--owner", owner, "--data", "Global/data/prepared/demo-v1")
    for name in ("shared001", "shared002"):
        run("./dev", "new", name)
        (member / "experiments" / name / "probe.py").write_text(
            "import json, sys, tabulate\n"
            "print(json.dumps({'prefix': sys.prefix, 'version': tabulate.__version__}))\n"
        )
    run("./dev", "env", "add", "shared001", "tabulate==0.9.0", timeout=300)
    stamp = (member / ".venv/pyvenv.cfg").stat().st_mtime_ns
    run("./dev", "env", "add", "shared001", "tabulate==0.9.0", timeout=300)
    run("./dev", "env", "sync", "shared002", timeout=300)
    assert (member / ".venv/pyvenv.cfg").stat().st_mtime_ns == stamp
    run("./dev", "new", "shared003", "--from", "shared001")
    cloned = member / "experiments/shared003"
    assert cloned.joinpath("probe.py").is_file()
    assert not cloned.joinpath(".venv").exists()
    run("./dev", "notebook", "shared003", timeout=300)
    notebook = json.loads((cloned / "analysis.ipynb").read_text())
    assert notebook["metadata"]["kernelspec"]["display_name"].endswith("/ shared003")
    assert "준비 완료" in run("./dev", "status")
    assert (member / ".venv/pyvenv.cfg").stat().st_mtime_ns == stamp
    for name in ("shared001", "shared002"):
        status = run("./dev", "env", "status", name)
        assert "개인 공통" in status
        probe = json.loads(run("./dev", "exec", name, "probe.py", timeout=300))
        assert probe == {"prefix": f"/workspace/Members/{owner}/.venv", "version": "0.9.0"}
        assert not (member / "experiments" / name / ".venv").exists()
    assert (member / ".venv/pyvenv.cfg").stat().st_mtime_ns == stamp
    # Two independently edited experiments can train concurrently in one member environment.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run, "./dev", "exec", name, "train.py", timeout=300)
            for name in ("shared001", "shared002")
        ]
        for future in futures:
            assert "Validation results saved" in future.result()
    assert (member / ".venv/pyvenv.cfg").stat().st_mtime_ns == stamp
    owner_locks = {
        filename: (member / filename).read_bytes()
        for filename in ("requirements.in", "requirements.lock", "environment.json")
    }
    run("./dev", "env", "isolate", "shared002", timeout=300)
    run("./dev", "env", "add", "shared002", "tabulate==0.8.10", timeout=300)
    assert "실험 전용" in run("./dev", "env", "status", "shared002")
    probe = json.loads(run("./dev", "exec", "shared002", "probe.py"))
    assert probe == {
        "prefix": f"/workspace/Members/{owner}/experiments/shared002/.venv",
        "version": "0.8.10",
    }
    assert json.loads(run("./dev", "exec", "shared001", "probe.py"))["version"] == "0.9.0"
    assert owner_locks == {filename: (member / filename).read_bytes() for filename in owner_locks}
    run("./dev", "exec", "shared002", "train.py", timeout=300)
    selected = []
    for name, version in (("shared001", "0.9.0"), ("shared002", "0.8.10")):
        saved = max(
            (workspace / "Global/artifacts" / owner / name).glob("*/result.json"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        result = json.loads(saved.read_text())
        assert result["software"]["packages"]["tabulate"] == version
        assert f"tabulate=={version}" in (saved.parent / "resources/requirements.lock").read_text()
        selected.append(saved.parent.relative_to(workspace).as_posix())
    run("./dev", "compare", "--runs", *selected, "--output", "Global/artifacts/shared-env.csv")
    assert "shared001" in run("./dev", "compare", "--experiment", "shared001", "--list")
    run("./dev", "restore", "restored001", "--run", selected[1])
    assert "실험 전용" in run("./dev", "env", "status", "restored001")
    run("./dev", "exec", "restored001", "train.py", timeout=300)
    restored = json.loads(run("./dev", "exec", "restored001", "probe.py"))
    assert restored["version"] == "0.8.10"
    assert owner_locks == {filename: (member / filename).read_bytes() for filename in owner_locks}
    assert json.loads(run("./dev", "exec", "shared001", "probe.py"))["version"] == "0.9.0"


def test_workspace_restart_preserves_results_and_rebuilds_changed_dependencies(docker_workspace):
    workspace, run, notebook_port = docker_workspace
    run("docker", "compose", "up", "--build", "-d", "--wait", timeout=900)
    container_id = run("docker", "compose", "ps", "-q", "cpu").strip()
    run("./dev", "demo", "--output", "Global/data/prepared/lifecycle-demo")
    run("./dev", "init", "--owner", "lifecycle", "--data", "Global/data/prepared/lifecycle-demo")
    runs = []
    for name in ("preserved001", "preserved002"):
        run("./dev", "new", name)
        run("./dev", "exec", name, "train.py")
        result = next((workspace / "Global/artifacts/lifecycle" / name).glob("*/result.json"))
        runs.append(result.parent.relative_to(workspace).as_posix())
    comparison = "Global/artifacts/lifecycle-comparison.csv"
    run("./dev", "compare", "--runs", *runs, "--output", comparison)
    preserved = {
        path.relative_to(workspace): path.read_bytes()
        for directory in ("Global/data/prepared/lifecycle-demo", "Global/artifacts/lifecycle")
        for path in (workspace / directory).rglob("*")
        if path.is_file()
    }
    preserved[Path(comparison)] = (workspace / comparison).read_bytes()
    run("docker", "compose", "down", "--remove-orphans")
    assert (workspace / "Global/data/prepared/lifecycle-demo/dataset.json").is_file()
    assert (workspace / comparison).is_file()
    assert len(list((workspace / "Global/artifacts/lifecycle").rglob("result.json"))) == 2
    assert len(list((workspace / "Global/artifacts/lifecycle").rglob("model.pkl"))) == 2
    assert all((workspace / path).read_bytes() == content for path, content in preserved.items())
    run("docker", "compose", "up", "-d", "--wait", timeout=120)
    assert run("docker", "compose", "ps", "-q", "cpu").strip() != container_id
    assert_new_terminal_uses_project_environment(run, notebook_port)
    run(
        "./dev", "compare", "--runs", *runs, "--output", "Global/artifacts/restarted-comparison.csv"
    )
    assert (workspace / "Global/artifacts/restarted-comparison.csv").is_file()

    # A running workspace must reject stale dependencies until a real rebuild.
    pyproject = workspace / "Global/pyproject.toml"
    original = pyproject.read_text(encoding="utf-8")
    changed, replacements = re.subn(
        r"^description = .*$",
        'description = "Docker dependency recovery check"',
        original,
        count=1,
        flags=re.MULTILINE,
    )
    assert replacements == 1 and changed != original
    pyproject.write_text(changed, encoding="utf-8")
    run("./dev", "shell", "-c", "true", expected_returncode=2)
    run("./dev", "deps", "lock")
    run("./dev", "deps", "export")
    before_rebuild = run("docker", "compose", "ps", "-q", "cpu").strip()
    run("docker", "compose", "up", "--build", "-d", "--wait", timeout=900)
    after_rebuild = run("docker", "compose", "ps", "-q", "cpu").strip()
    assert after_rebuild and after_rebuild != before_rebuild, (
        "Compose kept the old container after rebuilding changed dependencies. "
        f"Compose version: {run('docker', 'compose', 'version', '--short')}. "
        "Use the supported Compose version (2.40.3 or later)."
    )
    run("./dev", "check")
