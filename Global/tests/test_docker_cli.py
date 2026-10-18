"""Exercise command installation and a fresh Jupyter terminal in the real CPU image."""

import json
import os
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from test_docker_workflow import (
    assert_installed_project_commands,
    assert_new_terminal_uses_project_environment,
    docker_workspace,  # noqa: F401
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1",
    reason="Set RUN_DOCKER_TESTS=1 on a host with Docker to check installed commands",
)


def test_registered_commands_and_fresh_jupyter_terminal(docker_workspace):  # noqa: F811
    workspace, run, notebook_port = docker_workspace
    # Existing local files must move safely on the real bind mount before the worker starts.
    legacy_data = workspace / "data/raw/layout-check.txt"
    legacy_result = workspace / "artifacts/layout-check/preserved.txt"
    for path in (legacy_data, legacy_result):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic migration sentinel")
    run("docker", "compose", "up", "--build", "-d", "--wait", timeout=900)
    for previous in (legacy_data, legacy_result):
        moved = workspace / "Global" / previous.relative_to(workspace)
        assert moved.read_bytes() == b"synthetic migration sentinel"
        assert not previous.exists()
    container_id = run("docker", "compose", "ps", "-q", "cpu").strip()
    assert container_id
    running = json.loads(run("docker", "inspect", container_id))[0]
    assert running["State"]["Health"]["Status"] == "healthy"
    assert running["NetworkSettings"]["Ports"]["8888/tcp"] == [
        {"HostIp": "127.0.0.1", "HostPort": str(notebook_port)}
    ]
    with pytest.raises(HTTPError) as response:
        urlopen(f"http://127.0.0.1:{notebook_port}/api/contents", timeout=10)
    assert response.value.code in (401, 403)
    environment_check = workspace / "Global/artifacts/environment-check/cpu/environment.json"
    environment = json.loads(environment_check.read_text(encoding="utf-8"))
    assert environment["synthetic_data_and_ecg"] == "passed"
    assert environment["gpu"] == "not requested"
    assert_installed_project_commands(run)
    assert_new_terminal_uses_project_environment(run, notebook_port)
