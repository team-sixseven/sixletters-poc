"""Exercise command installation and a fresh Jupyter terminal in the real CPU image."""

import os

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
    _, run, notebook_port = docker_workspace
    run("docker", "compose", "up", "--build", "-d", "--wait", timeout=900)
    assert_installed_project_commands(run)
    assert_new_terminal_uses_project_environment(run, notebook_port)
