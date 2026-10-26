# ruff: noqa: F811
"""Real CPU workspace, fresh Jupyter terminal, notebook and persistence."""

import json
import os

import pytest
from docker_support import (
    assert_gpu_configuration,
    assert_new_terminal_uses_project_environment,
    docker_workspace,  # noqa: F401
)

pytestmark = pytest.mark.skipif(os.environ.get("RUN_DOCKER_TESTS") != "1", reason="Requires Docker")


def test_workspace_environment_and_notebook(docker_workspace):
    workspace, run, port = docker_workspace
    assert_gpu_configuration(docker_workspace)
    run("docker", "compose", "up", "--build", "-d", "--wait", "cpu")
    assert_new_terminal_uses_project_environment(run, port)
    run("./dev", "setup", "--owner", "alice")
    directory = workspace / "Members/alice/work"
    directory.mkdir()
    (directory / "probe.py").write_text(
        "from pathlib import Path\nimport pandas\nPath('Members/alice/proof.txt').write_text('ok')\n"
    )
    run("./dev", "exec", "Members/alice/work", "probe.py")
    assert (workspace / "Members/alice/proof.txt").read_text() == "ok"
    assert not (directory / ".venv").exists()
    run("./dev", "notebook", "Members/alice/work")
    assert json.loads((directory / "analysis.ipynb").read_text())["metadata"]["kernelspec"]["name"]
    notebook_path = directory / "analysis.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook["cells"][0]["source"] = [
        "import pandas, tabulate\n",
        "assert tabulate.__version__ == '0.9.0'\n",
    ]
    notebook_path.write_text(json.dumps(notebook))
    run("./dev", "env", "add", "Members/alice", "tabulate==0.9.0")
    assert (workspace / "Members/alice/.venv").is_dir()
    assert not (directory / ".venv").exists()
    run(
        "./dev",
        "shell",
        "-c",
        "jupyter nbconvert --to notebook --execute Members/alice/work/analysis.ipynb --output checked.ipynb",
    )
    run("docker", "compose", "restart", "cpu")
    run("docker", "compose", "up", "-d", "--wait", "cpu")
    assert "alice" in run("./dev", "status")
    assert (workspace / "Members/alice/proof.txt").read_text() == "ok"
