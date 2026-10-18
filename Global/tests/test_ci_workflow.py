"""Protect complete shard matrices and independently runnable Docker coverage."""

import ast
import re
from pathlib import Path

import pytest
import yaml

from scripts.ci_scope import DOCKER

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def jobs():
    return yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]


@pytest.mark.parametrize("job, mode", [("linux", "tests_mode"), ("compatibility", "native_mode")])
def test_full_platform_matrix_requires_all_four_shards(jobs, job, mode):
    selected = jobs[job]
    assert selected["strategy"]["matrix"]["shard"] == (
        "${{ fromJSON(needs.scope.outputs." + mode + " == '\"full\"' && '[0,1,2,3]' || '[0]') }}"
    )
    assert selected["strategy"]["fail-fast"] is False
    assert not selected.get("continue-on-error")
    command = next(step for step in selected["steps"] if "Run selected" in step.get("name", ""))
    assert '--tests-only full --shard-index "$SHARD_INDEX" --shard-count 4' in command["run"]
    assert command["env"]["SHARD_INDEX"] == "${{ matrix.shard }}"
    assert not command.get("if")
    assert not command.get("continue-on-error")
    smoke = next(step for step in selected["steps"] if "notebook" in step.get("name", ""))
    assert "matrix.shard == 0" in smoke["if"]
    assert job in jobs["final_gate"]["needs"]


def test_docker_matrix_runs_each_independent_scenario_once(jobs):
    matrix = jobs["docker"]["strategy"]["matrix"]
    assert set(matrix) == {"suite"}
    assert matrix["suite"] == "${{ fromJSON(needs.scope.outputs.docker || '[]') }}"
    command = next(
        step for step in jobs["docker"]["steps"] if step.get("name") == "Verify Docker integration"
    )
    assert set(re.findall(r"^\s+([a-z-]+)\)", command["run"], re.M)) == set(DOCKER)
    source = ast.parse((ROOT / "Global/tests/test_docker_workflow.py").read_text(encoding="utf-8"))
    tests = {
        node.name
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }
    targets = re.findall(r"Global/tests/test_docker_workflow\.py::(test_\w+)", command["run"])
    assert len(targets) == len(set(targets))
    assert set(targets) == tests
    for filename in ("test_docker_cli.py", "test_tracking_docker.py"):
        assert command["run"].count(f"Global/tests/{filename} ") == 1
    assert not jobs["docker"].get("continue-on-error")
    assert not command.get("continue-on-error")
    assert "docker" in jobs["final_gate"]["needs"]


def test_docker_images_read_shared_cache_without_waiting_for_cache_warm(jobs):
    assert jobs["docker"]["needs"] == "scope"
    steps = jobs["docker"]["steps"]
    workspace = next(step for step in steps if step.get("id") == "workspace_image")
    tracking = next(step for step in steps if step.get("id") == "tracking_image")
    assert "scope=workspace-shared-v1" in workspace["with"]["cache-from"]
    assert "scope=tracking-shared-v1" in tracking["with"]["cache-from"]
    assert "shared" not in workspace["with"]["cache-to"]
    assert "shared" not in tracking["with"]["cache-to"]
    assert "result-retry" in tracking["if"]
    assert "archive-retry" in tracking["if"]
    assert "tracking" in tracking["if"]
