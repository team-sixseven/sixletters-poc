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


def test_release_is_checked_before_merge_without_duplicate_push_run():
    # BaseLoader preserves the YAML 1.2 event name "on" instead of treating it as True.
    workflow = yaml.load(
        (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    events = workflow["on"]
    assert set(events) == {"pull_request", "workflow_dispatch"}
    assert set(events["pull_request"]["branches"]) == {"develop", "main"}
    assert set(events["pull_request"]["types"]) == {"opened", "synchronize", "reopened", "edited"}
    assert not {"paths", "paths-ignore"} & events["pull_request"].keys()


def test_every_pr_keeps_policy_and_selected_job_validation(jobs):
    assert "if" not in jobs["checks"]
    assert set(jobs["final_gate"]["needs"]) == {
        "scope",
        "checks",
        "linux",
        "compatibility",
        "docker",
    }
    assert "always()" in jobs["final_gate"]["if"]
    for job in jobs.values():
        assert not job.get("continue-on-error")


@pytest.mark.parametrize("job, mode", [("linux", "tests_mode"), ("compatibility", "native_mode")])
def test_full_platform_matrix_requires_all_four_shards(jobs, job, mode):
    selected = jobs[job]
    if job == "linux":
        assert selected["strategy"]["matrix"]["shard"] == (
            "${{ fromJSON(needs.scope.outputs."
            + mode
            + " == '\"full\"' && '[0,1,2,3]' || '[0]') }}"
        )
    else:
        assert selected["strategy"]["matrix"]["shard"] == (
            "${{ fromJSON(needs.scope.outputs.native_shard_count == '4' && '[0,1,2,3]' "
            "|| needs.scope.outputs.native_shard_count == '2' && '[0,1]' || '[0]') }}"
        )
        assert (
            jobs["scope"]["outputs"]["native_shard_count"]
            == "${{ steps.scope.outputs.native_shard_count }}"
        )
    assert selected["strategy"]["fail-fast"] is False
    assert not selected.get("continue-on-error")
    command = next(step for step in selected["steps"] if "Run selected" in step.get("name", ""))
    if job == "linux":
        assert '--tests-only full --shard-index "$SHARD_INDEX" --shard-count 4' in command["run"]
    else:
        assert (
            '--tests-only "$TESTS_MODE" --test-files "$TEST_FILES" --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT"'
            in command["run"]
        )
        assert 'if [[ "$SHARD_COUNT" != 1 ]]' in command["run"]
        assert command["env"]["SHARD_COUNT"] == "${{ needs.scope.outputs.native_shard_count }}"
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
