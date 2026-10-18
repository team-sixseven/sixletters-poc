"""Protect complete shard matrices and independently runnable Docker coverage."""

import ast
import itertools
import re
from pathlib import Path

import pytest
import yaml

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


def test_docker_matrix_runs_each_independent_phase_once(jobs):
    matrix = jobs["docker"]["strategy"]["matrix"]
    assert matrix["phase"] == ["core", "notebooks", "member"]
    pairs = [
        (suite, phase)
        for suite, phase in itertools.product(
            ("cli", "workspace", "recovery", "tracking"), matrix["phase"]
        )
        if {"suite": suite, "phase": phase} not in matrix["exclude"]
    ]
    assert pairs == [
        ("cli", "core"),
        ("workspace", "core"),
        ("workspace", "notebooks"),
        ("workspace", "member"),
        ("recovery", "core"),
        ("tracking", "core"),
    ]
    command = next(
        step for step in jobs["docker"]["steps"] if step.get("name") == "Verify Docker integration"
    )
    assert set(re.findall(r"^\s+([a-z]+/[a-z]+)\)", command["run"], re.M)) == {
        f"{suite}/{phase}" for suite, phase in pairs
    }
    source = ast.parse((ROOT / "Global/tests/test_docker_workflow.py").read_text(encoding="utf-8"))
    tests = {
        node.name
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }
    targets = re.findall(r"Global/tests/test_docker_workflow\.py::(test_\w+)", command["run"])
    assert len(targets) == len(set(targets))
    assert set(targets) == tests
    assert not jobs["docker"].get("continue-on-error")
    assert not command.get("continue-on-error")
    assert "docker" in jobs["final_gate"]["needs"]
