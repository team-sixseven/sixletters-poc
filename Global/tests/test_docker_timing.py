"""Timing output preserves test failures and excludes private invocation details."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import docker_timing
import pytest


@pytest.fixture(autouse=True)
def timing_state(monkeypatch):
    for name in ("_enabled", "_case", "_depth"):
        monkeypatch.setattr(docker_timing, name, getattr(docker_timing, name))
    monkeypatch.setattr(docker_timing, "_records", [])
    docker_timing.reset(enabled=True)
    yield


def test_failed_nested_phase_records_both_failures_without_exception_text(monkeypatch):
    clock = iter([10, 11, 13, 15])
    monkeypatch.setattr(docker_timing, "perf_counter", lambda: next(clock))
    docker_timing.set_case("test_example.py::test_recovery[private-parameter]")
    with pytest.raises(RuntimeError, match="private-error"):
        with docker_timing.stage("recovery"):
            with docker_timing.stage("upload"):
                raise RuntimeError("private-error")
    output = "\n".join(docker_timing.timing_lines())
    assert "| test_recovery | ↳ upload | 2.00 | failed |" in output
    assert "| test_recovery | recovery | 5.00 | failed |" in output
    assert "private" not in output
    assert "overlap" in output


def test_command_timing_records_lifecycle_without_arguments_or_output():
    with docker_timing.command_stage("tracking", ("up", "--password", "private-password")):
        pass
    with docker_timing.command_stage("workspace", ("docker", "compose", "up", "-d")):
        pass
    with docker_timing.command_stage("workspace", ("./dev", "exec", "private-experiment")):
        pass
    output = "\n".join(docker_timing.timing_lines())
    assert "tracking.up" in output and "workspace.up" in output
    assert "private" not in output and "--password" not in output


def test_overall_reports_include_setup_failures_and_skipped_tests():
    docker_timing.record_report(
        SimpleNamespace(
            nodeid="test_x.py::test_start", when="setup", duration=2.5, outcome="failed"
        )
    )
    docker_timing.record_report(
        SimpleNamespace(nodeid="test_x.py::test_skip", when="call", duration=0.0, outcome="skipped")
    )
    output = "\n".join(docker_timing.timing_lines())
    assert "pytest.setup | 2.50 | failed" in output
    assert "pytest.call | 0.00 | skipped" in output


def test_disabled_timing_does_not_record_normal_tests():
    docker_timing.reset()
    with docker_timing.stage("unused"):
        pass
    assert docker_timing.timing_lines() == []


def test_failed_pytest_run_keeps_failure_and_writes_phase_summary(tmp_path):
    source = Path(__file__).parent
    for name in ("conftest.py", "docker_timing.py"):
        shutil.copy2(source / name, tmp_path / name)
    (tmp_path / "test_sample.py").write_text(
        "from docker_timing import stage\n"
        "def test_failure():\n"
        "    with stage('recovery.remote_validation'):\n"
        "        assert False\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.md"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(tmp_path)],
        cwd=tmp_path,
        env={**os.environ, "RUN_DOCKER_TESTS": "1", "GITHUB_STEP_SUMMARY": str(summary)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Docker phase timings" in result.stdout
    report = summary.read_text(encoding="utf-8")
    assert "recovery.remote_validation" in report and "failed" in report
    assert "pytest.call" in report
