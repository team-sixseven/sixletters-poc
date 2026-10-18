"""Expose Docker phase timings in successful and failed CI runs."""

import os
from pathlib import Path

import pytest
from docker_timing import record_report, reset, set_case, timing_lines


def pytest_sessionstart(session):
    reset(enabled=os.environ.get("RUN_DOCKER_TESTS") == "1")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    set_case(item.nodeid)


def pytest_runtest_logreport(report):
    record_report(report)


def pytest_terminal_summary(terminalreporter):
    lines = timing_lines()
    if not lines:
        return
    terminalreporter.section("Docker phase timings")
    for line in lines[2:]:
        terminalreporter.write_line(line)
    if destination := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(destination).open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
