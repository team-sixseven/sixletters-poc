"""Split a complete pytest collection deterministically across independent CI jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def shard_nodeids(nodeids: list[str], index: int, count: int) -> tuple[list[str], list[str]]:
    """Return the canonical collection and one disjoint, nonempty partition."""
    if type(index) is not int or type(count) is not int or count < 2 or not 0 <= index < count:
        raise ValueError("Shard index must be within a shard count of at least two")
    if not nodeids or any(not isinstance(nodeid, str) or not nodeid for nodeid in nodeids):
        raise ValueError("Cannot shard an empty or invalid pytest collection")
    if len(nodeids) != len(set(nodeids)):
        raise ValueError("Cannot shard duplicate pytest node IDs")
    ordered = sorted(nodeids)
    if len(ordered) < count:
        raise ValueError("Every shard must contain at least one test")
    return ordered, ordered[index::count]


def pytest_addoption(parser):
    group = parser.getgroup("ci-shards")
    group.addoption("--ci-shard-index", type=int, default=None)
    group.addoption("--ci-shard-count", type=int, default=None)
    group.addoption("--ci-shard-report", default=None)


def pytest_configure(config):
    index = config.getoption("ci_shard_index")
    count = config.getoption("ci_shard_count")
    if index is None or count is None:
        raise pytest.UsageError("Both --ci-shard-index and --ci-shard-count are required")
    if count < 2 or not 0 <= index < count:
        raise pytest.UsageError("Shard index must be within a shard count of at least two")
    if config.option.keyword or config.option.markexpr or config.getoption("lf", default=False):
        raise pytest.UsageError("CI shards must collect the full suite without selection filters")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    index = config.getoption("ci_shard_index")
    count = config.getoption("ci_shard_count")
    try:
        ordered, selected = shard_nodeids([item.nodeid for item in items], index, count)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    selected_ids = set(selected)
    by_id = {item.nodeid: item for item in items}
    deselected = [item for item in items if item.nodeid not in selected_ids]
    items[:] = [by_id[nodeid] for nodeid in selected]
    config.hook.pytest_deselected(items=deselected)
    report = {
        "schema_version": 1,
        "shard_index": index,
        "shard_count": count,
        "collected_count": len(ordered),
        "selected_count": len(selected),
        "collection_sha256": hashlib.sha256(
            json.dumps(ordered, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "selected_nodeids": selected,
    }
    config._ci_shard_report = report
    report_path = config.getoption("ci_shard_report")
    if report_path:
        path = Path(report_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            raise pytest.UsageError(f"Cannot write shard report: {error}") from error


def pytest_terminal_summary(terminalreporter, config):
    report = getattr(config, "_ci_shard_report", None)
    if report:
        terminalreporter.write_line(
            f"CI shard {report['shard_index'] + 1}/{report['shard_count']}: "
            f"{report['selected_count']}/{report['collected_count']} collected tests; "
            f"collection {report['collection_sha256']}"
        )
