"""CI partitions execute every collected test exactly once, including parameters."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci_shards import shard_nodeids


def test_partitions_are_complete_disjoint_balanced_and_input_order_independent():
    nodeids = [f"test_sample.py::test_parameter[{number:03d}]" for number in range(29)]
    partitions = [shard_nodeids(nodeids, index, 4)[1] for index in range(4)]
    assert sorted(nodeid for group in partitions for nodeid in group) == sorted(nodeids)
    assert max(map(len, partitions)) - min(map(len, partitions)) == 1
    for index, partition in enumerate(partitions):
        assert shard_nodeids(list(reversed(nodeids)), index, 4)[1] == partition
        assert all(not set(partition) & set(other) for other in partitions[index + 1 :])


@pytest.mark.parametrize(
    ("index", "count"),
    [(None, 4), (0, None), (-1, 4), (4, 4), (0, 1), (0, 0), (True, 4), (0, 4.0)],
)
def test_invalid_shard_configuration_fails_closed(index, count):
    with pytest.raises(ValueError, match="index"):
        shard_nodeids(["one", "two", "three", "four"], index, count)


@pytest.mark.parametrize("nodeids", [[], ["same", "same"], ["only"], ["", "two"], [1, 2]])
def test_empty_duplicate_invalid_or_too_small_collections_fail_closed(nodeids):
    with pytest.raises(ValueError):
        shard_nodeids(nodeids, 0, 2)


@pytest.fixture
def sample_suite(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    config = suite / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    (suite / "conftest.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "def pytest_collection_finish(session):\n"
        "    Path(session.config.rootpath, 'collected.json').write_text(\n"
        "        json.dumps([item.nodeid for item in session.items]))\n",
        encoding="utf-8",
    )
    (suite / "test_sample.py").write_text(
        "import pytest\n"
        "from pathlib import Path\n"
        "@pytest.mark.parametrize('number', range(13))\n"
        "def test_parameter(number):\n"
        "    Path(__file__).with_name(f'executed-{number}').touch()\n"
        "class TestOther:\n"
        "    def test_member(self):\n"
        "        Path(__file__).with_name('executed-class').touch()\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    def run(*arguments):
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-c", str(config), "-q", *arguments, str(suite)],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )

    return suite, run


def test_real_pytest_shards_execute_exactly_the_unsharded_collection(sample_suite):
    suite, run = sample_suite
    baseline = run("--collect-only")
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    expected = json.loads((suite / "collected.json").read_text())
    reports = []
    executed = set()
    for index in range(4):
        report = suite / f"shard-{index}.json"
        result = run(
            "-p",
            "Global.scripts.ci_shards",
            "--ci-shard-index",
            str(index),
            "--ci-shard-count",
            "4",
            "--ci-shard-report",
            str(report),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        current = {path.name for path in suite.glob("executed-*")}
        assert not executed & current
        executed.update(current)
        for path in suite.glob("executed-*"):
            path.unlink()
        payload = json.loads(report.read_text())
        assert payload["selected_nodeids"] == json.loads((suite / "collected.json").read_text())
        assert payload["selected_count"] == len(payload["selected_nodeids"])
        assert payload["collected_count"] == len(expected)
        assert payload["shard_index"] == index
        assert payload["shard_count"] == 4
        reports.append(payload)
    assert len(executed) == len(expected)
    assert sorted(nodeid for report in reports for nodeid in report["selected_nodeids"]) == sorted(
        expected
    )
    assert len({report["collection_sha256"] for report in reports}) == 1


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--ci-shard-index", "0"],
        ["--ci-shard-count", "4"],
        ["--ci-shard-index", "4", "--ci-shard-count", "4"],
        ["--ci-shard-index", "0", "--ci-shard-count", "4", "-k", "parameter"],
        ["--ci-shard-index", "0", "--ci-shard-count", "4", "--lf"],
        ["--ci-shard-index", "0", "--ci-shard-count", "100"],
    ],
)
def test_real_pytest_rejects_incomplete_or_filtered_shards(sample_suite, arguments):
    _, run = sample_suite
    result = run("-p", "Global.scripts.ci_shards", *arguments)
    assert result.returncode == 4, result.stdout + result.stderr


def test_failing_test_keeps_pytest_failure_exit_status(sample_suite):
    suite, run = sample_suite
    (suite / "test_sample.py").write_text(
        "import pytest\n@pytest.mark.parametrize('number', range(4))\n"
        "def test_parameter(number):\n    assert number != 0\n",
        encoding="utf-8",
    )
    result = run("-p", "Global.scripts.ci_shards", "--ci-shard-index", "0", "--ci-shard-count", "4")
    assert result.returncode == 1, result.stdout + result.stderr
