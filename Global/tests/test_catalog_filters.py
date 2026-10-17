"""Run filters preserve explicit selection, correct identity and full comparison validation."""

import copy
import json
import sys

import pytest

from sixletters import __main__ as cli
from sixletters.run_catalog import filter_run_catalog


def entry(index, **changes):
    return {
        "run_id": f"{index:032x}",
        "local_run_id": f"local-{index:03}",
        "owner": "member",
        "experiment": "exp001",
        "dataset_version": "music-v1",
        "feature_set": "clinical",
        "model": "own",
        "parameters": {},
        "metrics": {"roc_auc": 1 - index / 1000},
        "created_at": f"2026-10-17T00:00:00.{index:06}Z",
        **changes,
    }


def write_local(root, item):
    directory = root / item["owner"] / item["experiment"] / item["local_run_id"]
    directory.mkdir(parents=True)
    (directory / "predictions.csv").touch()
    (directory / "result.json").write_text(
        json.dumps(
            {
                "owner": item["owner"],
                "experiment": item["experiment"],
                "run_id": item["local_run_id"],
                "created_at": item["created_at"],
                "feature_set": item["feature_set"],
                "synthetic": False,
                "comparison_context": {"dataset": {"dataset_id": item["dataset_version"]}},
                "training": {"model": item["model"], "parameters": item["parameters"]},
                "metrics": item["metrics"],
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_filters_apply_before_recent_limit_and_never_rank_by_score():
    runs = [entry(index) for index in range(1, 61)]
    runs += [entry(900, owner="another"), entry(901, dataset_version="music-hrv-v1")]
    runs += [entry(902, feature_set="ecg")]
    original = copy.deepcopy(runs)
    selected, count = filter_run_catalog(
        runs, owner="member", dataset="music-v1", feature_set="clinical", limit=2
    )
    assert count == 60
    assert [item["local_run_id"] for item in selected] == ["local-060", "local-059"]
    assert runs == original


def test_default_catalog_limit_bounds_six_hundred_runs_without_losing_match_count():
    selected, count = filter_run_catalog([entry(index) for index in range(600)])
    assert count == 600
    assert len(selected) == 20
    assert selected[0]["local_run_id"] == "local-599"
    assert selected[-1]["local_run_id"] == "local-580"


def test_timezones_ties_unknown_times_and_input_order_have_deterministic_order():
    runs = [
        entry(3, created_at=None),
        entry(2, created_at="2026-10-17T09:00:00+09:00"),
        entry(4, created_at="bad"),
        entry(1, created_at="2026-10-17T00:00:00Z"),
        entry(5, created_at="2026-10-17T10:00:00"),
    ]
    assert [item["local_run_id"] for item in filter_run_catalog(runs)[0]] == [
        f"local-{index:03}" for index in (1, 2, 3, 4, 5)
    ]
    assert filter_run_catalog(runs)[0] == filter_run_catalog(runs[::-1])[0]


@pytest.mark.parametrize("value", [0, -1, True, "5", None])
def test_catalog_limit_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="1 이상"):
        filter_run_catalog([entry(1)], limit=value)


def test_explicit_local_paths_still_skip_catalog_filters_and_keep_order(tmp_path, monkeypatch):
    paths = [tmp_path / "second", tmp_path / "first"]
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["poc", "compare", "--runs", *map(str, paths)])
    monkeypatch.setattr(cli, "choose_runs", lambda *_: pytest.fail("catalog requested"))
    seen = []

    def compare(runs, output, mode, reference):
        assert reference is None
        seen.extend(runs)
        return runs

    monkeypatch.setattr("sixletters.run_catalog.compare_with_reference", compare)
    monkeypatch.setattr(cli, "show_comparison_deltas", lambda _: None)
    assert cli.main() == 0
    assert seen == paths
