"""Run filters preserve explicit selection, correct identity and full comparison validation."""

import copy
import json
import sys

import pytest

from sixletters import __main__ as cli
from sixletters import tracking
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


def test_local_and_remote_filters_select_same_identity_and_print_omitted_count(
    tmp_path, monkeypatch, capsys
):
    runs = [entry(1), entry(2), entry(3), entry(4, owner="another")]
    locations = {item["local_run_id"]: write_local(tmp_path, item) for item in runs}
    options = {"owner": "member", "dataset": "music-v1", "feature_set": "clinical", "limit": 2}
    monkeypatch.setattr("builtins.input", lambda _: "2 1")
    assert cli.choose_runs(tmp_path, **options) == [locations["local-002"], locations["local-003"]]
    local_output = capsys.readouterr().out
    assert cli.choose_remote_runs(runs, **options) == [entry(2)["run_id"], entry(3)["run_id"]]
    remote_output = capsys.readouterr().out
    for output in (local_output, remote_output):
        assert "조건에 맞는 실행 3개 중 최근 2개 표시" in output
        assert "--limit" in output
        assert "local-001" not in output
        assert "another" not in output
        assert output.index("1. member / exp001 / local-003") < output.index(
            "2. member / exp001 / local-002"
        )
    assert f"서버 실행 ID: {entry(3)['run_id']}" in remote_output


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("filters", [{"owner": "missing"}, {"dataset": "data/music-v1"}])
def test_no_match_explains_filters_without_prompt_or_download(
    tmp_path, monkeypatch, capsys, local, filters
):
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("prompted"))
    write_local(tmp_path, entry(1))
    with pytest.raises(ValueError, match="dataset_id"):
        if local:
            cli.choose_runs(tmp_path, **filters)
        else:
            cli.choose_remote_runs([entry(1)], **filters)
    assert "조건에 맞는 실행 0개" in capsys.readouterr().out


def test_one_visible_run_prompts_to_widen_filters_and_does_not_auto_select(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("prompted"))
    with pytest.raises(ValueError, match="--limit를 2 이상"):
        cli.choose_remote_runs([entry(1), entry(2)], limit=1)


def test_archive_uses_same_filters_but_allows_one_visible_run(tmp_path, monkeypatch):
    write_local(tmp_path, entry(1))
    latest = write_local(tmp_path, entry(2))
    monkeypatch.setattr("builtins.input", lambda _: "1")
    assert cli.choose_runs(tmp_path, archive=True, owner="member", limit=1) == [latest]


@pytest.mark.parametrize("source", ["local", "remote", "archive"])
@pytest.mark.parametrize("custom_root", [False, True])
def test_cli_forwards_all_catalog_filters_to_requested_source(
    tmp_path, monkeypatch, source, custom_root
):
    arguments = ["archive"] if source == "archive" else ["compare"]
    if source == "remote":
        arguments += ["--remote"]
    result_root = tmp_path / ("reviewer-artifacts" if custom_root else "Global/artifacts")
    if custom_root:
        arguments += ["--root", str(result_root)]
    arguments += [
        "--owner",
        "member",
        "--dataset",
        "music-v1",
        "--feature-set",
        "ecg",
        "--limit",
        "5",
    ]
    paths = [tmp_path / "one", tmp_path / "two"]
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["dev", *arguments])
    calls = []

    def select(value, **filters):
        if source == "archive":
            assert filters.pop("archive") is True
        assert filters == {
            "owner": "member",
            "dataset": "music-v1",
            "feature_set": "ecg",
            "limit": 5,
        }
        assert value == (["metadata"] if source == "remote" else result_root)
        calls.append("selected")
        return ["a" * 32, "b" * 32] if source == "remote" else paths

    monkeypatch.setattr(cli, "choose_runs", select)
    monkeypatch.setattr(cli, "choose_remote_runs", select)
    monkeypatch.setattr(tracking, "list_remote_runs", lambda **_kwargs: ["metadata"])

    def fetch(root, *, run_ids, issues):
        assert root == result_root
        assert run_ids == ["a" * 32, "b" * 32]
        return paths

    monkeypatch.setattr(tracking, "fetch_runs", fetch)

    def compare(runs, output, mode, reference):
        assert reference is None
        assert runs == paths
        calls.append("compared")
        return runs

    def archive(path):
        assert path == paths[0]
        calls.append("archived")
        return "uploaded"

    monkeypatch.setattr("sixletters.run_catalog.compare_with_reference", compare)
    monkeypatch.setattr(cli, "show_comparison_deltas", lambda _: None)
    monkeypatch.setattr(tracking, "archive_run", archive)
    assert cli.main() == 0
    assert calls == ["selected", "archived" if source == "archive" else "compared"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["compare", "--remote", "--runs", "one", "two"],
        ["compare", "--run-ids", "a" * 32, "b" * 32],
        ["compare", "--remote", "--run-ids", "a" * 32],
        ["compare", "--remote", "--run-ids", "a" * 32, "a" * 32],
        ["compare", "--remote", "--run-ids", "../unsafe", "b" * 32],
        ["compare", "--runs", "one", "two", "--root", "elsewhere"],
        ["compare", "--runs", "one", "two", "--owner", "member"],
        ["compare", "--remote", "--run-ids", "a" * 32, "b" * 32, "--limit", "2"],
        ["compare", "--limit", "0"],
        ["compare", "--limit", "-1"],
        ["compare", "--limit", "2.5"],
        ["archive", "--run", "one", "--dataset", "music-v1"],
        ["archive", "--limit", "0"],
    ],
)
def test_invalid_or_ambiguous_cli_options_fail_before_any_io(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["dev", *arguments])
    monkeypatch.setattr(cli, "project_root", lambda: pytest.fail("I/O before argument validation"))
    with pytest.raises(SystemExit, match="2"):
        cli.main()


@pytest.mark.parametrize("custom_root", [False, True])
def test_explicit_remote_ids_skip_listing_and_keep_order_for_full_validation(
    tmp_path, monkeypatch, custom_root
):
    identifiers = ["b" * 32, "a" * 32]
    paths = [tmp_path / identifier for identifier in identifiers]
    result_root = tmp_path / ("reviewer-artifacts" if custom_root else "Global/artifacts")
    arguments = ["dev", "compare", "--remote", "--run-ids", *identifiers, "--mode", "inputs"]
    if custom_root:
        arguments += ["--root", str(result_root)]
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        arguments,
    )
    monkeypatch.setattr(tracking, "list_remote_runs", lambda **_kwargs: pytest.fail("listed"))
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("prompted"))
    calls = []

    def fetch(root, *, run_ids, issues):
        assert root == result_root
        assert run_ids == identifiers
        calls.append("verified download")
        return paths

    def compare(runs, output, mode, reference):
        assert reference is None
        assert runs == paths and mode == "inputs"
        assert calls == ["verified download"]
        calls.append("full comparison")
        return runs

    monkeypatch.setattr(tracking, "fetch_runs", fetch)
    monkeypatch.setattr("sixletters.run_catalog.compare_with_reference", compare)
    monkeypatch.setattr(cli, "show_comparison_deltas", lambda _: None)
    assert cli.main() == 0
    assert calls == ["verified download", "full comparison"]


def test_explicit_local_paths_still_skip_catalog_filters_and_keep_order(tmp_path, monkeypatch):
    paths = [tmp_path / "second", tmp_path / "first"]
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["dev", "compare", "--runs", *map(str, paths)])
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
