"""Retry lifecycle and public commands preserve local-only and shared workflows."""

import json
import sys
from pathlib import Path

import pytest

from scripts import tracking_worker
from sixletters import __main__ as cli
from sixletters import tracking


def test_worker_survives_connection_failure_and_stops_after_success(monkeypatch, capsys):
    calls = []

    class Stop:
        count = 0

        def is_set(self):
            return self.count == 2

        def wait(self, interval):
            assert interval == 60
            self.count += 1

    def sync(root, *, retry_only):
        assert retry_only is True
        calls.append(root)
        if len(calls) == 1:
            raise ConnectionError("password=SECRET")
        return {"uploaded": 1, "pending": 0, "invalid": 0}

    monkeypatch.setattr(tracking_worker, "sync_runs", sync)
    tracking_worker.run_worker(Path("artifacts"), 60, Stop())
    output = capsys.readouterr().out
    assert "SECRET" not in output
    assert "'uploaded': 1" in output
    assert len(calls) == 2


@pytest.mark.parametrize("interval", ["0", "nan", "bad", "3601"])
def test_worker_rejects_invalid_retry_interval(monkeypatch, interval):
    monkeypatch.setenv("POC_TRACKING_SYNC_INTERVAL", interval)
    monkeypatch.setattr(tracking_worker, "run_worker", lambda *args: pytest.fail("started"))
    assert tracking_worker.main() == 2


@pytest.mark.parametrize(("pending", "expected"), [(0, 0), (1, 1)])
def test_sync_cli_reports_pending_status(monkeypatch, tmp_path, pending, expected):
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["poc", "sync"])

    def sync(root):
        assert root == tmp_path / "Global/artifacts"
        return {"uploaded": 0, "already_synced": 0, "pending": pending, "invalid": 0}

    monkeypatch.setattr(tracking, "sync_runs", sync)
    assert cli.main() == expected


@pytest.fixture
def remote_catalog():
    return [
        {
            "run_id": f"remote-{index}",
            "local_run_id": f"local-{index}",
            "owner": f"member-{index}",
            "experiment": "exp001",
            "model": "own-model",
            "feature_set": "clinical",
            "dataset_version": "demo-v1",
            "synthetic": True,
        }
        for index in range(6)
    ]


def test_remote_compare_downloads_only_selected_runs_and_uses_shared_comparator(
    monkeypatch, tmp_path, remote_catalog, capsys
):
    remote_catalog[1]["synthetic"] = None
    remote = [tmp_path / ".remote" / str(index) for index in range(6)]
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["poc", "compare", "--remote", "--mode", "inputs"])
    monkeypatch.setattr(tracking, "list_remote_runs", lambda **_kwargs: remote_catalog)
    events = []

    def select(prompt):
        events.append("selected")
        return "2 3 4"

    def fetch(root, *, run_ids, issues):
        assert events == ["selected"]
        assert root == tmp_path / "Global/artifacts"
        assert run_ids == ["remote-1", "remote-2", "remote-3"]
        events.append("downloaded")
        return remote[1:4]

    def compare(runs, output, mode, reference):
        assert reference is None
        assert runs == remote[1:4]
        assert mode == "inputs"
        assert output.parent == tmp_path / "Global/artifacts/comparisons"
        assert events == ["selected", "downloaded"]
        return runs

    monkeypatch.setattr("builtins.input", select)
    monkeypatch.setattr(tracking, "fetch_runs", fetch)
    monkeypatch.setattr("sixletters.run_catalog.compare_with_reference", compare)
    monkeypatch.setattr(cli, "show_comparison_deltas", lambda _: None)
    assert cli.main() == 0
    assert "data type unknown" in capsys.readouterr().out


@pytest.mark.parametrize("selection", ["1", "1 1", "0 2", "7 2", "invalid", ""])
def test_remote_compare_rejects_invalid_selection_before_downloading(
    monkeypatch, tmp_path, remote_catalog, selection
):
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["poc", "compare", "--remote"])
    monkeypatch.setattr(tracking, "list_remote_runs", lambda **_kwargs: remote_catalog)
    monkeypatch.setattr("builtins.input", lambda _: selection)
    monkeypatch.setattr(tracking, "fetch_runs", lambda *args, **kwargs: pytest.fail("downloaded"))
    with pytest.raises(SystemExit, match="2"):
        cli.main()


def test_remote_cli_rejects_local_run_arguments(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["poc", "compare", "--remote", "--runs", "one", "two"])
    with pytest.raises(SystemExit, match="2"):
        cli.main()


def test_local_selection_does_not_include_remote_cache_or_nested_model_files(tmp_path, monkeypatch):
    run_dirs = [tmp_path / "member" / "exp001" / str(index) for index in range(2)]
    cached = tmp_path / ".remote" / "server" / "copy"
    nested = run_dirs[0] / "models" / "checkpoint"
    for directory in [*run_dirs, cached, nested]:
        directory.mkdir(parents=True)
        (directory / "predictions.csv").touch()
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "owner": "member",
                    "experiment": "one",
                    "run_id": directory.name,
                    "created_at": "2026-10-17T00:00:00Z",
                    "feature_set": "clinical",
                    "training": {"model": "independent"},
                    "synthetic": True,
                    "comparison_context": {"dataset": {"dataset_id": "demo-v1"}},
                }
            )
        )
    monkeypatch.setattr("builtins.input", lambda _: "1 2")
    assert cli.choose_runs(tmp_path) == run_dirs


@pytest.mark.parametrize(
    ("outcome", "exit_code"),
    [("uploaded", 0), ("already_synced", 0), ("pending", 1), ("disabled", 1)],
)
def test_archive_cli_uses_explicit_run_and_does_not_report_pending_as_success(
    monkeypatch, tmp_path, capsys, outcome, exit_code
):
    selected = tmp_path / "팀원 가" / "exp001" / "run"
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["poc", "archive", "--run", str(selected)])
    received = []

    def archive(run):
        received.append(run)
        return outcome

    monkeypatch.setattr(tracking, "archive_run", archive)
    assert cli.main() == exit_code
    assert received == [selected]
    if outcome == "pending":
        assert "poc sync" in capsys.readouterr().out


def test_archive_interactively_selects_a_single_local_run(monkeypatch, tmp_path):
    run = tmp_path / "Global/artifacts" / "팀원" / "exp001" / "run"
    run.mkdir(parents=True)
    (run / "predictions.csv").touch()
    (run / "result.json").write_text(
        json.dumps(
            {
                "owner": "팀원",
                "experiment": "exp001",
                "run_id": "run",
                "feature_set": "clinical",
                "training": {"model": "own-model"},
                "synthetic": True,
                "comparison_context": {"dataset": {"dataset_id": "demo"}},
            }
        )
    )
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["poc", "archive"])
    monkeypatch.setattr("builtins.input", lambda _: "1")
    received = []
    monkeypatch.setattr(tracking, "archive_run", lambda path: received.append(path) or "uploaded")
    assert cli.main() == 0
    assert received == [run]
    monkeypatch.setattr("builtins.input", lambda _: "1 1")
    with pytest.raises(SystemExit, match="2"):
        cli.main()
    assert received == [run]
    monkeypatch.setattr("builtins.input", lambda _: "")
    with pytest.raises(SystemExit, match="2"):
        cli.main()
    assert received == [run]


def test_archive_rejects_ambiguous_location(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["poc", "archive", "--run", "one", "--root", "two"])
    with pytest.raises(SystemExit, match="2"):
        cli.main()


def test_worker_logs_changed_failure_identity_and_cause_without_repeating(monkeypatch, capsys):
    def summary(run_id, code):
        return {
            "uploaded": 0,
            "pending": 1,
            "invalid": 0,
            "issues": [
                {
                    "owner": "one",
                    "experiment": "exp001",
                    "run_id": run_id,
                    "status": "pending",
                    "code": code,
                    "message": "안내",
                    "action": "조치",
                }
            ],
        }

    summaries = [
        summary("a" * 32, "network"),
        summary("a" * 32, "network"),
        summary("b" * 32, "network"),
        summary("b" * 32, "auth"),
        {"uploaded": 0, "pending": 0, "invalid": 0, "issues": []},
        {"uploaded": 0, "pending": 0, "invalid": 0, "issues": []},
    ]

    class Stop:
        count = 0

        def is_set(self):
            return self.count == len(summaries)

        def wait(self, interval):
            self.count += 1

    stop = Stop()
    monkeypatch.setattr(
        tracking_worker, "sync_runs", lambda *_args, **_kwargs: summaries[stop.count]
    )
    tracking_worker.run_worker(Path("artifacts"), 60, stop)
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 4
    assert "a" * 32 in output[0]
    assert "b" * 32 in output[1]
    assert "auth" in output[2]
    assert "'issues': []" in output[3]


@pytest.fixture
def duplicate_issue():
    return {
        "scope": "remote",
        "status": "excluded",
        "code": "duplicate_remote",
        "owner": "member",
        "experiment": "exp001",
        "run_id": "a" * 32,
        "remote_run_ids": ["b" * 32, "c" * 32],
        "message": "같은 로컬 실행이 서버에 여러 개 있습니다.",
        "action": "서버 관리자에게 중복 실행 ID를 확인받으세요.",
    }


def test_remote_compare_discloses_duplicates_once_before_selecting_healthy_runs(
    monkeypatch, tmp_path, remote_catalog, duplicate_issue, capsys
):
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["poc", "compare", "--remote"])

    def catalog(*, issues):
        issues.append(duplicate_issue)
        return remote_catalog

    def select(_):
        output = capsys.readouterr().out
        assert "비교 제외" in output and "duplicate_remote" in output
        assert "b" * 32 in output and "c" * 32 in output
        return "1 2"

    paths = [tmp_path / "healthy-one", tmp_path / "healthy-two"]

    def fetch(root, *, run_ids, issues):
        assert run_ids == ["remote-0", "remote-1"]
        issues.append(dict(duplicate_issue))
        return paths

    monkeypatch.setattr(tracking, "list_remote_runs", catalog)
    monkeypatch.setattr(tracking, "fetch_runs", fetch)
    monkeypatch.setattr("builtins.input", select)
    monkeypatch.setattr(
        "sixletters.run_catalog.compare_with_reference", lambda runs, *_args, **_kwargs: runs
    )
    monkeypatch.setattr(cli, "show_comparison_deltas", lambda _: None)
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "비교 제외" not in output
    assert "Compared 2 validation runs" in output


@pytest.mark.parametrize("explicit", [False, True])
def test_new_duplicate_during_fetch_is_reported_even_when_compare_cannot_continue(
    monkeypatch, tmp_path, remote_catalog, duplicate_issue, capsys, explicit
):
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    arguments = ["poc", "compare", "--remote"]
    if explicit:
        arguments += ["--run-ids", "b" * 32, "d" * 32]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(tracking, "list_remote_runs", lambda **_kwargs: remote_catalog)
    monkeypatch.setattr("builtins.input", lambda _: "1 2")

    def fetch(root, *, run_ids, issues):
        issues.append(duplicate_issue)
        raise tracking.BundleError("Selected run has duplicate copies")

    monkeypatch.setattr(tracking, "fetch_runs", fetch)
    monkeypatch.setattr(
        "sixletters.run_catalog.compare_with_reference",
        lambda *_args, **_kwargs: pytest.fail("compared"),
    )
    with pytest.raises(SystemExit, match="2"):
        cli.main()
    output = capsys.readouterr().out
    assert "비교 제외" in output
    assert "duplicate_remote" in output and "b" * 32 in output


def test_sync_cli_reports_unreadable_directory_and_nonzero_exit(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["poc", "sync"])
    issue = {
        "scope": "scan",
        "path": "member/exp001",
        "owner": "member",
        "experiment": "exp001",
        "run_id": "확인 불가",
        "status": "pending",
        "code": "local_permission",
        "message": "로컬 결과 폴더에 접근할 수 없습니다.",
        "action": "폴더의 읽기·탐색 권한을 확인하세요.",
    }
    monkeypatch.setattr(
        tracking, "sync_runs", lambda _: {"uploaded": 1, "pending": 1, "issues": [issue]}
    )
    assert cli.main() == 1
    output = capsys.readouterr().out
    assert "전송 완료: 1" in output
    assert "탐색 실패: member/exp001 [local_permission]" in output
    assert "모두 확인하지 못했습니다" in output


def test_worker_announces_changed_scan_path_and_recovery(monkeypatch, capsys):
    def summary(path):
        return {
            "pending": 1,
            "issues": [
                {
                    "scope": "scan",
                    "path": path,
                    "owner": "unknown",
                    "experiment": "unknown",
                    "run_id": "unknown",
                    "status": "pending",
                    "code": "local_permission",
                }
            ],
        }

    summaries = [summary("one"), summary("one"), summary("two"), {"issues": []}]

    class Stop:
        count = 0

        def is_set(self):
            return self.count == len(summaries)

        def wait(self, interval):
            self.count += 1

    stop = Stop()
    monkeypatch.setattr(
        tracking_worker, "sync_runs", lambda *_args, **_kwargs: summaries[stop.count]
    )
    tracking_worker.run_worker(Path("artifacts"), 60, stop)
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 3
    assert "'path': 'one'" in output[0]
    assert "'path': 'two'" in output[1]
    assert "'issues': []" in output[2]
