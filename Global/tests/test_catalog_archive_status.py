"""Selection metadata exposes archive readiness without fetching models or changing results."""

import json

import pytest
from test_run_catalog import entry
from test_tracking import setup_tracking  # noqa: F401

from sixletters import tracking
from sixletters.run_catalog import filter_run_catalog, show_run_catalog


def test_experiment_and_archive_filters_apply_before_recent_limit():
    entries = [
        entry(1, experiment="first", archive_status="complete"),
        entry(2, experiment="second", archive_status="complete"),
        entry(3, experiment="first", archive_status="none"),
        entry(4, experiment="first", archive_status="complete"),
    ]
    selected, count = filter_run_catalog(entries, experiment="first", archived_only=True, limit=1)
    assert count == 2
    assert len(selected) == 1
    assert selected[0]["experiment"] == "first"
    assert selected[0]["archive_status"] == "complete"
    assert len(entries) == 4
    assert filter_run_catalog(entries, experiment="FIRST")[1] == 0


def test_missing_and_legacy_archive_status_never_claim_readiness(capsys):
    entries = [entry(), entry(2, archive_status="legacy"), entry(3, archive_status="pending")]
    assert filter_run_catalog(entries, archived_only=True) == ([], 0)
    show_run_catalog(entries)
    output = capsys.readouterr().out
    assert "모델 보관: 확인되지 않음" in output
    assert "이전 형식 전체 파일" in output
    assert "모델 보관: 전송 대기" in output


def test_remote_and_local_archive_states_follow_selected_upload(setup_tracking, monkeypatch):  # noqa: F811
    root, client, make_run, enable = setup_tracking
    run = make_run()
    assert tracking.local_archive_status(run) == "unknown"
    enable()
    tracking.sync_runs(root)
    assert tracking.list_remote_runs()[0]["archive_status"] == "none"
    assert tracking.local_archive_status(run) == "none"
    tracking.archive_run(run)
    assert tracking.list_remote_runs()[0]["archive_status"] == "complete"
    assert tracking.local_archive_status(run) == "complete"
    before = {p: p.read_bytes() for p in (root / ".tracking").rglob("*") if p.is_file()}
    monkeypatch.setattr(tracking, "_client", lambda *_: pytest.fail("Local state must not connect"))
    assert tracking.local_archive_status(run) == "complete"
    assert before == {p: p.read_bytes() for p in (root / ".tracking").rglob("*") if p.is_file()}
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "another-destination")
    assert tracking.local_archive_status(run) == "unknown"
    assert client.runs


def test_local_archive_state_does_not_expose_invalid_configuration_or_mixed_identity(
    setup_tracking,  # noqa: F811
    monkeypatch,
):
    root, _, make_run, enable = setup_tracking
    run = make_run()
    enable()
    tracking.sync_runs(root)
    state_path = next((root / ".tracking").glob(f"*/{run.name}.json"))
    state = json.loads(state_path.read_text())
    state.update(archive_status="complete", local_identity=["another", "experiment", run.name])
    state_path.write_text(json.dumps(state))
    assert tracking.local_archive_status(run) == "unknown"
    state_path.write_text("[]")
    assert tracking.local_archive_status(run) == "unknown"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://secret:password@example.invalid")
    assert tracking.local_archive_status(run) == "unknown"
