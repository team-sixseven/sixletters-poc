import json
from pathlib import Path

import pytest
from docker_completion import (
    assert_remote_completion,
    published_runs_from_response,
    wait_for_local_completion,
)

from sixletters.tracking import _destination_key

CONFIGURATION = ("http://tracking:5000", "recovery-test")
LOCAL_ID = "a" * 32
REMOTE_ID = "b" * 32


@pytest.fixture
def saved_state(tmp_path):
    directory = tmp_path / "Global/artifacts/alice/exp001" / LOCAL_ID
    directory.mkdir(parents=True)
    path = (
        directory.parents[2] / ".tracking" / _destination_key(*CONFIGURATION) / f"{LOCAL_ID}.json"
    )
    path.parent.mkdir(parents=True)
    state = {
        "schema_version": 2,
        "local_identity": ["alice", "exp001", LOCAL_ID],
        "remote_run_id": REMOTE_ID,
        "status": "synced",
        "archive_requested": True,
        "archive_status": "complete",
    }
    path.write_text(json.dumps(state), encoding="utf-8")
    return {LOCAL_ID: {"directory": directory}}, path, state


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = 0.0
        sleeps = []
        on_sleep = None

        def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now += seconds
            if self.on_sleep:
                self.on_sleep()

    clock = Clock()
    monkeypatch.setattr("docker_completion.time.monotonic", lambda: clock.now)
    monkeypatch.setattr("docker_completion.time.sleep", clock.sleep)
    return clock


def remote_run():
    return {
        "info": {"run_id": REMOTE_ID, "experiment_id": "17", "status": "FINISHED"},
        "data": {
            "tags": [
                {"key": "sixletters.local_run_id", "value": LOCAL_ID},
                {"key": "sixletters.complete", "value": "true"},
                {"key": "sixletters.archive_status", "value": "complete"},
            ]
        },
    }


def test_completion_wait_observes_atomic_local_update_without_network(saved_state, clock):
    runs, path, state = saved_state
    path.unlink()

    def publish():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state), encoding="utf-8")
        temporary.replace(path)

    clock.on_sleep = publish
    assert wait_for_local_completion(runs, CONFIGURATION) == {LOCAL_ID: REMOTE_ID}
    assert clock.sleeps == [0.25]


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 1},
        {"local_identity": ["bob", "exp001", LOCAL_ID]},
        {"local_identity": ["alice", "other-experiment", LOCAL_ID]},
        {"local_identity": ["alice", "exp001", "c" * 32]},
        {"remote_run_id": "not-a-run-id"},
        {"remote_run_id": None},
        {"status": "pending"},
    ],
)
def test_wrong_or_incomplete_local_state_never_passes(saved_state, clock, change):
    runs, path, state = saved_state
    state.update(change)
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(AssertionError, match="Background retry did not finish"):
        wait_for_local_completion(runs, CONFIGURATION)
    assert clock.now == 180


@pytest.mark.parametrize("bad_contents", ['{"password": "secret"', "[]", "null"])
def test_invalid_state_times_out_without_printing_contents(saved_state, clock, bad_contents):
    runs, path, _ = saved_state
    path.write_text(bad_contents, encoding="utf-8")
    with pytest.raises(AssertionError) as error:
        wait_for_local_completion(runs, CONFIGURATION, timeout=1)
    assert "secret" not in str(error.value)
    assert str(path) not in str(error.value)


def test_matching_run_at_another_destination_does_not_pass(saved_state, clock):
    runs, path, _ = saved_state
    other = path.parent.parent / _destination_key("http://other-server:5000", CONFIGURATION[1])
    other.mkdir()
    path.replace(other / path.name)
    with pytest.raises(AssertionError, match="unavailable=1"):
        wait_for_local_completion(runs, CONFIGURATION, timeout=1)


@pytest.mark.parametrize(
    "change",
    [
        {"archive_requested": False},
        {"archive_status": "pending"},
        {"remote_run_id": "c" * 32},
    ],
)
def test_archive_wait_requires_selection_completion_and_original_remote_id(
    saved_state, clock, change
):
    runs, path, state = saved_state
    state.update(change)
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(AssertionError, match="Background retry did not finish"):
        wait_for_local_completion(
            runs,
            CONFIGURATION,
            archive=True,
            expected_remote_ids={LOCAL_ID: REMOTE_ID},
            timeout=1,
        )


def test_local_completion_still_requires_independent_remote_result(saved_state):
    runs, _, _ = saved_state
    expected = wait_for_local_completion(runs, CONFIGURATION)
    with pytest.raises(AssertionError, match="Server did not confirm"):
        published_runs_from_response({"runs": []}, expected, "17")
    remote = remote_run()
    remote["info"]["status"] = "RUNNING"
    with pytest.raises(AssertionError, match="not finished"):
        published_runs_from_response({"runs": [remote]}, expected, "17")
    remote["info"]["status"] = "FINISHED"
    remote["data"]["tags"][1]["value"] = "false"
    with pytest.raises(AssertionError, match="incomplete"):
        published_runs_from_response({"runs": [remote]}, expected, "17")


@pytest.mark.parametrize("field,value", [("run_id", "c" * 32), ("experiment_id", "18")])
def test_remote_completion_rejects_other_destination_or_run(field, value):
    remote = remote_run()
    remote["info"][field] = value
    with pytest.raises(AssertionError, match="differs"):
        published_runs_from_response({"runs": [remote]}, {LOCAL_ID: REMOTE_ID}, "17")


def test_remote_completion_rejects_duplicate_or_wrong_local_identity():
    remote = remote_run()
    with pytest.raises(AssertionError, match="Duplicate remote"):
        published_runs_from_response({"runs": [remote, remote]}, {LOCAL_ID: REMOTE_ID}, "17")
    remote["data"]["tags"][0]["value"] = "c" * 32
    with pytest.raises(AssertionError, match="Server did not confirm"):
        published_runs_from_response({"runs": [remote]}, {LOCAL_ID: REMOTE_ID}, "17")


def test_completed_local_archive_cannot_hide_incomplete_server_archive(saved_state):
    runs, _, _ = saved_state
    expected = wait_for_local_completion(
        runs, CONFIGURATION, archive=True, expected_remote_ids={LOCAL_ID: REMOTE_ID}
    )
    remote = remote_run()
    remote["data"]["tags"][2]["value"] = "pending"
    with pytest.raises(AssertionError, match="archive state differs"):
        assert_remote_completion(
            remote,
            local_id=LOCAL_ID,
            remote_id=expected[LOCAL_ID],
            experiment_id="17",
            archive_status="complete",
        )


def test_all_local_runs_must_finish_without_shortening_deadline(saved_state, clock):
    runs, _, _ = saved_state
    other = "c" * 32
    runs[other] = {"directory": Path(runs[LOCAL_ID]["directory"]).with_name(other)}
    with pytest.raises(AssertionError, match="1/2 local states ready"):
        wait_for_local_completion(runs, CONFIGURATION)
    assert clock.now == 180
