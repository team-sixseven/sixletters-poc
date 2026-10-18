"""Observe local retry markers cheaply, then require independent server verification."""

import json
import re
import time
from collections import Counter
from pathlib import Path

from sixletters.tracking import _destination_key

_REMOTE_ID = re.compile(r"[0-9a-f]{32}")


def wait_for_local_completion(
    local_runs,
    configuration,
    *,
    archive=False,
    expected_remote_ids=None,
    timeout=180,
    poll_interval=0.25,
):
    """Wait for atomically written queue state, not repeated authenticated HTTP calls.

    This is only a completion signal. Callers must still verify the actual remote
    runs and artifacts; a local marker alone never proves a successful upload.
    """
    assert local_runs, "At least one saved run is required"
    assert timeout > 0 and poll_interval > 0
    if expected_remote_ids is not None:
        assert set(expected_remote_ids) == set(local_runs)
    destination = _destination_key(*configuration)
    paths = {}
    for local_id, saved in local_runs.items():
        directory = Path(saved["directory"])
        assert directory.name == local_id, "Saved run identity does not match its directory"
        paths[local_id] = (
            directory.parents[2] / ".tracking" / destination / f"{local_id}.json",
            [directory.parent.parent.name, directory.parent.name, local_id],
        )
    deadline = time.monotonic() + timeout
    while True:
        ready, pending = {}, Counter()
        for local_id, (path, identity) in paths.items():
            # The publisher uses atomic replacement. Do not hold its file lock
            # while waiting: the worker must remain free to publish the result.
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError):
                pending["unavailable"] += 1
                continue
            if not isinstance(state, dict) or state.get("schema_version") != 2:
                pending["schema"] += 1
            elif state.get("local_identity") != identity:
                pending["identity"] += 1
            elif not isinstance(state.get("remote_run_id"), str) or not _REMOTE_ID.fullmatch(
                state["remote_run_id"]
            ):
                pending["remote identity"] += 1
            elif (
                expected_remote_ids is not None
                and state["remote_run_id"] != expected_remote_ids[local_id]
            ):
                pending["remote identity"] += 1
            elif state.get("status") != "synced":
                pending["result pending"] += 1
            elif archive and (
                state.get("archive_requested") is not True
                or state.get("archive_status") != "complete"
            ):
                pending["archive pending"] += 1
            else:
                ready[local_id] = state["remote_run_id"]
        if len(ready) == len(paths):
            assert len(set(ready.values())) == len(ready), "Remote run identities are duplicated"
            return ready
        remaining = deadline - time.monotonic()
        assert remaining > 0, (
            f"Background retry did not finish: {len(ready)}/{len(paths)} local states ready; "
            + ", ".join(f"{reason}={count}" for reason, count in sorted(pending.items()))
        )
        time.sleep(min(poll_interval, remaining))


def assert_remote_completion(remote, *, local_id, remote_id, experiment_id, archive_status=None):
    """A local marker must agree with the independently retrieved server state."""
    info = remote["info"]
    tags = {tag["key"]: tag["value"] for tag in remote["data"].get("tags", [])}
    assert info["run_id"] == remote_id, "Remote run identity differs from the local state"
    assert info["experiment_id"] == experiment_id, "Remote experiment differs"
    assert tags.get("sixletters.local_run_id") == local_id, "Remote local identity differs"
    assert info["status"] == "FINISHED", "Remote run is not finished"
    assert tags.get("sixletters.complete") == "true", "Remote result upload is incomplete"
    if archive_status is not None:
        assert tags.get("sixletters.archive_status") == archive_status, (
            "Remote model archive state differs"
        )


def published_runs_from_response(response, expected_remote_ids, experiment_id):
    """Require exactly one fully published server run for every observed local run."""
    published = {}
    for remote in response.get("runs", []):
        tags = {tag["key"]: tag["value"] for tag in remote["data"].get("tags", [])}
        local_id = tags.get("sixletters.local_run_id")
        if local_id not in expected_remote_ids:
            continue
        assert local_id not in published, "Duplicate remote runs identify the same local result"
        assert_remote_completion(
            remote,
            local_id=local_id,
            remote_id=expected_remote_ids[local_id],
            experiment_id=experiment_id,
        )
        published[local_id] = remote
    assert set(published) == set(expected_remote_ids), "Server did not confirm all local results"
    return published
