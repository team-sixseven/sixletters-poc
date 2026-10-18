import builtins
import copy
import json
import re
import shutil
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit
from uuid import uuid4

import pandas as pd
import pytest

from sixletters import tracking
from sixletters.comparison import compare_runs
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment
from sixletters.workspace import init_experiment


class FakeClient:
    """The server stores independent copies, including writes with lost responses."""

    def __init__(self, root):
        self.root = root
        self.experiments = {}
        self.runs = {}
        self.metrics = {}
        self.parameters = {}
        self.batches = []
        self.batch_tags = []
        self.tag_calls = []
        self.fail_create_response = False
        self.fail_upload = False
        self.fail_finish_response = False
        self.fail_batch_response = False
        self.fail_archive_tag_response = False
        self.uploads = []
        self.offline = False
        self.downloads = []
        self.create_count = 0

    def bind(self, monkeypatch):
        """Route both MLflow API boundaries to this fake server's storage."""
        from mlflow import artifacts

        monkeypatch.setattr(tracking, "_client", lambda uri: self)

        def download(*, artifact_uri, dst_path, tracking_uri):
            assert tracking_uri == tracking._configuration()[0]
            parts = unquote(urlsplit(artifact_uri).path).lstrip("/").split("/")
            assert parts[2] == "artifacts"
            return self.download_artifacts(parts[1], "/".join(parts[3:]), dst_path)

        monkeypatch.setattr(artifacts, "download_artifacts", download)

    def get_experiment_by_name(self, name):
        if self.offline:
            raise ConnectionError("Authorization: password=SECRET http://secret-server")
        identifier = self.experiments.get(name)
        return SimpleNamespace(experiment_id=identifier) if identifier else None

    def create_experiment(self, name):
        identifier = str(len(self.experiments) + 1)
        self.experiments[name] = identifier
        return identifier

    def create_run(self, experiment_id, tags):
        identifier = uuid4().hex
        run = SimpleNamespace(
            info=SimpleNamespace(
                run_id=identifier,
                status="RUNNING",
                experiment_id=experiment_id,
                lifecycle_stage="active",
                artifact_uri=f"mlflow-artifacts:/{experiment_id}/{identifier}/artifacts",
            ),
            data=SimpleNamespace(tags=dict(tags), metrics={}, params={}),
        )
        self.runs[identifier] = run
        self.create_count += 1
        if self.fail_create_response:
            self.fail_create_response = False
            raise ConnectionError("create response lost SECRET")
        return copy.deepcopy(run)

    def get_run(self, run_id):
        return copy.deepcopy(self.runs[run_id])

    def search_runs(self, experiment_ids, filter_string, max_results, page_token, run_view_type=1):
        found = [run for run in self.runs.values() if run.info.experiment_id in experiment_ids]
        if run_view_type != 3:
            stage = "active" if run_view_type == 1 else "deleted"
            found = [run for run in found if run.info.lifecycle_stage == stage]
        for tag, value in re.findall(r"tags\.`([^`]+)` = '([^']+)'", filter_string):
            found = [run for run in found if run.data.tags.get(tag) == value]
        if "attributes.status = 'FINISHED'" in filter_string:
            found = [run for run in found if run.info.status == "FINISHED"]
        return copy.deepcopy(found)

    def log_batch(self, run_id, *, metrics, params, tags, synchronous):
        assert synchronous is True
        self.batches.append((run_id, metrics, params))
        self.batch_tags.append(tags)
        for metric in metrics:
            assert metric.timestamp > 0
            assert metric.step == 0
            self.metrics.setdefault(run_id, {})[metric.key] = metric.value
            self.runs[run_id].data.metrics[metric.key] = metric.value
        for parameter in params:
            previous = self.parameters.setdefault(run_id, {}).setdefault(
                parameter.key, parameter.value
            )
            assert previous == parameter.value
            self.runs[run_id].data.params[parameter.key] = parameter.value
        for tag in tags:
            assert len(tag.value) <= 8000
            self.runs[run_id].data.tags[tag.key] = tag.value
        if self.fail_batch_response:
            self.fail_batch_response = False
            raise ConnectionError("batch response lost SECRET")

    def log_artifact(self, run_id, local_path, artifact_path):
        if self.fail_upload:
            self.fail_upload = False
            raise ConnectionError("upload interrupted SECRET")
        self.uploads.append((run_id, artifact_path, Path(local_path).name))
        destination = self.root / run_id / artifact_path / Path(local_path).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, destination)

    def set_tag(self, run_id, key, value):
        self.tag_calls.append((key, value))
        self.runs[run_id].data.tags[key] = value
        if (
            self.fail_archive_tag_response
            and key == tracking.PREFIX + "archive_status"
            and value == "complete"
        ):
            self.fail_archive_tag_response = False
            raise ConnectionError("archive response lost SECRET")

    def set_terminated(self, run_id, status):
        self.runs[run_id].info.status = status
        if self.fail_finish_response:
            self.fail_finish_response = False
            raise ConnectionError("finish response lost SECRET")

    def download_artifacts(self, run_id, path, dst_path):
        self.downloads.append(path)
        destination = Path(dst_path) / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / run_id / path, destination)
        return str(destination)


@pytest.fixture
def setup_tracking(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    data = save_prepared_dataset(make_demo_dataset(), tmp_path / "data", synthetic=True)
    root = tmp_path / "artifacts"
    model = tmp_path / "weights.txt"
    model.write_text("opaque model, never deserialize")
    client = FakeClient(tmp_path / "server")
    client.bind(monkeypatch)

    def make_run(owner="팀원가", feature_set="clinical"):
        config = init_experiment(
            owner, "exp001", root=tmp_path / "experiments", feature_set=feature_set
        )
        experiment = Experiment(data, config, output_root=root)
        prediction = pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5})
        return experiment.save(
            prediction,
            model_files={"weights.txt": model},
            training={"model": "ConstantScore", "seed": None, "parameters": {"score": 0.5}},
        )

    def enable():
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example:5000")

    return root, client, make_run, enable


def test_unconfigured_tracking_does_not_connect_or_change_saved_bundle(setup_tracking):
    root, client, make_run, _ = setup_tracking
    run = make_run()
    before = {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    assert tracking.auto_publish(run) == "disabled"
    assert tracking.sync_runs(root)["disabled"]
    assert not client.runs
    assert not (root / ".tracking").exists()
    assert {
        path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()
    } == before


@pytest.mark.parametrize(
    "uri", ["http://tracking.example:5000", "https://member:SECRET@invalid.example"]
)
def test_save_does_not_import_tracking_or_wait_for_its_configuration(
    setup_tracking, monkeypatch, uri
):
    root, client, make_run, enable = setup_tracking
    original_import = builtins.__import__

    def reject_tracking(name, *args, **kwargs):
        if name in {"tracking", "sixletters.tracking"} or name.startswith("mlflow"):
            pytest.fail("Local save must not load the upload client")
        return original_import(name, *args, **kwargs)

    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    with monkeypatch.context() as local:
        local.setattr(builtins, "__import__", reject_tracking)
        run = make_run()
    assert (run / "result.json").is_file()
    assert not client.runs
    assert not (root / ".tracking").exists()
    # A later invocation can discover the saved result even after the caller exits.
    enable()
    assert tracking.sync_runs(root, retry_only=True)["uploaded"] == 1
    assert tracking.sync_runs(root, retry_only=True)["already_synced"] == 1
    assert client.create_count == 1


def test_worker_cannot_observe_a_partly_written_completion_marker(setup_tracking, monkeypatch):
    root, client, make_run, enable = setup_tracking
    enable()
    original_replace = Path.replace
    completion = []

    def observe_marker(path, target):
        if path.name == ".result.json.pending":
            assert not Path(target).exists()
            assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == path.parent.name
            assert tracking.sync_runs(root, retry_only=True)["uploaded"] == 0
            assert not client.runs
            result = original_replace(path, target)
            assert tracking.sync_runs(root, retry_only=True)["uploaded"] == 1
            completion.append(target)
            return result
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", observe_marker)
    run = make_run()
    assert completion == [run / "result.json"]
    assert not (run / ".result.json.pending").exists()
    assert tracking.sync_runs(root, retry_only=True)["already_synced"] == 1


def test_worker_publishes_only_comparison_files_after_local_completion(setup_tracking):
    root, client, make_run, enable = setup_tracking
    enable()
    run = make_run()
    assert not client.runs
    assert tracking.sync_runs(root, retry_only=True)["uploaded"] == 1
    remote = next(iter(client.runs.values()))
    assert remote.info.status == "FINISHED"
    assert remote.data.tags[tracking.PREFIX + "complete"] == "true"
    assert remote.data.tags[tracking.PREFIX + "owner"] == "팀원가"
    uploaded = client.root / remote.info.run_id / tracking.REMOTE_FOLDER
    assert {path.name for path in uploaded.iterdir()} == {*tracking.BASIC_FILES, tracking.MANIFEST}
    for name in tracking.BASIC_FILES:
        assert (uploaded / name).read_bytes() == (run / name).read_bytes()
    assert (run / "models/weights.txt").is_file()
    assert (run / "source/train.py").is_file()
    assert remote.data.tags[tracking.PREFIX + "archive_status"] == "none"
    assert (uploaded / tracking.MANIFEST).exists()
    assert not list(uploaded.rglob("dataset.json"))
    assert set(client.metrics[remote.info.run_id]) == {
        "roc_auc",
        "average_precision",
        "brier_score",
    }
    assert client.parameters[remote.info.run_id]["model"] == "ConstantScore"
    assert len(client.batches) == 1
    assert client.batches[0][0] == remote.info.run_id
    assert len({metric.timestamp for metric in client.batches[0][1]}) == 1
    assert {tag.key for tag in client.batch_tags[0]} == {
        tracking.PREFIX + "parameters_complete",
        tracking.PREFIX + "validation_summary_v1",
        tracking.PREFIX + "target_definition",
    }
    assert client.tag_calls == [(tracking.PREFIX + "complete", "true")]
    assert tracking.sync_runs(root)["already_synced"] == 1
    assert client.create_count == 1
    assert len(client.batches) == 1


def test_offline_save_succeeds_without_secrets_and_sync_recovers(setup_tracking, capsys):
    root, client, make_run, enable = setup_tracking
    enable()
    client.offline = True
    with warnings.catch_warnings(record=True) as captured:
        run = make_run()
    assert not captured
    assert not client.runs
    assert (run / "result.json").exists()
    pending = tracking.sync_runs(root)
    assert pending["pending"] == 1
    assert pending["issues"][0]["code"] == "network"
    assert "SECRET" not in str(pending)
    assert "secret-server" not in str(pending)
    client.offline = False
    assert tracking.sync_runs(root)["uploaded"] == 1
    assert client.create_count == 1
    assert "SECRET" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "failure",
    ["fail_create_response", "fail_upload", "fail_finish_response", "fail_batch_response"],
)
def test_uncertain_or_partial_upload_is_idempotently_retried(setup_tracking, failure):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    setattr(client, failure, True)
    with pytest.warns(RuntimeWarning):
        assert tracking.auto_publish(run) == "pending"
    first_id = next(iter(client.runs))
    outcome = tracking.sync_runs(root)
    assert outcome["uploaded"] + outcome["already_synced"] == 1
    assert set(client.runs) == {first_id}
    assert client.create_count == 1
    assert tracking.auto_publish(run) == "already_synced"
    assert len(client.batches) == 1


def test_partial_upload_is_not_visible_for_comparison(setup_tracking):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    client.fail_upload = True
    with pytest.warns(RuntimeWarning):
        tracking.auto_publish(run)
    assert tracking.fetch_runs(root) == []
    assert not client.downloads
    assert tracking.sync_runs(root)["uploaded"] == 1
    assert len(tracking.fetch_runs(root)) == 1


@pytest.mark.parametrize("modified", list(tracking.BASIC_FILES))
def test_changed_bundle_cannot_overwrite_existing_remote_run(setup_tracking, modified):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    assert tracking.auto_publish(run) == "uploaded"
    remote_id = next(iter(client.runs))
    original = (client.root / remote_id / tracking.REMOTE_FOLDER / modified).read_bytes()
    (run / modified).write_bytes(b"modified")
    outcome = tracking.sync_runs(root)
    assert outcome["invalid"] == 1
    assert client.create_count == 1
    assert (client.root / remote_id / tracking.REMOTE_FOLDER / modified).read_bytes() == original


def test_added_raw_data_is_rejected_before_upload(setup_tracking):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    (run / "dataset.json").write_text("private raw data")
    enable()
    assert tracking.sync_runs(root)["invalid"] == 1
    assert not client.runs


def test_six_owners_publish_compare_and_remote_cache_is_never_republished(setup_tracking):
    root, client, make_run, enable = setup_tracking
    local = [make_run(f"팀원{number}") for number in range(1, 7)]
    enable()
    assert tracking.sync_runs(root)["uploaded"] == 6
    paths = tracking.fetch_runs(root)
    assert len(paths) == 6
    for path in paths:
        assert path.is_relative_to(root / ".remote")
        assert {file.name for file in path.iterdir()} == {
            tracking.MANIFEST,
            "result.json",
            "predictions.csv",
        }
    assert set(client.downloads) == {
        "bundle/manifest.json",
        "bundle/result.json",
        "bundle/predictions.csv",
    }
    comparison = compare_runs(paths, root / "comparisons" / "six.csv")
    assert set(comparison.owner) == {f"팀원{number}" for number in range(1, 7)}
    assert (
        len(
            {
                client.runs[key].data.tags[tracking.PREFIX + "comparison_group"]
                for key in client.runs
            }
        )
        == 1
    )
    assert tracking.sync_runs(root)["already_synced"] == 6
    assert client.create_count == len(local)
    # A cache-only machine must never upload received results again.
    for run in local:
        shutil.rmtree(run)
    assert tracking.sync_runs(root)["uploaded"] == 0
    assert client.create_count == 6


@pytest.mark.parametrize("offline", [False, True])
def test_sync_shares_one_connection_lookup_and_retries_next_pass(
    setup_tracking, monkeypatch, offline
):
    root, client, make_run, enable = setup_tracking
    for owner in ("one", "two", "three"):
        make_run(owner)
    enable()
    lookups = []
    original = client.get_experiment_by_name

    def lookup(name):
        lookups.append(name)
        return original(name)

    monkeypatch.setattr(client, "get_experiment_by_name", lookup)
    client.offline = offline
    assert tracking.sync_runs(root)["pending" if offline else "uploaded"] == 3
    assert len(lookups) == 1
    client.offline = False
    assert tracking.sync_runs(root)["uploaded" if offline else "already_synced"] == 3
    assert len(lookups) == 2
    assert client.create_count == 3


@pytest.mark.parametrize("archived", [False, True])
def test_worker_skips_confirmed_runs_but_manual_sync_still_verifies(
    setup_tracking, monkeypatch, archived
):
    from sixletters import comparison

    root, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    if archived:
        assert tracking.archive_run(run) == "uploaded"
    else:
        assert tracking.auto_publish(run) == "uploaded"
    verified = []
    lookups = []
    original_verify = comparison._read_verified_result
    original_lookup = client.get_experiment_by_name

    def verify(path):
        verified.append(path)
        return original_verify(path)

    def lookup(name):
        lookups.append(name)
        return original_lookup(name)

    monkeypatch.setattr(comparison, "_read_verified_result", verify)
    monkeypatch.setattr(client, "get_experiment_by_name", lookup)
    monkeypatch.setattr(tracking.os, "replace", lambda *args: pytest.fail("Unchanged state write"))
    assert tracking.sync_runs(root, retry_only=True)["already_synced"] == 1
    assert verified == lookups == []
    assert tracking.sync_runs(root)["already_synced"] == 1
    assert verified == [run]
    assert len(lookups) == 1


@pytest.mark.parametrize("modified", list(tracking.BASIC_FILES))
def test_worker_rejects_modified_completed_comparison_files(setup_tracking, modified):
    root, client, make_run, enable = setup_tracking
    enable()
    run = make_run()
    assert tracking.sync_runs(root)["uploaded"] == 1
    path = run / modified
    path.write_bytes(path.read_bytes() + b"\n")
    assert tracking.sync_runs(root, retry_only=True)["invalid"] == 1
    assert client.create_count == 1


@pytest.mark.parametrize("state_missing", [False, True])
def test_worker_recovers_preoptimization_or_missing_local_state(setup_tracking, state_missing):
    root, client, make_run, enable = setup_tracking
    enable()
    run = make_run()
    assert tracking.sync_runs(root)["uploaded"] == 1
    state_path = next((root / ".tracking").glob(f"*/{run.name}.json"))
    if state_missing:
        state_path.unlink()
    else:
        state = tracking._json(state_path)
        del state["local_identity"]
        tracking._write_json(state_path, state)
    assert tracking.sync_runs(root, retry_only=True)["already_synced"] == 1
    assert tracking._json(state_path)["local_identity"] == ["팀원가", "exp001", run.name]
    assert client.create_count == 1


@pytest.mark.parametrize("local_state", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_deleted_remote_result_is_reported_without_recreation_and_recovers_after_restore(
    setup_tracking, local_state, legacy
):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    if legacy:
        remote = make_legacy_run(root, client, run, completed=True, local_state=True)
    else:
        assert tracking.auto_publish(run) == "uploaded"
        remote = next(iter(client.runs.values()))
    state_path = next((root / ".tracking").glob(f"*/{run.name}.json"))
    if not local_state:
        state_path.unlink()
    remote.info.lifecycle_stage = "deleted"
    original_uploads = list(client.uploads)

    summary = tracking.sync_runs(root)
    assert summary["already_synced"] == summary["uploaded"] == summary["invalid"] == 0
    assert summary["pending"] == 1
    assert summary["issues"][0]["code"] == "remote_deleted"
    assert summary["issues"][0]["run_id"] == run.name
    assert tracking.list_remote_runs() == []
    assert tracking._json(state_path)["status"] == "pending"
    assert tracking._json(state_path)["remote_run_id"] == remote.info.run_id
    # After detection, the worker cannot skip the deleted run as already complete.
    assert tracking.sync_runs(root, retry_only=True)["issues"][0]["code"] == "remote_deleted"
    with pytest.warns(RuntimeWarning, match="remote_deleted"):
        assert tracking.archive_run(run) == "pending"
    assert remote.info.lifecycle_stage == "deleted"
    assert client.create_count == 1
    assert client.uploads == original_uploads

    remote.info.lifecycle_stage = "active"
    recovered = tracking.sync_runs(root, retry_only=True)
    assert recovered["pending"] == recovered["invalid"] == 0
    assert recovered["uploaded"] + recovered["already_synced"] == 1
    assert len(tracking.list_remote_runs()) == 1
    assert client.create_count == 1
    assert tracking._json(state_path)["status"] == "synced"


def test_worker_does_not_skip_a_new_archive_request_on_a_completed_run(setup_tracking):
    root, client, make_run, enable = setup_tracking
    enable()
    run = make_run()
    assert tracking.sync_runs(root)["uploaded"] == 1
    state_path = next((root / ".tracking").glob(f"*/{run.name}.json"))
    state = tracking._json(state_path)
    tracking._write_json(
        state_path.with_suffix(".archive-request"),
        {"schema_version": 1, "bundle_hash": state["bundle_hash"]},
    )
    assert tracking.sync_runs(root, retry_only=True)["uploaded"] == 1
    remote = next(iter(client.runs.values()))
    assert remote.data.tags[tracking.PREFIX + "archive_status"] == "complete"
    assert tracking.sync_runs(root, retry_only=True)["already_synced"] == 1


def test_worker_does_not_skip_a_run_moved_under_another_owner(setup_tracking):
    root, client, make_run, enable = setup_tracking
    enable()
    run = make_run()
    assert tracking.sync_runs(root)["uploaded"] == 1
    destination = root / "different-owner" / run.parent.name / run.name
    destination.parent.mkdir(parents=True)
    run.rename(destination)
    assert tracking.sync_runs(root, retry_only=True)["invalid"] == 1
    assert client.create_count == 1


def test_remote_selection_downloads_only_chosen_runs_in_selected_order(setup_tracking):
    root, client, make_run, enable = setup_tracking
    enable()
    for owner in ("one", "two", "three"):
        make_run(owner)
    assert tracking.sync_runs(root)["uploaded"] == 3
    listed = tracking.list_remote_runs()
    assert len(listed) == 3
    assert not client.downloads
    assert {item["owner"] for item in listed} == {"one", "two", "three"}
    assert {item["model"] for item in listed} == {"ConstantScore"}
    assert all(item["synthetic"] is True for item in listed)
    assert all(item["feature_set"] == "clinical" for item in listed)
    chosen = [listed[2]["run_id"], listed[0]["run_id"]]
    # Invalid artifacts in an unselected run do not block the chosen comparison.
    (client.root / listed[1]["run_id"] / "bundle/manifest.json").write_text("invalid")
    paths = tracking.fetch_runs(root, run_ids=chosen)
    assert [path.name for path in paths] == chosen
    assert len(client.downloads) == 6
    assert len(compare_runs(paths, root / "selected.csv")) == 2
    assert tracking.fetch_runs(root, run_ids=chosen) == paths
    assert len(client.downloads) == 6


@pytest.mark.parametrize("selection", [[], ["bad"], ["a" * 32, "a" * 32], "a" * 32])
def test_invalid_remote_selection_is_rejected_before_download(setup_tracking, selection):
    root, client, _, enable = setup_tracking
    enable()
    with pytest.raises(tracking.TrackingError, match="distinct"):
        tracking.fetch_runs(root, run_ids=selection)
    assert not client.downloads


@pytest.mark.parametrize("change", ["missing", "unfinished", "other_experiment", "protocol"])
def test_remote_selection_is_rechecked_before_downloading(setup_tracking, change):
    root, client, make_run, enable = setup_tracking
    enable()
    make_run()
    assert tracking.sync_runs(root)["uploaded"] == 1
    remote_id = tracking.list_remote_runs()[0]["run_id"]
    remote = client.runs[remote_id]
    if change == "missing":
        del client.runs[remote_id]
    elif change == "unfinished":
        remote.info.status = "RUNNING"
    elif change == "other_experiment":
        remote.info.experiment_id = client.create_experiment("different")
    else:
        remote.data.tags[tracking.PREFIX + "protocol"] = "unsupported"
    with pytest.raises(tracking.TrackingError, match="no longer available"):
        tracking.fetch_runs(root, run_ids=[remote_id])
    assert not client.downloads


def test_selected_artifacts_are_verified_after_listing(setup_tracking):
    root, client, make_run, enable = setup_tracking
    enable()
    make_run()
    assert tracking.sync_runs(root)["uploaded"] == 1
    remote_id = tracking.list_remote_runs()[0]["run_id"]
    (client.root / remote_id / "bundle/predictions.csv").write_text("modified")
    with pytest.raises(tracking.BundleError, match="hashes"):
        tracking.fetch_runs(root, run_ids=[remote_id])


@pytest.mark.parametrize("modified", ["manifest.json", "result.json", "predictions.csv"])
def test_remote_tampering_prevents_comparison(setup_tracking, modified):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    tracking.auto_publish(run)
    remote_id = next(iter(client.runs))
    path = client.root / remote_id / tracking.REMOTE_FOLDER / modified
    path.write_bytes(path.read_bytes() + b" ")
    # Whitespace changes JSON meaning only minimally, so hashes must protect bytes.
    if modified == "manifest.json":
        manifest = json.loads(path.read_text())
        manifest["files"]["predictions.csv"]["bytes"] += 1
        path.write_text(json.dumps(manifest))
    with pytest.raises(tracking.TrackingError):
        tracking.fetch_runs(root)


def test_modified_cached_predictions_are_rejected(setup_tracking):
    root, _, make_run, enable = setup_tracking
    run = make_run()
    enable()
    tracking.auto_publish(run)
    cache = tracking.fetch_runs(root)[0]
    (cache / "predictions.csv").write_text("modified")
    with pytest.raises(tracking.BundleError):
        tracking.fetch_runs(root)


def test_incompatible_feature_sets_remain_rejected_by_common_comparison(setup_tracking):
    root, _, make_run, enable = setup_tracking
    make_run("팀원가", "clinical")
    make_run("팀원나", "ecg")
    enable()
    assert tracking.sync_runs(root)["uploaded"] == 2
    with pytest.raises(ValueError, match="same feature_set"):
        compare_runs(tracking.fetch_runs(root), root / "comparison.csv")


def test_state_is_scoped_by_server_url(setup_tracking, monkeypatch):
    root, first, make_run, enable = setup_tracking
    run = make_run()
    enable()
    assert tracking.auto_publish(run) == "uploaded"
    first_download = tracking.fetch_runs(root)[0]
    second = FakeClient(root / "other-server")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://second.example")
    second.bind(monkeypatch)
    assert tracking.auto_publish(run) == "uploaded"
    second_download = tracking.fetch_runs(root)[0]
    assert first_download != second_download
    assert len(first.downloads) == len(second.downloads) == 3
    assert (first_download / "result.json").read_bytes() == (
        second_download / "result.json"
    ).read_bytes()
    assert first.create_count == second.create_count == 1
    assert len(list((root / ".tracking").iterdir())) == 2


def test_same_server_different_experiment_has_independent_upload_state(setup_tracking, monkeypatch):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    assert tracking.auto_publish(run) == "uploaded"
    first = next(iter(client.runs.values()))
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "another-project")
    assert tracking.auto_publish(run) == "uploaded"
    second = list(client.runs.values())[-1]
    assert first.info.experiment_id != second.info.experiment_id
    assert client.create_count == 2
    assert tracking.sync_runs(root)["already_synced"] == 1
    assert len(tracking.fetch_runs(root)) == 1
    assert len(list((root / ".tracking").iterdir())) == 2


def test_warning_as_error_cannot_fail_successful_local_save(setup_tracking):
    _, client, make_run, enable = setup_tracking
    enable()
    client.offline = True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run = make_run()
    assert (run / "result.json").exists()


def test_concurrent_save_and_sync_cannot_create_duplicate_remote_runs(setup_tracking):
    _, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    barrier = threading.Barrier(2)

    def publish():
        barrier.wait()
        return tracking.auto_publish(run)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: publish(), range(2)))
    assert sorted(results) == ["already_synced", "uploaded"]
    assert client.create_count == 1


@pytest.mark.parametrize(
    "uri",
    [
        "file:///tmp/server",
        "sqlite:///tmp/server",
        "http://user:SECRET@example.com",
        "https://example.com?token=SECRET",
        "https://example.com#SECRET",
        "https://example.com:invalid",
        "http://",
        "https://exam ple.com",
    ],
)
def test_invalid_uris_fail_without_exposing_credentials(setup_tracking, monkeypatch, uri):
    root, client, make_run, _ = setup_tracking
    run = make_run()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    with pytest.warns(RuntimeWarning) as captured:
        assert tracking.auto_publish(run) == "pending"
    assert "SECRET" not in str(captured[0].message)
    with pytest.raises(tracking.TrackingError) as error:
        tracking.sync_runs(root)
    assert "SECRET" not in str(error.value)
    assert not client.runs


def test_remote_duplicates_are_quarantined_while_healthy_runs_remain_available(setup_tracking):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    healthy = make_run("healthy")
    other = make_run("other")
    enable()
    tracking.auto_publish(run)
    original = next(iter(client.runs.values()))
    duplicate = client.create_run(original.info.experiment_id, original.data.tags)
    client.set_terminated(duplicate.info.run_id, "FINISHED")
    tracking.auto_publish(healthy)
    tracking.auto_publish(other)
    issues = []
    listed = tracking.list_remote_runs(issues=issues)
    assert [item["local_run_id"] for item in listed] == [healthy.name, other.name]
    assert len(issues) == 1
    issue = issues[0]
    assert issue["code"] == "duplicate_remote"
    assert issue["status"] == "excluded"
    assert issue["run_id"] == run.name
    assert issue["remote_run_ids"] == sorted([original.info.run_id, duplicate.info.run_id])
    assert issue["message"] and issue["action"]
    fetched_issues = []
    paths = tracking.fetch_runs(root, issues=fetched_issues)
    assert fetched_issues == issues
    assert [json.loads((path / "result.json").read_text())["run_id"] for path in paths] == [
        healthy.name,
        other.name,
    ]
    assert len(compare_runs(paths, root / "comparison.csv")) == 2
    for remote_id in issue["remote_run_ids"]:
        with pytest.raises(tracking.BundleError, match="excluded.*duplicate"):
            tracking.fetch_runs(root, run_ids=[remote_id], issues=[])
    with pytest.warns(RuntimeWarning, match="duplicate_remote"):
        assert len(tracking.list_remote_runs()) == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(tracking.list_remote_runs()) == 2
    assert client.create_count == 4


def test_same_run_first_uploaded_from_two_roots_does_not_block_other_team_results(
    setup_tracking, monkeypatch, tmp_path
):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    healthy = make_run("healthy")
    other = make_run("other")
    copied = tmp_path / "copied" / run.relative_to(root)
    shutil.copytree(run, copied)
    enable()
    client.create_experiment("sixseven-poc")
    barrier = threading.Barrier(2)
    search = tracking._search

    def simultaneous_search(client, experiment_id, filters, **options):
        found = search(client, experiment_id, filters, **options)
        if run.name in filters:
            assert found == []
            barrier.wait(timeout=10)
        return found

    with monkeypatch.context() as racing:
        racing.setattr(tracking, "_search", simultaneous_search)
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(tracking.auto_publish, [run, copied]))
    assert outcomes == ["uploaded", "uploaded"]
    assert client.create_count == 2
    assert tracking.auto_publish(healthy) == "uploaded"
    assert tracking.auto_publish(other) == "uploaded"
    issues = []
    assert [item["local_run_id"] for item in tracking.list_remote_runs(issues=issues)] == [
        healthy.name,
        other.name,
    ]
    assert len(issues) == 1
    assert len(issues[0]["remote_run_ids"]) == 2
    paths = tracking.fetch_runs(root, issues=[])
    assert len(compare_runs(paths, root / "comparison.csv")) == 2
    assert len(client.runs) == 4  # Quarantine never deletes or rewrites remote records.


def test_selected_run_adds_full_reproduction_without_changing_comparison(setup_tracking):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    assert tracking.auto_publish(run) == "uploaded"
    remote = next(iter(client.runs.values()))
    cached = tracking.fetch_runs(root)[0]
    original_manifest = (cached / tracking.MANIFEST).read_bytes()
    original_hash = remote.data.tags[tracking.PREFIX + "bundle_hash"]
    assert tracking.archive_run(run) == "uploaded"
    assert client.create_count == 1
    assert len(client.batches) == 1
    assert remote.info.status == "FINISHED"
    assert remote.data.tags[tracking.PREFIX + "complete"] == "true"
    assert remote.data.tags[tracking.PREFIX + "archive_status"] == "complete"
    assert remote.data.tags[tracking.PREFIX + "bundle_hash"] == original_hash
    archived = client.root / remote.info.run_id / tracking.ARCHIVE_FOLDER
    for path in run.rglob("*"):
        if path.is_file():
            assert (archived / path.relative_to(run)).read_bytes() == path.read_bytes()
    manifest = json.loads((archived / tracking.MANIFEST).read_text())
    assert manifest["comparison_hash"] == original_hash
    assert tracking._hash_json(manifest) == remote.data.tags[tracking.PREFIX + "archive_hash"]
    assert tracking.fetch_runs(root) == [cached]
    assert (cached / tracking.MANIFEST).read_bytes() == original_manifest
    count = len(client.uploads)
    assert tracking.archive_run(run) == "already_synced"
    assert tracking.sync_runs(root)["already_synced"] == 1
    assert len(client.uploads) == count
    assert len(client.batches) == 1


@pytest.mark.parametrize("retry_only", [False, True])
@pytest.mark.parametrize("failure", ["fail_upload", "fail_archive_tag_response"])
def test_archive_failure_retries_without_hiding_comparison_or_duplicating_run(
    setup_tracking, failure, retry_only
):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    tracking.auto_publish(run)
    remote = next(iter(client.runs.values()))
    setattr(client, failure, True)
    with pytest.warns(RuntimeWarning, match=r"\[network\]"):
        assert tracking.archive_run(run) == "pending"
    assert len(tracking.fetch_runs(root)) == 1
    assert remote.data.tags[tracking.PREFIX + "complete"] == "true"
    assert remote.info.status == "FINISHED"
    recovered = tracking.sync_runs(root, retry_only=retry_only)
    assert recovered["uploaded"] + recovered["already_synced"] == 1
    assert remote.data.tags[tracking.PREFIX + "archive_status"] == "complete"
    assert client.create_count == len(client.batches) == 1


def test_offline_selection_is_retried_but_other_runs_are_never_promoted(setup_tracking):
    root, client, make_run, enable = setup_tracking
    selected = make_run("팀원가")
    other = make_run("팀원나")
    enable()
    client.offline = True
    with pytest.warns(RuntimeWarning, match=r"\[network\]"):
        assert tracking.archive_run(selected) == "pending"
    assert len(list((root / ".tracking").glob("*/*.archive-request"))) == 1
    client.offline = False
    assert tracking.sync_runs(root)["uploaded"] == 2
    by_id = {run.data.tags[tracking.PREFIX + "local_run_id"]: run for run in client.runs.values()}
    assert by_id[selected.name].data.tags[tracking.PREFIX + "archive_status"] == "complete"
    assert by_id[other.name].data.tags[tracking.PREFIX + "archive_status"] == "none"
    assert not (client.root / by_id[other.name].info.run_id / tracking.ARCHIVE_FOLDER).exists()
    assert client.create_count == len(client.batches) == 2


def test_unconfigured_archive_is_disabled_without_queuing_selection(setup_tracking):
    root, client, make_run, _ = setup_tracking
    run = make_run()
    assert tracking.archive_run(run) == "disabled"
    assert not (root / ".tracking").exists()
    assert not client.runs


def test_invalid_archive_uri_raises_without_claiming_selection_is_queued(
    setup_tracking, monkeypatch
):
    root, _, make_run, _ = setup_tracking
    run = make_run()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://user:SECRET@example.com")
    with pytest.raises(tracking.TrackingError, match="without credentials"):
        tracking.archive_run(run)
    assert not (root / ".tracking").exists()


def test_busy_uploader_cannot_lose_an_explicit_archive_request(setup_tracking):
    from filelock import FileLock

    root, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    tracking.auto_publish(run)
    state = next((root / ".tracking").glob("*/*.json"))
    with FileLock(str(state.with_suffix(".lock"))):
        with pytest.warns(RuntimeWarning, match=r"\[locked\]"):
            assert tracking.archive_run(run) == "pending"
    assert state.with_suffix(".archive-request").exists()
    assert tracking.sync_runs(root)["uploaded"] == 1
    remote = next(iter(client.runs.values()))
    assert remote.data.tags[tracking.PREFIX + "archive_status"] == "complete"
    assert client.create_count == len(client.batches) == 1


@pytest.mark.parametrize(
    "modified", ["models/weights.txt", "source/train.py", "shared_source/tracking.py"]
)
def test_changed_reproduction_is_rejected_without_breaking_basic_comparison(
    setup_tracking, modified
):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    tracking.auto_publish(run)
    (run / modified).write_bytes(b"changed after local save")
    with pytest.raises(tracking.BundleError):
        tracking.archive_run(run)
    assert len(tracking.fetch_runs(root)) == 1
    assert tracking.sync_runs(root)["invalid"] == 1
    assert client.create_count == 1


def test_unselected_sync_never_opens_models_or_source_snapshots(setup_tracking, monkeypatch):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    actual_hash = tracking.file_hash

    def basic_only(path):
        assert path.parent == run
        return actual_hash(path)

    monkeypatch.setattr(tracking, "file_hash", basic_only)
    monkeypatch.setattr(tracking, "source_hash", lambda sources: pytest.fail("Source read"))
    enable()
    assert tracking.auto_publish(run) == "uploaded"
    assert tracking.sync_runs(root)["already_synced"] == 1
    assert client.create_count == 1


def test_completed_archive_worker_does_not_rehash_large_models(setup_tracking, monkeypatch):
    root, _, make_run, enable = setup_tracking
    run = make_run()
    enable()
    assert tracking.archive_run(run) == "uploaded"
    actual_hash = tracking.file_hash

    def basic_only(path):
        assert path.parent == run
        return actual_hash(path)

    monkeypatch.setattr(tracking, "file_hash", basic_only)
    assert tracking.sync_runs(root)["already_synced"] == 1


def test_concurrent_auto_upload_and_archive_share_one_remote_run(setup_tracking):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    barrier = threading.Barrier(2)

    def upload(function):
        barrier.wait()
        return function(run)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(upload, [tracking.auto_publish, tracking.archive_run]))
    assert set(outcomes).issubset({"uploaded", "already_synced"})
    assert tracking.sync_runs(root)["already_synced"] == 1
    remote = next(iter(client.runs.values()))
    assert remote.data.tags[tracking.PREFIX + "archive_status"] == "complete"
    assert client.create_count == len(client.batches) == 1


def make_legacy_run(root, client, run_dir, *, completed, local_state=True):
    result, manifest = tracking._full_manifest(run_dir)
    bundle_hash = tracking._hash_json(manifest)
    experiment = client.create_experiment("sixseven-poc")
    remote = client.create_run(
        experiment,
        tags={
            tracking.PREFIX + "protocol": "1",
            tracking.PREFIX + "local_run_id": run_dir.name,
            tracking.PREFIX + "bundle_hash": bundle_hash,
            tracking.PREFIX + "complete": str(completed).lower(),
            tracking.PREFIX + "comparison_group": tracking._group(result),
        },
    )
    directory = (
        root
        / ".tracking"
        / tracking._destination_key("http://tracking.example:5000", "sixseven-poc")
    )
    directory.mkdir(parents=True, exist_ok=True)
    if local_state:
        tracking._write_json(
            directory / f"{run_dir.name}.json",
            {
                "schema_version": 1,
                "bundle_hash": bundle_hash,
                "remote_run_id": remote.info.run_id,
                "status": "synced" if completed else "pending",
            },
        )
    if completed:
        tracking._upload_files(
            client, remote.info.run_id, run_dir, manifest, tracking.REMOTE_FOLDER
        )
        tracking._upload_manifest(
            client, remote.info.run_id, manifest, tracking.REMOTE_FOLDER, directory
        )
        client.set_terminated(remote.info.run_id, "FINISHED")
    return client.runs[remote.info.run_id]


@pytest.mark.parametrize("local_state", [False, True])
def test_completed_legacy_run_is_preserved_and_compares_alongside_new_runs(
    setup_tracking, local_state
):
    root, client, make_run, enable = setup_tracking
    legacy = make_run("팀원가")
    remote = make_legacy_run(root, client, legacy, completed=True, local_state=local_state)
    original_tags = dict(remote.data.tags)
    uploaded = len(client.uploads)
    enable()
    assert tracking.auto_publish(legacy) == "already_synced"
    assert tracking.archive_run(legacy) == "already_synced"
    assert len(client.uploads) == uploaded
    assert remote.data.tags == original_tags
    make_run("팀원나")
    assert tracking.sync_runs(root)["uploaded"] == 1
    paths = tracking.fetch_runs(root)
    assert len(paths) == 2
    assert len(compare_runs(paths, root / "compared.csv")) == 2
    assert client.create_count == 2


@pytest.mark.parametrize("retry_only", [False, True])
@pytest.mark.parametrize("local_state", [False, True])
def test_pending_legacy_run_resumes_as_basic_only_on_same_remote_identity(
    setup_tracking, local_state, retry_only
):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    remote = make_legacy_run(root, client, run, completed=False, local_state=local_state)
    enable()
    assert tracking.sync_runs(root, retry_only=retry_only)["uploaded"] == 1
    assert client.create_count == 1
    assert remote.data.tags[tracking.PREFIX + "protocol"] == "2"
    assert remote.data.tags[tracking.PREFIX + "archive_status"] == "none"
    uploaded = client.root / remote.info.run_id / tracking.REMOTE_FOLDER
    assert {path.name for path in uploaded.iterdir()} == {*tracking.BASIC_FILES, tracking.MANIFEST}
    assert not (client.root / remote.info.run_id / tracking.ARCHIVE_FOLDER).exists()
    assert len(tracking.fetch_runs(root)) == 1


@pytest.mark.parametrize("tag", ["legacy_bundle_hash", "bundle_hash", "protocol"])
def test_interrupted_legacy_tag_migration_reuses_run_and_stays_basic(
    setup_tracking, monkeypatch, tag
):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    make_legacy_run(root, client, run, completed=False)
    setter = client.set_tag
    failed = False

    def lose_response(run_id, key, value):
        nonlocal failed
        setter(run_id, key, value)
        if key == tracking.PREFIX + tag and not failed:
            failed = True
            raise ConnectionError("migration response lost")

    monkeypatch.setattr(client, "set_tag", lose_response)
    enable()
    with pytest.warns(RuntimeWarning):
        assert tracking.auto_publish(run) == "pending"
    assert tracking.sync_runs(root)["uploaded"] == 1
    assert client.create_count == 1
    assert not any(folder.startswith(tracking.ARCHIVE_FOLDER) for _, folder, _ in client.uploads)
    assert len(tracking.fetch_runs(root)) == 1


def test_sync_issues_identify_each_run_and_preserve_cached_connection_cause(setup_tracking):
    root, client, make_run, enable = setup_tracking
    runs = [make_run(owner) for owner in ("one", "two", "three")]
    enable()
    client.offline = True
    summary = tracking.sync_runs(root)
    assert summary["pending"] == 3
    assert [(item["owner"], item["experiment"], item["run_id"]) for item in summary["issues"]] == [
        (run.parent.parent.name, "exp001", run.name) for run in sorted(runs)
    ]
    assert {item["code"] for item in summary["issues"]} == {"network"}
    assert {item["status"] for item in summary["issues"]} == {"pending"}
    assert all(item["message"] and item["action"] for item in summary["issues"])
    assert not any(value in json.dumps(summary) for value in ("SECRET", "http://", str(root)))
    client.offline = False
    recovered = tracking.sync_runs(root)
    assert recovered["uploaded"] == 3
    assert recovered["issues"] == []
    assert client.create_count == 3


@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        ("UNAUTHENTICATED", "auth"),
        ("CUSTOMER_UNAUTHORIZED", "auth"),
        ("PERMISSION_DENIED", "permission"),
        ("DEADLINE_EXCEEDED", "timeout"),
        ("TEMPORARILY_UNAVAILABLE", "server"),
        ("INTERNAL_ERROR", "unknown"),
    ],
)
def test_structured_mlflow_failures_are_actionable_without_exposing_messages(
    setup_tracking, monkeypatch, error_code, expected
):
    from mlflow.exceptions import RestException

    root, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    error = RestException(
        {"error_code": error_code, "message": "SECRET https://secret.example/token"}
    )

    def fail(*_):
        raise error

    monkeypatch.setattr(client, "get_experiment_by_name", fail)
    summary = tracking.sync_runs(root)
    assert summary["issues"][0]["code"] == expected
    assert summary["issues"][0]["run_id"] == run.name
    with pytest.warns(RuntimeWarning) as warnings_seen:
        assert tracking.auto_publish(run) == "pending"
        assert tracking.archive_run(run) == "pending"
    for text in (json.dumps(summary), *(str(w.message) for w in warnings_seen)):
        assert "SECRET" not in text
        assert "secret.example" not in text
        assert expected in text
        assert run.name in text


@pytest.mark.parametrize(
    ("http_status", "expected"),
    [(401, "auth"), (403, "permission"), (503, "server"), (504, "timeout")],
)
def test_http_status_classification_ignores_exception_text(http_status, expected):
    from requests import HTTPError, Response

    response = Response()
    response.status_code = http_status
    response.url = "https://secret.example/password=SECRET"
    error = HTTPError("SECRET 401 forbidden timeout", response=response)
    wrapped = RuntimeError("SECRET wrapper")
    wrapped.__cause__ = error
    details = tracking.failure_details(wrapped)
    assert details["code"] == expected
    assert "SECRET" not in str(details)
    assert "secret.example" not in str(details)


def test_typed_failure_chain_prioritizes_timeout_and_does_not_guess_from_text():
    from filelock import Timeout
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import ReadTimeout

    message = "SECRET https://secret.example/path 401 permission denied"
    error = RequestsConnectionError(message)
    error.__context__ = ReadTimeout(message)
    assert tracking.failure_details(error)["code"] == "timeout"
    assert tracking.failure_details(Timeout("/secret/credential/file"))["code"] == "locked"
    assert tracking.failure_details(PermissionError(message))["code"] == "local_permission"
    assert tracking.failure_details(RuntimeError(message))["code"] == "unknown"
    # Cycles from nested wrappers must not prevent a safe diagnostic.
    error.__cause__ = error
    assert tracking.failure_details(error)["code"] == "timeout"


def test_invalid_run_issue_is_actionable_and_remains_distinct_from_pending(setup_tracking):
    root, _, make_run, enable = setup_tracking
    run = make_run()
    enable()
    (run / "predictions.csv").write_text("modified result", encoding="utf-8")
    summary = tracking.sync_runs(root)
    assert summary["invalid"] == 1
    assert summary["pending"] == 0
    assert summary["issues"][0]["status"] == "invalid"
    assert summary["issues"][0]["code"] == "bundle_changed"
    assert summary["issues"][0]["run_id"] == run.name


def test_remote_catalog_has_exact_typed_summary_and_no_artifact_downloads(setup_tracking):
    root, client, make_run, enable = setup_tracking
    enable()
    make_run()
    assert tracking.sync_runs(root)["uploaded"] == 1
    remote = next(iter(client.runs.values()))
    remote.info.start_time = 1_700_000_000_000
    remote.data.params.update(
        {
            "training.flag": "false",
            "training.optional": "null",
            "training.layers": "[64, 32]",
            "training.nested": '{"rate": 0.02}',
            "unrelated.secret": '"do not parse"',
            "seed": "None",
        }
    )
    item = tracking.list_remote_runs()[0]
    assert item["metrics"] == remote.data.metrics
    assert item["parameters"]["flag"] is False
    assert item["parameters"]["optional"] is None
    assert item["parameters"]["layers"] == [64, 32]
    assert item["parameters"]["nested"] == {"rate": 0.02}
    assert "unrelated.secret" not in item["parameters"]
    assert item["parameters_complete"] is True
    assert item["seed"] is None and item["seed_known"] is True
    assert item["created_at"] == "2023-11-14T22:13:20+00:00"
    assert item["created_at_source"] == "tracking_start"
    summary = item["validation_summary"]
    assert summary["patients"] == summary["positive"] + summary["negative"]
    assert summary["repeats"] == 2000
    assert summary["intervals"]["roc_auc"] == {"lower": 0.5, "upper": 0.5}
    assert item["validation_summary_status"] == "available"
    assert "Synthetic binary target" in item["target_definition"]
    assert client.downloads == []


@pytest.mark.parametrize(
    "encoded", [None, "broken", "NaN", '{"patients": -1}', "[" * 1100 + "]" * 1100]
)
def test_old_or_invalid_remote_uncertainty_does_not_block_verified_comparison(
    setup_tracking, encoded
):
    root, client, make_run, enable = setup_tracking
    enable()
    first, second = make_run(), make_run(owner="팀원나")
    before = {run: (run / "result.json").read_bytes() for run in (first, second)}
    assert tracking.sync_runs(root)["uploaded"] == 2
    for remote in client.runs.values():
        key = tracking.PREFIX + "validation_summary_v1"
        if encoded is None:
            remote.data.tags.pop(key)
        else:
            remote.data.tags[key] = encoded
    catalog = tracking.list_remote_runs()
    assert all(item["validation_summary"] is None for item in catalog)
    assert all(
        item["validation_summary_status"] == ("missing" if encoded is None else "invalid")
        for item in catalog
    )
    assert client.downloads == []
    paths = tracking.fetch_runs(root)
    table = compare_runs(paths, root / "uncertainty.csv")
    assert table.patient_count.gt(0).all()
    assert table.roc_auc_ci_lower.eq(0.5).all()
    assert table.roc_auc_ci_upper.eq(0.5).all()
    assert all((run / "result.json").read_bytes() == before[run] for run in (first, second))


def test_long_endpoint_definition_still_uploads_without_changing_original(
    setup_tracking, monkeypatch
):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    result_file = run / "result.json"
    result = json.loads(result_file.read_text(encoding="utf-8"))
    definition = "Custom outcome definition " * 400
    result["comparison_context"]["dataset"]["target_definition"] = definition
    result_file.write_text(json.dumps(result))
    original = result_file.read_bytes()
    set_tag = client.set_tag

    def bounded_tag(run_id, key, value):
        assert len(value) <= 8000
        set_tag(run_id, key, value)

    monkeypatch.setattr(client, "set_tag", bounded_tag)
    enable()
    assert tracking.sync_runs(root)["uploaded"] == 1
    assert result_file.read_bytes() == original
    remote = tracking.list_remote_runs()[0]
    assert remote["target_definition"] == definition[:4000]
    uploaded = client.root / remote["run_id"] / tracking.REMOTE_FOLDER / "result.json"
    assert uploaded.read_bytes() == original


@pytest.mark.parametrize("change", ["old_summary", "malformed", "nonfinite"])
def test_remote_catalog_marks_missing_or_truncated_metadata_unknown(setup_tracking, change):
    root, client, make_run, enable = setup_tracking
    enable()
    make_run()
    assert tracking.sync_runs(root)["uploaded"] == 1
    remote = next(iter(client.runs.values()))
    if change == "old_summary":
        remote.data.tags.pop(tracking.PREFIX + "parameters_complete")
    else:
        remote.data.params["training.broken"] = '{"value":' if change == "malformed" else "NaN"
    remote.data.params.pop("seed")
    item = tracking.list_remote_runs()[0]
    assert item["parameters_complete"] is False
    assert "broken" not in item["parameters"]
    assert item["created_at"] is None
    assert item["seed"] is None and item["seed_known"] is False
    assert client.downloads == []


@pytest.mark.parametrize("parameters", [{"unsupported/key": 2}, {"large": "x" * 6000}])
def test_upload_marks_filtered_or_truncated_parameters_incomplete(setup_tracking, parameters):
    _, client, make_run, enable = setup_tracking
    run = make_run()
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    result["training"]["parameters"] = parameters
    enable()
    remote = client.create_run("1", {})
    tracking._log_summary(client, remote, result)
    assert (
        client.runs[remote.info.run_id].data.tags[tracking.PREFIX + "parameters_complete"]
        == "false"
    )


@pytest.mark.parametrize(
    ("failure", "expected", "status"),
    [("corrupt", "bundle_changed", "invalid"), ("permission", "local_permission", "pending")],
)
def test_sync_distinguishes_invalid_local_state_from_local_permission_error(
    setup_tracking, monkeypatch, failure, expected, status
):
    root, _, make_run, enable = setup_tracking
    enable()
    run = make_run()
    assert tracking.sync_runs(root)["uploaded"] == 1
    state_path = next((root / ".tracking").glob("*/*.json"))
    if failure == "corrupt":
        state_path.write_text("SECRET malformed JSON", encoding="utf-8")
    else:
        original = tracking._json

        def read(path):
            if path == state_path:
                raise PermissionError("SECRET /private/config/password")
            return original(path)

        monkeypatch.setattr(tracking, "_json", read)
    summary = tracking.sync_runs(root)
    assert summary[status] == 1
    assert summary["issues"][0]["run_id"] == run.name
    assert summary["issues"][0]["code"] == expected
    assert "SECRET" not in str(summary)


@pytest.mark.parametrize("encoded", ['"' + "x" * 5998 + '"', "1e999", '{"rate": 1e999}'])
def test_catalog_never_interprets_truncated_or_nonfinite_setting_as_exact(setup_tracking, encoded):
    root, client, make_run, enable = setup_tracking
    enable()
    make_run()
    assert tracking.sync_runs(root)["uploaded"] == 1
    remote = next(iter(client.runs.values()))
    remote.data.params["training.broken"] = encoded
    item = tracking.list_remote_runs()[0]
    assert "broken" not in item["parameters"]
    assert item["parameters_complete"] is False
    assert not client.downloads


def test_archive_prequeue_permission_error_is_sanitized_in_cli_and_never_queued(
    setup_tracking, monkeypatch, capsys
):
    import sys

    from sixletters import __main__ as cli

    root, client, make_run, enable = setup_tracking
    run = make_run()
    enable()

    def denied(_run_dir):
        raise PermissionError("SECRET /private/credentials/file https://secret.example/password")

    monkeypatch.setattr(tracking, "_local_manifest", denied)
    monkeypatch.setattr(cli, "project_root", lambda: root.parent)
    monkeypatch.setattr(sys, "argv", ["dev", "archive", "--run", str(run)])
    with pytest.raises(SystemExit, match="2"):
        cli.main()
    output = capsys.readouterr().err
    assert "[local_permission]" in output
    assert "권한" in output
    assert "dev archive" in output
    assert "선택 요청은 저장되지 않았습니다" in output
    for private in ("SECRET", "/private/credentials", "https://secret.example"):
        assert private not in output
    assert not (root / ".tracking").exists()
    assert not client.runs


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
@pytest.mark.parametrize("retry_only", [False, True])
def test_sync_reports_failed_directory_scan_and_continues_readable_results(
    setup_tracking, monkeypatch, depth, retry_only
):
    root, client, make_run, enable = setup_tracking
    blocked = make_run("blocked")
    make_run("healthy")
    directory = [root, blocked.parent.parent, blocked.parent, blocked][depth]
    scandir = tracking.os.scandir

    def guarded_scan(path):
        if not isinstance(path, int) and Path(path) == directory:
            raise PermissionError("SECRET https://secret.example password=SECRET")
        return scandir(path)

    enable()
    with monkeypatch.context() as guarded:
        guarded.setattr(tracking.os, "scandir", guarded_scan)
        summary = tracking.sync_runs(root, retry_only=retry_only)
    assert summary["uploaded"] == (0 if depth == 0 else 1)
    assert summary["pending"] == 1
    assert summary["invalid"] == 0
    assert len(summary["issues"]) == 1
    issue = summary["issues"][0]
    assert issue["scope"] == "scan"
    assert issue["path"] == directory.relative_to(root).as_posix()
    assert issue["code"] == "local_permission"
    assert issue["run_id"] == (blocked.name if depth == 3 else "unknown")
    assert "SECRET" not in json.dumps(summary)
    assert str(root) not in json.dumps(summary)
    recovered = tracking.sync_runs(root, retry_only=retry_only)
    assert recovered["issues"] == []
    assert recovered["uploaded"] == (2 if depth == 0 else 1)
    assert len(client.runs) == 2


def test_sync_reports_actual_unreadable_directory_without_losing_healthy_run(setup_tracking):
    root, _, make_run, enable = setup_tracking
    blocked = make_run("blocked").parent.parent
    make_run("healthy")
    original_mode = blocked.stat().st_mode
    enable()
    try:
        blocked.chmod(0)
        try:
            with tracking.os.scandir(blocked):
                pass
        except PermissionError:
            pass
        else:
            pytest.skip("This platform/user does not enforce directory read permissions")
        summary = tracking.sync_runs(root)
        assert summary["uploaded"] == 1
        assert summary["pending"] == 1
        assert summary["issues"][0]["path"] == "blocked"
        assert summary["issues"][0]["code"] == "local_permission"
    finally:
        blocked.chmod(original_mode)


def test_sync_missing_root_and_incomplete_folders_do_not_report_false_failures(setup_tracking):
    root, client, _, enable = setup_tracking
    enable()
    assert tracking.sync_runs(root)["issues"] == []
    (root / "owner" / "experiment" / "incomplete").mkdir(parents=True)
    summary = tracking.sync_runs(root)
    assert summary["issues"] == []
    assert summary["pending"] == summary["invalid"] == 0
    assert not client.runs


def test_sync_does_not_follow_symlinks_or_discover_nested_results(setup_tracking):
    root, client, make_run, enable = setup_tracking
    run = make_run("owner")
    shutil.copytree(run, root / ".remote" / "server" / "cached")
    shutil.copytree(run, root / "other" / "experiment" / "container" / "nested")
    (root / "linked-owner").symlink_to(run.parent.parent, target_is_directory=True)
    (run.parent / "linked-run").symlink_to(run, target_is_directory=True)
    enable()
    summary = tracking.sync_runs(root)
    assert summary["uploaded"] == 1
    assert summary["invalid"] == 2
    assert {issue["path"] for issue in summary["issues"]} == {
        "linked-owner",
        "owner/exp001/linked-run",
    }
    assert len(client.runs) == 1


def test_duplicate_remote_diagnostics_do_not_echo_untrusted_metadata(setup_tracking):
    _, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    tracking.auto_publish(run)
    original = next(iter(client.runs.values()))
    original.data.tags[tracking.PREFIX + "owner"] = "https://secret.example/SECRET"
    original.data.tags[tracking.PREFIX + "experiment"] = "line\nbreak"
    duplicate = client.create_run(original.info.experiment_id, original.data.tags)
    client.set_terminated(duplicate.info.run_id, "FINISHED")
    issues = []
    assert tracking.list_remote_runs(issues=issues) == []
    assert issues[0]["owner"] == issues[0]["experiment"] == "unknown"
    assert "SECRET" not in json.dumps(issues)


def test_partial_summary_batch_retries_only_missing_values(setup_tracking, monkeypatch):
    root, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    original = client.log_batch
    failed = False

    def partially_record(run_id, *, metrics, params, tags, synchronous):
        nonlocal failed
        if not failed:
            failed = True
            original(run_id, metrics=metrics, params=params, tags=[], synchronous=synchronous)
            raise ConnectionError("summary response lost")
        return original(run_id, metrics=metrics, params=params, tags=tags, synchronous=synchronous)

    monkeypatch.setattr(client, "log_batch", partially_record)
    assert tracking.sync_runs(root)["pending"] == 1
    remote = next(iter(client.runs.values()))
    assert remote.data.tags[tracking.PREFIX + "complete"] == "false"
    assert client.uploads == []
    assert tracking.sync_runs(root)["uploaded"] == 1
    assert client.batches[1][1:] == ([], [])
    assert len(client.batch_tags[1]) == 3
    assert sum(len(batch[1]) for batch in client.batches) == 3
    assert tracking.auto_publish(run) == "already_synced"
    assert len(client.batches) == 2


@pytest.mark.parametrize(
    "artifact_uri",
    [
        None,
        "file:///private/result",
        "/private/result",
        "s3://bucket/result",
        "runs:/other-run/result",
        "http://tracking.example:5000/api/2.0/mlflow-artifacts/artifacts/1/run/artifacts",
        "mlflow-artifacts://outside.example/1/run/artifacts",
        "mlflow-artifacts://tracking.example:5001/1/run/artifacts",
        "mlflow-artifacts://user:SECRET@tracking.example:5000/1/run/artifacts",
        "mlflow-artifacts:/1/../run/artifacts",
        "mlflow-artifacts:/1/./run/artifacts",
        "mlflow-artifacts:/1/%2e%2e/run/artifacts",
        "mlflow-artifacts:/1/run/artifacts?redirect=SECRET",
        "mlflow-artifacts:/1/run/artifacts#fragment",
        "mlflow-artifacts:/1\\run/artifacts",
        "mlflow-artifacts:/1/\nrun/artifacts",
        "mlflow-artifacts:/",
    ],
)
def test_fetch_rejects_unsafe_artifact_location_before_download(setup_tracking, artifact_uri):
    root, client, make_run, enable = setup_tracking
    make_run()
    enable()
    assert tracking.sync_runs(root)["uploaded"] == 1
    remote = next(iter(client.runs.values()))
    remote.info.artifact_uri = artifact_uri
    with pytest.raises(tracking.BundleError, match="safe proxy"):
        tracking.fetch_runs(root)
    assert client.downloads == []
    assert not list((root / ".remote").glob("*/*/result.json"))


@pytest.mark.parametrize(
    "relative",
    [
        "/result.json",
        "../result.json",
        "bundle/../result.json",
        "bundle//result.json",
        "bundle\\result.json",
        "bundle/\nresult.json",
    ],
)
def test_download_rejects_unsafe_relative_path(tmp_path, monkeypatch, relative):
    import mlflow.artifacts

    monkeypatch.setattr(
        mlflow.artifacts, "download_artifacts", lambda **kwargs: pytest.fail("No request allowed")
    )
    remote = SimpleNamespace(info=SimpleNamespace(artifact_uri="mlflow-artifacts:/1/run/artifacts"))
    with pytest.raises(tracking.BundleError, match="safe proxy"):
        tracking._download_artifact(remote, relative, tmp_path, "http://tracking.example:5000")


def test_direct_artifact_download_uses_authenticated_proxy_without_run_or_model_lookups(
    tmp_path, monkeypatch
):
    import base64
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests = []
    body = b"verified artifact bytes"
    secret = base64.b64encode(b"member:password").decode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            if path == "/team/api/3.0/mlflow/server-info":
                response = b"{}"
            elif path == "/team/api/2.0/mlflow-artifacts/artifacts":
                response = b'{"files": []}'
            elif path in {
                "/team/api/2.0/mlflow-artifacts/artifacts/1/run/artifacts/bundle/result.json",
                "/team/api/2.0/mlflow-artifacts/artifacts/1/run/artifacts/reproduction/models/모델 1.txt",
            }:
                response = body
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "member")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "password")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    monkeypatch.delenv("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", raising=False)
    uri = f"http://127.0.0.1:{server.server_port}/team"
    try:
        for authority, filename in (
            ("", "bundle/result.json"),
            (f"//127.0.0.1:{server.server_port}", "reproduction/models/모델 1.txt"),
        ):
            remote = SimpleNamespace(
                info=SimpleNamespace(artifact_uri=f"mlflow-artifacts:{authority}/1/run/artifacts")
            )
            actual = tracking._download_artifact(remote, filename, tmp_path, uri)
            assert actual.read_bytes() == body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert all(auth == f"Basic {secret}" for _, auth in requests)
    assert len(requests) == 5  # Two listings + two files + one cached capability lookup.
    assert sum("?path=" in path for path, _ in requests) == 2
    assert sum(path == "/team/api/3.0/mlflow/server-info" for path, _ in requests) == 1
    assert not any("/runs/" in path or "/logged-models/" in path for path, _ in requests)
