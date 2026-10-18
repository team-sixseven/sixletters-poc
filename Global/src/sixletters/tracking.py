"""Publish immutable local result bundles and retrieve verified comparison inputs.

Training never depends on tracking availability. The local completion marker stays
authoritative; upload state and downloaded results live outside saved run bundles.

CI transfer scope measurement.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import time
import warnings
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

from filelock import FileLock

from .contracts import _hash_json
from .provenance import file_hash, source_hash, validate_resource_metadata
from .tracking_diagnostics import failure_details

PREFIX = "sixletters."
PROTOCOL = "2"
LEGACY_PROTOCOL = "1"
MANIFEST = "manifest.json"
REMOTE_FOLDER = "bundle"
COMPARISON_FILES = ("result.json", "predictions.csv")
BASIC_FILES = (*COMPARISON_FILES, "config.toml")
ARCHIVE_FOLDER = "reproduction"
_UUID = re.compile(r"[0-9a-f]{32}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_PARAMETER = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,179}\Z")


class TrackingError(ValueError):
    """A sanitized tracking failure, safe to display without connection secrets."""


class BundleError(TrackingError):
    """A local or remote result failed integrity checks."""


class ConfigurationError(TrackingError):
    """Invalid tracking settings, without including their original values."""

    diagnostic_code = "configuration"


class RemoteDeletedError(TrackingError):
    """A deleted remote result needs an explicit server-side restoration."""

    diagnostic_code = "remote_deleted"


def _configuration() -> tuple[str, str] | None:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        return None
    try:
        parsed = urlsplit(uri)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and not any(character.isspace() for character in uri)
        )
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ConfigurationError(
            "MLFLOW_TRACKING_URI must be an HTTP(S) URL without credentials or query"
        )
    uri = urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))
    experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "sixseven-poc").strip()
    if not experiment or len(experiment) > 250:
        raise ConfigurationError("MLFLOW_EXPERIMENT_NAME must contain 1 to 250 characters")
    return uri, experiment


def _client(uri: str):
    # These environment settings are MLflow's supported per-process HTTP controls.
    # Users can set longer timeouts for large model uploads in their local .env.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT", "30")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def _server_key(uri: str) -> str:
    return _hash_json(uri)[:24]


def _destination_key(uri: str, experiment: str) -> str:
    return _hash_json({"server": uri, "experiment": experiment})[:24]


def _safe_file(path: Path, parent: Path) -> None:
    if not path.is_file() or not path.resolve().is_relative_to(parent.resolve()):
        raise BundleError("Result files must stay inside their bundle")
    current = path
    while current != parent.parent:
        if current.is_symlink():
            raise BundleError("Result bundles must not contain symlinks")
        current = current.parent


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise BundleError("Saved tracking metadata is invalid") from error
    if not isinstance(value, dict):
        raise BundleError("Expected a JSON object")
    return value


def _write_json(path: Path, value: dict) -> None:
    content = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if path.is_file() and path.read_bytes() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _group(result: dict) -> str:
    return _hash_json(
        {
            "comparison_context": result["comparison_context"],
            "synthetic": result["synthetic"],
            "feature_groups": result["feature_groups"],
            "feature_set": result["feature_set"],
        }
    )


def _full_manifest(run_dir: Path) -> tuple[dict, dict]:
    # Keep comparison verification imports local to the upload boundary.
    from .comparison import _read_verified_result

    try:
        if run_dir.is_symlink():
            raise BundleError("Result directory must not be a symlink")
        result = _read_verified_result(run_dir)
        if (
            not _UUID.fullmatch(result["run_id"])
            or run_dir.name != result["run_id"]
            or run_dir.parent.name != result["experiment"]
            or run_dir.parent.parent.name != result["owner"]
        ):
            raise BundleError("Result identity differs from its local directory")
        entries = {}
        for path in sorted(run_dir.rglob("*")):
            relative = path.relative_to(run_dir).as_posix()
            if path.is_symlink():
                raise BundleError("Result bundles must not contain symlinks")
            if not path.is_file():
                continue
            if relative not in {*COMPARISON_FILES, "config.toml"} and path.relative_to(
                run_dir
            ).parts[0] not in {"models", "source", "shared_source", "resources"}:
                raise BundleError("Unexpected file in result bundle; raw data must not be uploaded")
            if path.relative_to(run_dir).parts[0] in {"source", "shared_source"} and (
                path.suffix != ".py" or "__pycache__" in path.relative_to(run_dir).parts
            ):
                raise BundleError("Source snapshots must contain Python source files only")
            _safe_file(path, run_dir)
            entries[relative] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        if not {*COMPARISON_FILES, "config.toml"}.issubset(entries) or not any(
            name.startswith("source/") and name.endswith(".py") for name in entries
        ):
            raise BundleError("Result bundle is incomplete")
        if {
            name.removeprefix("models/"): info
            for name, info in entries.items()
            if name.startswith("models/")
        } != result["model_files"]:
            raise BundleError("Saved model files differ from their recorded hashes")
        resources = result.get("resources", {})
        validate_resource_metadata(resources)
        if {
            name.removeprefix("resources/"): info
            for name, info in entries.items()
            if name.startswith("resources/")
        } != resources:
            raise BundleError("Saved resource files differ from their recorded hashes")
        for folder, key in (("source", "source_hash"), ("shared_source", "shared_code_hash")):
            # Validate the saved snapshot, whose collection rules may predate
            # current live-source exclusions such as build/ or hidden files.
            sources = {
                name.removeprefix(folder + "/"): (run_dir / name).read_bytes()
                for name in entries
                if name.startswith(folder + "/")
            }
            if source_hash(sources) != result["training"][key]:
                raise BundleError("Saved source files differ from their recorded hashes")
        return result, {"schema_version": 1, "local_run_id": result["run_id"], "files": entries}
    except PermissionError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise BundleError("Local result bundle is incomplete or modified") from error


def _experiment_id(client, name: str, *, create: bool) -> str | None:
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    if not create:
        return None
    try:
        return client.create_experiment(name)
    except Exception:
        # Concurrent clients may both see the project missing before creation.
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            raise
        return experiment.experiment_id


def _search(client, experiment_id: str, filters: str, *, include_deleted: bool = False) -> list:
    found = []
    token = None
    options = {}
    if include_deleted:
        from mlflow.entities import ViewType

        options["run_view_type"] = ViewType.ALL
    while True:
        page = client.search_runs(
            [experiment_id], filter_string=filters, max_results=1000, page_token=token, **options
        )
        found.extend(page)
        token = getattr(page, "token", None)
        if not token:
            return found


def _comparison_manifest(run_dir: Path) -> dict:
    if run_dir.is_symlink():
        raise BundleError("Result directory must not be a symlink")
    entries = {}
    for name in BASIC_FILES:
        path = run_dir / name
        _safe_file(path, run_dir)
        entries[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    # Never traverse unselected model/source directories during basic sync.
    if any(
        path.name not in {*BASIC_FILES, "models", "source", "shared_source", "resources"}
        for path in run_dir.iterdir()
    ):
        raise BundleError("Unexpected file in result bundle; raw data must not be uploaded")
    return {
        "schema_version": 2,
        "kind": "comparison",
        "local_run_id": run_dir.name,
        "files": entries,
    }


def _local_manifest(run_dir: Path) -> tuple[dict, dict]:
    """Validate comparison data without opening saved models or Python snapshots."""
    from .comparison import _read_verified_result

    try:
        manifest = _comparison_manifest(run_dir)
        result = _read_verified_result(run_dir)
        if (
            not _UUID.fullmatch(result["run_id"])
            or run_dir.name != result["run_id"]
            or run_dir.parent.name != result["experiment"]
            or run_dir.parent.parent.name != result["owner"]
        ):
            raise BundleError("Result identity differs from its local directory")
        return result, manifest
    except PermissionError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise BundleError("Local comparison result is incomplete or modified") from error


def _legacy_manifest(client, remote, state_dir: Path) -> dict:
    with tempfile.TemporaryDirectory(dir=state_dir) as temporary:
        path = Path(
            client.download_artifacts(remote.info.run_id, f"{REMOTE_FOLDER}/{MANIFEST}", temporary)
        )
        _safe_file(path, Path(temporary))
        return _checked_manifest(path, remote)


def _check_remote(remote, local_id: str, bundle_hash: str) -> None:
    if (
        remote.data.tags.get(PREFIX + "protocol") != PROTOCOL
        or remote.data.tags.get(PREFIX + "local_run_id") != local_id
        or remote.data.tags.get(PREFIX + "bundle_hash") != bundle_hash
    ):
        raise BundleError("The existing remote run has different comparison content")


def _log_summary(client, remote, result: dict, *, validation_summary: dict | None = None) -> None:
    parameters = {
        "model": result["training"]["model"],
        "seed": result["training"]["seed"],
        "feature_set": result["feature_set"],
        "feature_count": result["feature_count"],
    }
    parameters_complete = True
    for name, value in result["training"]["parameters"].items():
        encoded = json.dumps(value, ensure_ascii=False)
        if _PARAMETER.fullmatch(name):
            parameters[f"training.{name}"] = encoded
        if not _PARAMETER.fullmatch(name) or len(encoded) >= 6000:
            parameters_complete = False
    parameters = {name: str(value)[:6000] for name, value in parameters.items()}
    recorded_metrics = remote.data.metrics
    recorded_parameters = remote.data.params
    if any(
        name in recorded_metrics and recorded_metrics[name] != float(value)
        for name, value in result["metrics"].items()
    ) or any(
        name in recorded_parameters and recorded_parameters[name] != value
        for name, value in parameters.items()
    ):
        raise BundleError("Remote summary differs from the saved result")
    from mlflow.entities import Metric, Param

    timestamp = time.time_ns() // 1_000_000
    metrics = [
        Metric(name, float(value), timestamp, step=0)
        for name, value in result["metrics"].items()
        if name not in recorded_metrics
    ]
    params = [
        Param(name, value) for name, value in parameters.items() if name not in recorded_parameters
    ]
    # Querying persisted values also handles a successful log_batch whose HTTP
    # response was lost, without adding another metric-history point on retry.
    if metrics or params:
        client.log_batch(remote.info.run_id, metrics=metrics, params=params, synchronous=True)
    client.set_tag(
        remote.info.run_id, PREFIX + "parameters_complete", str(parameters_complete).lower()
    )
    if validation_summary is not None:
        # Derived, aggregate display metadata; immutable run files and the
        # original three metric histories remain unchanged on retries.
        client.set_tag(
            remote.info.run_id,
            PREFIX + "validation_summary_v1",
            json.dumps(validation_summary, sort_keys=True, allow_nan=False),
        )
        client.set_tag(
            remote.info.run_id,
            PREFIX + "target_definition",
            # Only a bounded endpoint hint is needed by the selection display.
            # The full definition remains in the immutable result artifact.
            result["comparison_context"]["dataset"]["target_definition"][:4000],
        )


def _upload_manifest(client, remote_id: str, manifest: dict, folder: str, state_dir: Path) -> None:
    with tempfile.TemporaryDirectory(dir=state_dir) as temporary:
        path = Path(temporary) / MANIFEST
        _write_json(path, manifest)
        client.log_artifact(remote_id, str(path), artifact_path=folder)


def _upload_files(client, remote_id: str, run_dir: Path, manifest: dict, folder: str) -> None:
    for name in manifest["files"]:
        destination = PurePosixPath(folder) / PurePosixPath(name).parent
        client.log_artifact(remote_id, str(run_dir / name), artifact_path=destination.as_posix())


def _archive(
    client,
    remote,
    run_dir: Path,
    base_manifest: dict,
    state: dict,
    state_path: Path,
    *,
    verify_completed: bool,
) -> str:
    remote_id = remote.info.run_id
    tags = remote.data.tags
    if (
        tags.get(PREFIX + "archive_status") == "complete"
        and not verify_completed
        and state.get("archive_hash")
    ):
        if state["archive_hash"] != tags.get(PREFIX + "archive_hash"):
            raise BundleError("The remote reproduction archive has different content")
        state["archive_status"] = "complete"
        _write_json(state_path, state)
        return "already_synced"
    _, manifest = _full_manifest(run_dir)
    if any(manifest["files"].get(name) != entry for name, entry in base_manifest["files"].items()):
        raise BundleError("Local comparison files changed before reproduction upload")
    manifest.update(
        {"schema_version": 2, "kind": "reproduction", "comparison_hash": _hash_json(base_manifest)}
    )
    archive_hash = _hash_json(manifest)
    if state.get("archive_hash") not in {None, archive_hash} or tags.get(
        PREFIX + "archive_hash"
    ) not in {None, archive_hash}:
        raise BundleError("A previously selected reproduction archive was modified")
    state.update({"archive_hash": archive_hash, "archive_status": "pending"})
    _write_json(state_path, state)
    if tags.get(PREFIX + "archive_status") == "complete":
        state["archive_status"] = "complete"
        _write_json(state_path, state)
        return "already_synced"
    client.set_tag(remote_id, PREFIX + "archive_hash", archive_hash)
    client.set_tag(remote_id, PREFIX + "archive_status", "pending")
    _upload_files(client, remote_id, run_dir, manifest, ARCHIVE_FOLDER)
    _, after = _full_manifest(run_dir)
    if after["files"] != manifest["files"]:
        raise BundleError("Local reproduction files changed during upload")
    _upload_manifest(client, remote_id, manifest, ARCHIVE_FOLDER, state_path.parent)
    client.set_tag(remote_id, PREFIX + "archive_status", "complete")
    state["archive_status"] = "complete"
    _write_json(state_path, state)
    return "uploaded"


class _PublishSession:
    """Share one connection lookup, including failures, within a sync pass."""

    def __init__(self, configuration: tuple[str, str]):
        self.configuration = configuration
        self.connection = None
        self.failure = None

    def connect(self):
        if self.failure is not None:
            raise TrackingError("The shared tracking connection is unavailable") from self.failure
        if self.connection is None:
            try:
                uri, name = self.configuration
                client = _client(uri)
                self.connection = (client, _experiment_id(client, name, create=True))
            except Exception as error:
                self.failure = error
                raise
        return self.connection


def _publish(
    run_dir: Path,
    configuration: tuple[str, str],
    *,
    request_archive: bool = False,
    session: _PublishSession | None = None,
) -> str:
    uri, experiment_name = configuration
    run_dir = run_dir.absolute()
    root = run_dir.parents[2]
    state_dir = root / ".tracking" / _destination_key(uri, experiment_name)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{run_dir.name}.json"
    with FileLock(str(state_path.with_suffix(".lock")), timeout=1):
        result, manifest = _local_manifest(run_dir)
        bundle_hash = _hash_json(manifest)
        state = _json(state_path) if state_path.exists() else {}
        if state.get("schema_version") == 1:
            # One-time integrity verification while converting a pre-v2 queue.
            # Its saved full hash does not identify the smaller comparison bundle.
            if _hash_json(_full_manifest(run_dir)[1]) != state.get("bundle_hash"):
                raise BundleError("A previously queued result was modified; save a new run")
            state["legacy_bundle_hash"] = state["bundle_hash"]
            state["bundle_hash"] = bundle_hash
        if state and state.get("bundle_hash") != bundle_hash:
            raise BundleError("A previously queued result was modified; save a new run")
        state.update(
            {
                "schema_version": 2,
                "bundle_hash": bundle_hash,
                "local_identity": [result["owner"], result["experiment"], result["run_id"]],
            }
        )
        state.setdefault("status", "pending")
        request_path = state_path.with_suffix(".archive-request")
        if request_path.exists():
            if _json(request_path).get("bundle_hash") != bundle_hash:
                raise BundleError("The selected result was modified before its upload")
            state["archive_requested"] = True
        _write_json(state_path, state)
        client, experiment_id = (session or _PublishSession(configuration)).connect()
        local_id = result["run_id"]
        if state.get("remote_run_id"):
            remote = client.get_run(state["remote_run_id"])
        else:
            matches = _search(
                client,
                experiment_id,
                f"tags.`{PREFIX}local_run_id` = '{local_id}'",
                include_deleted=True,
            )
            if len(matches) > 1:
                raise BundleError("Duplicate remote run identities require administrator review")
            remote = matches[0] if matches else None
        if remote is not None:
            if (
                remote.info.experiment_id != experiment_id
                or remote.data.tags.get(PREFIX + "local_run_id") != local_id
            ):
                raise BundleError("The existing remote run belongs to another result or experiment")
            if remote.info.lifecycle_stage == "deleted":
                state.update({"remote_run_id": remote.info.run_id, "status": "pending"})
                _write_json(state_path, state)
                raise RemoteDeletedError("The remote run is deleted; restore it before syncing")
            if remote.data.tags.get(PREFIX + "protocol") == LEGACY_PROTOCOL:
                legacy_hash = remote.data.tags.get(
                    PREFIX + "legacy_bundle_hash", remote.data.tags.get(PREFIX + "bundle_hash")
                )
                if (
                    remote.info.status == "FINISHED"
                    and remote.data.tags.get(PREFIX + "complete") == "true"
                ):
                    legacy = _legacy_manifest(client, remote, state_dir)
                    if any(
                        legacy["files"].get(name) != entry
                        for name, entry in manifest["files"].items()
                    ):
                        raise BundleError("Existing legacy comparison files differ")
                    if request_archive and _hash_json(_full_manifest(run_dir)[1]) != legacy_hash:
                        raise BundleError("Existing legacy reproduction files differ")
                    state.update(
                        {
                            "remote_run_id": remote.info.run_id,
                            "status": "synced",
                            "archive_status": "complete",
                        }
                    )
                    _write_json(state_path, state)
                    return "already_synced"
                if state.get("legacy_bundle_hash") != legacy_hash:
                    if _hash_json(_full_manifest(run_dir)[1]) != legacy_hash:
                        raise BundleError("Existing legacy result has different content")
                    state["legacy_bundle_hash"] = legacy_hash
                    _write_json(state_path, state)
                # Preserve the old identity before changing tags, so an interrupted
                # migration can resume without creating a run or uploading models.
                client.set_tag(remote.info.run_id, PREFIX + "legacy_bundle_hash", legacy_hash)
                client.set_tag(remote.info.run_id, PREFIX + "bundle_hash", bundle_hash)
                client.set_tag(remote.info.run_id, PREFIX + "protocol", PROTOCOL)
                client.set_tag(remote.info.run_id, PREFIX + "archive_status", "none")
                remote = client.get_run(remote.info.run_id)
            _check_remote(remote, local_id, bundle_hash)
        else:
            dataset = result["comparison_context"]["dataset"]
            remote = client.create_run(
                experiment_id,
                tags={
                    "mlflow.runName": f"{result['owner']}/{result['experiment']}/{local_id[:8]}",
                    PREFIX + "protocol": PROTOCOL,
                    PREFIX + "local_run_id": local_id,
                    PREFIX + "bundle_hash": bundle_hash,
                    PREFIX + "complete": "false",
                    PREFIX + "archive_status": "none",
                    PREFIX + "owner": result["owner"],
                    PREFIX + "experiment": result["experiment"],
                    PREFIX + "feature_set": result["feature_set"],
                    PREFIX + "dataset_version": dataset["dataset_id"],
                    PREFIX + "comparison_group": _group(result),
                    PREFIX + "synthetic": str(result["synthetic"]).lower(),
                },
            )
        remote_id = remote.info.run_id
        state["remote_run_id"] = remote_id
        _write_json(state_path, state)
        outcome = "already_synced"
        if not (
            remote.info.status == "FINISHED" and remote.data.tags.get(PREFIX + "complete") == "true"
        ):
            from .run_catalog import summary_from_run

            _log_summary(client, remote, result, validation_summary=summary_from_run(run_dir))
            _upload_files(client, remote_id, run_dir, manifest, REMOTE_FOLDER)
            if _local_manifest(run_dir)[1] != manifest:
                raise BundleError("Local comparison files changed during upload")
            _upload_manifest(client, remote_id, manifest, REMOTE_FOLDER, state_dir)
            client.set_tag(remote_id, PREFIX + "complete", "true")
            client.set_terminated(remote_id, status="FINISHED")
            outcome = "uploaded"
        state["status"] = "synced"
        _write_json(state_path, state)
        if state.get("archive_requested"):
            archived = _archive(
                client,
                remote,
                run_dir,
                manifest,
                state,
                state_path,
                verify_completed=request_archive,
            )
            if archived == "uploaded":
                outcome = "uploaded"
        return outcome


def archive_run(run_dir: Path) -> str:
    """Queue full reproduction files on the existing run, then try to upload them.

    No destination means no request is queued. Invalid configuration and integrity
    failures raise sanitized errors; network/lock failures return pending only
    after selection has been durably recorded for ``sync_runs``.
    """
    configuration = _configuration()
    if configuration is None:
        return "disabled"
    run_dir = Path(run_dir).absolute()
    try:
        _, manifest = _local_manifest(run_dir)
    except PermissionError as error:
        details = failure_details(error)
        raise TrackingError(
            f"[{details['code']}] {details['message']} "
            "선택 요청은 저장되지 않았습니다. artifacts 폴더 권한을 확인한 뒤 dev archive를 다시 실행하세요."
        ) from error
    uri, experiment_name = configuration
    request_path = (
        run_dir.parents[2]
        / ".tracking"
        / _destination_key(uri, experiment_name)
        / f"{run_dir.name}.archive-request"
    )
    request = {"schema_version": 1, "bundle_hash": _hash_json(manifest)}
    try:
        if request_path.exists() and _json(request_path) != request:
            raise BundleError("The selected result was modified before its upload")
        # Separate from mutable synchronization state: selection survives a busy
        # uploader holding the run lock, or a server call with a lost response.
        _write_json(request_path, request)
    except OSError as error:
        raise TrackingError("Could not queue the selected result; retry dev archive") from error
    try:
        return _publish(run_dir, configuration, request_archive=True)
    except BundleError:
        raise
    except Exception as error:
        _warn_pending(run_dir, error)
        return "pending"


def _safe_component(value: str) -> str:
    from .workspace import validate_component

    try:
        return validate_component(value, "run identity")
    except ValueError:
        return "unknown"


def _run_issue(run_dir: Path, error: BaseException) -> dict[str, str]:
    run_dir = Path(run_dir).absolute()
    invalid = isinstance(error, BundleError)
    return {
        "run_id": run_dir.name if _UUID.fullmatch(run_dir.name) else "unknown",
        "owner": _safe_component(run_dir.parent.parent.name),
        "experiment": _safe_component(run_dir.parent.name),
        "status": "invalid" if invalid else "pending",
        **failure_details(error, bundle_error=invalid),
    }


def _warn_pending(run_dir: Path, error: BaseException) -> None:
    issue = _run_issue(run_dir, error)
    try:
        warnings.warn(
            "MLflow upload pending. Local result is saved; "
            f"{issue['owner']}/{issue['experiment']}/{issue['run_id']} "
            f"[{issue['code']}]: {issue['message']} {issue['action']}",
            RuntimeWarning,
            stacklevel=3,
        )
    except Warning:
        # A warnings-as-errors policy must not invalidate completed local training.
        pass


def auto_publish(run_dir: Path) -> str:
    """Explicitly attempt one synchronous upload of an existing local result.

    Retained for callers of the earlier helper. Experiment.save does not call it;
    the workspace worker and manual sync_runs discover completion markers instead.
    """
    try:
        configuration = _configuration()
        if configuration is None:
            return "disabled"
        return _publish(Path(run_dir), configuration)
    except Exception as error:
        # Exceptions from HTTP libraries can contain URLs, headers or credentials.
        _warn_pending(run_dir, error)
        return "pending"


def _retry_complete(run_dir: Path, configuration: tuple[str, str]) -> bool:
    """Skip confirmed immutable results; manual sync still verifies server state."""
    state_path = (
        run_dir.parents[2] / ".tracking" / _destination_key(*configuration) / f"{run_dir.name}.json"
    )
    if not state_path.is_file():
        return False
    with FileLock(str(state_path.with_suffix(".lock")), timeout=1):
        state = _json(state_path)
        if (
            state.get("schema_version") != 2
            or state.get("status") != "synced"
            or state.get("local_identity")
            != [run_dir.parent.parent.name, run_dir.parent.name, run_dir.name]
            or not _UUID.fullmatch(str(state.get("remote_run_id", "")))
            or state.get("bundle_hash") != _hash_json(_comparison_manifest(run_dir))
        ):
            return False
        request_path = state_path.with_suffix(".archive-request")
        if request_path.exists():
            if _json(request_path) != {
                "schema_version": 1,
                "bundle_hash": state["bundle_hash"],
            }:
                return False
            if not state.get("archive_requested"):
                return False
        return not state.get("archive_requested") or state.get("archive_status") == "complete"


def _scan_issue(path: Path, root: Path, error: BaseException) -> dict[str, str]:
    parts = path.relative_to(root).parts
    invalid = isinstance(error, BundleError)
    return {
        "scope": "scan",
        "path": "/".join(_safe_component(part) for part in parts) or ".",
        "owner": _safe_component(parts[0]) if parts else "unknown",
        "experiment": _safe_component(parts[1]) if len(parts) > 1 else "unknown",
        "run_id": parts[2] if len(parts) > 2 and _UUID.fullmatch(parts[2]) else "unknown",
        "status": "invalid" if invalid else "pending",
        **failure_details(error, bundle_error=invalid),
    }


def _completed_markers(root: Path, issues: list[dict]) -> Iterator[Path]:
    """Inspect only root/owner/experiment/run; report failed scans without following links."""

    def scan(directory: Path, depth: int) -> Iterator[Path]:
        try:
            if directory.is_symlink():
                raise BundleError("Result directories must not be symlinks")
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except FileNotFoundError as error:
            # A clean checkout has no result directory before its first save.
            if depth:
                issues.append(_scan_issue(directory, root, error))
            return
        except (OSError, BundleError) as error:
            issues.append(_scan_issue(directory, root, error))
            return
        for entry in entries:
            path = directory / entry.name
            if entry.name.startswith(".") or depth == 3 and entry.name != "result.json":
                continue
            try:
                if entry.is_symlink():
                    raise BundleError("Result paths must not be symlinks")
                if depth < 3:
                    if entry.is_dir(follow_symlinks=False):
                        yield from scan(path, depth + 1)
                elif entry.is_file(follow_symlinks=False):
                    yield path
                else:
                    raise BundleError("Result completion marker must be a file")
            except (OSError, BundleError) as error:
                issues.append(_scan_issue(path, root, error))

    yield from scan(root, 0)


def sync_runs(root: Path, *, retry_only: bool = False) -> dict:
    """Retry local completed runs, excluding remote caches and nested result copies."""
    summary = {
        "uploaded": 0,
        "already_synced": 0,
        "pending": 0,
        "invalid": 0,
        "disabled": False,
        "issues": [],
    }
    configuration = _configuration()
    if configuration is None:
        summary["disabled"] = True
        return summary
    root = Path(root).absolute()
    session = _PublishSession(configuration)
    scan_issues = []
    for marker in _completed_markers(root, scan_issues):
        try:
            outcome = (
                "already_synced"
                if retry_only and _retry_complete(marker.parent, configuration)
                else _publish(marker.parent, configuration, session=session)
            )
            summary[outcome] += 1
        except BundleError as error:
            summary["invalid"] += 1
            summary["issues"].append(_run_issue(marker.parent, error))
        except Exception as error:
            summary["pending"] += 1
            summary["issues"].append(_run_issue(marker.parent, error))
    for issue in scan_issues:
        summary[issue["status"]] += 1
        summary["issues"].append(issue)
    return summary


def _checked_manifest(path: Path, remote) -> dict:
    manifest = _json(path)
    tags = remote.data.tags
    if (
        (
            (
                tags.get(PREFIX + "protocol") == LEGACY_PROTOCOL
                and manifest.get("schema_version") != 1
            )
            or (
                tags.get(PREFIX + "protocol") == PROTOCOL
                and (
                    manifest.get("schema_version") != 2
                    or manifest.get("kind") != "comparison"
                    or set(manifest.get("files", {})) != set(BASIC_FILES)
                )
            )
            or tags.get(PREFIX + "protocol") not in {LEGACY_PROTOCOL, PROTOCOL}
        )
        or manifest.get("local_run_id") != tags.get(PREFIX + "local_run_id")
        or _hash_json(manifest) != tags.get(PREFIX + "bundle_hash")
        or not isinstance(manifest.get("files"), dict)
        or not set(COMPARISON_FILES).issubset(manifest["files"])
    ):
        raise BundleError("Remote bundle manifest does not match its completed run")
    for name, entry in manifest["files"].items():
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or relative.as_posix() != name
            or any(part in {"", ".", ".."} for part in relative.parts)
            or "\\" in name
            or not isinstance(entry, dict)
            or set(entry) != {"sha256", "bytes"}
            or not isinstance(entry["sha256"], str)
            or not _HASH.fullmatch(entry["sha256"])
            or type(entry["bytes"]) is not int
            or entry["bytes"] < 0
        ):
            raise BundleError("Remote bundle manifest contains invalid file entries")
    return manifest


def _check_download(directory: Path, remote) -> None:
    from .comparison import _read_verified_result

    manifest = _checked_manifest(directory / MANIFEST, remote)
    for name in COMPARISON_FILES:
        path = directory / name
        _safe_file(path, directory)
        if {"sha256": file_hash(path), "bytes": path.stat().st_size} != manifest["files"][name]:
            raise BundleError("Downloaded comparison files do not match their bundle hashes")
    result = _read_verified_result(directory)
    if result["run_id"] != manifest["local_run_id"] or _group(result) != remote.data.tags.get(
        PREFIX + "comparison_group"
    ):
        raise BundleError("Downloaded result identity or comparison group differs")


def _available_runs(*, issues: list[dict] | None = None):
    """Read current completed run identities without downloading artifacts."""
    try:
        configuration = _configuration()
        if configuration is None:
            raise TrackingError("Set MLFLOW_TRACKING_URI before comparing team results")
        uri, experiment_name = configuration
        client = _client(uri)
        experiment_id = _experiment_id(client, experiment_name, create=False)
        if experiment_id is None:
            return client, uri, []
        runs = _search(
            client,
            experiment_id,
            f"attributes.status = 'FINISHED' and tags.`{PREFIX}complete` = 'true'",
        )
        by_identity = {}
        for remote in runs:
            if remote.data.tags.get(PREFIX + "protocol") not in {LEGACY_PROTOCOL, PROTOCOL}:
                continue
            if (
                remote.info.status != "FINISHED"
                or remote.data.tags.get(PREFIX + "complete") != "true"
            ):
                continue
            if remote.info.experiment_id != experiment_id:
                raise BundleError("A listed run belongs to another experiment")
            remote_id = remote.info.run_id
            local_id = remote.data.tags.get(PREFIX + "local_run_id", "")
            if not _UUID.fullmatch(remote_id) or not _UUID.fullmatch(local_id):
                raise BundleError("Remote run has an invalid identity")
            by_identity.setdefault(local_id, []).append(remote)
        available = []
        for local_id, copies in by_identity.items():
            if len(copies) == 1:
                available.extend(copies)
                continue
            _report_remote_issue(
                {
                    "scope": "remote",
                    "status": "excluded",
                    "code": "duplicate_remote",
                    "run_id": local_id,
                    "remote_run_ids": sorted(remote.info.run_id for remote in copies),
                    "owner": _safe_component(copies[0].data.tags.get(PREFIX + "owner", "")),
                    "experiment": _safe_component(
                        copies[0].data.tags.get(PREFIX + "experiment", "")
                    ),
                    "message": "동일한 실행 ID의 원격 결과가 중복되어 해당 결과 모두를 비교에서 제외했습니다.",
                    "action": "표시된 원격 실행 ID를 서버 관리자에게 전달해 중복을 확인하세요. 다른 실행 결과는 비교할 수 있습니다.",
                },
                issues,
            )
        return client, uri, available
    except TrackingError:
        raise
    except Exception as error:
        raise TrackingError("Team results could not be listed; check the server") from error


def _remote_selection(remote) -> dict:
    from .run_catalog import validation_metadata

    def reject_constant(_):
        raise ValueError("Non-finite JSON value")

    def finite_float(encoded):
        value = float(encoded)
        if not math.isfinite(value):
            raise ValueError("Non-finite JSON value")
        return value

    parameters = {}
    parameters_complete = remote.data.tags.get(PREFIX + "parameters_complete") == "true"
    for name, encoded in remote.data.params.items():
        if not name.startswith("training."):
            continue
        try:
            if (
                not _PARAMETER.fullmatch(name.removeprefix("training."))
                or not isinstance(encoded, str)
                or len(encoded) >= 6000
            ):
                raise ValueError
            parameters[name.removeprefix("training.")] = json.loads(
                encoded, parse_constant=reject_constant, parse_float=finite_float
            )
        except (TypeError, ValueError):
            # Truncated/malformed summaries are unknown, never a guessed setting.
            parameters_complete = False
    seed = None
    seed_known = False
    if "seed" in remote.data.params:
        raw_seed = remote.data.params["seed"]
        try:
            parsed_seed = None if raw_seed == "None" else json.loads(raw_seed)
            if parsed_seed is None or type(parsed_seed) is int and parsed_seed >= 0:
                seed, seed_known = parsed_seed, True
        except (TypeError, ValueError):
            pass
    created_at = None
    start_time = getattr(remote.info, "start_time", None)
    if type(start_time) in (int, float) and math.isfinite(start_time):
        try:
            created_at = datetime.fromtimestamp(start_time / 1000, UTC).isoformat()
        except (ValueError, OverflowError, OSError):
            pass
    metrics = {
        name: float(value)
        for name, value in remote.data.metrics.items()
        if type(value) in (int, float) and math.isfinite(value)
    }
    encoded_summary = remote.data.tags.get(PREFIX + "validation_summary_v1")
    validation_summary = None
    if isinstance(encoded_summary, str) and len(encoded_summary) <= 4096:
        try:
            validation_summary = validation_metadata(
                json.loads(
                    encoded_summary, parse_constant=reject_constant, parse_float=finite_float
                )
            )
        except (ValueError, TypeError, RecursionError):
            pass
    return {
        "run_id": remote.info.run_id,
        "local_run_id": remote.data.tags[PREFIX + "local_run_id"],
        "owner": remote.data.tags.get(PREFIX + "owner", "unknown"),
        "experiment": remote.data.tags.get(PREFIX + "experiment", "unknown"),
        "model": remote.data.params.get("model", "unknown"),
        "feature_set": remote.data.tags.get(PREFIX + "feature_set", "unknown"),
        "dataset_version": remote.data.tags.get(PREFIX + "dataset_version", "unknown"),
        "synthetic": {"true": True, "false": False}.get(remote.data.tags.get(PREFIX + "synthetic")),
        "metrics": metrics,
        "validation_summary": validation_summary,
        "validation_summary_status": (
            "available"
            if validation_summary is not None
            else "missing"
            if encoded_summary is None
            else "invalid"
        ),
        "target_definition": remote.data.tags.get(PREFIX + "target_definition"),
        "parameters": parameters,
        "parameters_complete": parameters_complete,
        "seed": seed,
        "seed_known": seed_known,
        "created_at": created_at,
        "created_at_source": "tracking_start",
        "archive_status": (
            "legacy"
            if remote.data.tags.get(PREFIX + "protocol") == LEGACY_PROTOCOL
            else remote.data.tags.get(PREFIX + "archive_status")
            if remote.data.tags.get(PREFIX + "archive_status") in {"none", "pending", "complete"}
            else "unknown"
        ),
        "archive_status_source": "server",
    }


def _report_remote_issue(issue: dict, issues: list[dict] | None) -> None:
    if issues is not None:
        issues.append(issue)
    else:
        try:
            warnings.warn(
                f"[{issue['code']}] {issue['message']} "
                f"local={issue['run_id']} remote={','.join(issue['remote_run_ids'])} "
                f"{issue['action']}",
                RuntimeWarning,
                stacklevel=3,
            )
        except Warning:
            # A caller's warnings-as-errors setting must not hide healthy results.
            pass


def local_archive_status(run_dir: Path) -> str:
    """Read the current destination's last local record without connecting or writing."""
    try:
        configuration = _configuration()
        if configuration is None:
            return "unknown"
        run_dir = Path(run_dir).absolute()
        if not _UUID.fullmatch(run_dir.name):
            return "unknown"
        identity = [run_dir.parents[1].name, run_dir.parent.name, run_dir.name]
        directory = run_dir.parents[2] / ".tracking" / _destination_key(*configuration)
        state_path = directory / f"{run_dir.name}.json"
        request_path = directory / f"{run_dir.name}.archive-request"
        for path in (state_path, request_path):
            if any(part.is_symlink() for part in (path, *path.parents)):
                return "unknown"
            if path.exists() and (not path.is_file() or path.stat().st_size > 1024 * 1024):
                return "unknown"
        state = _json(state_path) if state_path.is_file() else {}
        if state and state.get("local_identity") != identity:
            return "unknown"
        if state.get("archive_status") == "complete" and state.get("status") == "synced":
            return "complete"
        if state.get("archive_requested") or request_path.is_file():
            return "pending"
        return "none" if state else "unknown"
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return "unknown"


def list_remote_runs(*, issues: list[dict] | None = None) -> list[dict]:
    """Read selection metadata; exclude ambiguous identities and report diagnostics."""
    _, _, runs = _available_runs(issues=issues)
    return [_remote_selection(remote) for remote in runs]


def fetch_runs(
    root: Path, *, run_ids: list[str] | None = None, issues: list[dict] | None = None
) -> list[Path]:
    """Download and verify selected runs, or all completed runs when IDs are omitted."""
    try:
        if run_ids is not None and (
            not isinstance(run_ids, list | tuple)
            or not run_ids
            or any(not isinstance(value, str) or not _UUID.fullmatch(value) for value in run_ids)
            or len(set(run_ids)) != len(run_ids)
        ):
            raise TrackingError("Select distinct completed remote run IDs")
        excluded = []
        client, uri, runs = _available_runs(issues=excluded)
        for issue in excluded:
            _report_remote_issue(issue, issues)
        if run_ids is not None:
            quarantined = {remote_id for issue in excluded for remote_id in issue["remote_run_ids"]}
            if quarantined.intersection(run_ids):
                raise BundleError(
                    "Selected runs are excluded because of duplicate local run identities"
                )
            available = {remote.info.run_id: remote for remote in runs}
            if any(identifier not in available for identifier in run_ids):
                raise TrackingError("A selected run is no longer available; refresh the run list")
            runs = [available[identifier] for identifier in run_ids]
        cache = Path(root).absolute() / ".remote" / _server_key(uri)
        cache.mkdir(parents=True, exist_ok=True)
        paths = []
        for remote in runs:
            remote_id = remote.info.run_id
            destination = cache / remote_id
            with FileLock(str(cache / f"{remote_id}.lock"), timeout=1):
                if not destination.exists():
                    with tempfile.TemporaryDirectory(dir=cache) as temporary:
                        staging = Path(temporary) / "verified"
                        staging.mkdir()
                        for name in (MANIFEST, *COMPARISON_FILES):
                            downloaded = Path(
                                client.download_artifacts(
                                    remote_id, f"{REMOTE_FOLDER}/{name}", temporary
                                )
                            )
                            _safe_file(downloaded, Path(temporary))
                            shutil.copyfile(downloaded, staging / name)
                        _check_download(staging, remote)
                        staging.rename(destination)
                else:
                    _check_download(destination, remote)
                paths.append(destination)
        return paths
    except TrackingError:
        raise
    except Exception as error:
        raise TrackingError(
            "Team results could not be fetched or verified; check the server"
        ) from error
