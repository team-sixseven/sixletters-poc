"""Restore verified experiment files without loading models or executing saved code."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from filelock import FileLock

from .contracts import _hash_json
from .data.prepared import BUNDLE_FILENAME, load_prepared_dataset
from .experiment import _load_config
from .provenance import _check_unique_paths, file_hash, resource_metadata
from .workspace import _reject_ambiguous_name, validate_component

_UUID = re.compile(r"[0-9a-f]{32}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_BASIC_FILES = {"result.json", "predictions.csv", "config.toml"}
_BUNDLE_FOLDERS = {"models", "source", "shared_source", "resources"}
_RESTORE_OUTPUT = Path("outputs/.restore")


def _no_symlinks(path: Path) -> Path:
    path = Path(path).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("복원 경로와 상위 폴더에는 심볼릭 링크를 사용할 수 없습니다.")
    return path


def _portable_path(name: str) -> PurePosixPath:
    if not isinstance(name, str):
        raise ValueError("재현 파일 경로는 문자열이어야 합니다.")
    path = PurePosixPath(name)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() != name
        or any(validate_component(part, "reproduction filename") != part for part in path.parts)
    ):
        raise ValueError("재현 파일은 허용된 상대 경로를 사용해야 합니다.")
    return path


def _validate_entries(entries: dict) -> None:
    if not isinstance(entries, dict) or not _BASIC_FILES.issubset(entries):
        raise ValueError(
            "모델·코드를 포함한 전체 재현 파일이 필요합니다. 먼저 poc archive를 실행하세요."
        )
    for name, info in entries.items():
        path = _portable_path(name)
        if (
            name not in _BASIC_FILES
            and (len(path.parts) < 2 or path.parts[0] not in _BUNDLE_FOLDERS)
            or not isinstance(info, dict)
            or set(info) != {"sha256", "bytes"}
            or not isinstance(info["sha256"], str)
            or not _HASH.fullmatch(info["sha256"])
            or type(info["bytes"]) is not int
            or info["bytes"] < 0
        ):
            raise ValueError("재현 파일 목록의 경로·크기·해시를 확인할 수 없습니다.")
    _check_unique_paths(entries)


def _copy_verified(source: Path, destination: Path, expected: dict) -> None:
    _no_symlinks(source)
    if not source.is_file():
        raise ValueError("복원할 원본 파일이 없습니다.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    actual = {"sha256": file_hash(destination), "bytes": destination.stat().st_size}
    if actual != expected:
        raise ValueError("복원 중 원본 파일이 변경되었습니다. 원본을 확인한 뒤 다시 시도하세요.")


def _toml_value(value) -> str:
    if isinstance(value, dict):
        return (
            "{ "
            + ", ".join(
                f"{json.dumps(key, ensure_ascii=False)} = {_toml_value(item)}"
                for key, item in value.items()
            )
            + " }"
        )
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _restored_config(raw: bytes, *, owner: str, name: str) -> bytes:
    config = tomllib.loads(raw.decode("utf-8"))
    config["experiment"].update({"owner": owner, "name": name})
    lines = []
    for section, values in config.items():
        lines.extend(
            [
                f"[{section}]",
                *(
                    f"{json.dumps(key, ensure_ascii=False)} = {_toml_value(value)}"
                    for key, value in values.items()
                ),
                "",
            ]
        )
    content = "\n".join(lines).encode("utf-8")
    if tomllib.loads(content.decode("utf-8")) != config:
        raise ValueError("실험 설정을 손실 없이 복원할 수 없습니다.")
    return content


def restore_run(
    run_dir: Path,
    *,
    owner: str,
    name: str,
    root: Path,
    data: Path,
) -> Path:
    """Copy a complete local/archive bundle into a new named experiment.

    The supplied common data must exactly match the original fingerprint. Saved
    code is copied, never imported; models and original shared code remain hidden
    below outputs so they cannot enter the next experiment's source snapshot.
    """
    from .tracking import _full_manifest

    run_dir = _no_symlinks(run_dir)
    result, manifest = _full_manifest(run_dir)
    _validate_entries(manifest["files"])
    owner = validate_component(owner, "owner")
    name = validate_component(name, "experiment name")
    root = _no_symlinks(root)
    data_file = Path(data) / BUNDLE_FILENAME if Path(data).is_dir() else Path(data)
    data_hash = file_hash(data_file)
    dataset, synthetic = load_prepared_dataset(data_file)
    if file_hash(data_file) != data_hash:
        raise ValueError(
            "복원 확인 중 공통 데이터가 변경되었습니다. --data를 확인하고 다시 시도하세요."
        )
    original_data = result["comparison_context"]["dataset"]
    if (
        dataset.fingerprint() != original_data
        or synthetic != result["synthetic"]
        or {key: list(value) for key, value in dataset.feature_groups.items()}
        != result["feature_groups"]
    ):
        expected_id = str(original_data.get("dataset_id", "unknown"))[:150]
        raise ValueError(
            f"복원 데이터가 원본 {expected_id}의 내용·환자 분할·정답 정의와 다릅니다. "
            "해당 버전의 공통 데이터를 준비하고 --data로 지정하세요. "
            "원본 파일이나 분할을 수정해 맞추지 마세요."
        )
    raw, settings, _ = _load_config(run_dir / "config.toml")
    if any(
        settings[key] != result[field]
        for key, field in (
            ("owner", "owner"),
            ("name", "experiment"),
            ("feature_set", "feature_set"),
        )
    ) or ("features" in settings and settings["features"] != result["features"]):
        raise ValueError(
            "보관된 설정과 원본 실행 정보가 일치하지 않습니다. 원본 설정을 확인하세요."
        )
    config_filename = result.get("config_filename", "config.toml")
    if (
        not isinstance(config_filename, str)
        or validate_component(config_filename, "configuration filename") != config_filename
    ):
        raise ValueError("원래 설정 파일 이름이 올바르지 않습니다.")
    rewritten = _restored_config(raw, owner=owner, name=name)
    copies = {}
    for filename, info in manifest["files"].items():
        relative = PurePosixPath(filename)
        folder = relative.parts[0]
        if folder in {"source", "resources"}:
            destination = Path(*relative.parts[1:])
        elif folder in {"models", "shared_source"}:
            destination = _RESTORE_OUTPUT / folder / Path(*relative.parts[1:])
        else:
            continue
        key = destination.as_posix()
        if key in copies:
            raise ValueError("원본 코드와 추가 설정의 복원 경로가 충돌합니다.")
        copies[key] = (run_dir / filename, info)
    trace_name = (_RESTORE_OUTPUT / "trace.json").as_posix()
    configs = {config_filename: rewritten, "config.toml": rewritten}
    rewritten_configs = [config_filename]
    if config_filename != "config.toml" and "config.toml" in copies:
        default_source, default_info = copies.pop("config.toml")
        try:
            default_raw, _, _ = _load_config(default_source)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(
                "추가 설정 config.toml이 실험 설정 형식과 충돌합니다. "
                "이름을 구분한 원본 실험을 보관한 뒤 복원하세요."
            ) from error
        configs["config.toml"] = _restored_config(default_raw, owner=owner, name=name)
        copies[(_RESTORE_OUTPUT / "original_default_config.toml").as_posix()] = (
            default_source,
            default_info,
        )
        rewritten_configs.append("config.toml")
    generated_files = dict(configs)
    # Even a common-only run must not adopt the receiving member's extra packages.
    # An isolated scope without package records uses the common runtime directly.
    if "environment.toml" in generated_files:
        raise ValueError("실험 설정 파일명 environment.toml이 환경 선택 파일과 충돌합니다.")
    # A reproduction must use its archived package state, never replace or
    # inherit the receiving member's independently maintained environment.
    if "environment.toml" in copies:
        original_environment = (_RESTORE_OUTPUT / "original_environment.toml").as_posix()
        if original_environment in copies:
            raise ValueError("원본 환경 설정의 복원 경로가 충돌합니다.")
        copies[original_environment] = copies.pop("environment.toml")
    generated_files["environment.toml"] = b'scope = "isolated"\n'
    generated = set(generated_files) | {trace_name}
    original_config = (_RESTORE_OUTPUT / "original_config.toml").as_posix()
    if set(copies).intersection(generated | {original_config}):
        raise ValueError(
            "원본 코드·추가 설정이 복원용 파일과 충돌합니다. "
            "특히 이전 실행의 설정 파일명이 보관되지 않은 경우 원본 작성자에게 확인하세요. "
            "기존 파일은 덮어쓰지 않았습니다."
        )
    _check_unique_paths([*copies, *generated, original_config])
    trace = {
        "schema_version": 1,
        "restored_at": datetime.now(UTC).isoformat(),
        "original": {
            "owner": result["owner"],
            "experiment": result["experiment"],
            "run_id": result["run_id"],
            "manifest_hash": _hash_json(manifest),
            "dataset": original_data,
            "synthetic": result["synthetic"],
            "training": result["training"],
            "software": result["software"],
            "shared_code_scope": result.get("shared_code_scope"),
            "execution": result.get("execution"),
            "config_filename": config_filename,
        },
        "restored": {"owner": owner, "experiment": name},
        "models": (_RESTORE_OUTPUT / "models").as_posix(),
        "shared_source": (_RESTORE_OUTPUT / "shared_source").as_posix(),
        "identity_updated_configs": rewritten_configs,
        "environment_isolated": True,
        "automatically_executed": False,
    }
    trace_raw = (json.dumps(trace, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    # Bound the operational trace without treating it as a training resource.
    resource_metadata({"trace.json": trace_raw})
    destination = root / owner / "experiments" / name
    if destination.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("원본 실행 폴더 안에는 개인 실험을 복원할 수 없습니다.")
    for parent, component in ((root, owner), (destination.parent, name)):
        _reject_ambiguous_name(parent, component)
    _no_symlinks(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # Never replace an existing experiment, including an empty one.
    try:
        for relative, (source, expected) in copies.items():
            _copy_verified(source, destination / relative, expected)
        for filename, content in generated_files.items():
            (destination / filename).write_bytes(content)
        original_path = destination / original_config
        original_path.parent.mkdir(parents=True, exist_ok=True)
        original_path.write_bytes(raw)
        (destination / trace_name).write_bytes(trace_raw)
        if _full_manifest(run_dir)[1] != manifest:
            raise ValueError("복원 중 원본 실행이 변경되었습니다. 원본 확인 후 다시 시도하세요.")
        if file_hash(data_file) != data_hash:
            raise ValueError(
                "복원 중 공통 데이터가 변경되었습니다. --data를 확인하고 다시 시도하세요."
            )
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination / config_filename


def _archive_manifest(path: Path, remote) -> dict:
    from .tracking import PREFIX, _json

    manifest = _json(path)
    if (
        set(manifest) != {"schema_version", "kind", "comparison_hash", "local_run_id", "files"}
        or manifest["schema_version"] != 2
        or manifest["kind"] != "reproduction"
        or manifest["local_run_id"] != remote.data.tags.get(PREFIX + "local_run_id")
        or manifest["comparison_hash"] != remote.data.tags.get(PREFIX + "bundle_hash")
        or _hash_json(manifest) != remote.data.tags.get(PREFIX + "archive_hash")
    ):
        raise ValueError("서버의 모델 보관 목록과 완료 정보가 일치하지 않습니다.")
    _validate_entries(manifest["files"])
    return manifest


def _verify_cached_archive(container: Path, remote) -> Path:
    from .tracking import _full_manifest, _json

    manifest = _archive_manifest(_no_symlinks(container / "manifest.json"), remote)
    identity = _json(_no_symlinks(container / "identity.json"))
    owner = validate_component(identity.get("owner"), "archived owner")
    name = validate_component(identity.get("experiment"), "archived experiment")
    local_id = manifest["local_run_id"]
    if not isinstance(local_id, str) or not _UUID.fullmatch(local_id):
        raise ValueError("서버의 원본 실행 ID가 올바르지 않습니다.")
    run = container / "run" / owner / name / local_id
    _no_symlinks(run)
    result, actual = _full_manifest(run)
    if actual["files"] != manifest["files"] or result["run_id"] != local_id:
        raise ValueError("내려받은 재현 파일이 서버의 기록과 다릅니다.")
    return run


def fetch_archive(cache_root: Path, remote_run_id: str) -> Path:
    """Download one completed MLflow reproduction archive; never deserialize models."""
    from . import tracking

    if not isinstance(remote_run_id, str) or not _UUID.fullmatch(remote_run_id):
        raise ValueError("서버 목록에 표시된 MLflow 실행 ID를 지정하세요.")
    try:
        client, uri, runs = tracking._available_runs()
        remote = next((run for run in runs if run.info.run_id == remote_run_id), None)
        if remote is None:
            raise ValueError("해당 실행을 조회할 수 없습니다. 서버 목록과 실행 ID를 확인하세요.")
        if (
            remote.data.tags.get(tracking.PREFIX + "protocol") != tracking.PROTOCOL
            or remote.data.tags.get(tracking.PREFIX + "archive_status") != "complete"
        ):
            raise ValueError(
                "모델·재현 파일 보관이 완료되지 않았습니다. "
                "전체 원본 결과가 있는 컴퓨터에서 poc archive를 완료하세요."
            )
        cache = _no_symlinks(cache_root) / ".reproduction" / tracking._server_key(uri)
        _no_symlinks(cache)
        cache.mkdir(parents=True, exist_ok=True)
        destination = cache / remote_run_id
        with FileLock(str(cache / f"{remote_run_id}.lock"), timeout=1):
            _no_symlinks(destination)
            if destination.exists():
                return _verify_cached_archive(destination, remote)
            with tempfile.TemporaryDirectory(prefix=".restore-", dir=cache) as temporary:
                temporary = Path(temporary)
                downloads = temporary / "downloads"
                downloads.mkdir()
                manifest_path = Path(
                    client.download_artifacts(
                        remote_run_id,
                        f"{tracking.ARCHIVE_FOLDER}/{tracking.MANIFEST}",
                        str(downloads),
                    )
                )
                tracking._safe_file(manifest_path, downloads)
                manifest = _archive_manifest(manifest_path, remote)
                local_id = manifest["local_run_id"]
                if not isinstance(local_id, str) or not _UUID.fullmatch(local_id):
                    raise ValueError("서버의 원본 실행 ID가 올바르지 않습니다.")
                staged = temporary / "verified"
                staged.mkdir()
                (staged / "manifest.json").write_bytes(manifest_path.read_bytes())
                run = None
                ordered = [
                    "result.json",
                    *(key for key in manifest["files"] if key != "result.json"),
                ]
                for filename in ordered:
                    downloaded = Path(
                        client.download_artifacts(
                            remote_run_id, f"{tracking.ARCHIVE_FOLDER}/{filename}", str(downloads)
                        )
                    )
                    tracking._safe_file(downloaded, downloads)
                    expected = manifest["files"][filename]
                    if {
                        "sha256": file_hash(downloaded),
                        "bytes": downloaded.stat().st_size,
                    } != expected:
                        raise ValueError("내려받은 재현 파일의 크기·해시가 원본과 다릅니다.")
                    if run is None:
                        result = tracking._json(downloaded)
                        owner = validate_component(result.get("owner"), "archived owner")
                        name = validate_component(result.get("experiment"), "archived experiment")
                        if result.get("run_id") != local_id:
                            raise ValueError("보관된 결과와 실행 ID가 일치하지 않습니다.")
                        run = staged / "run" / owner / name / local_id
                        run.mkdir(parents=True)
                        (staged / "identity.json").write_text(
                            json.dumps({"owner": owner, "experiment": name})
                        )
                    _copy_verified(downloaded, run / filename, expected)
                verified = _verify_cached_archive(staged, remote)
                relative = verified.relative_to(staged)
                staged.rename(destination)
                return destination / relative
    except (ValueError, tracking.TrackingError):
        raise
    except Exception as error:
        raise tracking.TrackingError(
            "모델·재현 파일을 내려받지 못했습니다. 서버 연결·권한을 확인한 뒤 다시 시도하세요."
        ) from error


def _status_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _prediction_steps(
    directory: Path, run_id: str, candidate: dict, root: Path, data: Path
) -> tuple[str, list[str]]:
    """Suggest reviewed commands; never execute code or guess a custom model loader."""
    from .local_settings import read_settings
    from .personal_env import LOCK_NAMES, environment_resources

    owner = validate_component(candidate["owner"], "owner")
    name = validate_component(candidate["experiment"], "experiment")
    frozen = _no_symlinks(directory / candidate["directory"])
    standard = Path(__file__).resolve().parents[2] / "templates/experiment/predict.py"
    predictor = frozen / "source/predict.py"
    template_matches = (
        standard.is_file()
        and predictor.is_file()
        and predictor.read_bytes() == standard.read_bytes()
        and (frozen / "models/model.pkl").is_file()
    )
    local = root / "Members" / owner / "experiments" / name
    local_matches = False
    if template_matches:
        try:
            _no_symlinks(local / "predict.py")
            _, settings, _ = _load_config(_no_symlinks(local / "config.toml"))
            expected = {
                name: (frozen / "resources" / name).read_bytes()
                for name in LOCK_NAMES
                if (frozen / "resources" / name).is_file()
            }
            local_matches = (
                settings["owner"] == owner
                and settings["name"] == name
                and (local / "predict.py").read_bytes() == predictor.read_bytes()
                and environment_resources(local) == expected
            )
        except (OSError, ValueError, KeyError, TypeError):
            pass
    plan_path = _status_path(directory, root)
    data_path = _status_path(data, root)
    if local_matches:
        return (
            "보관된 표준 예측 코드·패키지 기록과 일치합니다. 모델을 검토한 뒤 예측만 실행하세요.",
            [
                shlex.join(
                    [
                        "./dev",
                        "exec",
                        "--owner",
                        owner,
                        name,
                        "predict.py",
                        "--plan",
                        plan_path,
                        "--run-id",
                        run_id,
                        "--data",
                        data_path,
                    ]
                )
            ],
        )
    try:
        target_owner = read_settings(root).get("owner") or owner
    except (OSError, ValueError):
        target_owner = owner
    target_owner = validate_component(target_owner, "owner")
    restored_name = f"final-{run_id[:12]}"
    suffix = 1
    while (root / "Members" / target_owner / "experiments" / restored_name).exists() or (
        root / "Members" / target_owner / "experiments" / restored_name
    ).is_symlink():
        restored_name = f"final-{run_id[:12]}-{suffix}"
        suffix += 1
    commands = [
        shlex.join(
            [
                "./dev",
                "restore",
                restored_name,
                "--owner",
                target_owner,
                "--run",
                _status_path(frozen, root),
                "--data",
                data_path,
            ]
        ),
        shlex.join(["./dev", "env", "sync", restored_name, "--owner", target_owner]),
    ]
    if template_matches:
        commands.append(
            shlex.join(
                [
                    "./dev",
                    "exec",
                    "--owner",
                    target_owner,
                    restored_name,
                    "predict.py",
                    "--plan",
                    plan_path,
                    "--run-id",
                    run_id,
                    "--data",
                    data_path,
                ]
            )
        )
        return (
            "현재 개인 코드·패키지 기록을 그대로 사용할 수 없어 고정한 원본을 새 실험에 복원하는 명령입니다. 모델 검토 후 순서대로 실행하세요.",
            commands,
        )
    return (
        "원본을 복원한 뒤 모델 형식에 맞는 예측 코드를 검토하세요. 사용자 정의 코드는 실행 명령을 추측하지 않습니다. 재학습하지 마세요.",
        commands,
    )


def final_plan_status(plan_directory: Path, data: Path, root: Path) -> dict:
    """Inspect every frozen candidate and show progress without computing test metrics."""
    from .final_evaluation import FinalEvaluation, _data, _plan, _read_json, _replay_tolerance

    directory, root, data = (_no_symlinks(Path(path)) for path in (plan_directory, root, data))
    plan = _plan(directory)
    dataset, synthetic = _data(data)
    if dataset.fingerprint() != plan["dataset"] or synthetic != plan["synthetic"]:
        raise ValueError("최종 평가 계획과 같은 데이터·환자 분할을 지정하세요.")
    candidates = []
    counts = {"complete": 0, "missing": 0, "invalid": 0}
    report = _no_symlinks(directory / "report")
    for run_id, candidate in plan["candidates"].items():
        item = {key: candidate[key] for key in ("owner", "experiment")}
        item.update(
            run_id=run_id,
            status="invalid",
            commands=[],
            message="고정한 원본 또는 예측 기록을 확인하세요. 기존 파일을 덮어쓰지 마세요.",
        )
        try:
            if not isinstance(run_id, str) or not _UUID.fullmatch(run_id):
                raise ValueError("Invalid candidate identity")
            FinalEvaluation(directory, run_id, data)
            stored = _no_symlinks(directory / "predictions" / run_id)
            if not stored.exists():
                if report.exists():
                    raise ValueError("Report exists without all candidate predictions")
                item["status"] = "missing"
                item["message"], item["commands"] = _prediction_steps(
                    directory, run_id, candidate, root, data
                )
            else:
                receipt = _read_json(_no_symlinks(stored / "receipt.json"))
                predictions = _no_symlinks(stored / "predictions.csv")
                if (
                    receipt.get("plan_hash") != plan["plan_hash"]
                    or receipt.get("run_id") != run_id
                    or receipt.get("predictions_sha256") != file_hash(predictions)
                    or receipt.get("validation_replay") != "matched"
                    or (
                        plan["schema_version"] >= 2
                        and receipt.get("validation_replay_tolerance") != _replay_tolerance(plan)
                    )
                ):
                    raise ValueError("Saved prediction receipt differs")
                item.update(
                    status="complete",
                    message="검증 재현과 예측 파일의 완료 기록·해시가 일치합니다. 다시 예측하지 않습니다.",
                )
        except (OSError, ValueError, KeyError, TypeError):
            item.update(status="invalid", commands=[])
        counts[item["status"]] += 1
        candidates.append(item)
    commands = []
    if candidates and counts["complete"] == len(candidates) and not report.exists():
        commands.append(
            shlex.join(
                ["./dev", "final", "report", directory.name, "--data", _status_path(data, root)]
            )
        )
    if _plan(directory) != plan:
        raise ValueError("상태를 확인하는 중 최종 평가 계획이 변경되었습니다.")
    return {
        "name": directory.name,
        "plan": str(directory),
        "candidates": candidates,
        "counts": counts,
        "report": str(report) if report.exists() else None,
        "next_commands": commands,
    }
