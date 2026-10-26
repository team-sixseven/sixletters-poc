"""Upload user-authored results; never train, evaluate, or deserialize models."""

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

from filelock import FileLock
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from .local_settings import connection, read_settings, validate_owner


class UploadError(ValueError):
    pass


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def regular(path: Path) -> Path:
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"일반 파일만 등록할 수 있습니다: {path.name}")
    if (
        path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}
        or any(p.startswith(".") for p in path.parts if p not in {".", ".."})
        or re.search(r"(^|[-_.])(secret|credential|id_rsa|id_ed25519)([-_.]|$)", path.name, re.I)
    ):
        raise ValueError(f"숨김 파일·인증 파일은 등록할 수 없습니다: {path.name}")
    return path


def json_object(path: Path) -> dict:
    regular(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError(f"{path.name}: JSON 설정은 1 MiB 이하여야 합니다.")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"중복 JSON 키: {key}")
            value[key] = item
        return value

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: JSON 객체가 필요합니다.")
    for key in value:
        if (
            not re.fullmatch(r"[\w .\-/]{1,200}", key)
            or key.startswith("/")
            or ".." in key.split("/")
        ):
            raise ValueError("지표·설정 이름은 1~200자의 문자·숫자·공백·_·-·.·/를 사용하세요.")
    return value


def bundle(directory: Path, model: Path | None = None) -> tuple[dict, dict, dict[str, Path]]:
    directory = directory.absolute()
    metrics = json_object(directory / "metrics.json")
    if not metrics or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values()
    ):
        raise ValueError("metrics.json에는 하나 이상의 유한한 숫자 지표가 필요합니다.")
    params_file = directory / "params.json"
    params = json_object(params_file) if params_file.exists() else {}
    encoded = {}
    for key, value in params.items():
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, allow_nan=False)
        )
        if len(text) > 6000:
            raise ValueError(f"설정 값이 너무 깁니다: {key}")
        encoded[key] = text
    files = {"metrics.json": directory / "metrics.json"}
    if params_file.exists():
        files["params.json"] = params_file
    artifacts = directory / "artifacts"
    if artifacts.exists() or artifacts.is_symlink():
        if artifacts.is_symlink() or not artifacts.is_dir():
            raise ValueError("artifacts는 실제 폴더여야 합니다.")
        for file in sorted(artifacts.rglob("*")):
            if file.is_symlink():
                raise ValueError("artifacts 안에 심볼릭 링크를 사용할 수 없습니다.")
            if file.is_file():
                files[file.relative_to(directory).as_posix()] = regular(file)
    if model is not None:
        model = model.absolute()
        if any(p.is_symlink() for p in (model, *model.parents)) or not model.exists():
            raise ValueError("모델 파일 또는 폴더를 확인하세요. 심볼릭 링크는 지원하지 않습니다.")
        candidates = [model] if model.is_file() else list(model.rglob("*"))
        count = 0
        for file in sorted(candidates):
            if file.is_symlink():
                raise ValueError("모델 폴더 안에 심볼릭 링크를 사용할 수 없습니다.")
            if file.is_file():
                relative = file.name if model.is_file() else file.relative_to(model).as_posix()
                files[f"models/{relative}"] = regular(file)
                count += 1
        if not count:
            raise ValueError("모델 폴더가 비어 있습니다.")
    return metrics, encoded, files


def atomic_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".upload-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def upload(root: Path, directory: Path, model: Path | None = None) -> dict:
    root, directory = root.resolve(), directory.absolute()
    metrics, params, files = bundle(directory, model)
    directory = directory.resolve()
    owner = validate_owner(read_settings(root).get("owner", ""))
    if not directory.resolve().is_relative_to(root / "Members" / owner):
        raise ValueError("setup에 등록한 본인의 Members/<ID>/ 폴더 안의 결과를 선택하세요.")
    config = connection(root)
    relative = directory.relative_to(root).as_posix()
    identity = hashlib.sha256(
        json.dumps([config["uri"].rstrip("/"), config["experiment"], owner, relative]).encode()
    ).hexdigest()
    cache = root / ".poc-cache" / "uploads"
    if any(p.is_symlink() for p in (cache, *cache.parents)):
        raise ValueError("전송 상태 폴더는 심볼릭 링크일 수 없습니다.")
    cache.mkdir(parents=True, exist_ok=True)
    state_file = cache / f"{identity}.json"
    with FileLock(str(cache / f"{identity}.lock"), timeout=0):
        if state_file.is_symlink():
            raise ValueError("전송 상태 파일은 심볼릭 링크일 수 없습니다.")
        hashes = {name: digest(file) for name, file in files.items()}
        state = (
            json.loads(state_file.read_text())
            if state_file.exists()
            else {"hashes": hashes, "done": []}
        )
        old_results = {k: v for k, v in state["hashes"].items() if not k.startswith("models/")}
        results = {k: v for k, v in hashes.items() if not k.startswith("models/")}
        old_models = {k: v for k, v in state["hashes"].items() if k.startswith("models/")}
        models = {k: v for k, v in hashes.items() if k.startswith("models/")}
        if old_results != results or (models and old_models and models != old_models):
            raise ValueError(
                "이미 등록을 시작한 결과가 변경됐습니다. 새 결과 폴더로 저장해 등록하세요."
            )
        state["hashes"].update(hashes)
        atomic_json(state_file, state)
        # MLflow resolves authentication from the process environment; never include it in artifacts.
        previous = {
            k: os.environ.get(k)
            for k in (
                "MLFLOW_TRACKING_USERNAME",
                "MLFLOW_TRACKING_PASSWORD",
                "MLFLOW_HTTP_REQUEST_MAX_RETRIES",
            )
        }
        try:
            for key, name in (
                ("MLFLOW_TRACKING_USERNAME", "username"),
                ("MLFLOW_TRACKING_PASSWORD", "password"),
            ):
                if config[name]:
                    os.environ[key] = config[name]
                else:
                    os.environ.pop(key, None)
            # An ambiguous create response must be recovered by searching the upload ID,
            # not by blindly retrying the POST inside the HTTP client.
            os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
            client = MlflowClient(tracking_uri=config["uri"])
            experiment = client.get_experiment_by_name(config["experiment"])
            if experiment is None:
                try:
                    experiment_id = client.create_experiment(config["experiment"])
                except MlflowException:
                    experiment = client.get_experiment_by_name(config["experiment"])
                    if experiment is None:
                        raise
                    experiment_id = experiment.experiment_id
            else:
                experiment_id = experiment.experiment_id
            result_hash = hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()
            if not state.get("run_id"):
                existing = client.search_runs(
                    [experiment_id],
                    filter_string=f"tags.`sixseven.upload_id` = '{identity}'",
                    max_results=2,
                )
                if len(existing) > 1:
                    raise UploadError(
                        "동일 등록 ID가 서버에 여러 개 있습니다. 서버 관리자에게 확인하세요."
                    )
                run = (
                    existing[0]
                    if existing
                    else client.create_run(
                        experiment_id,
                        tags={
                            "mlflow.runName": directory.name,
                            "sixseven.owner": owner,
                            "sixseven.upload_id": identity,
                            "sixseven.result_sha256": result_hash,
                        },
                    )
                )
                state["run_id"] = run.info.run_id
                atomic_json(state_file, state)
            run_id = state["run_id"]
            remote = client.get_run(run_id)
            if remote.data.tags.get("sixseven.result_sha256") != result_hash:
                raise UploadError("서버에 등록한 결과가 다릅니다. 새 결과 폴더를 사용하세요.")
            if models:
                model_hash = hashlib.sha256(json.dumps(models, sort_keys=True).encode()).hexdigest()
                prior_model = remote.data.tags.get("sixseven.model_sha256")
                if prior_model and prior_model != model_hash:
                    raise UploadError(
                        "서버에 다른 모델이 등록돼 있습니다. 새 결과 폴더를 사용하세요."
                    )
                client.set_tag(run_id, "sixseven.model_sha256", model_hash)
            if remote.info.lifecycle_stage != "active":
                raise UploadError("서버에서 삭제된 실행입니다. 서버에서 복원 후 다시 시도하세요.")
            for key, value in params.items():
                if key in remote.data.params and remote.data.params[key] != value:
                    raise UploadError("서버의 설정 값이 다릅니다. 새 결과 폴더를 사용하세요.")
                if key not in remote.data.params:
                    client.log_param(run_id, key, value)
            for key, value in metrics.items():
                if key in remote.data.metrics and remote.data.metrics[key] != value:
                    raise UploadError("서버의 지표 값이 다릅니다. 새 결과 폴더를 사용하세요.")
                if key not in remote.data.metrics:
                    client.log_metric(run_id, key, value)
            for name, file in files.items():
                if name in state["done"]:
                    continue
                # Snapshot before upload so concurrent edits cannot change a registered artifact.
                with tempfile.TemporaryDirectory(prefix="sixseven-upload-") as temp:
                    snapshot = Path(temp) / file.name
                    shutil.copyfile(file, snapshot)
                    if digest(snapshot) != hashes[name]:
                        raise UploadError(
                            "전송 중 파일이 변경됐습니다. 원래 파일을 복구 후 재시도하세요."
                        )
                    parent = Path(name).parent.as_posix()
                    client.log_artifact(run_id, str(snapshot), None if parent == "." else parent)
                state["done"].append(name)
                atomic_json(state_file, state)
            if set(state["hashes"]) - set(state["done"]):
                raise UploadError(
                    "아직 전송되지 않은 모델이 있습니다. 같은 --model 경로를 지정해 재시도하세요."
                )
            client.set_terminated(run_id, "FINISHED")
            return {
                "run_id": run_id,
                "url": f"{config['uri'].rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}",
                "files": len(files),
                "metrics": len(metrics),
                "params": len(params),
            }
        except UploadError:
            raise
        except Exception as error:
            code = getattr(error, "error_code", type(error).__name__)
            # Remote exception messages may echo authorization or request URLs.
            raise UploadError(
                f"전송 실패 ({code}). 서버 주소·접속·계정 권한을 확인하고 같은 명령으로 재시도하세요. 결과 파일과 전송 상태는 유지됩니다."
            ) from None
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
