"""Private, checkout-local owner and tracking connection settings."""

import json
import os
import re
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

SETTINGS = ".poc.local.toml"


def project_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "Global/pyproject.toml").is_file():
            return candidate
    raise ValueError("저장소 안에서 실행하세요.")


def read_settings(root: Path) -> dict:
    file = root / SETTINGS
    if file.is_symlink():
        raise ValueError("설정 파일은 심볼릭 링크일 수 없습니다.")
    return tomllib.loads(file.read_text(encoding="utf-8")) if file.exists() else {}


def validate_owner(owner: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", owner):
        raise ValueError("이름은 영문 소문자로 시작하는 1~40자의 영문·숫자·_·- ID를 사용하세요.")
    return owner


def connection(root: Path) -> dict[str, str]:
    settings = read_settings(root)
    result = {
        key: (os.environ.get(env) or settings.get(key, ""))
        for key, env in {
            "uri": "MLFLOW_TRACKING_URI",
            "username": "MLFLOW_TRACKING_USERNAME",
            "password": "MLFLOW_TRACKING_PASSWORD",
            "experiment": "MLFLOW_EXPERIMENT_NAME",
        }.items()
    }
    result["experiment"] = result["experiment"] or "sixseven-poc"
    parsed = urlsplit(result["uri"])
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "dev setup에서 http(s) 서버 주소를 설정하세요. 주소에 계정·비밀번호를 넣지 마세요."
        )
    return result


def write_settings(root: Path, values: dict[str, str]) -> Path:
    validate_owner(values["owner"])
    destination = root / SETTINGS
    if destination.is_symlink():
        raise ValueError("설정 파일은 심볼릭 링크일 수 없습니다.")
    fd, temporary = tempfile.mkstemp(prefix=".settings-", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(
                "\n".join(
                    f"{key} = {json.dumps(value, ensure_ascii=False)}"
                    for key, value in values.items()
                )
                + "\n"
            )
        Path(temporary).replace(destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination
