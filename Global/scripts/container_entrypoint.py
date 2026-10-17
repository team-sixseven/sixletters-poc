#!/usr/bin/env python3
"""Run container commands as the owner of the mounted checkout."""

from __future__ import annotations

import hashlib
import os
import pwd
import sys
from pathlib import Path

WORKSPACE = Path("/workspace")
DEPENDENCY_HASHES = Path("/opt/poc/dependencies.sha256")


def identity_value(name: str, default: int) -> int:
    value = os.environ.get(name, "")
    if not value:
        return default
    try:
        number = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name}에는 숫자 UID 또는 GID를 지정하세요.") from exc
    if number < 0:
        raise RuntimeError(f"{name}에는 0 이상의 UID 또는 GID를 지정하세요.")
    return number


def use_checkout_owner(workspace: Path = WORKSPACE) -> None:
    owner = workspace.stat()
    # Desktop may report the current container UID for a writable host mount.
    # Probe it as a regular user; a real root-owned Linux checkout stays blocked.
    default_uid = owner.st_uid or 1000
    default_gid = owner.st_gid if owner.st_uid else 1000
    uid = identity_value("POC_UID", default_uid)
    gid = identity_value("POC_GID", default_gid)
    if uid == 0:
        raise RuntimeError(
            "POC_UID=0으로 실행할 수 없습니다. 쓰기 권한이 있는 일반 사용자 UID를 지정하세요."
        )
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
    elif os.geteuid() != uid or os.getegid() != gid:
        raise RuntimeError("컨테이너 사용자와 저장소 소유자가 다릅니다. UID/GID 설정을 확인하세요.")
    if not os.access(workspace, os.W_OK | os.X_OK):
        raise RuntimeError("저장소에 쓸 수 없습니다. WSL 홈의 저장소 소유자와 권한을 확인하세요.")
    try:
        username = pwd.getpwuid(uid).pw_name
    except KeyError:
        username = f"poc-{uid}"
    # Libraries such as Jupyter/getpass also support UIDs without a passwd entry.
    os.environ["USER"] = username
    os.environ["LOGNAME"] = username
    os.environ.setdefault("HOME", "/tmp/poc-home")


def check_dependencies(workspace: Path = WORKSPACE, manifest: Path = DEPENDENCY_HASHES) -> None:
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, filename = line.split(maxsplit=1)
        with (workspace / filename.lstrip("*")).open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(
                "패키지 설정이 이미지와 다릅니다. docker compose up --build를 다시 실행하세요. "
                "GPU: docker compose up --build gpu"
            )


def main() -> int:
    try:
        use_checkout_owner()
        os.chdir(WORKSPACE)
        if os.environ.get("POC_SKIP_DEPENDENCY_CHECK") != "1":
            check_dependencies()
        if len(sys.argv) < 2:
            raise RuntimeError("컨테이너에서 실행할 명령이 필요합니다.")
        os.execvp(sys.argv[1], sys.argv[1:])
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"컨테이너 실행 실패: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
