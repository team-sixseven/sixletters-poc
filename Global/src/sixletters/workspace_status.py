"""Read-only readiness summaries; never connect, install, upload, or repair."""

import hashlib
import json
from pathlib import Path

from .local_settings import read_settings
from .run_catalog import display_text


def show_environment_status(status: dict) -> None:
    labels = {
        "ready": "준비 완료",
        "install-needed": "설치 필요",
        "update-needed": "환경 갱신 필요",
    }
    print(f"상태: {labels.get(status['state'], status['state'])}")
    if status.get("python"):
        print(f"Python: {display_text(str(status['python']), 500)}")
    if status.get("reason"):
        print(display_text(status["reason"], 500))
    for item in status.get("active", []):
        print(
            f"사용 중: {display_text(item.get('experiment', 'unknown'))} · "
            f"{display_text(item.get('kind', 'process'))} · PID {item.get('pid', '?')}"
        )
    if status.get("next_command"):
        print(f"다음: {status['next_command']}")


def image_dependency_status(root: Path, manifest: Path) -> str:
    if not manifest.is_file():
        return "outside-container"
    try:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, name = line.split(maxsplit=1)
            relative = Path(name.lstrip("*"))
            if relative.is_absolute() or ".." in relative.parts:
                return "update-needed"
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                return "update-needed"
    except (OSError, ValueError):
        return "update-needed"
    return "ready"


def transfer_status(root: Path) -> dict:
    """Summarize local delivery receipts for the configured destination only."""
    from .tracking import _completed_markers, _configuration, _destination_key

    configuration = _configuration()
    summary = {
        "enabled": configuration is not None,
        "saved": 0,
        "synced": 0,
        "pending": 0,
        "archive_pending": 0,
        "invalid": 0,
    }
    issues = []
    for marker in _completed_markers(root, issues):
        summary["saved"] += 1
        if configuration is None:
            continue
        state = root / ".tracking" / _destination_key(*configuration) / f"{marker.parent.name}.json"
        try:
            if state.is_symlink() or any(
                path.is_symlink() for path in (state.parent, state.parent.parent)
            ):
                raise ValueError("Unsafe receipt")
            record = json.loads(state.read_text(encoding="utf-8")) if state.is_file() else {}
            identity = list(marker.parent.relative_to(root).parts)
            if (
                record.get("schema_version") == 2
                and record.get("local_identity") == identity
                and record.get("status") == "synced"
            ):
                summary["synced"] += 1
            else:
                summary["pending"] += 1
            if (
                record.get("archive_requested") or state.with_suffix(".archive-request").is_file()
            ) and record.get("archive_status") != "complete":
                summary["archive_pending"] += 1
        except (OSError, ValueError, TypeError, AttributeError):
            summary["invalid"] += 1
    summary["invalid"] += len(issues)
    return summary


def show_workspace_status(
    root: Path, *, manifest: Path = Path("/opt/poc/dependencies.sha256")
) -> None:
    image = image_dependency_status(root, manifest)
    print(
        "공통 이미지: "
        + {
            "ready": "현재 패키지 설정과 일치",
            "update-needed": "갱신 필요",
            "outside-container": "컨테이너 밖에서 실행하여 확인 불가",
        }[image]
    )
    if image == "update-needed":
        print("다음: 학습·저장 후 ./dev down → ./dev up -d --wait (GPU는 --gpu)")
    try:
        settings = read_settings(root)
    except (ValueError, OSError):
        settings = {}
        print("개인 설정을 읽을 수 없습니다. 다음: ./dev init")
    if not settings:
        print("이름·데이터: 미등록. 다음: ./dev music → ./dev init")
    else:
        print(f"이름: {display_text(settings['owner'])}")
        location = root / settings["data"]
        ready = (location / "dataset.json").is_file()
        print(
            f"데이터: {display_text(settings['data'], 240)} · "
            + ("준비 파일 있음" if ready else "준비 필요")
        )
        print("데이터 무결성·환자 분할은 학습·저장 시 다시 검증합니다.")
        if not ready:
            print("다음: ./dev music 후 ./dev init에서 준비된 데이터 경로 확인")
        from .personal_env import _env_command, env_status

        experiments = root / "Members" / settings["owner"] / "experiments"
        if experiments.is_dir() and not experiments.is_symlink():
            for directory in sorted(experiments.iterdir()):
                if (
                    directory.is_dir()
                    and not directory.is_symlink()
                    and (directory / "config.toml").is_file()
                ):
                    print(f"실험: {display_text(directory.name)}")
                    try:
                        show_environment_status(env_status(directory))
                    except (ValueError, OSError):
                        print(f"환경 기록 확인 필요. 다음: {_env_command(directory, 'status')}")
    try:
        status = transfer_status(root / "Global/artifacts")
        if not status["enabled"]:
            print(f"MLflow: 미연결 · 로컬 결과 {status['saved']}개. 로컬 학습·비교 가능.")
            print("팀 공유 시 Global/docs/tracking.md의 연결 설정을 적용하세요.")
        else:
            print(
                f"MLflow: 연결 설정 있음 (서버 접속은 확인하지 않음)\n"
                f"로컬 전송 기록: 완료 {status['synced']} · 대기 {status['pending']} · "
                f"모델 보관 대기 {status['archive_pending']} · 확인 필요 {status['invalid']}"
            )
            print("서버의 현재 상태와 실패 원인 확인: ./dev sync")
    except (ValueError, OSError):
        print("MLflow 설정·전송 기록 확인 필요. .env 설정 확인 후 ./dev sync")
