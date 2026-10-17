"""Fixed, credential-free explanations for tracking failures."""

import errno
import socket

from filelock import Timeout as LockTimeout

_DETAILS = {
    "auth": (
        "MLflow 인증에 실패했습니다.",
        ".env의 계정·인증 정보를 확인하세요. 변경했다면 학습·저장 후 환경을 종료하고 다시 만든 뒤 dev sync를 실행하세요.",
    ),
    "permission": (
        "MLflow 실험에 쓰기 권한이 없습니다.",
        "서버 관리자에게 해당 실험의 쓰기 권한을 확인받은 뒤 dev sync를 실행하세요.",
    ),
    "local_permission": (
        "로컬 결과 폴더에 접근할 수 없습니다.",
        "artifacts 폴더의 읽기·쓰기 권한을 확인한 뒤 dev sync를 실행하세요.",
    ),
    "network": (
        "MLflow 서버에 연결할 수 없습니다.",
        "서버·네트워크/VPN·접속 주소를 확인하세요. .env를 바꿨다면 환경을 종료하고 다시 만든 뒤 dev sync를 실행하세요.",
    ),
    "server": (
        "MLflow 서버가 요청을 처리하지 못했습니다.",
        "서버 관리자에게 상태를 확인받은 뒤 dev sync를 실행하세요.",
    ),
    "timeout": (
        "MLflow 요청의 대기 시간이 초과됐습니다.",
        "서버 응답을 확인하세요. .env의 MLFLOW_HTTP_REQUEST_TIMEOUT을 바꿨다면 환경을 종료하고 다시 만든 뒤 dev sync를 실행하세요.",
    ),
    "locked": (
        "다른 프로세스가 이 실행 결과를 전송 중입니다.",
        "현재 전송이 끝날 때까지 기다리세요. 자동 재전송 또는 dev sync로 다시 확인할 수 있습니다.",
    ),
    "bundle_changed": (
        "저장 파일이 누락·변경됐거나 서버의 결과와 일치하지 않습니다.",
        "원래 저장 파일을 복원하거나 새 실행으로 저장하세요. 완료된 실행 폴더는 직접 수정하지 마세요.",
    ),
    "remote_deleted": (
        "MLflow에서 이 실행이 삭제되어 공유·비교할 수 없습니다.",
        "계속 공유할 결과라면 서버 관리자에게 해당 실행의 복원을 요청한 뒤 dev sync를 실행하세요. 자동으로 복원하거나 새 실행을 만들지 않습니다.",
    ),
    "configuration": (
        "MLflow 연결 설정이 올바르지 않습니다.",
        ".env의 MLFLOW_TRACKING_URI와 MLFLOW_EXPERIMENT_NAME을 확인하고 환경을 종료·재생성한 뒤 dev sync를 실행하세요.",
    ),
    "unknown": (
        "전송에 실패했지만 원인을 안전하게 식별할 수 없습니다.",
        "dev sync로 재시도하세요. 반복되면 실행 ID를 기준으로 서버 관리자에게 확인을 요청하세요.",
    ),
}


def failure_details(error: BaseException, *, bundle_error: bool = False) -> dict[str, str]:
    """Classify structured codes/types only; never read an exception message."""
    if bundle_error:
        code = "bundle_changed"
    else:
        code = _classify_chain(error)
    message, action = _DETAILS[code]
    return {"code": code, "message": message, "action": action}


def _classify_chain(error: BaseException) -> str:
    visited = set()
    codes = set()
    pending = [error]
    while pending and len(visited) < 32:
        current = pending.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))
        code = _classify(current)
        if code is not None:
            codes.add(code)
        # HTTP wrappers often retain the useful exception only in __context__.
        for nested in (current.__cause__, current.__context__, getattr(current, "reason", None)):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return next(
        (
            code
            for code in (
                "configuration",
                "remote_deleted",
                "auth",
                "permission",
                "local_permission",
                "locked",
                "timeout",
                "network",
                "server",
            )
            if code in codes
        ),
        "unknown",
    )


def _classify(error: BaseException) -> str | None:
    if isinstance(error, LockTimeout):
        return "locked"
    diagnostic_code = getattr(error, "diagnostic_code", None)
    if diagnostic_code in ("configuration", "remote_deleted"):
        return diagnostic_code
    server_code = getattr(error, "error_code", None)
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", getattr(error, "status_code", None))
    if server_code in ("UNAUTHENTICATED", "CUSTOMER_UNAUTHORIZED") or status == 401:
        return "auth"
    if server_code == "PERMISSION_DENIED" or status == 403:
        return "permission"
    if server_code == "DEADLINE_EXCEEDED" or status in (408, 504):
        return "timeout"
    if server_code in ("TEMPORARILY_UNAVAILABLE", "UNAVAILABLE") or status in (500, 502, 503):
        return "server"
    kinds = {f"{kind.__module__}.{kind.__name__}" for kind in type(error).__mro__}
    if isinstance(error, TimeoutError) or kinds & {
        "requests.exceptions.Timeout",
        "urllib3.exceptions.TimeoutError",
        "httpx.TimeoutException",
    }:
        return "timeout"
    if isinstance(error, PermissionError):
        return "local_permission"
    if (
        isinstance(error, (ConnectionError, socket.gaierror))
        or kinds
        & {
            "requests.exceptions.ConnectionError",
            "urllib3.exceptions.NewConnectionError",
            "urllib3.exceptions.ProtocolError",
            "httpx.NetworkError",
        }
        or isinstance(error, OSError)
        and error.errno
        in {
            errno.ECONNREFUSED,
            errno.ECONNRESET,
            errno.ENETUNREACH,
            errno.EHOSTUNREACH,
            errno.ENETDOWN,
            errno.ENOTCONN,
        }
    ):
        return "network"
    return None
