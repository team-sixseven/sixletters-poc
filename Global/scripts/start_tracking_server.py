"""Run the dedicated, authenticated MLflow server and create team accounts."""

from __future__ import annotations

import argparse
import configparser
import getpass
import os
import re
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path("/var/lib/mlflow")
LOCAL_URI = "http://127.0.0.1:5000"


def validate_username(username: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", username):
        raise ValueError("계정 이름은 영문·숫자·밑줄·점·하이픈 1~64자로 입력하세요.")
    return username


def validate_password(password: str) -> str:
    if (
        len(password) < 12
        or password.lower() in {"password1234", "changeme12345", "your-password"}
        or len(set(password)) < 4
    ):
        raise ValueError("기본 예시가 아닌 12자 이상의 비밀번호를 설정하세요.")
    return password


def configure_server(data_dir: Path = DATA_DIR) -> list[str]:
    """Build persistent, secret-free auth settings before starting MLflow."""
    admin = validate_username(os.environ.get("MLFLOW_AUTH_ADMIN_USERNAME", ""))
    password = os.environ.get("MLFLOW_AUTH_ADMIN_PASSWORD", "")
    auth_db = data_dir / "auth.db"
    if password:
        validate_password(password)
    elif not auth_db.is_file() or auth_db.stat().st_size == 0:
        raise ValueError("최초 실행에는 MLFLOW_AUTH_ADMIN_PASSWORD가 필요합니다.")
    secret = os.environ.get("MLFLOW_FLASK_SERVER_SECRET_KEY", "")
    if len(secret) < 32 or len(set(secret)) < 8:
        raise ValueError("MLFLOW_FLASK_SERVER_SECRET_KEY에 32자 이상의 무작위 값을 설정하세요.")
    allowed_hosts = os.environ.get("MLFLOW_SERVER_ALLOWED_HOSTS", "localhost:*,127.0.0.1:*")
    if not allowed_hosts or any(host.strip() == "*" for host in allowed_hosts.split(",")):
        raise ValueError("MLFLOW_SERVER_ALLOWED_HOSTS에는 허용할 호스트를 직접 지정하세요.")

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "artifacts").mkdir(exist_ok=True)
    config_path = data_dir / "auth.ini"
    config = configparser.ConfigParser(interpolation=None)
    config["mlflow"] = {
        "default_permission": "EDIT",
        "database_uri": f"sqlite:///{auth_db.as_posix()}",
        "admin_username": admin,
        "authorization_function": "mlflow.server.auth:authenticate_request_basic_auth",
        "grant_default_workspace_access": "true",
    }
    with config_path.open("w", encoding="utf-8") as stream:
        config.write(stream)
    config_path.chmod(0o600)
    os.environ["MLFLOW_AUTH_CONFIG_PATH"] = str(config_path)
    os.environ["MLFLOW_SERVER_ALLOWED_HOSTS"] = allowed_hosts
    return [
        "mlflow",
        "server",
        "--app-name",
        "basic-auth",
        "--host",
        "0.0.0.0",
        "--port",
        "5000",
        "--workers",
        "1",
        "--backend-store-uri",
        f"sqlite:///{(data_dir / 'tracking.db').as_posix()}",
        "--serve-artifacts",
        "--artifacts-destination",
        str(data_dir / "artifacts"),
        "--default-artifact-root",
        "mlflow-artifacts:/",
    ]


def create_user(username: str) -> None:
    """Create a non-admin user without placing passwords in shell history."""
    validate_username(username)
    admin = validate_username(os.environ.get("MLFLOW_AUTH_ADMIN_USERNAME", ""))
    if username == admin:
        raise ValueError("관리자와 다른 팀원 계정 이름을 입력하세요.")
    admin_password = getpass.getpass("관리자 비밀번호: ")
    password = validate_password(getpass.getpass("새 팀원 비밀번호 (12자 이상): "))
    if password != getpass.getpass("새 팀원 비밀번호 확인: "):
        raise ValueError("비밀번호 확인이 일치하지 않습니다.")
    # Imported only by this action so configuration checks need no MLflow dependency.
    from mlflow.exceptions import MlflowException
    from mlflow.server import get_app_client

    os.environ["MLFLOW_TRACKING_USERNAME"] = admin
    os.environ["MLFLOW_TRACKING_PASSWORD"] = admin_password
    try:
        client = get_app_client("basic-auth", tracking_uri=LOCAL_URI)
        client.create_user(username=username, password=password)
    except MlflowException as exc:
        raise ValueError(
            f"계정 생성 실패 ({exc.error_code}). 관리자 인증과 계정 중복을 확인하세요."
        ) from None
    finally:
        os.environ.pop("MLFLOW_TRACKING_PASSWORD", None)
    print(f"팀원 계정 생성 완료: {username} (관리자 권한 없음)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("serve")
    subparsers.add_parser("health")
    user_parser = subparsers.add_parser("user")
    user_parser.add_argument("username")
    args = parser.parse_args()
    try:
        if args.action == "serve":
            command = configure_server()
            os.execvp(command[0], command)
        elif args.action == "user":
            create_user(args.username)
        else:
            with urllib.request.urlopen(f"{LOCAL_URI}/health", timeout=3) as response:
                return 0 if response.status == 200 else 1
    except (ValueError, OSError) as exc:
        print(f"MLflow: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
