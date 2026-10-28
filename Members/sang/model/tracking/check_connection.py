"""Check the effective tracking address and personal account before training."""

from __future__ import annotations

import base64
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sixletters.local_settings import connection, read_settings, validate_owner

from ..config import REPO_ROOT


def check_mlflow_connection() -> dict:
    """Verify authentication and shared experiment access without creating a run."""
    settings = read_settings(REPO_ROOT)
    owner = validate_owner(settings.get("owner", ""))
    config = connection(REPO_ROOT)
    if not config["username"] or not config["password"]:
        raise ValueError("MLflow 개인 계정·비밀번호를 먼저 dev setup으로 설정하세요.")
    token = base64.b64encode(f"{config['username']}:{config['password']}".encode()).decode()

    def request(endpoint, body=None):
        req = Request(
            config["uri"].rstrip("/") + "/api/2.0/mlflow/" + endpoint,
            data=None if body is None else json.dumps(body).encode(),
            headers={
                "Authorization": "Basic " + token,
                "Content-Type": "application/json",
            },
            method="GET" if body is None else "POST",
        )
        try:
            with urlopen(req, timeout=10) as response:
                return json.loads(response.read())
        except HTTPError as error:
            raise ConnectionError(
                f"MLflow HTTP {error.code}: 주소·개인 계정·비밀번호·권한을 확인하세요."
            ) from None
        except (URLError, TimeoutError):
            raise ConnectionError(
                "MLflow에 연결할 수 없습니다. 서버 연결 상태·주소·TCP 5000을 확인하세요."
            ) from None

    user = request("users/get?" + urlencode({"username": config["username"]}))["user"]
    if user["username"] != config["username"]:
        raise ValueError("MLflow 인증 계정이 설정과 다릅니다.")
    experiments = request("experiments/search", {"max_results": 1000})["experiments"]
    if not any(e["name"] == config["experiment"] for e in experiments):
        raise ValueError("설정한 공용 MLflow 실험을 조회할 수 없습니다.")
    result = {
        "uri": config["uri"],
        "username": config["username"],
        "owner": owner,
        "experiment": config["experiment"],
        "environment_overrides": [
            key
            for key in (
                "MLFLOW_TRACKING_URI",
                "MLFLOW_TRACKING_USERNAME",
                "MLFLOW_TRACKING_PASSWORD",
                "MLFLOW_EXPERIMENT_NAME",
            )
            if os.environ.get(key)
        ],
    }
    print("MLflow 연결 확인:", json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    check_mlflow_connection()
