# ruff: noqa: F811
"""Actual authenticated MLflow registration, retry and model file persistence."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError

import pytest
from docker_support import tracking_server  # noqa: F401
from filelock import Timeout
from mlflow import MlflowClient

from sixletters.local_settings import write_settings
from sixletters.upload import UploadError, upload

pytestmark = pytest.mark.skipif(os.environ.get("RUN_DOCKER_TESTS") != "1", reason="Requires Docker")


def test_registration_retry_models_and_auth(tracking_server, tmp_path, monkeypatch):
    server = tracking_server
    for key in list(os.environ):
        if key.startswith("MLFLOW_TRACKING_") or key == "MLFLOW_EXPERIMENT_NAME":
            monkeypatch.delenv(key)
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    with pytest.raises(HTTPError):
        server["request"](
            "/api/2.0/mlflow/experiments/search", {"max_results": 10}, credentials=None
        )
    root = tmp_path.resolve()
    settings = {
        "owner": "alice",
        "uri": server["uri"],
        "username": server["admin"],
        "password": server["password"],
    }
    write_settings(root, settings)
    directory = root / "Members/alice/results/run1"
    (directory / "artifacts").mkdir(parents=True)
    (directory / "metrics.json").write_text('{"roc_auc": 0.75, "sensitivity": 0.8}')
    (directory / "params.json").write_text('{"model": "custom", "layers": [8, 4]}')
    (directory / "artifacts/report.txt").write_text("Synthetic integration report")
    first = upload(root, directory)
    assert upload(root, directory)["run_id"] == first["run_id"]

    def send():
        try:
            return upload(root, directory)["run_id"]
        except Timeout:
            return "locked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert set(pool.map(lambda _: send(), range(2))) <= {first["run_id"], "locked"}
    model = root / "model.bin"
    model.write_bytes(b"test-only model artifact; never executed")
    assert upload(root, directory, model)["run_id"] == first["run_id"]
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", server["admin"])
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", server["password"])
    client = MlflowClient(tracking_uri=server["uri"])
    run = client.get_run(first["run_id"])
    assert run.data.metrics["roc_auc"] == 0.75
    assert run.data.params["layers"] == "[8, 4]"
    assert len(client.search_runs([run.info.experiment_id])) == 1
    assert len(client.get_metric_history(first["run_id"], "roc_auc")) == 1
    downloaded = client.download_artifacts(first["run_id"], "models/model.bin", str(tmp_path))
    assert Path(downloaded).read_bytes() == model.read_bytes()
    # A failed connection can be retried without creating another run.
    server["run"]("stop")
    with pytest.raises(UploadError):
        upload(root, directory)
    server["run"]("up", "-d", "--wait")
    assert upload(root, directory)["run_id"] == first["run_id"]
    assert len(client.search_runs([run.info.experiment_id])) == 1
    assert json.loads((directory / "metrics.json").read_text())["roc_auc"] == 0.75

    (root / "Global").mkdir()
    (root / "Global/pyproject.toml").touch()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "sixletters", "upload", str(directory)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert first["run_id"] in completed.stdout
    assert server["password"] not in completed.stdout + completed.stderr
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "incorrect-password")
    with pytest.raises(UploadError):
        upload(root, directory)
