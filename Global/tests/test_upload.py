"""Result validation, recovery and public repository boundaries."""

from types import SimpleNamespace

import pytest
from filelock import Timeout

from scripts.check_repository import violations
from sixletters import upload as module
from sixletters.local_settings import connection, read_settings, write_settings


@pytest.fixture
def result(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    for key in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_EXPERIMENT_NAME",
    ):
        monkeypatch.delenv(key, raising=False)
    write_settings(
        root, {"owner": "alice", "uri": "http://localhost:5000", "password": "private-value"}
    )
    directory = root / "Members/alice/results/one"
    (directory / "artifacts").mkdir(parents=True)
    (directory / "metrics.json").write_text('{"roc_auc":0.7}')
    (directory / "params.json").write_text('{"model":"custom"}')
    (directory / "artifacts/report.txt").write_text("report")
    return root, directory


class Client:
    def __init__(self):
        self.runs = []
        self.files = {}
        self.failure = False
        self.creates = 0
        self.metric_calls = 0

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="1")

    def search_runs(self, *args, **kwargs):
        return self.runs

    def create_run(self, experiment_id, tags):
        self.creates += 1
        run = SimpleNamespace(
            info=SimpleNamespace(run_id="run1", lifecycle_stage="active"),
            data=SimpleNamespace(params={}, metrics={}, tags=tags),
        )
        self.runs.append(run)
        return run

    def get_run(self, run_id):
        return self.runs[0]

    def log_param(self, run_id, key, value):
        self.runs[0].data.params[key] = value

    def log_metric(self, run_id, key, value):
        self.runs[0].data.metrics[key] = value
        self.metric_calls += 1

    def set_tag(self, run_id, key, value):
        self.runs[0].data.tags[key] = value

    def log_artifact(self, run_id, file, parent):
        from pathlib import Path

        if self.failure:
            self.failure = False
            raise RuntimeError("password=do-not-print")
        path = Path(file)
        self.files[(parent, path.name)] = path.read_bytes()

    def set_terminated(self, *args):
        pass


@pytest.fixture
def client(monkeypatch):
    value = Client()
    monkeypatch.setattr(module, "MlflowClient", lambda **kwargs: value)
    return value


def test_repeat_retry_and_later_model(result, client):
    root, directory = result
    client.failure = True
    with pytest.raises(module.UploadError) as error:
        module.upload(root, directory)
    assert "do-not-print" not in str(error.value)
    assert module.upload(root, directory)["run_id"] == "run1"
    model = root / "weights.bin"
    model.write_bytes(b"opaque-model")
    assert module.upload(root, directory, model)["run_id"] == "run1"
    assert module.upload(root, directory)["run_id"] == "run1"
    assert client.creates == 1 and client.metric_calls == 1
    assert client.files[("models", "weights.bin")] == b"opaque-model"
    assert all(b"private-value" not in data for data in client.files.values())
    model.write_bytes(b"different")
    with pytest.raises(ValueError, match="변경"):
        module.upload(root, directory, model)


def test_remote_manifest_rejects_changed_result_after_local_state_loss(result, client):
    root, directory = result
    module.upload(root, directory)
    for file in (root / ".poc-cache/uploads").glob("*.json"):
        file.unlink()
    (directory / "metrics.json").write_text('{"roc_auc":0.8}')
    with pytest.raises(module.UploadError, match="서버에 등록한 결과"):
        module.upload(root, directory)
    assert client.creates == 1


def test_changed_files_and_removed_files_require_new_folder(result, client):
    root, directory = result
    module.upload(root, directory)
    (directory / "artifacts/report.txt").unlink()
    with pytest.raises(ValueError, match="변경"):
        module.upload(root, directory)


@pytest.mark.parametrize(
    "text", ["{}", "[]", '{"x":true}', '{"x":NaN}', '{"x":Infinity}', '{"x":".7"}', '{"x":1,"x":2}']
)
def test_invalid_metrics_do_not_contact_server(result, client, text):
    root, directory = result
    (directory / "metrics.json").write_text(text)
    with pytest.raises(ValueError):
        module.upload(root, directory)
    assert client.creates == 0


def test_symlink_and_secret_files_rejected(result):
    root, directory = result
    secret = root / "outside"
    secret.write_text("secret")
    link = directory / "artifacts/link.txt"
    link.symlink_to(secret)
    with pytest.raises(ValueError):
        module.bundle(directory)
    link.unlink()
    (directory / "artifacts/.env").write_text("PASSWORD=secret")
    with pytest.raises(ValueError):
        module.bundle(directory)


def test_same_workspace_concurrent_upload_is_locked(result, client, monkeypatch):
    root, directory = result
    original = client.log_artifact

    def attempt(*args):
        with pytest.raises(Timeout):
            module.upload(root, directory)
        original(*args)

    monkeypatch.setattr(client, "log_artifact", attempt)
    module.upload(root, directory)
    assert client.creates == 1


def test_blank_container_env_falls_back_to_local_settings(result, monkeypatch):
    root, _ = result
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    assert connection(root)["uri"] == "http://localhost:5000"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://other.example")
    assert connection(root)["uri"] == "https://other.example"
    assert read_settings(root)["owner"] == "alice"


def test_personal_results_and_models_cannot_be_committed():
    from pathlib import Path

    for name in ("Members/alice/results/one/metrics.json", "Members/alice/models/weights.json"):
        assert violations(Path(name), b"{}")


def test_model_retry_without_model_does_not_claim_complete(result, client):
    root, directory = result
    module.upload(root, directory)
    model = root / "model.bin"
    model.write_bytes(b"model")
    client.failure = True
    with pytest.raises(module.UploadError):
        module.upload(root, directory, model)
    with pytest.raises(module.UploadError, match="--model"):
        module.upload(root, directory)
    assert module.upload(root, directory, model)["run_id"] == "run1"


def test_model_directory_retains_structure(result, client):
    root, directory = result
    model = root / "model"
    (model / "nested").mkdir(parents=True)
    (model / "nested/weights.bin").write_bytes(b"weights")
    (model / "config.json").write_text("{}")
    module.upload(root, directory, model)
    assert client.files[("models/nested", "weights.bin")] == b"weights"
    assert client.files[("models", "config.json")] == b"{}"


def test_ambiguous_create_response_recovers_existing_run(result, client, monkeypatch):
    root, directory = result
    original = client.create_run

    def ambiguous(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("response lost")

    monkeypatch.setattr(client, "create_run", ambiguous)
    with pytest.raises(module.UploadError):
        module.upload(root, directory)
    monkeypatch.setattr(client, "create_run", original)
    assert module.upload(root, directory)["run_id"] == "run1"
    assert client.creates == 1


def test_http_create_retries_disabled_and_environment_restored(result, client, monkeypatch):
    import os

    root, directory = result
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "6")

    def factory(**kwargs):
        assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "0"
        return client

    monkeypatch.setattr(module, "MlflowClient", factory)
    module.upload(root, directory)
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "6"
