"""Server bootstrap validates credentials without copying secrets into config."""

import configparser
import importlib.util
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/start_tracking_server.py"
spec = importlib.util.spec_from_file_location("tracking_startup", SCRIPT)
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("MLFLOW_AUTH_CONFIG_PATH", "")
    monkeypatch.setenv("MLFLOW_AUTH_ADMIN_USERNAME", "team-admin")
    monkeypatch.setenv("MLFLOW_AUTH_ADMIN_PASSWORD", "OnlyForTests-Long-314159")
    monkeypatch.setenv("MLFLOW_FLASK_SERVER_SECRET_KEY", "test-only-0123456789-abcdefghijklmnop")
    monkeypatch.setenv("MLFLOW_SERVER_ALLOWED_HOSTS", "localhost:*,127.0.0.1:*,100.64.1.2:5000")


def test_bootstrap_uses_auth_proxy_artifacts_and_separate_persistent_stores(tmp_path, configured):
    command = startup.configure_server(tmp_path)
    assert command[:4] == ["mlflow", "server", "--app-name", "basic-auth"]
    assert command[command.index("--backend-store-uri") + 1] == (
        f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    )
    assert command[command.index("--artifacts-destination") + 1] == str(tmp_path / "artifacts")
    assert command[command.index("--default-artifact-root") + 1] == "mlflow-artifacts:/"
    config_path = tmp_path / "auth.ini"
    config = configparser.ConfigParser()
    config.read(config_path)
    assert config["mlflow"]["database_uri"] == f"sqlite:///{(tmp_path / 'auth.db').as_posix()}"
    assert config["mlflow"]["default_permission"] == "EDIT"
    text = config_path.read_text()
    assert os.environ["MLFLOW_AUTH_ADMIN_PASSWORD"] not in text
    assert os.environ["MLFLOW_FLASK_SERVER_SECRET_KEY"] not in text
    assert os.environ["MLFLOW_AUTH_CONFIG_PATH"] == str(config_path)


@pytest.mark.parametrize("password", ["", "password1234", "changeme12345", "x" * 20, "short"])
def test_bootstrap_rejects_missing_default_or_weak_password(
    tmp_path, configured, monkeypatch, password
):
    monkeypatch.setenv("MLFLOW_AUTH_ADMIN_PASSWORD", password)
    with pytest.raises(ValueError):
        startup.configure_server(tmp_path)
    assert not (tmp_path / "auth.ini").exists()


@pytest.mark.parametrize("secret", ["", "short", "x" * 40])
def test_bootstrap_rejects_weak_csrf_secret(tmp_path, configured, monkeypatch, secret):
    monkeypatch.setenv("MLFLOW_FLASK_SERVER_SECRET_KEY", secret)
    with pytest.raises(ValueError, match="MLFLOW_FLASK_SERVER_SECRET_KEY"):
        startup.configure_server(tmp_path)


def test_restart_keeps_stores_and_accepts_removed_bootstrap_password(
    tmp_path, configured, monkeypatch
):
    (tmp_path / "auth.db").write_bytes(b"existing authentication database")
    (tmp_path / "tracking.db").write_bytes(b"existing experiment database")
    monkeypatch.delenv("MLFLOW_AUTH_ADMIN_PASSWORD")
    startup.configure_server(tmp_path)
    assert (tmp_path / "auth.db").read_bytes() == b"existing authentication database"
    assert (tmp_path / "tracking.db").read_bytes() == b"existing experiment database"


@pytest.mark.parametrize("username", ["", "team:admin", "name\nconfig=true", "name with spaces"])
def test_bootstrap_rejects_invalid_usernames(tmp_path, configured, monkeypatch, username):
    monkeypatch.setenv("MLFLOW_AUTH_ADMIN_USERNAME", username)
    with pytest.raises(ValueError):
        startup.configure_server(tmp_path)


def test_unrestricted_host_wildcard_is_rejected(tmp_path, configured, monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ALLOWED_HOSTS", "localhost:*,*")
    with pytest.raises(ValueError, match="MLFLOW_SERVER_ALLOWED_HOSTS"):
        startup.configure_server(tmp_path)


def test_user_creation_rejects_mismatched_passwords_before_network(configured, monkeypatch):
    passwords = iter(["admin-password", "UniqueNewMember-123", "wrong-confirmation"])
    monkeypatch.setattr(startup.getpass, "getpass", lambda _: next(passwords))
    with pytest.raises(ValueError, match="일치"):
        startup.create_user("member1")


def test_example_has_no_preset_credentials():
    example = (SCRIPT.parents[2] / ".env.tracking.example").read_text()
    assignments = dict(
        line.split("=", 1) for line in example.splitlines() if line and not line.startswith("#")
    )
    for name in (
        "MLFLOW_AUTH_ADMIN_USERNAME",
        "MLFLOW_AUTH_ADMIN_PASSWORD",
        "MLFLOW_FLASK_SERVER_SECRET_KEY",
    ):
        assert assignments[name] == ""
    assert assignments["POC_MLFLOW_BIND_HOST"] == "127.0.0.1"
