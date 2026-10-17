"""Convenience commands remain read-only until an explicit result selection."""

import hashlib
import json
import sys
import tomllib
from pathlib import Path

import pytest

from sixletters import __main__ as cli
from sixletters import reproduction, tracking, workspace_status
from sixletters.local_settings import write_settings
from sixletters.workspace import init_experiment


def test_interactive_mistake_retries_same_catalog_and_cancel_is_explicit(monkeypatch):
    answers = iter(["9", "a", "1 1", "2 1"])
    monkeypatch.setattr(cli, "prompt", lambda *_: next(answers))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert cli._choose_indexes(2) == [1, 0]
    monkeypatch.setattr(cli, "prompt", lambda *_: "q")
    with pytest.raises(cli.SelectionCancelled):
        cli._choose_indexes(2)


def test_noninteractive_invalid_selection_fails_without_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "prompt", lambda *_: calls.append(1) or "9")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(ValueError):
        cli._choose_indexes(2)
    assert calls == [1]


def test_interactive_empty_input_retries_but_eof_fails(monkeypatch):
    answers = iter(["", "1 2"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert cli._choose_indexes(2) == [0, 1]

    def end_input(*_):
        raise EOFError

    monkeypatch.setattr("builtins.input", end_input)
    with pytest.raises(ValueError, match="No input"):
        cli._choose_indexes(2)


def test_final_status_does_not_migrate_legacy_data(tmp_path, monkeypatch):
    migration = tmp_path / "Global/scripts/migrate_layout.py"
    migration.parent.mkdir(parents=True)
    migration.write_text("raise AssertionError('must not migrate')")
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["dev", "final", "status", "missing", "--data", "Global/data/demo"]
    )
    monkeypatch.setattr(cli.subprocess, "run", lambda *_a, **_kw: pytest.fail("migration"))
    monkeypatch.setattr(
        reproduction,
        "final_plan_status",
        lambda *_: (_ for _ in ()).throw(ValueError("missing plan")),
    )
    with pytest.raises(SystemExit, match="2"):
        cli.main()


@pytest.mark.parametrize(
    "arguments",
    [
        ["compare", "--remote", "--list"],
        ["archive", "--list"],
        ["restore", "retry001", "--remote", "--list"],
        ["final", "freeze", "planned", "--remote", "--list"],
    ],
)
def test_catalog_list_never_prompts_uploads_downloads_or_creates_plan(
    tmp_path, monkeypatch, arguments
):
    write_settings(tmp_path, "member", Path("Global/data/demo"))
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["dev", *arguments])
    monkeypatch.setattr(cli, "prompt", lambda *_: pytest.fail("prompted"))
    monkeypatch.setattr(tracking, "list_remote_runs", lambda **_: [])
    monkeypatch.setattr(tracking, "fetch_runs", lambda *_a, **_k: pytest.fail("downloaded"))
    monkeypatch.setattr(tracking, "archive_run", lambda *_: pytest.fail("uploaded"))
    monkeypatch.setattr(reproduction, "fetch_archive", lambda *_: pytest.fail("downloaded archive"))
    assert cli.main() == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == [".poc.local.toml"]


def test_restore_selects_only_complete_archives_and_preserves_destination_owner(
    tmp_path, monkeypatch
):
    write_settings(tmp_path, "destination", Path("Global/data/demo"))
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dev",
            "restore",
            "retry001",
            "--remote",
            "--source-owner",
            "source",
            "--experiment",
            "exp001",
        ],
    )
    base = {
        "owner": "source",
        "experiment": "exp001",
        "dataset_version": "v1",
        "feature_set": "clinical",
        "metrics": {},
        "parameters": {},
        "model": "m",
        "created_at": "2026-10-18",
    }
    entries = [
        {**base, "run_id": "a" * 32, "local_run_id": "1", "archive_status": "none"},
        {**base, "run_id": "b" * 32, "local_run_id": "2", "archive_status": "complete"},
    ]
    monkeypatch.setattr(tracking, "list_remote_runs", lambda **_: entries)
    monkeypatch.setattr(cli, "prompt", lambda *_: "1")
    downloaded = []
    monkeypatch.setattr(
        reproduction,
        "fetch_archive",
        lambda _, run_id: downloaded.append(run_id) or tmp_path / "run",
    )
    restored = []
    monkeypatch.setattr(
        reproduction,
        "restore_run",
        lambda source, **kwargs: restored.append((source, kwargs)) or tmp_path / "config.toml",
    )
    assert cli.main() == 0
    assert downloaded == ["b" * 32]
    assert restored[0][1]["owner"] == "destination"
    assert restored[0][1]["name"] == "retry001"


def test_clone_cli_keeps_personal_code_and_new_identity(tmp_path, monkeypatch):
    write_settings(tmp_path, "member", Path("Global/data/demo"))
    config = init_experiment("member", "old", root=tmp_path / "Members", script="fit.py")
    config.with_name("helper.py").write_text("VALUE = 42\n")
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["dev", "new", "new", "--from", "old"])
    assert cli.main() == 0
    target = config.parent.parent / "new"
    assert target.joinpath("helper.py").read_text() == "VALUE = 42\n"
    assert target.joinpath("fit.py").is_file()
    assert tomllib.loads(target.joinpath("config.toml").read_text())["experiment"]["name"] == "new"


def test_status_is_read_only_and_does_not_print_connection_secrets(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://private.example")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "never-print-me")
    monkeypatch.setattr(tracking, "_client", lambda *_: pytest.fail("network"))
    before = list(tmp_path.rglob("*"))
    workspace_status.show_workspace_status(tmp_path, manifest=tmp_path / "absent")
    text = capsys.readouterr().out
    assert "never-print-me" not in text and "private.example" not in text
    assert "미등록" in text and "서버 접속은 확인하지 않음" in text
    assert list(tmp_path.rglob("*")) == before


def test_status_matches_current_destination_only(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://one.example")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "team")
    root = tmp_path / "artifacts"
    run = root / "member" / "exp001" / ("a" * 32)
    run.mkdir(parents=True)
    (run / "result.json").write_text("{}")
    state_dir = root / ".tracking" / tracking._destination_key("https://one.example", "team")
    state_dir.mkdir(parents=True)
    (state_dir / f"{run.name}.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "local_identity": ["member", "exp001", run.name],
                "status": "synced",
            }
        )
    )
    assert workspace_status.transfer_status(root)["synced"] == 1
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://two.example")
    assert workspace_status.transfer_status(root)["pending"] == 1


def test_image_readiness_explains_changed_dependencies(tmp_path):
    dependency = tmp_path / "requirements.txt"
    dependency.write_text("one")
    manifest = tmp_path / "manifest"
    manifest.write_text(hashlib.sha256(b"one").hexdigest() + "  requirements.txt\n")
    assert workspace_status.image_dependency_status(tmp_path, manifest) == "ready"
    dependency.write_text("two")
    assert workspace_status.image_dependency_status(tmp_path, manifest) == "update-needed"
