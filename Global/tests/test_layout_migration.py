"""Existing local datasets and experiments survive the layout upgrade byte for byte."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_layout.py"
spec = importlib.util.spec_from_file_location("migrate_layout", SCRIPT)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def write(root, relative, content=b"original"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_old_data_results_and_member_files_keep_exact_bytes(tmp_path):
    files = {
        "data/prepared/music/features.csv": "환자,특징\nA,1\n".encode(),
        "artifacts/조성률/baseline/run/result.json": b'{"source": "experiments/old/train.py"}',
        "artifacts/조성률/baseline/run/model.pkl": b"\x00\xffbinary model",
        "artifacts/조성률/baseline/run/source/train.py": b"old source snapshot\n",
        "experiments/조성률/baseline/train.py": b"# retain provenance\nDATA = 'data/prepared'\n",
        "experiments/조성률/baseline/config.toml": b'name = "baseline"\n',
        "notebooks/조성률/explore.ipynb": b'{"cells": []}',
    }
    for name, content in files.items():
        write(tmp_path, name, content)
    write(tmp_path, "Global/data/README.md", b"tracked guide")
    write(tmp_path, "Members/조성률/experiments/other/train.py", b"other experiment")

    assert "7 files moved" in migration.migrate_layout(tmp_path)
    for name, content in files.items():
        parts = Path(name).parts
        if parts[0] in {"data", "artifacts"}:
            destination = tmp_path / "Global" / name
        else:
            destination = tmp_path / "Members" / parts[1] / parts[0] / Path(*parts[2:])
        assert destination.read_bytes() == content
        assert not (tmp_path / name).exists()
    assert (tmp_path / "Global/data/README.md").read_bytes() == b"tracked guide"
    assert (
        tmp_path / "Members/조성률/experiments/other/train.py"
    ).read_bytes() == b"other experiment"
    assert migration.migrate_layout(tmp_path) == "Local layout: 0 files moved"


def test_late_conflict_prevents_every_move_and_settings_edit(tmp_path):
    data = write(tmp_path, "data/prepared/features.csv")
    result = write(tmp_path, "artifacts/owner/experiment/run/result.json", b"old result")
    destination = write(
        tmp_path, "Global/artifacts/owner/experiment/run/result.json", b"new result"
    )
    settings = write(tmp_path, ".poc.local.toml", b'owner = "owner"\ndata = "data/prepared"\n')

    with pytest.raises(ValueError, match="Migration conflict"):
        migration.migrate_layout(tmp_path)
    assert data.read_bytes() == b"original"
    assert result.read_bytes() == b"old result"
    assert destination.read_bytes() == b"new result"
    assert not (tmp_path / "Global/data").exists()
    assert settings.read_bytes() == b'owner = "owner"\ndata = "data/prepared"\n'


def test_settings_preserve_owner_comments_and_update_after_moving_data(tmp_path):
    write(tmp_path, "data/prepared/music/features.csv")
    original = '# local\nowner = "조성률"\ndata = "data/prepared/music" # chosen dataset\n'
    settings = write(tmp_path, ".poc.local.toml", original.encode())
    assert "local data setting updated" in migration.migrate_layout(tmp_path)
    assert settings.read_text(encoding="utf-8") == original.replace('"data/', '"Global/data/')
    assert list(tmp_path.glob(".poc.local.*")) == [settings]


@pytest.mark.parametrize("data", ["data/missing", "custom/prepared", "../data/prepared"])
def test_missing_or_unrelated_configured_data_is_not_rewritten(tmp_path, data):
    write(tmp_path, "data/other/features.csv")
    original = f'owner = "owner"\ndata = "{data}"\n'.encode()
    settings = write(tmp_path, ".poc.local.toml", original)
    migration.migrate_layout(tmp_path)
    assert settings.read_bytes() == original


def test_settings_retry_can_finish_after_all_files_have_already_moved(tmp_path):
    write(tmp_path, "Global/data/prepared/features.csv")
    settings = write(tmp_path, ".poc.local.toml", b'owner = "owner"\ndata = "data/prepared"\n')
    assert "local data setting updated" in migration.migrate_layout(tmp_path)
    assert settings.read_bytes() == b'owner = "owner"\ndata = "Global/data/prepared"\n'


@pytest.mark.parametrize(
    "link_path",
    ["data", "data/nested", "Global", "Global/data", "Members/owner", ".poc.local.toml"],
)
def test_symlinks_are_rejected_before_any_files_move(tmp_path, link_path):
    source = write(tmp_path, "artifacts/owner/experiment/run/result.json")
    link = tmp_path / link_path
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    except OSError:
        pytest.skip("Creating symbolic links is not available on this host")
    if link_path == "Members/owner":
        write(tmp_path, "experiments/owner/example/train.py")
    if link_path in {"Global", "Global/data"}:
        write(tmp_path, "data/features.csv")
    with pytest.raises(ValueError, match="Symbolic links"):
        migration.migrate_layout(tmp_path)
    assert source.read_bytes() == b"original"
    assert link.is_symlink()


def test_unowned_notebook_requires_explicit_destination_without_partial_move(tmp_path):
    source = write(tmp_path, "data/features.csv")
    notebook = write(tmp_path, "notebooks/explore.ipynb")
    with pytest.raises(ValueError, match=r"Members/<name>/notebooks/"):
        migration.migrate_layout(tmp_path)
    assert source.exists()
    assert notebook.exists()


def test_cache_and_template_leftovers_are_neither_moved_nor_removed(tmp_path):
    cache = write(tmp_path, "experiments/owner/example/__pycache__/train.pyc")
    template = write(tmp_path, "experiments/_template/train.py")
    notebook_cache = write(tmp_path, "notebooks/.ipynb_checkpoints/old.ipynb")
    train = write(tmp_path, "experiments/owner/example/train.py")
    summary = migration.migrate_layout(tmp_path)
    assert "1 files moved" in summary
    assert "3 cache/template paths left unchanged" in summary
    assert all(path.exists() for path in (cache, template, notebook_cache))
    assert not train.exists()


def test_stopped_link_unlink_is_resumable_without_overwriting(tmp_path, monkeypatch):
    source = write(tmp_path, "data/prepared/features.csv")
    unlink = Path.unlink

    def interrupt(path, *args, **kwargs):
        if path == source:
            raise OSError("simulated interruption after link")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupt)
    with pytest.raises(OSError, match="simulated interruption"):
        migration.migrate_layout(tmp_path)
    target = tmp_path / "Global/data/prepared/features.csv"
    assert os.path.samefile(source, target)
    monkeypatch.setattr(Path, "unlink", unlink)
    assert "1 files moved" in migration.migrate_layout(tmp_path)
    assert not source.exists()
    assert target.read_bytes() == b"original"


def test_new_destination_created_after_preflight_is_not_overwritten(tmp_path, monkeypatch):
    source = write(tmp_path, "data/features.csv")
    link = os.link

    def race(old, new):
        new.write_bytes(b"new unrelated data")
        return link(old, new)

    monkeypatch.setattr(migration.os, "link", race)
    with pytest.raises(ValueError, match="Could not safely move"):
        migration.migrate_layout(tmp_path)
    assert source.read_bytes() == b"original"
    assert (tmp_path / "Global/data/features.csv").read_bytes() == b"new unrelated data"


def test_member_case_collision_across_old_roots_is_rejected(tmp_path):
    experiment = write(tmp_path, "experiments/Alice/example/train.py")
    notebook = write(tmp_path, "notebooks/alice/explore.ipynb")
    with pytest.raises(ValueError, match="Case or Unicode name collision"):
        migration.migrate_layout(tmp_path)
    assert experiment.exists()
    assert notebook.exists()
    assert not (tmp_path / "Members").exists()


def test_malformed_settings_block_migration_before_any_file_moves(tmp_path):
    source = write(tmp_path, "data/features.csv")
    write(tmp_path, ".poc.local.toml", b'owner = "unfinished')
    with pytest.raises(ValueError, match="Invalid .poc.local.toml"):
        migration.migrate_layout(tmp_path)
    assert source.exists()


def test_cli_quiet_suppresses_only_no_op_output(tmp_path):
    command = [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--quiet"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.stdout == ""
    write(tmp_path, "data/features.csv")
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "1 files moved" in result.stdout
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.stdout == ""


@pytest.mark.parametrize("already_moved", [False, True])
def test_settings_can_point_to_dataset_manifest_file(tmp_path, already_moved):
    relative = "data/prepared/music/dataset.json"
    manifest = write(
        tmp_path, f"Global/{relative}" if already_moved else relative, b'{"schema": 1}'
    )
    settings = write(
        tmp_path, ".poc.local.toml", f'owner = "owner"\ndata = "{relative}"\n'.encode()
    )
    assert "local data setting updated" in migration.migrate_layout(tmp_path)
    assert settings.read_text(encoding="utf-8") == f'owner = "owner"\ndata = "Global/{relative}"\n'
    assert (tmp_path / "Global" / relative).read_bytes() == b'{"schema": 1}'
    assert manifest.exists() is already_moved
    assert migration.migrate_layout(tmp_path) == "Local layout: 0 files moved"


def test_malformed_settings_do_not_block_no_op_or_init_repair(tmp_path):
    original = b'owner = "unfinished'
    settings = write(tmp_path, ".poc.local.toml", original)
    write(tmp_path, "Global/data/prepared/dataset.json", b"{}")
    assert migration.migrate_layout(tmp_path) == "Local layout: 0 files moved"
    command = [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--quiet"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.stdout == ""
    assert settings.read_bytes() == original


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_settings_update_preserves_original_newline_bytes(tmp_path, newline):
    write(tmp_path, "data/prepared/music/dataset.json", b"{}")
    original = newline.join(
        [
            "# 팀 설정".encode(),
            'owner = "조성률"'.encode(),
            b'data = "data/prepared/music" # selected dataset',
            b"",
        ]
    )
    settings = write(tmp_path, ".poc.local.toml", original)
    migration.migrate_layout(tmp_path)
    assert settings.read_bytes() == original.replace(b'"data/', b'"Global/data/')


def test_member_case_collision_with_existing_destination_is_rejected(tmp_path):
    source = write(tmp_path, "experiments/alice/example/train.py")
    existing = write(tmp_path, "Members/Alice/notebooks/existing.ipynb")
    with pytest.raises(ValueError, match="Case or Unicode name collision"):
        migration.migrate_layout(tmp_path)
    assert source.read_bytes() == b"original"
    assert existing.read_bytes() == b"original"
    assert not (tmp_path / "Members/Alice/experiments").exists()


def test_identically_spelled_member_names_across_old_roots_merge(tmp_path):
    write(tmp_path, "experiments/Alice/example/train.py", b"training source")
    write(tmp_path, "notebooks/Alice/explore.ipynb", b"notebook")
    assert "2 files moved" in migration.migrate_layout(tmp_path)
    assert (
        tmp_path / "Members/Alice/experiments/example/train.py"
    ).read_bytes() == b"training source"
    assert (tmp_path / "Members/Alice/notebooks/explore.ipynb").read_bytes() == b"notebook"
