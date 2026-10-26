"""Setup preserves existing data and personal work across repeated calls."""

import sys

import pytest

from sixletters.__main__ import main


def test_setup_preserves_files_and_connection(tmp_path, monkeypatch):
    (tmp_path / "Global").mkdir()
    (tmp_path / "Global/pyproject.toml").touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["dev", "setup", "--owner", "alice", "--uri", "http://localhost:5000"]
    )
    assert main() == 0
    data = tmp_path / "Global/data/DATA/clinical/split/train.csv"
    data.write_text("synthetic fixture")
    code = tmp_path / "Members/alice/experiments/custom.py"
    code.write_text("print('preserved')")
    monkeypatch.setattr(sys, "argv", ["dev", "setup", "--owner", "alice"])
    assert main() == 0
    assert data.read_text() == "synthetic fixture"
    assert code.read_text() == "print('preserved')"
    assert "http://localhost:5000" in (tmp_path / ".poc.local.toml").read_text()
    for name in ("notebooks", "experiments", "results", "models"):
        assert (tmp_path / "Members/alice" / name).is_dir()
    assert (tmp_path / "Global/data/DATA/ECG_num/split").is_dir()
    assert (tmp_path / "Global/data/DATA/ECG_tem/ECG_tem_clean").is_dir()


def test_setup_rejects_link_without_writing_outside(tmp_path, monkeypatch):
    (tmp_path / "Global").mkdir()
    (tmp_path / "Global/pyproject.toml").touch()
    member = tmp_path / "Members/alice"
    member.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (member / "models").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation unavailable on this platform")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["dev", "setup", "--owner", "alice"])
    assert main() == 2
    assert not (tmp_path / ".poc.local.toml").exists()
    assert list(outside.iterdir()) == []
