"""Personal walkthroughs preserve local work, identity and common data selection."""

import json
import os
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

from sixletters.local_settings import write_settings
from sixletters.workspace import PRACTICE_NOTEBOOK, init_practice


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "한글 팀 프로젝트"
    (root / "Global").mkdir(parents=True)
    (root / "Global/pyproject.toml").touch()
    return root


def cli(root, *args):
    return subprocess.run(
        [sys.executable, "-m", "sixletters", "practice", *args],
        cwd=root,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_practice_binds_owner_and_preserves_existing_work(checkout):
    first = init_practice("팀원 가", root=checkout / "Members")
    second = init_practice("팀원 나", root=checkout / "Members")
    notebook = nbformat.read(first, as_version=4)
    nbformat.validate(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert 'OWNER = "팀원 가"' in code
    assert 'OWNER = "__POC_OWNER__"' not in code
    assert first.parent == checkout / "Members/팀원 가/notebooks"
    assert second.parent == checkout / "Members/팀원 나/notebooks"
    assert all(
        cell.execution_count is None and cell.outputs == []
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    notebook.cells.append(nbformat.v4.new_markdown_cell("직접 작성한 해석"))
    nbformat.write(notebook, first)
    before = first.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        init_practice("팀원 가", root=checkout / "Members")
    assert first.read_bytes() == before
    assert not (checkout / "Global/data").exists()
    assert not (checkout / "Global/artifacts").exists()


def test_practice_uses_local_owner_from_nested_directory_without_replacing_settings(checkout):
    settings = write_settings(checkout, "팀원 가", Path("Global/data/prepared/custom-v2"))
    before = settings.read_bytes()
    nested = checkout / "Members/팀원 가/notebooks"
    nested.mkdir(parents=True)
    result = cli(nested)
    assert result.returncode == 0, result.stderr
    assert f"Members/팀원 가/notebooks/{PRACTICE_NOTEBOOK}" in result.stdout
    assert (nested / PRACTICE_NOTEBOOK).is_file()
    assert settings.read_bytes() == before
    again = cli(nested)
    assert again.returncode == 2
    assert "already exists" in again.stderr


def test_explicit_owner_does_not_require_or_change_local_settings(checkout):
    result = cli(checkout, "--owner", "팀원 나")
    assert result.returncode == 0, result.stderr
    assert not (checkout / ".poc.local.toml").exists()
    missing = cli(checkout)
    assert missing.returncode == 2
    assert "dev init" in missing.stderr


@pytest.mark.parametrize("owner", ["../outside", "a/b", "CON", " leading", 'bad"name'])
def test_invalid_practice_owner_cannot_write_files(checkout, owner):
    with pytest.raises(ValueError):
        init_practice(owner, root=checkout / "Members")
    assert not (checkout / "Members").exists()


def test_practice_normalizes_korean_and_rejects_case_collisions(checkout):
    root = checkout / "Members"
    path = init_practice("조성률", root=root)
    assert path.parent.parent.name == "조성률"
    init_practice("Alice", root=root)
    with pytest.raises(ValueError, match="collides"):
        init_practice("alice", root=root)


@pytest.mark.skipif(os.name == "nt", reason="Windows symlink creation requires extra privileges")
@pytest.mark.parametrize("link", ["Members", "Members/팀원", "Members/팀원/notebooks"])
def test_practice_does_not_write_through_symlink(checkout, tmp_path, link):
    outside = tmp_path / "outside"
    outside.mkdir()
    path = checkout / link
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        init_practice("팀원", root=checkout / "Members")
    assert not list(outside.iterdir())


def test_practice_template_is_valid_and_does_not_contain_outputs():
    path = Path(__file__).resolve().parents[1] / "templates/practice" / PRACTICE_NOTEBOOK
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert all(
        cell.execution_count is None and not cell.outputs
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    # A template's outputs must never carry real MUSIC records into version control.
    assert "image/png" not in json.dumps(notebook)
