"""Notebook preparation keeps edits while selecting the managed experiment kernel."""

import nbformat
import pytest
from nbclient import NotebookClient

from sixletters import notebook_workspace as nw


@pytest.fixture
def notebook_workspace(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    directory = root / "Members/팀원/experiments/exp001"
    directory.mkdir(parents=True)
    (directory / "config.toml").touch()
    specification = root / ".poc-cache/kernels/sixletters-test/kernel.json"
    calls = []

    def register(path, *, relock=False):
        calls.append((path, relock))
        return "SixSeven · 팀원 / exp001", specification

    monkeypatch.setattr(nw, "env_kernel", register)
    return directory, calls


def test_new_notebook_is_valid_blank_and_uses_registered_kernel(notebook_workspace):
    directory, calls = notebook_workspace
    result = nw.prepare_notebook(directory)
    assert result == {
        "path": directory / "analysis.ipynb",
        "kernel": "SixSeven · 팀원 / exp001",
        "created": True,
    }
    notebook = nbformat.read(result["path"], as_version=4)
    nbformat.validate(notebook)
    assert len(notebook.cells) == 1
    assert notebook.cells[0].source == ""
    assert notebook.metadata.kernelspec.name == "sixletters-test"
    assert calls == [(directory, False)]
    # The blank scaffold executes in an available local kernel without a data dependency.
    NotebookClient(notebook, timeout=30, kernel_name="python3").execute()


def test_repeated_preparation_preserves_cells_outputs_metadata_and_bytes(notebook_workspace):
    directory, _ = notebook_workspace
    path = nw.prepare_notebook(directory)["path"]
    value = nbformat.read(path, as_version=4)
    value.cells[0].source = "print('edited')"
    value.cells[0].execution_count = 1
    value.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="edited\n")]
    value.metadata["personal"] = {"preserve": True}
    nbformat.write(value, path)
    before = path.read_bytes()
    assert nw.prepare_notebook(directory)["created"] is False
    assert path.read_bytes() == before


def test_existing_notebook_changes_only_kernel_metadata(notebook_workspace):
    directory, _ = notebook_workspace
    path = directory / "exploration.ipynb"
    original = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("x = 7")])
    original.metadata["personal"] = "unchanged"
    original.metadata["kernelspec"] = {"name": "old", "display_name": "Old", "language": "python"}
    nbformat.write(original, path)
    result = nw.prepare_notebook(directory, filename=path.name, relock=True)
    value = nbformat.read(path, as_version=4)
    assert result["created"] is False
    assert value.cells == original.cells
    assert value.metadata.personal == "unchanged"
    assert value.metadata.kernelspec.name == "sixletters-test"


@pytest.mark.parametrize(
    "filename", ["../escape.ipynb", ".hidden.ipynb", "folder/a.ipynb", "train.py"]
)
def test_unsafe_notebook_names_rejected(notebook_workspace, filename):
    directory, calls = notebook_workspace
    with pytest.raises(ValueError):
        nw.prepare_notebook(directory, filename=filename)
    assert calls == []


@pytest.mark.parametrize("content", [b"not json", b"{}", b"\xff"])
def test_invalid_existing_notebook_is_not_changed(notebook_workspace, content):
    directory, calls = notebook_workspace
    path = directory / "analysis.ipynb"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="invalid"):
        nw.prepare_notebook(directory)
    assert path.read_bytes() == content
    assert calls == []


def test_symlink_notebook_is_not_followed(notebook_workspace):
    directory, calls = notebook_workspace
    outside = directory.parents[3] / "outside.ipynb"
    outside.write_text("unchanged")
    (directory / "analysis.ipynb").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic"):
        nw.prepare_notebook(directory)
    assert outside.read_text() == "unchanged"
    assert calls == []


def test_environment_failure_leaves_no_notebook(notebook_workspace, monkeypatch):
    directory, _ = notebook_workspace

    def fail(*args, **kwargs):
        raise ValueError("package conflict")

    monkeypatch.setattr(nw, "env_kernel", fail)
    with pytest.raises(ValueError, match="package conflict"):
        nw.prepare_notebook(directory)
    assert not (directory / "analysis.ipynb").exists()


def test_editor_change_during_preparation_is_not_overwritten(notebook_workspace, monkeypatch):
    directory, _ = notebook_workspace
    path = directory / "analysis.ipynb"
    value = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("original = 1")])
    nbformat.write(value, path)

    def edited(*args, **kwargs):
        changed = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("edited = 2")])
        nbformat.write(changed, path)
        return "Display", directory / "spec/kernel.json"

    monkeypatch.setattr(nw, "env_kernel", edited)
    with pytest.raises(ValueError, match="changed during"):
        nw.prepare_notebook(directory)
    assert nbformat.read(path, as_version=4).cells[0].source == "edited = 2"


def test_notebook_case_collision_does_not_create_second_file(notebook_workspace):
    directory, calls = notebook_workspace
    existing = directory / "Analysis.ipynb"
    nbformat.write(nbformat.v4.new_notebook(), existing)
    with pytest.raises(ValueError, match="collides"):
        nw.prepare_notebook(directory)
    assert calls == []
