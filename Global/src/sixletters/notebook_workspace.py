"""Prepare editable personal notebooks with their experiment's registered kernel."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import nbformat
from filelock import FileLock, Timeout

from .personal_env import _experiment, env_kernel
from .workspace import _reject_ambiguous_name, validate_component

MAX_NOTEBOOK_BYTES = 16 * 1024 * 1024


def _read_notebook(path: Path):
    if path.is_symlink():
        raise ValueError("Notebook files must not be symbolic links")
    if not path.exists():
        return None, None
    if not path.is_file() or path.stat().st_size > MAX_NOTEBOOK_BYTES:
        raise ValueError("Notebook must be a regular file no larger than 16 MiB")
    with path.open("rb") as handle:
        raw = handle.read(MAX_NOTEBOOK_BYTES + 1)
    if len(raw) > MAX_NOTEBOOK_BYTES:
        raise ValueError("Notebook exceeds the 16 MiB limit")
    try:
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict) or value.get("nbformat") != 4:
            raise ValueError("Use a valid version 4 notebook")
        notebook = nbformat.from_dict(value)
        nbformat.validate(notebook)
    except (ValueError, UnicodeDecodeError, nbformat.ValidationError) as error:
        raise ValueError("Existing notebook is invalid; its contents were not changed") from error
    return notebook, raw


def prepare_notebook(
    experiment_dir: Path, *, filename: str = "analysis.ipynb", relock: bool = False
) -> dict:
    """Keep user cells intact while preparing the correct Python and kernel metadata."""
    directory = _experiment(experiment_dir)
    filename = validate_component(filename, "notebook filename")
    if not filename.endswith(".ipynb"):
        raise ValueError("Notebook filename must end with .ipynb")
    _reject_ambiguous_name(directory, filename)
    path = directory / filename
    guard = directory / ".poc-notebook.lock"
    if guard.is_symlink():
        raise ValueError("Notebook preparation lock must not be a symbolic link")
    try:
        with FileLock(guard, timeout=0):
            notebook, original = _read_notebook(path)
            display, specification = env_kernel(directory, relock=relock)
            created = notebook is None
            if created:
                notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("")])
            kernel = {
                "name": specification.parent.name,
                "display_name": display,
                "language": "python",
            }
            changed = created or notebook.metadata.get("kernelspec") != kernel
            if changed:
                notebook.metadata["kernelspec"] = kernel
                nbformat.validate(notebook)
                content = nbformat.writes(notebook, version=4).encode("utf-8")
                if len(content) > MAX_NOTEBOOK_BYTES:
                    raise ValueError("Prepared notebook exceeds the 16 MiB limit")
                # An editor does not share this lock. Do not replace a newly saved edit.
                _, current = _read_notebook(path)
                if current != original:
                    raise ValueError("Notebook changed during preparation; run the command again")
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(content)
                try:
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            return {"path": path, "kernel": display, "created": created}
    except Timeout as error:
        raise ValueError(
            "This notebook is already being prepared; wait for it to finish"
        ) from error
