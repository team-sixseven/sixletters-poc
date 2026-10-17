"""Local convenience settings; these never select or execute a training method."""

import json
import tomllib
from pathlib import Path

from .workspace import validate_component

SETTINGS_FILE = ".poc.local.toml"


def project_root(start: Path | None = None) -> Path:
    """Find the checkout even when an IDE starts a script in its own folder."""
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "Global/pyproject.toml").is_file() or (
            (candidate / "pyproject.toml").is_file() and (candidate / "experiments").is_dir()
        ):
            return candidate
    return Path(__file__).resolve().parents[3]


def resolve_workspace_path(root: Path, path: Path) -> Path:
    """Resolve former shared paths without changing arbitrary custom locations."""
    root = Path(root).resolve()
    path = Path(path)
    absolute = (path if path.is_absolute() else root / path).resolve()
    if not (root / "Global/pyproject.toml").is_file():
        return absolute
    try:
        relative = absolute.relative_to(root)
    except ValueError:
        return absolute
    if relative.parts and relative.parts[0] in {"data", "artifacts"}:
        destination = root / "Global" / relative
        if absolute.exists():
            raise ValueError(
                "Legacy data/results remain at the repository root. Run "
                "python Global/scripts/migrate_layout.py before continuing."
            )
        return destination
    return absolute


def read_settings(root: Path) -> dict[str, str]:
    path = root / SETTINGS_FILE
    if not path.exists():
        return {}
    settings = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(settings) != {"owner", "data"} or any(
        not isinstance(value, str) or not value.strip() for value in settings.values()
    ):
        raise ValueError(f"Invalid {SETTINGS_FILE}; use poc init to register owner and data")
    settings["owner"] = validate_component(settings["owner"], "owner")
    settings["data"] = _relative_data(root, Path(settings["data"]))
    return settings


def _relative_data(root: Path, data: Path) -> str:
    resolved = resolve_workspace_path(root, data).resolve()
    if not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
        raise ValueError("The configured data path must be inside this repository")
    return resolved.relative_to(root.resolve()).as_posix()


def write_settings(root: Path, owner: str, data: Path) -> Path:
    owner = validate_component(owner, "owner")
    relative = _relative_data(root, data)
    path = root / SETTINGS_FILE
    if path.is_symlink():
        raise ValueError(f"{SETTINGS_FILE} must not be a symbolic link")
    path.write_text(
        f"owner = {json.dumps(owner, ensure_ascii=False)}\n"
        f"data = {json.dumps(relative, ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    return path


def configured_data(root: Path) -> Path:
    settings = read_settings(root)
    if "data" not in settings:
        raise ValueError("Specify --data or register a data path with poc init first")
    return root / settings["data"]
