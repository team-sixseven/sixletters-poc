"""Record experiment files without importing or executing personal code."""

import hashlib
import os
import re
import subprocess
import unicodedata
from pathlib import Path, PurePosixPath

from .contracts import _hash_json
from .workspace import validate_component

PACKAGE = Path(__file__).resolve().parent
SHARED_CODE_SCOPE_VERSION = 4
# These modules only launch work, present results or transfer completed runs.
# All other Python sources, including newly added modules, remain checked.
_SCOPE_V1_OPERATIONAL_SOURCE_PATHS = (
    "__main__.py",
    "local_settings.py",
    "run_catalog.py",
    "tracking.py",
    "tracking_diagnostics.py",
    "workspace.py",
)
_SCOPE_V2_OPERATIONAL_SOURCE_PATHS = (*_SCOPE_V1_OPERATIONAL_SOURCE_PATHS, "personal_env.py")
# Final-test code has its own frozen engine identity, independent of validation runs.
_SCOPE_V3_OPERATIONAL_SOURCE_PATHS = (
    *_SCOPE_V2_OPERATIONAL_SOURCE_PATHS,
    "execution.py",
    "reproduction.py",
    "final_evaluation.py",
    "final_metrics.py",
)
OPERATIONAL_SOURCE_PATHS = (
    *_SCOPE_V3_OPERATIONAL_SOURCE_PATHS,
    "notebook_workspace.py",
    "workspace_status.py",
)
_SCOPE_OPERATIONAL_SOURCE_PATHS = {
    1: _SCOPE_V1_OPERATIONAL_SOURCE_PATHS,
    2: _SCOPE_V2_OPERATIONAL_SOURCE_PATHS,
    3: _SCOPE_V3_OPERATIONAL_SOURCE_PATHS,
    4: OPERATIONAL_SOURCE_PATHS,
}
MAX_RESOURCE_BYTES = 1024 * 1024
MAX_TOTAL_RESOURCE_BYTES = 8 * MAX_RESOURCE_BYTES
MAX_RESOURCE_FILES = 256
RESOURCE_EXTENSIONS = frozenset({".toml", ".yaml", ".yml", ".json", ".ini", ".cfg"})
RESOURCE_FILENAMES = frozenset({"requirements.in", "requirements.txt", "requirements.lock"})
_IGNORED_DIRECTORIES = frozenset(
    {
        "__pycache__",
        "__pypackages__",
        "venv",
        "venvs",
        "env",
        "envs",
        "virtualenv",
        "site-packages",
        "node_modules",
        "build",
        "dist",
    }
)
_RESOURCE_OUTPUT_DIRECTORIES = frozenset(
    {
        "data",
        "artifacts",
        "models",
        "model",
        "outputs",
        "output",
        "results",
        "reports",
        "checkpoints",
    }
)
_SENSITIVE_RESOURCE_NAME = re.compile(
    r"(?:^|[._-])(?:secrets?|credentials?|passwords?|passwd|tokens?|api[_-]?keys?|"
    r"private[_-]?keys?|service[_-]?account|id_rsa|id_ed25519)(?:$|[._-])",
    re.IGNORECASE,
)


def source_hash(sources: dict[str, bytes]) -> str:
    """Identify Python sources independently of operating-system line endings."""
    return _hash_json(
        {
            name: hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
            for name, raw in sorted(sources.items())
        }
    )


def scientific_sources(
    sources: dict[str, bytes], *, version: int = SHARED_CODE_SCOPE_VERSION
) -> dict[str, bytes]:
    """Exclude only known operational modules from scientific identity checks."""
    if type(version) is not int or version not in _SCOPE_OPERATIONAL_SOURCE_PATHS:
        raise ValueError("Unsupported shared code scope version")
    excluded = _SCOPE_OPERATIONAL_SOURCE_PATHS[version]
    return {name: raw for name, raw in sources.items() if name not in excluded}


def shared_code_scope(
    sources: dict[str, bytes], *, version: int = SHARED_CODE_SCOPE_VERSION
) -> dict:
    """Record the exact scope alongside the full source snapshot and its hash."""
    checked = scientific_sources(sources, version=version)
    return {
        "version": version,
        "excluded": list(_SCOPE_OPERATIONAL_SOURCE_PATHS[version]),
        "hash": source_hash(checked),
    }


def validate_shared_code_scope(scope: dict) -> None:
    if (
        not isinstance(scope, dict)
        or set(scope) != {"version", "excluded", "hash"}
        or type(scope["version"]) is not int
        or scope["version"] not in _SCOPE_OPERATIONAL_SOURCE_PATHS
        or scope["excluded"] != list(_SCOPE_OPERATIONAL_SOURCE_PATHS[scope["version"]])
        or not isinstance(scope["hash"], str)
        or re.fullmatch(r"[0-9a-f]{64}", scope["hash"]) is None
    ):
        raise ValueError("Unsupported or malformed shared code scope; use compatible saved runs")


def _ignored_directory(name: str, *, resources: bool = False) -> bool:
    return (
        name.startswith(".")
        or name.casefold() in _IGNORED_DIRECTORIES
        or name.casefold().endswith((".egg-info", ".dist-info"))
        or resources
        and (
            name.casefold() in _RESOURCE_OUTPUT_DIRECTORIES
            or _SENSITIVE_RESOURCE_NAME.search(name) is not None
        )
    )


def _snapshot_files(directory: Path, *, resources: bool = False):
    """Prune installed dependencies before walking; never follow symbolic links."""
    directory = directory.resolve()

    def failed_directory(error):
        raise ValueError("Snapshot directory could not be read") from error

    for current, directories, filenames in os.walk(
        directory, followlinks=False, onerror=failed_directory
    ):
        retained = []
        for name in sorted(directories):
            if _ignored_directory(name, resources=resources):
                continue
            path = Path(current) / name
            if path.is_symlink():
                raise ValueError("Snapshot directories must not be symbolic links")
            retained.append(name)
        directories[:] = retained
        for name in sorted(filenames):
            if not name.startswith("."):
                yield directory, Path(current) / name


def _checked_snapshot_path(directory: Path, path: Path) -> str:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory):
        raise ValueError("Snapshot files must be regular files inside their source directory")
    name = path.relative_to(directory).as_posix()
    if any(
        validate_component(part, "snapshot filename") != part
        for part in path.relative_to(directory).parts
    ):
        raise ValueError("Snapshot filenames must be portable relative paths")
    return name


def _check_unique_paths(names) -> None:
    files = set()
    directories = set()
    for name in names:
        parts = tuple(
            unicodedata.normalize("NFC", part).casefold() for part in PurePosixPath(name).parts
        )
        parents = {parts[:index] for index in range(1, len(parts))}
        if parts in files or parts in directories or parents & files:
            raise ValueError("Snapshot filenames collide across operating systems")
        files.add(parts)
        directories.update(parents)


def python_sources(directory: Path) -> dict[str, bytes]:
    sources = {}
    for root, path in _snapshot_files(directory):
        if path.suffix == ".py":
            name = _checked_snapshot_path(root, path)
            sources[name] = path.read_bytes()
    _check_unique_paths(sources)
    return sources


def _is_resource_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(path.parts)
        and not path.is_absolute()
        and path.as_posix() == name
        and not any(_ignored_directory(part, resources=True) for part in path.parts[:-1])
        and not path.name.startswith(".")
        and _SENSITIVE_RESOURCE_NAME.search(path.name) is None
        and (path.suffix.lower() in RESOURCE_EXTENSIONS or path.name.lower() in RESOURCE_FILENAMES)
    )


def validate_resource_metadata(metadata: dict) -> None:
    """Validate untrusted saved resource paths without reading or executing files."""
    if not isinstance(metadata, dict) or len(metadata) > MAX_RESOURCE_FILES:
        raise ValueError("Resource metadata must be a bounded filename mapping")
    total = 0
    for name, info in metadata.items():
        if (
            not isinstance(name, str)
            or not _is_resource_name(name)
            or any(
                validate_component(part, "resource filename") != part
                for part in PurePosixPath(name).parts
            )
        ):
            raise ValueError("Resource filenames must be permitted portable relative paths")
        if (
            not isinstance(info, dict)
            or set(info) != {"sha256", "bytes"}
            or not isinstance(info["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", info["sha256"]) is None
            or type(info["bytes"]) is not int
            or not 0 <= info["bytes"] <= MAX_RESOURCE_BYTES
        ):
            raise ValueError("Resource metadata requires a SHA256 hash and bounded byte size")
        total += info["bytes"]
    if total > MAX_TOTAL_RESOURCE_BYTES:
        raise ValueError("Experiment resources exceed the 8 MiB total limit")
    _check_unique_paths(metadata)


def resource_metadata(resources: dict[str, bytes]) -> dict:
    metadata = {
        name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        for name, raw in sorted(resources.items())
    }
    validate_resource_metadata(metadata)
    return metadata


def experiment_resources(
    directory: Path, *, config_name: str = "config.toml", exclude: set[str] | None = None
) -> dict[str, bytes]:
    """Capture small UTF-8 configuration/manifests, excluding data and credentials."""
    resources = {}
    total = 0
    for root, path in _snapshot_files(directory, resources=True):
        name = path.relative_to(root).as_posix()
        if name == config_name or (exclude and name in exclude) or not _is_resource_name(name):
            continue
        name = _checked_snapshot_path(root, path)
        if path.stat().st_size > MAX_RESOURCE_BYTES:
            raise ValueError(f"Experiment resource exceeds the 1 MiB file limit: {name}")
        with path.open("rb") as handle:
            raw = handle.read(MAX_RESOURCE_BYTES + 1)
        if len(raw) > MAX_RESOURCE_BYTES:
            raise ValueError(f"Experiment resource exceeds the 1 MiB file limit: {name}")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Experiment resource must contain UTF-8 text: {name}") from error
        if any(ord(char) < 32 and char not in "\t\r\n" for char in text):
            raise ValueError(f"Experiment resource must contain text, not binary data: {name}")
        total += len(raw)
        if total > MAX_TOTAL_RESOURCE_BYTES or len(resources) >= MAX_RESOURCE_FILES:
            raise ValueError("Experiment resources exceed the 8 MiB or 256 file total limit")
        resources[name] = raw
    resource_metadata(resources)
    return resources


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git_state() -> dict:
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PACKAGE, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=PACKAGE, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"code_commit": commit, "code_dirty": dirty}
