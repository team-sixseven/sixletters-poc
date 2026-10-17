"""Move local files from the legacy checkout layout without changing their contents."""

import argparse
import json
import os
import re
import stat
import tempfile
import tomllib
import unicodedata
from pathlib import Path

IGNORED_NAMES = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".DS_Store"}


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _check_path(path: Path, root: Path) -> None:
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError(f"Symbolic links cannot be migrated: {item.relative_to(root)}")
        if item == root:
            break


def _key(path: Path) -> str:
    return unicodedata.normalize("NFC", path.as_posix()).casefold()


def _settings_update(root: Path, destinations: set[Path]) -> tuple[Path, str] | None:
    path = root / ".poc.local.toml"
    _check_path(path, root)
    if not path.exists():
        return None
    original = path.read_bytes().decode("utf-8")
    try:
        data = tomllib.loads(original).get("data")
    except tomllib.TOMLDecodeError as error:
        if not destinations:
            return None  # Keep dev init available to repair unrelated invalid local settings.
        raise ValueError("Invalid .poc.local.toml; fix it before migrating local files") from error
    if not isinstance(data, str):
        return None
    old = Path(data)
    if old.is_absolute() or not old.parts or old.parts[0] != "data" or ".." in old.parts:
        return None
    target = root / "Global" / old
    _check_path(target, root)
    if target not in destinations and not (target.is_dir() or target.is_file()):
        return None
    replacement = json.dumps(target.relative_to(root).as_posix(), ensure_ascii=False)
    updated, count = re.subn(
        r"""(?m)^([ \t]*data[ \t]*=[ \t]*)(?:"(?:[^"\\\n]|\\.)*"|'[^'\n]*')""",
        lambda match: match[1] + replacement,
        original,
        count=1,
    )
    if count != 1:
        raise ValueError("Use a single-line data value in .poc.local.toml before migrating")
    return path, updated


def migrate_layout(root: Path) -> str:
    """Preflight and migrate local data, results and named member work; safe to retry."""
    root = Path(root).absolute()
    if root.is_symlink():
        raise ValueError("The repository root must not be a symbolic link")
    files: list[tuple[Path, Path]] = []
    directories: set[Path] = set()
    old_directories: set[Path] = set()
    reservations: dict[str, str] = {}
    skipped: list[str] = []

    def reserve(target: Path, *, directory: bool) -> None:
        _check_path(target, root)
        for parent in (target, *target.parents):
            if parent == root:
                break
            key = _key(parent)
            spelling = parent.as_posix()
            if key in reservations and reservations[key] != spelling:
                raise ValueError(f"Case or Unicode name collision: {parent.relative_to(root)}")
            reservations[key] = spelling
            if _exists(parent) and parent != target and not parent.is_dir():
                raise ValueError(f"Destination parent is not a directory: {parent}")
            if parent.parent.is_dir():
                for sibling in parent.parent.iterdir():
                    if sibling.as_posix() != spelling and _key(sibling) == key:
                        raise ValueError(f"Case or Unicode name collision: {parent}")
        if _exists(target) and directory and not target.is_dir():
            raise ValueError(f"Destination is not a directory: {target.relative_to(root)}")

    def plan(source: Path, target: Path) -> None:
        _check_path(source, root)
        if source.is_dir():
            reserve(target, directory=True)
            directories.add(target)
            old_directories.add(source)
            for child in sorted(source.iterdir()):
                if child.name in IGNORED_NAMES:
                    skipped.append(child.relative_to(root).as_posix())
                else:
                    plan(child, target / child.name)
        elif source.is_file():
            reserve(target, directory=False)
            if _exists(target) and (not target.is_file() or not os.path.samefile(source, target)):
                raise ValueError(
                    f"Migration conflict: {source.relative_to(root)} -> "
                    f"{target.relative_to(root)} already exists; move or rename one copy first"
                )
            files.append((source, target))
        else:
            raise ValueError(f"Unsupported local file: {source.relative_to(root)}")

    for name in ("data", "artifacts"):
        source = root / name
        _check_path(source, root)
        if _exists(source):
            if not source.is_dir():
                raise ValueError(f"Legacy {name}/ must be a directory")
            plan(source, root / "Global" / name)

    for name in ("experiments", "notebooks"):
        source = root / name
        _check_path(source, root)
        if not _exists(source):
            continue
        if not source.is_dir():
            raise ValueError(f"Legacy {name}/ must be a directory")
        old_directories.add(source)
        for owner in sorted(source.iterdir()):
            if owner.name in IGNORED_NAMES or (name == "experiments" and owner.name == "_template"):
                skipped.append(owner.relative_to(root).as_posix())
                continue
            _check_path(owner, root)
            if not owner.is_dir():
                raise ValueError(
                    f"Cannot determine an owner for {owner.relative_to(root)}; move it manually "
                    f"to Members/<name>/{name}/ (or Global/notebooks/ for shared notebooks)"
                )
            plan(owner, root / "Members" / owner.name / name)

    settings = _settings_update(root, directories | {target for _, target in files})
    # No file is moved until every source, destination and settings edit passes preflight.
    for target in sorted(directories, key=lambda path: (len(path.parts), path.as_posix())):
        target.mkdir(parents=True, exist_ok=True)
    for source, target in files:
        if not _exists(target):
            # Exclusive creation cannot overwrite a file added after preflight. Linking also
            # preserves bytes and metadata; a stopped link/unlink is recognized on the next run.
            try:
                os.link(source, target)
            except OSError as error:
                raise ValueError(
                    f"Could not safely move {source.relative_to(root)}; keep source and destination "
                    "on the same filesystem and retry after checking permissions"
                ) from error
        elif not target.is_file() or not os.path.samefile(source, target):
            raise ValueError(f"Destination changed during migration: {target.relative_to(root)}")
        source.unlink()
    if settings:
        path, content = settings
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=root, prefix=".poc.local.", delete=False
        ) as temporary:
            pending = Path(temporary.name)
            temporary.write(content.encode("utf-8"))
        try:
            pending.chmod(stat.S_IMODE(path.stat().st_mode))
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    for source in sorted(old_directories, key=lambda path: len(path.parts), reverse=True):
        try:
            source.rmdir()  # Never remove nonempty folders or skipped user files.
        except OSError:
            pass
    details = [f"Local layout: {len(files)} files moved"]
    if settings:
        details.append("local data setting updated")
    if skipped:
        details.append(f"{len(skipped)} cache/template paths left unchanged")
    return "; ".join(details)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--quiet", action="store_true", help="Suppress unchanged-layout messages")
    args = parser.parse_args()
    try:
        summary = migrate_layout(args.root)
        if (
            not args.quiet
            or not summary.startswith("Local layout: 0 files moved")
            or ("local data setting updated" in summary)
        ):
            print(summary)
    except (ValueError, OSError) as error:
        print(f"Local layout migration stopped: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
