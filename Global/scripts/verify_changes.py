"""Run repository policy checks and tests selected from committed and local changes."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from .ci_scope import PERSONAL_TESTS, registered_commands_changed, select_scope
except ImportError:
    from ci_scope import PERSONAL_TESTS, registered_commands_changed, select_scope


@dataclass(frozen=True)
class Changes:
    paths: tuple[str, ...]
    certain: bool
    base: str | None


def _git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=False)
    if check and result.returncode:
        raise ValueError("Git 변경 범위를 확인하지 못했습니다. 저장소와 기준 브랜치를 확인하세요.")
    return result


def _paths(result: subprocess.CompletedProcess) -> set[str]:
    return {value for value in result.stdout.decode("utf-8").split("\0") if value}


def _revision(root: Path, name: str) -> str | None:
    result = _git(
        root, "rev-parse", "--verify", "--end-of-options", f"{name}^{{commit}}", check=False
    )
    return result.stdout.decode().strip() if result.returncode == 0 else None


def changed_files(root: Path, *, base: str | None = None, staged: bool = False) -> Changes:
    """Include branch commits, staged/unstaged edits and untracked files by default."""
    if staged and base:
        raise ValueError("--staged와 --base는 함께 사용할 수 없습니다.")
    _git(root, "rev-parse", "--show-toplevel")
    if staged:
        files = _paths(_git(root, "diff", "--cached", "--name-only", "--no-renames", "-z", "--"))
        return Changes(tuple(sorted(files)), True, None)

    selected = base
    revision = _revision(root, selected) if selected else None
    if selected and revision is None:
        raise ValueError(f"기준 브랜치를 찾을 수 없습니다: {selected}")
    if selected is None:
        for candidate in ("origin/develop", "develop"):
            revision = _revision(root, candidate)
            if revision is not None:
                selected = candidate
                break
    files = _paths(_git(root, "ls-files", "--others", "--exclude-standard", "-z"))
    head = _revision(root, "HEAD")
    if head is None:
        files.update(_paths(_git(root, "ls-files", "-z")))
        return Changes(tuple(sorted(files)), False, selected)
    files.update(_paths(_git(root, "diff", "--name-only", "--no-renames", "-z", "HEAD", "--")))
    if revision is None:
        return Changes(tuple(sorted(files)), False, None)
    common = _git(root, "merge-base", revision, head, check=False)
    if common.returncode:
        return Changes(tuple(sorted(files)), False, selected)
    ancestor = common.stdout.decode().strip()
    files.update(
        _paths(_git(root, "diff", "--name-only", "--no-renames", "-z", ancestor, head, "--"))
    )
    return Changes(tuple(sorted(files)), True, selected)


def test_command(
    mode: str,
    test_files: list[str] | None = None,
    *,
    shard_index: int | None = None,
    shard_count: int | None = None,
    shard_report: str | None = None,
) -> list[str] | None:
    if mode not in {"none", "personal", "focused", "full"}:
        raise ValueError("Unknown verification test mode")
    files = test_files or []
    if mode == "focused":
        if (
            not isinstance(files, list)
            or not files
            or any(
                not isinstance(name, str)
                or re.fullmatch(r"Global/tests/test_[A-Za-z0-9_]+\.py", name) is None
                for name in files
            )
        ):
            raise ValueError("Focused tests must list repository test files")
    elif files:
        raise ValueError("Test files are only accepted for focused verification")
    sharded = shard_index is not None or shard_count is not None or shard_report is not None
    if sharded and (
        mode != "full"
        or type(shard_index) is not int
        or type(shard_count) is not int
        or shard_count < 2
        or not 0 <= shard_index < shard_count
    ):
        raise ValueError("Only full tests accept a valid shard index and count together")
    if mode == "none":
        return None
    command = [sys.executable, "-m", "pytest", "-c", "Global/pyproject.toml", "-q"]
    if sharded:
        command.extend(
            [
                "-p",
                "Global.scripts.ci_shards",
                "--ci-shard-index",
                str(shard_index),
                "--ci-shard-count",
                str(shard_count),
                "--durations=10",
            ]
        )
        if shard_report is not None:
            if not isinstance(shard_report, str) or not shard_report.strip():
                raise ValueError("Shard report path must not be empty")
            command.extend(["--ci-shard-report", shard_report])
    if mode == "personal":
        command.extend(PERSONAL_TESTS)
    elif mode == "focused":
        command.extend(sorted(set(files)))
    return command


def run_tests(
    root: Path,
    mode: str,
    test_files: list[str] | None = None,
    *,
    shard_index: int | None = None,
    shard_count: int | None = None,
    shard_report: str | None = None,
) -> int:
    command = test_command(
        mode,
        test_files,
        shard_index=shard_index,
        shard_count=shard_count,
        shard_report=shard_report,
    )
    if mode == "focused":
        for name in test_files or []:
            file = root / name
            if (
                not file.is_file()
                or file.is_symlink()
                or not file.resolve().is_relative_to(root.resolve())
            ):
                raise ValueError(f"검사 파일을 찾을 수 없거나 저장소 밖을 가리킵니다: {name}")
    return subprocess.run(command, cwd=root, check=False).returncode if command else 0


def local_scope(root: Path, changes: Changes, *, full: bool = False, staged: bool = False) -> dict:
    scripts_only = False
    if "Global/pyproject.toml" in changes.paths and changes.certain:
        if staged:
            scripts_only = registered_commands_changed(root, "HEAD", "")
        elif changes.base:
            common = _git(root, "merge-base", changes.base, "HEAD", check=False)
            if common.returncode == 0:
                scripts_only = registered_commands_changed(root, common.stdout.decode().strip())
    return select_scope(
        list(changes.paths),
        full=full or not changes.certain,
        root=root,
        scripts_only=scripts_only,
    )


def verify(root: Path, *, full: bool = False, staged: bool = False, base: str | None = None) -> int:
    changes = changed_files(root, staged=staged, base=base)
    scope = local_scope(root, changes, full=full, staged=staged)
    mode = scope["tests_mode"]
    scope_name = {
        "none": "문서·정책 검사",
        "personal": "개인 실험 공통 계약 검사",
        "full": "전체 회귀 검사",
        "focused": "변경 테스트와 관련 검사",
    }[mode]
    print(f"검증 범위: {scope_name} · 변경 파일 {len(changes.paths)}개", flush=True)
    if changes.base:
        print(f"기준: {changes.base}와 공통 조상 이후 커밋 + 로컬 변경", flush=True)
    if not changes.certain:
        print("기준을 확정할 수 없어 전체 검사를 실행합니다.", flush=True)
    command = [sys.executable, "-m", "pre_commit", "run"]
    files = [name for name in changes.paths if (root / name).is_file()]
    if staged:
        # Let pre-commit stash unstaged edits and inspect the actual index contents.
        pass
    elif full or not changes.certain or not files:
        command.append("--all-files")
    else:
        command.extend(["--files", *files])
    environment = os.environ.copy()
    environment.pop("SKIP", None)
    checked = subprocess.run(command, cwd=root, env=environment, check=False)
    if checked.returncode:
        print(
            "정책 검사 또는 자동 정리로 파일이 변경됐습니다. 확인 후 다시 검증하세요.", flush=True
        )
        return checked.returncode
    return run_tests(root, mode, scope["test_files"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true", help="Run policy checks and all regression tests"
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--staged", action="store_true", help="Inspect only the staged changes")
    scope.add_argument("--base", help="Compare against this Git revision instead of develop")
    parser.add_argument(
        "--tests-only", choices=("none", "personal", "focused", "full"), help=argparse.SUPPRESS
    )
    parser.add_argument("--test-files", default="[]", help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--shard-report", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        if args.tests_only:
            return run_tests(
                root,
                args.tests_only,
                json.loads(args.test_files),
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                shard_report=args.shard_report,
            )
        if any(
            value is not None for value in (args.shard_index, args.shard_count, args.shard_report)
        ):
            raise ValueError("Shard options require --tests-only full")
        return verify(root, full=args.full, staged=args.staged, base=args.base)
    except (ValueError, OSError) as error:
        print(f"검증 실패: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
