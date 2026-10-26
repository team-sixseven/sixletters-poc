"""Run repository policy and the retained environment/upload tests."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="검사 전체 실행 (기본값과 동일)")
    parser.add_argument(
        "--staged", action="store_true", help="커밋할 파일 정책 검사 후 전체 테스트"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.pop("SKIP", None)
    command = [sys.executable, "-m", "pre_commit", "run"]
    if not args.staged:
        command.append("--all-files")
    result = subprocess.run(command, cwd=root, env=env, check=False)
    if result.returncode:
        return result.returncode
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-c", "Global/pyproject.toml", "-q"], cwd=root, check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
