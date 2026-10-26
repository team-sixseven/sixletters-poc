"""Select checks from the complete PR diff; unknown files select all checks."""

import json
import os
import subprocess
from pathlib import Path


def select_scope(paths: list[str], *, full: bool = False) -> dict:
    tests = full
    native = full
    suites = {"workspace", "tracking"} if full else set()
    for path in paths:
        if path.endswith(".md") or path.startswith("Global/docs/assets/"):
            continue
        tests = True
        if path.startswith("Members/"):
            continue
        if path in {"Global/src/sixletters/upload.py", "Global/tests/test_upload.py"}:
            suites.add("tracking")
        elif path in {
            "Global/Dockerfile.tracking",
            "compose.tracking.yaml",
            "Global/scripts/start_tracking_server.py",
            "Global/tests/test_tracking_server.py",
            "Global/tests/test_tracking_integration.py",
        }:
            suites.add("tracking")
        elif path.startswith("Global/src/sixletters/") or path in {
            "dev",
            "Global/Dockerfile",
            "compose.yaml",
            "Global/scripts/start_workspace.py",
            "Global/scripts/container_entrypoint.py",
            "Global/scripts/workspace_health.py",
            "Global/tests/test_workspace_integration.py",
        }:
            suites.add("workspace")
            native = True
            if path.endswith(("__main__.py", "local_settings.py")):
                suites.add("tracking")
        elif path.startswith("Global/tests/test_personal") or path.startswith(
            "Global/tests/test_notebook"
        ):
            suites.add("workspace")
            native = True
        elif path.startswith("Global/tests/test_repository"):
            continue
        else:
            native = True
            suites.update({"workspace", "tracking"})
    return {
        "tests": tests,
        "native": ["windows-2025", "macos-26"] if native else [],
        "docker": sorted(suites),
    }


def event_scope(event_name: str, event: dict) -> dict:
    if event_name != "pull_request" or event["pull_request"]["base"]["ref"] != "develop":
        return select_scope([], full=True)
    pr = event["pull_request"]
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            f"{pr['base']['sha']}...{pr['head']['sha']}",
            "--",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        return select_scope([], full=True)
    return select_scope([p for p in result.stdout.decode().split("\0") if p])


if __name__ == "__main__":
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    scope = event_scope(os.environ["GITHUB_EVENT_NAME"], event)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        for key, value in scope.items():
            stream.write(f"{key}={json.dumps(value)}\n")
    print(json.dumps(scope))
