"""Select tests from the complete PR; keep data checks and integration boundaries explicit."""

import json
import os
import subprocess
from pathlib import Path, PurePosixPath

NATIVE = ["windows-2025", "macos-26"]
DOCKER = ["workspace", "recovery", "tracking"]
DEPENDENCIES = {
    "Global/pyproject.toml",
    "Global/uv.lock",
    "Global/requirements.txt",
    "Global/.python-version",
    ".gitattributes",
}
CI_FILES = {
    "Global/scripts/ci_scope.py",
    "Global/scripts/ci_validate.py",
    "Global/scripts/verify_changes.py",
    "Global/tests/test_ci_scope.py",
    "Global/tests/test_ci_validation.py",
    "Global/tests/test_verification.py",
}
WORKSPACE_ENVIRONMENT = {
    "Global/Dockerfile",
    "compose.yaml",
    ".dockerignore",
    ".env.example",
    "dev",
    "Global/scripts/start_workspace.py",
    "Global/scripts/container_entrypoint.py",
    "Global/scripts/workspace_health.py",
    "Global/scripts/check_environment.py",
    "Global/scripts/tracking_worker.py",
}
TRACKING_ENVIRONMENT = {
    "Global/Dockerfile.tracking",
    "compose.tracking.yaml",
    ".env.tracking.example",
    "Global/scripts/start_tracking_server.py",
}
RESULT_PROTOCOL = {
    "Global/src/sixletters/__init__.py",
    "Global/src/sixletters/__main__.py",
    "Global/src/sixletters/experiment.py",
    "Global/src/sixletters/provenance.py",
    "Global/src/sixletters/workspace.py",
    "Global/src/sixletters/personal_env.py",
    "Global/src/sixletters/notebook_workspace.py",
    "Global/src/sixletters/execution.py",
    "Global/src/sixletters/local_settings.py",
    "Global/scripts/migrate_layout.py",
    "Global/tests/conftest.py",
    "conftest.py",
}
TRACKING_CLIENT = {
    "Global/src/sixletters/tracking.py",
    "Global/src/sixletters/tracking_diagnostics.py",
    "Global/src/sixletters/reproduction.py",
    "Global/tests/test_tracking_docker.py",  # Also supplies the recovery server fixture.
}
SMOKE = {
    "Global/scripts/smoke_check.py",
    "Global/tests/test_smoke_check.py",
    "Global/notebooks/00_environment_check.ipynb",
}


def full_scope() -> dict:
    return {
        "tests": True,
        "tests_mode": "full",
        "smoke": True,
        "native": NATIVE.copy(),
        "docker": DOCKER.copy(),
    }


def select_scope(paths: list[str], *, full: bool = False) -> dict:
    if full or not paths:
        return full_scope()
    tests = smoke = personal = False
    suites = set()
    for path in paths:
        if (
            path in DEPENDENCIES | CI_FILES
            or path.startswith(".github/workflows/")
            or path.startswith("Global/requirements/")
        ):
            return full_scope()
        if (
            PurePosixPath(path).suffix.lower() in {".md", ".rst"}
            or path.startswith(".github/ISSUE_TEMPLATE/")
            or (
                path.startswith(("Global/docs/", "Global/reports/"))
                and PurePosixPath(path).suffix.lower()
                in {
                    ".pdf",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".svg",
                    ".webp",
                    ".docx",
                    ".pptx",
                    ".csv",
                    ".json",
                    ".txt",
                }
            )
        ):
            continue
        if path.startswith("Members/"):
            personal = True
        elif path in WORKSPACE_ENVIRONMENT:
            tests = smoke = True
            suites.update(("workspace", "recovery"))
        elif path in TRACKING_ENVIRONMENT:
            tests = smoke = True
            suites.update(("recovery", "tracking"))
        elif path in RESULT_PROTOCOL or path.startswith("Global/templates/experiment/"):
            tests = True
            suites.update(DOCKER)
        elif path in TRACKING_CLIENT:
            tests = True
            suites.update(("recovery", "tracking"))
        elif path == "Global/src/sixletters/comparison.py":
            tests = True
            suites.add("tracking")
        elif path == "Global/tests/test_docker_workflow.py":
            tests = True
            suites.update(("workspace", "recovery"))
        elif path in SMOKE:
            tests = smoke = True
        elif path in {".pre-commit-config.yaml", ".gitignore"} or path.startswith(
            (
                "Global/src/",
                "Global/scripts/",
                "Global/tests/",
                "Global/templates/",
                "Global/notebooks/",
            )
        ):
            tests = True
        else:
            # Unclassified infrastructure must not silently bypass integration checks.
            return full_scope()
    return {
        "tests": tests or personal,
        "tests_mode": "full" if tests else "personal" if personal else "none",
        "smoke": smoke,
        "native": [],
        "docker": [suite for suite in DOCKER if suite in suites],
    }


def event_scope(event_name: str, event: dict) -> dict:
    if not isinstance(event, dict) or event_name != "pull_request":
        # Release pushes and manual checks exercise the whole supported environment.
        return full_scope()
    try:
        pr = event["pull_request"]
        if pr["base"]["ref"] != "develop":
            return full_scope()
        before, after = pr["base"]["sha"], pr["head"]["sha"]
    except (KeyError, TypeError):
        return full_scope()
    if not isinstance(before, str) or not isinstance(after, str) or not before or not after:
        return full_scope()
    changed = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", f"{before}...{after}", "--"],
        capture_output=True,
        check=False,
    )
    if changed.returncode:
        return full_scope()
    paths = changed.stdout.decode("utf-8", errors="replace").rstrip("\0").split("\0")
    return select_scope([path for path in paths if path])


def main() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    scope = event_scope(os.environ["GITHUB_EVENT_NAME"], event)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as handle:
        for key, value in scope.items():
            handle.write(f"{key}={json.dumps(value)}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        rows = [
            "## Selected checks",
            "",
            "- Quality and repository file rules: required",
            f"- Linux tests: {scope['tests_mode']}",
            f"- Environment smoke check: {'required' if scope['smoke'] else 'not selected'}",
            f"- Native compatibility: {', '.join(scope['native']) or 'not selected'}",
            f"- Docker integration: {', '.join(scope['docker']) or 'not selected'}",
        ]
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write("\n".join(rows) + "\n")
    print(json.dumps(scope))


if __name__ == "__main__":
    main()
