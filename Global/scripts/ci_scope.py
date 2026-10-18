"""Select tests from the complete PR; keep data checks and integration boundaries explicit."""

import ast
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path, PurePosixPath

NATIVE = ["windows-2025", "macos-26"]
# Every scenario starts independently; no scenario implicitly replaces another.
DOCKER = [
    "cli",
    "training",
    "lifecycle",
    "notebooks",
    "member",
    "result-retry",
    "archive-retry",
    "tracking",
]
DOCKER_SUITES = DOCKER
WORKSPACE_SCENARIOS = set(DOCKER) - {"tracking"}
TRANSFER_SCENARIOS = {"result-retry", "archive-retry", "tracking"}
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
    "Global/scripts/ci_docker_cache.py",
    "Global/scripts/ci_shards.py",
    "Global/tests/test_ci_docker_cache.py",
    "Global/tests/test_ci_shards.py",
    "Global/tests/test_ci_workflow.py",
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
}
TRACKING_ENVIRONMENT = {
    "Global/Dockerfile.tracking",
    "compose.tracking.yaml",
    ".env.tracking.example",
    "Global/scripts/start_tracking_server.py",
}
# Select the real container boundaries affected by each shared feature. Result
# format/environment metadata consumers include archive restoration, not only save.
SOURCE_DOCKER = {
    "Global/src/sixletters/__init__.py": set(DOCKER),
    "Global/src/sixletters/__main__.py": set(DOCKER),
    "Global/src/sixletters/experiment.py": {"training", *TRANSFER_SCENARIOS},
    "Global/src/sixletters/provenance.py": {"training", *TRANSFER_SCENARIOS},
    "Global/src/sixletters/workspace.py": {"training", "notebooks", "member", "tracking"},
    "Global/src/sixletters/personal_env.py": {
        "training",
        "notebooks",
        "member",
        *TRANSFER_SCENARIOS,
    },
    "Global/src/sixletters/notebook_workspace.py": {"notebooks"},
    "Global/src/sixletters/execution.py": {"training", "member", "tracking"},
    "Global/src/sixletters/local_settings.py": {
        "cli",
        "training",
        "notebooks",
        "member",
        "tracking",
    },
    "Global/scripts/migrate_layout.py": {"cli", "lifecycle"},
    "Global/scripts/tracking_worker.py": {"result-retry", "archive-retry"},
    "Global/src/sixletters/tracking.py": TRANSFER_SCENARIOS,
    "Global/src/sixletters/tracking_diagnostics.py": TRANSFER_SCENARIOS,
    "Global/src/sixletters/reproduction.py": {"member", "tracking"},
    "Global/src/sixletters/comparison.py": {"tracking"},
}
EXPERIMENT_TEMPLATE_SCENARIOS = {"training", "notebooks", "member", *TRANSFER_SCENARIOS}
# These areas have no Docker/process boundary of their own. Unrecognized common
# code is deliberately not included: adding a new integration must fail closed.
LINUX_ONLY = {
    ".pre-commit-config.yaml",
    ".gitignore",
    "Global/scripts/check_repository.py",
    "Global/src/sixletters/contracts.py",
    "Global/src/sixletters/evaluation.py",
    "Global/src/sixletters/final_evaluation.py",
    "Global/src/sixletters/final_metrics.py",
    "Global/src/sixletters/run_catalog.py",
    "Global/src/sixletters/workspace_status.py",
}
LINUX_ONLY_PREFIXES = (
    "Global/src/sixletters/data/",
    "Global/src/sixletters/features/",
    "Global/templates/practice/",
)

CLI_TESTS = {
    "Global/tests/test_cli_registration.py",
    "Global/tests/test_local_workflow.py",
    "Global/tests/test_usability_commands.py",
    "Global/tests/test_workspace_launcher.py",
}
# These tests exercise OS paths, process locks, interpreters or notebook kernels.
NATIVE_TEST_WORDS = (
    "environment",
    "personal_env",
    "execution",
    "workspace",
    "resources",
    "notebook",
    "local_workflow",
    "layout",
    "reproduction",
)
NATIVE_SOURCE_TESTS = {
    "Global/src/sixletters/personal_env.py": {
        "Global/tests/test_personal_env.py",
        "Global/tests/test_personal_env_publication.py",
        "Global/tests/test_notebook_environment.py",
        "Global/tests/test_shared_environment_resources.py",
    },
    "Global/src/sixletters/notebook_workspace.py": {
        "Global/tests/test_notebook_workspace.py",
        "Global/tests/test_notebook_environment.py",
    },
    "Global/src/sixletters/execution.py": {
        "Global/tests/test_execution_metadata.py",
        "Global/tests/test_personal_environment_cli.py",
    },
    "Global/src/sixletters/local_settings.py": {"Global/tests/test_local_workflow.py"},
    "Global/src/sixletters/workspace.py": {
        "Global/tests/test_workspace.py",
        "Global/tests/test_experiment_clone.py",
    },
    "Global/src/sixletters/experiment.py": {
        "Global/tests/test_experiment_resources.py",
        "Global/tests/test_model_resources.py",
    },
    "Global/src/sixletters/provenance.py": {"Global/tests/test_experiment_provenance.py"},
    "Global/src/sixletters/reproduction.py": {"Global/tests/test_reproduction.py"},
}
# Conservative develop-PR fallback still checks every native boundary, without
# repeating platform-independent statistical/data tests on each operating system.
NATIVE_FALLBACK_TESTS = sorted(set().union(*NATIVE_SOURCE_TESTS.values(), CLI_TESTS))

SMOKE = {
    "Global/scripts/smoke_check.py",
    "Global/tests/test_smoke_check.py",
    "Global/notebooks/00_environment_check.ipynb",
}


def full_scope() -> dict:
    return {
        "tests": True,
        "tests_mode": "full",
        "test_files": [],
        "smoke": True,
        "native": NATIVE.copy(),
        "native_mode": "full",
        "native_shard_count": 4,
        "native_tests": [],
        "docker": DOCKER.copy(),
    }


def pr_full_scope() -> dict:
    """Unknown/shared infrastructure changes need all boundaries, not all OS tests."""
    return {
        **full_scope(),
        "native_mode": "focused",
        "native_shard_count": 2,
        "native_tests": NATIVE_FALLBACK_TESTS.copy(),
    }


def scripts_only_change(before: str, after: str) -> bool:
    """Only installed command registration may bypass dependency-wide validation."""
    try:
        old, new = tomllib.loads(before), tomllib.loads(after)
        old_scripts = old["project"].pop("scripts", {})
        new_scripts = new["project"].pop("scripts", {})
        if not new_scripts or not isinstance(new_scripts, dict):
            return False
        for name, target in new_scripts.items():
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
                return False
            if not isinstance(target, str) or ":" not in target:
                return False
        return old == new and old_scripts != new_scripts
    except (KeyError, TypeError, AttributeError, tomllib.TOMLDecodeError):
        return False


def registered_commands_changed(root: Path, before: str, after: str | None = None) -> bool:
    """Compare the PR merge base with its head (or the current local file)."""

    def contents(revision: str) -> str:
        result = subprocess.run(
            ["git", "show", f"{revision}:Global/pyproject.toml"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise ValueError("Cannot read project metadata")
        return result.stdout.decode("utf-8")

    try:
        current = (
            contents(after)
            if after is not None
            else (root / "Global/pyproject.toml").read_text(encoding="utf-8")
        )
        return scripts_only_change(contents(before), current)
    except (OSError, ValueError, UnicodeError):
        return False


def affected_tests(root: Path, changed: set[str]) -> set[str] | None:
    """Include transitive test-module consumers; uncertainty requires a full run."""
    paths = sorted((root / "Global/tests").glob("test_*.py"))
    modules = {path.stem: path.relative_to(root).as_posix() for path in paths}
    known = set(modules.values())
    if not changed <= known or any(path.is_symlink() for path in paths):
        return None
    dependencies = {}
    try:
        for path in paths:
            imports = set()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == "pytest_plugins":
                    return None
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or "", *(alias.name for alias in node.names)]
                elif isinstance(node, ast.Call) and (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "__import__"
                    or isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                ):
                    if not node.args or not isinstance(node.args[0], ast.Constant):
                        return None
                    names = [str(node.args[0].value)]
                for name in names:
                    for component in name.split("."):
                        if component.startswith("test_"):
                            if component not in modules:
                                return None
                            imports.add(modules[component])
            dependencies[path.relative_to(root).as_posix()] = imports
    except (OSError, UnicodeError, SyntaxError):
        return None
    selected = changed.copy()
    while True:
        expanded = selected | {name for name, imports in dependencies.items() if imports & selected}
        if expanded == selected:
            return selected
        selected = expanded


def select_scope(
    paths: list[str],
    *,
    full: bool = False,
    root: Path | None = None,
    scripts_only: bool = False,
) -> dict:
    if full:
        return full_scope()
    if not paths:
        return pr_full_scope()
    root = root or Path(__file__).resolve().parents[2]
    if any(
        path in CI_FILES
        or path in {"Global/tests/conftest.py", "conftest.py"}
        or path.startswith(".github/workflows/")
        or path.startswith("Global/requirements/")
        or path in DEPENDENCIES
        and not (path == "Global/pyproject.toml" and scripts_only)
        for path in paths
    ):
        return pr_full_scope()
    changed_tests = {path for path in paths if path.startswith("Global/tests/")}
    selected = affected_tests(root, changed_tests) if changed_tests else set()
    if selected is None:
        return pr_full_scope()
    tests = smoke = personal = False
    suites = set()
    native_tests = (
        sorted(
            name
            for name in selected
            if any(word in PurePosixPath(name).stem for word in NATIVE_TEST_WORDS)
            and name
            not in {
                "Global/tests/test_docker_workflow.py",
                "Global/tests/test_docker_workspace_copy.py",
                "Global/tests/test_practice_execution.py",
            }
        )
        if changed_tests
        else []
    )
    if "Global/pyproject.toml" in paths and scripts_only:
        if not all((root / name).is_file() for name in CLI_TESTS):
            return pr_full_scope()
        selected.update(CLI_TESTS)
        suites.add("cli")
    native_selected = set(native_tests)
    for path in paths:
        native_selected.update(NATIVE_SOURCE_TESTS.get(path, set()))
    if "Global/pyproject.toml" in paths and scripts_only:
        native_selected.add("Global/tests/test_cli_registration.py")
    if any(not (root / name).is_file() for name in native_selected):
        return pr_full_scope()
    native_tests = sorted(native_selected)
    for path in [*paths, *selected]:
        if path == "Global/pyproject.toml" and scripts_only:
            continue
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
        if path.startswith("Global/tests/"):
            if path == "Global/tests/test_docker_workflow.py":
                suites.update(WORKSPACE_SCENARIOS)
            elif path in {
                "Global/tests/test_docker_cli.py",
                "Global/tests/test_cli_registration.py",
            }:
                suites.add("cli")
            elif path == "Global/tests/test_tracking_docker.py":
                suites.update(TRANSFER_SCENARIOS)
            elif path == "Global/tests/test_smoke_check.py":
                smoke = True
            continue
        if path.startswith("Members/"):
            personal = True
        elif path in WORKSPACE_ENVIRONMENT:
            tests = smoke = True
            suites.update(WORKSPACE_SCENARIOS)
        elif path in TRACKING_ENVIRONMENT:
            tests = smoke = True
            suites.update(TRANSFER_SCENARIOS)
        elif path in SOURCE_DOCKER:
            tests = True
            suites.update(SOURCE_DOCKER[path])
        elif path.startswith("Global/templates/experiment/"):
            tests = True
            suites.update(EXPERIMENT_TEMPLATE_SCENARIOS)
        elif path in SMOKE:
            tests = smoke = True
        elif path in LINUX_ONLY or path.startswith(LINUX_ONLY_PREFIXES):
            tests = True
        else:
            return pr_full_scope()
    # Mixed personal and focused changes still exercise the shared personal contract.
    if personal and selected:
        selected.update(PERSONAL_TESTS)
    mode = "full" if tests else "focused" if selected else "personal" if personal else "none"
    return {
        "tests": mode != "none",
        "tests_mode": mode,
        "test_files": sorted(selected) if mode == "focused" else [],
        "smoke": smoke,
        "native": NATIVE.copy() if native_tests else [],
        "native_mode": "focused" if native_tests else "none",
        "native_shard_count": 1 if native_tests else 0,
        "native_tests": native_tests,
        "docker": [suite for suite in DOCKER_SUITES if suite in suites],
    }


# Shared by local verification and CI; never execute arbitrary contributor models.
PERSONAL_TESTS = (
    "Global/tests/test_data_contract.py",
    "Global/tests/test_prepared.py",
    "Global/tests/test_experiment_identity.py",
    "Global/tests/test_personal_experiments.py",
    "Global/tests/test_training_metadata.py",
    "Global/tests/test_environment_comparison.py",
)


def event_scope(event_name: str, event: dict) -> dict:
    if not isinstance(event, dict) or event_name != "pull_request":
        # Release pushes and manual checks exercise the whole supported environment.
        return full_scope()
    try:
        pr = event["pull_request"]
        if pr["base"]["ref"] != "develop":
            return full_scope()
    except (KeyError, TypeError):
        return full_scope()
    try:
        before, after = pr["base"]["sha"], pr["head"]["sha"]
    except (KeyError, TypeError):
        return pr_full_scope()
    if not isinstance(before, str) or not isinstance(after, str) or not before or not after:
        return pr_full_scope()
    changed = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", f"{before}...{after}", "--"],
        capture_output=True,
        check=False,
    )
    if changed.returncode:
        return pr_full_scope()
    paths = changed.stdout.decode("utf-8", errors="replace").rstrip("\0").split("\0")
    root = Path.cwd()
    ancestor = subprocess.run(
        ["git", "merge-base", before, after],
        capture_output=True,
        check=False,
    )
    scripts_only = False
    if ancestor.returncode == 0 and "Global/pyproject.toml" in paths:
        scripts_only = registered_commands_changed(root, ancestor.stdout.decode().strip(), after)
    return select_scope([path for path in paths if path], root=root, scripts_only=scripts_only)


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
            f"- Focused test files: {', '.join(scope['test_files']) or 'not selected'}",
            f"- Environment smoke check: {'required' if scope['smoke'] else 'not selected'}",
            f"- Native compatibility: {', '.join(scope['native']) or 'not selected'} ({scope['native_mode']}; {scope['native_shard_count']} shards)",
            f"- Docker integration: {', '.join(scope['docker']) or 'not selected'}",
        ]
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write("\n".join(rows) + "\n")
    print(json.dumps(scope))


if __name__ == "__main__":
    main()
