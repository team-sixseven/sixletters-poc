"""Keep selective CI conservative across complete GitHub changes and real Git history."""

import json
import subprocess

import pytest

from scripts import ci_scope

DOCS = {"tests": False, "tests_mode": "none", "smoke": False, "native": [], "docker": []}
CODE = {"tests": True, "tests_mode": "full", "smoke": False, "native": [], "docker": []}
WORKSPACE = {
    "tests": True,
    "tests_mode": "full",
    "smoke": True,
    "native": [],
    "docker": ["workspace", "recovery"],
}
TRACKING = {
    "tests": True,
    "tests_mode": "full",
    "smoke": False,
    "native": [],
    "docker": ["recovery", "tracking"],
}
TRACKING_SERVER = {**TRACKING, "smoke": True}
PERSONAL = {**CODE, "tests_mode": "personal"}
FULL = {
    "tests": True,
    "tests_mode": "full",
    "smoke": True,
    "native": ["windows-2025", "macos-26"],
    "docker": ["workspace", "recovery", "tracking"],
}


@pytest.mark.parametrize(
    "paths",
    [
        ["README.md", "CONTRIBUTING.md", "Global/docs/tracking.md"],
        [
            "Members/조성률/README.md",
            "Global/src/sixletters/README.md",
            "Global/templates/experiment/README.md",
        ],
        ["Global/reports/결과 보고서.pdf", "Global/docs/architecture.png"],
        [".github/ISSUE_TEMPLATE/experiment.yml", ".github/pull_request_template.md"],
    ],
)
def test_document_only_changes_do_not_install_training_or_container_environments(paths):
    assert ci_scope.select_scope(paths) == DOCS


@pytest.mark.parametrize(
    "paths",
    [
        [
            "Members/조성률/experiments/exp001/train.py",
            "Members/조성률/experiments/exp001/config.toml",
        ],
        [
            "Members/member/experiments/helpers.py",
            "Members/member/notebooks/personal_analysis.ipynb",
        ],
        ["Global/src/sixletters/data/music.py", "Global/tests/test_music.py"],
        ["Global/src/sixletters/evaluation.py", "Global/src/sixletters/features/ecg.py"],
        ["Global/templates/practice/01_music_practice.ipynb"],
        ["Global/scripts/check_repository.py", "Global/tests/test_repository_checks.py"],
        [".gitignore", ".pre-commit-config.yaml"],
    ],
)
def test_personal_and_model_code_keep_linux_tests_without_platform_or_docker_repeats(paths):
    expected = PERSONAL if all(path.startswith("Members/") for path in paths) else CODE
    assert ci_scope.select_scope(["README.md", *paths]) == expected


@pytest.mark.parametrize(
    "path",
    [
        "Global/Dockerfile",
        "compose.yaml",
        ".dockerignore",
        ".env.example",
        "dev",
        "Global/scripts/start_workspace.py",
        "Global/scripts/container_entrypoint.py",
        "Global/scripts/workspace_health.py",
        "Global/scripts/tracking_worker.py",
        "Global/scripts/check_environment.py",
    ],
)
def test_workspace_changes_keep_startup_and_recovery_without_native_platform_repeats(path):
    assert ci_scope.select_scope([path]) == WORKSPACE


@pytest.mark.parametrize(
    "path",
    [
        "Global/Dockerfile.tracking",
        "compose.tracking.yaml",
        ".env.tracking.example",
        "Global/scripts/start_tracking_server.py",
    ],
)
def test_server_changes_keep_recovery_and_tracking_integration(path):
    assert ci_scope.select_scope([path]) == TRACKING_SERVER


@pytest.mark.parametrize(
    "path",
    [
        "Global/src/sixletters/tracking.py",
        "Global/src/sixletters/tracking_diagnostics.py",
        "Global/src/sixletters/reproduction.py",
        "Global/tests/test_tracking_docker.py",
    ],
)
def test_tracking_behavior_keeps_server_and_imported_recovery_fixture_coverage(path):
    assert ci_scope.select_scope([path]) == TRACKING


@pytest.mark.parametrize(
    "path",
    [
        "Global/scripts/smoke_check.py",
        "Global/tests/test_smoke_check.py",
        "Global/notebooks/00_environment_check.ipynb",
    ],
)
def test_environment_check_changes_execute_the_check_without_native_or_container_builds(path):
    assert ci_scope.select_scope([path]) == {**CODE, "smoke": True}


def test_comparison_changes_exercise_uploaded_results_without_workspace_rebuild():
    assert ci_scope.select_scope(["Global/src/sixletters/comparison.py"]) == {
        **CODE,
        "docker": ["tracking"],
    }


def test_docker_test_changes_keep_the_suites_that_use_its_fixtures():
    scope = ci_scope.select_scope(["Global/tests/test_docker_workflow.py"])
    assert scope["tests"] is True
    assert scope["native"] == []
    assert scope["docker"] == ["workspace", "recovery"]


@pytest.mark.parametrize(
    "path",
    [
        "Global/src/sixletters/experiment.py",
        "Global/src/sixletters/provenance.py",
        "Global/src/sixletters/__main__.py",
        "Global/src/sixletters/__init__.py",
        "Global/src/sixletters/local_settings.py",
        "Global/src/sixletters/workspace.py",
        "Global/templates/experiment/train.py",
        "Global/tests/conftest.py",
    ],
)
def test_shared_experiment_and_fixture_changes_keep_all_container_consumers(path):
    scope = ci_scope.select_scope([path])
    assert scope["tests"] is True
    assert scope["native"] == []
    assert scope["docker"] == ["workspace", "recovery", "tracking"]


@pytest.mark.parametrize(
    "path",
    [
        "Global/pyproject.toml",
        "Global/uv.lock",
        "Global/requirements.txt",
        "Global/requirements/gpu.txt",
        "Global/.python-version",
        ".gitattributes",
        ".github/workflows/ci.yml",
        "Global/scripts/ci_scope.py",
        "Global/tests/test_ci_scope.py",
        "Global/scripts/ci_validate.py",
        "Global/tests/test_ci_validation.py",
        "Global/reports/generate.py",
        "Global/docs/build.toml",
        "new-infrastructure/service.toml",
        "Makefile",
    ],
)
def test_dependency_validation_and_unrecognized_paths_require_full_checks(path):
    assert ci_scope.select_scope(["Members/member/experiments/train.py", path]) == FULL


def test_mixed_changes_union_required_suites_without_reintroducing_native_checks():
    paths = ["README.md", "Global/Dockerfile", "Global/src/sixletters/tracking.py"]
    expected = {**FULL, "native": []}
    assert ci_scope.select_scope(paths) == expected
    assert ci_scope.select_scope(list(reversed(paths)) + paths) == expected
    assert ci_scope.select_scope([*paths, "Global/uv.lock"]) == FULL


def test_empty_changes_and_explicit_full_request_do_not_skip_integration():
    assert ci_scope.select_scope([]) == FULL
    assert ci_scope.select_scope(["README.md"], full=True) == FULL


@pytest.mark.parametrize(
    ("event_name", "event"),
    [
        ("workflow_dispatch", {}),
        ("push", {"ref": "refs/heads/main"}),
        ("pull_request", {"pull_request": {"base": {"ref": "main"}}}),
        ("schedule", {}),
        ("unexpected_event", {}),
        ("push", {"ref": "refs/heads/develop"}),
    ],
)
def test_release_manual_and_unexpected_events_always_run_full_checks(event_name, event):
    assert ci_scope.event_scope(event_name, event) == FULL


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"pull_request": {}},
        {"pull_request": "invalid"},
        {"pull_request": {"base": "invalid"}},
        {"pull_request": {"base": {"ref": "develop"}, "head": {"sha": "a" * 40}}},
        {"pull_request": {"base": {"ref": "develop", "sha": "a" * 40}}},
        {
            "pull_request": {
                "base": {"ref": "develop", "sha": None},
                "head": {"sha": "a" * 40},
            }
        },
        {
            "pull_request": {
                "base": {"ref": "develop", "sha": ""},
                "head": {"sha": "a" * 40},
            }
        },
    ],
)
def test_missing_pr_revision_metadata_does_not_reduce_checks(event):
    assert ci_scope.event_scope("pull_request", event) == FULL


@pytest.fixture
def history(tmp_path, monkeypatch):
    """Use actual Git ranges, including a feature branch with more than one commit."""
    root = tmp_path / "repository"
    root.mkdir()

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        ).stdout.strip()

    git("init", "-b", "develop")
    git("config", "user.name", "CI Scope Test")
    git("config", "user.email", "scope-test@example.invalid")
    git("config", "core.hooksPath", str(tmp_path / "no-hooks"))
    git("config", "diff.renames", "true")

    def commit(files):
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if content is None:
                path.unlink()
            else:
                path.write_text(content, encoding="utf-8")
        git("add", "--all")
        git("commit", "--no-gpg-sign", "-m", "scope fixture")
        return git("rev-parse", "HEAD")

    base = commit(
        {
            "README.md": "Initial documentation\n",
            "Global/pyproject.toml": '[project]\nname = "fixture"\n',
            "Global/.python-version": "3.12.12\n",
            "Global/Dockerfile": 'FROM python:3.12-slim\nWORKDIR /app\nCMD ["python"]\n',
        }
    )
    monkeypatch.chdir(root)
    return root, git, commit, base


def pr_event(base, head):
    return {
        "pull_request": {
            "base": {"ref": "develop", "sha": base},
            "head": {"sha": head},
        }
    }


def test_complete_pr_keeps_environment_checks_after_a_documentation_only_followup(history):
    _, git, commit, base = history
    git("switch", "-c", "feature")
    commit({"Global/.python-version": "3.12.13\n"})
    documentation_commit = commit({"README.md": "Explain the new environment\n"})

    assert ci_scope.event_scope("pull_request", pr_event(base, documentation_commit)) == FULL


def test_document_only_pr_skips_training_environments(history):
    _, git, commit, base = history
    git("switch", "-c", "feature")
    head = commit({"README.md": "Updated team instructions\n"})
    assert ci_scope.event_scope("pull_request", pr_event(base, head)) == DOCS


def test_pr_uses_merge_base_so_unrelated_develop_changes_do_not_expand_coverage(history):
    _, git, commit, base = history
    git("switch", "-c", "feature")
    personal_head = commit({"Members/팀원 가/experiments/train.py": "print('personal')\n"})
    git("switch", "develop")
    advanced_base = commit({"Global/.python-version": "3.12.13\n"})
    assert ci_scope.event_scope("pull_request", pr_event(advanced_base, personal_head)) == PERSONAL
    assert ci_scope.event_scope("pull_request", pr_event(base, personal_head)) == PERSONAL


@pytest.mark.parametrize("change", ["delete", "rename"])
def test_deleted_or_renamed_infrastructure_cannot_hide_as_documentation(history, change):
    root, git, commit, base = history
    git("switch", "-c", "feature")
    dockerfile = (root / "Global/Dockerfile").read_text(encoding="utf-8")
    files = {"Global/Dockerfile": None}
    if change == "rename":
        files["Global/docs/container.md"] = dockerfile
    head = commit(files)
    if change == "rename":
        # Git's normal rename detection reports only the new documentation name.
        assert git("diff", "--name-only", f"{base}...{head}") == "Global/docs/container.md"
    assert ci_scope.event_scope("pull_request", pr_event(base, head)) == WORKSPACE


def test_missing_git_history_and_empty_diff_require_full_checks(history):
    _, _, _, base = history
    assert ci_scope.event_scope("pull_request", pr_event(base, "f" * 40)) == FULL
    assert ci_scope.event_scope("pull_request", pr_event(base, base)) == FULL


def test_scope_command_appends_machine_readable_github_outputs(
    history, tmp_path, monkeypatch, capsys
):
    _, git, commit, base = history
    git("switch", "-c", "feature")
    head = commit({"Global/docs/작업 방법.md": "Team instructions\n"})
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(pr_event(base, head)), encoding="utf-8")
    output_path = tmp_path / "output"
    output_path.write_text("previous=value\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_path))

    ci_scope.main()

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "previous=value"
    assert {
        key: json.loads(value) for key, value in (line.split("=", 1) for line in lines[1:])
    } == DOCS
    assert json.loads(capsys.readouterr().out) == DOCS
