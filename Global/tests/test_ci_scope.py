"""Keep selective CI conservative across complete GitHub changes and real Git history."""

import json
import subprocess

import pytest

from scripts import ci_scope

EMPTY = {"test_files": [], "native_mode": "none", "native_tests": []}

DOCS = {**EMPTY, "tests": False, "tests_mode": "none", "smoke": False, "native": [], "docker": []}
CODE = {**EMPTY, "tests": True, "tests_mode": "full", "smoke": False, "native": [], "docker": []}
WORKSPACE = {
    **EMPTY,
    "tests": True,
    "tests_mode": "full",
    "smoke": True,
    "native": [],
    "docker": ["workspace", "recovery"],
}
TRACKING = {
    **EMPTY,
    "tests": True,
    "tests_mode": "full",
    "smoke": False,
    "native": [],
    "docker": ["recovery", "tracking"],
}
TRACKING_SERVER = {**TRACKING, "smoke": True}
PERSONAL = {**CODE, "tests_mode": "personal"}
FULL = {
    **EMPTY,
    "native_mode": "full",
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
    scope = ci_scope.select_scope([path])
    if path.startswith("Global/tests/"):
        assert scope["tests_mode"] == "focused"
        assert "Global/tests/test_docker_workflow.py" in scope["test_files"]
        assert scope["docker"] == ci_scope.DOCKER
    elif path == "Global/src/sixletters/reproduction.py":
        assert scope == {
            **TRACKING,
            "native": ci_scope.NATIVE,
            "native_mode": "focused",
            "native_tests": ["Global/tests/test_reproduction.py"],
        }
    else:
        assert scope == TRACKING


@pytest.mark.parametrize(
    "path",
    [
        "Global/scripts/smoke_check.py",
        "Global/tests/test_smoke_check.py",
        "Global/notebooks/00_environment_check.ipynb",
    ],
)
def test_environment_check_changes_execute_the_check_without_native_or_container_builds(path):
    scope = ci_scope.select_scope([path])
    if path.startswith("Global/tests/"):
        assert scope == {**CODE, "tests_mode": "focused", "test_files": [path], "smoke": True}
    else:
        assert scope == {**CODE, "smoke": True}


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
    expects_native = path.endswith("conftest.py") or path in ci_scope.NATIVE_SOURCE_TESTS
    assert scope["native"] == (ci_scope.NATIVE if expects_native else [])
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
        "Global/scripts/ci_shards.py",
        "Global/tests/test_ci_shards.py",
        "Global/tests/test_ci_workflow.py",
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
    expected = {**FULL, "native": [], "native_mode": "none"}
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
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary.md"))

    ci_scope.main()

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "previous=value"
    assert {
        key: json.loads(value) for key, value in (line.split("=", 1) for line in lines[1:])
    } == DOCS
    assert json.loads(capsys.readouterr().out) == DOCS


def test_unit_test_edit_selects_consumers_without_repeating_unrelated_platforms():
    scope = ci_scope.select_scope(["Global/tests/test_experiments.py"])
    assert scope == {
        **DOCS,
        "tests": True,
        "tests_mode": "focused",
        "test_files": [
            "Global/tests/test_comparison_diagnostics.py",
            "Global/tests/test_comparison_uncertainty.py",
            "Global/tests/test_experiments.py",
        ],
    }


def test_file_lock_test_edit_keeps_focused_windows_and_macos_checks():
    path = "Global/tests/test_personal_env.py"
    scope = ci_scope.select_scope([path])
    assert scope["tests_mode"] == scope["native_mode"] == "focused"
    assert scope["test_files"] == scope["native_tests"] == [path]
    assert scope["native"] == ci_scope.NATIVE
    assert scope["docker"] == []


def test_mixed_personal_and_test_edits_keep_both_validation_contracts():
    scope = ci_scope.select_scope(["Members/member/train.py", "Global/tests/test_final_metrics.py"])
    assert scope["tests_mode"] == "focused"
    assert set(scope["test_files"]) == {
        "Global/tests/test_final_metrics.py",
        *ci_scope.PERSONAL_TESTS,
    }


@pytest.mark.parametrize("invalid", ["deleted", "helper", "syntax", "unknown_import", "plugin"])
def test_uncertain_test_graph_falls_back_to_full_checks(tmp_path, invalid):
    directory = tmp_path / "Global/tests"
    directory.mkdir(parents=True)
    changed = "Global/tests/test_changed.py"
    (directory / "test_changed.py").write_text("def test_value():\n    assert True\n")
    if invalid == "deleted":
        (directory / "test_changed.py").unlink()
    elif invalid == "helper":
        changed = "Global/tests/helpers.py"
        (directory / "helpers.py").write_text("VALUE = 1\n")
    elif invalid == "syntax":
        (directory / "test_changed.py").write_text("def test_value(\n")
    elif invalid == "unknown_import":
        (directory / "test_changed.py").write_text("from test_missing import helper\n")
    else:
        (directory / "test_changed.py").write_text('pytest_plugins = ["helpers"]\n')
    assert ci_scope.select_scope([changed], root=tmp_path) == FULL


def test_test_import_consumers_are_transitive(tmp_path):
    directory = tmp_path / "Global/tests"
    directory.mkdir(parents=True)
    for name, source in {
        "test_one.py": "VALUE = 1\n",
        "test_two.py": "from test_one import VALUE\n",
        "test_three.py": "import test_two\n",
        "test_other.py": "VALUE = 2\n",
    }.items():
        (directory / name).write_text(source)
    scope = ci_scope.select_scope(["Global/tests/test_one.py"], root=tmp_path)
    assert scope["test_files"] == [
        "Global/tests/test_one.py",
        "Global/tests/test_three.py",
        "Global/tests/test_two.py",
    ]


@pytest.mark.parametrize(
    "extra",
    [
        '\n[build-system]\nrequires = ["hatchling"]\n',
        '\n[tool.pytest.ini_options]\naddopts = "-x"\n',
        '\n[project.optional-dependencies]\necgs = ["wfdb"]\n',
    ],
)
def test_registration_change_cannot_hide_build_test_or_dependency_changes(extra):
    before = '[project]\nname = "fixture"\n[project.scripts]\nold = "package:main"\n'
    after = before.replace("old =", "new =")
    assert ci_scope.scripts_only_change(before, after)
    assert not ci_scope.scripts_only_change(before, after + extra)
    assert not ci_scope.scripts_only_change(before, after.replace('"fixture"', '"renamed"'))
    assert not ci_scope.scripts_only_change(before, "not valid toml")


def test_registered_command_change_runs_actual_entrypoint_and_fresh_terminal_tests():
    scope = ci_scope.select_scope(["Global/pyproject.toml"], scripts_only=True)
    assert scope == {
        **DOCS,
        "tests": True,
        "tests_mode": "focused",
        "test_files": sorted(ci_scope.CLI_TESTS),
        "docker": ["cli"],
        "native": ci_scope.NATIVE,
        "native_mode": "focused",
        "native_tests": ["Global/tests/test_cli_registration.py"],
    }
    assert (
        ci_scope.select_scope(["Global/pyproject.toml", "Global/uv.lock"], scripts_only=True)
        == FULL
    )


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ([], ["cli"]),
        (["Global/tests/test_cli_registration.py"], ["cli"]),
        (["Global/tests/test_docker_cli.py"], ["cli"]),
        (["Global/tests/test_docker_workflow.py"], ["workspace", "recovery"]),
        (["compose.yaml"], ["workspace", "recovery"]),
        (["Global/src/sixletters/tracking.py"], ["cli", "recovery", "tracking"]),
        (["compose.tracking.yaml"], ["cli", "recovery", "tracking"]),
        (["Global/src/sixletters/comparison.py"], ["cli", "tracking"]),
        (["Global/src/sixletters/experiment.py"], ci_scope.DOCKER),
        (["Global/uv.lock"], ci_scope.DOCKER),
    ],
)
def test_command_registration_keeps_other_changes_required_docker_checks(extra, expected):
    scope = ci_scope.select_scope(["Global/pyproject.toml", *extra], scripts_only=True)
    assert scope["docker"] == expected


@pytest.mark.parametrize(
    "path",
    ["Global/tests/test_docker_cli.py", "Global/tests/test_cli_registration.py"],
)
def test_command_test_edits_keep_actual_container_validation(path):
    scope = ci_scope.select_scope([path])
    assert scope["tests_mode"] == "focused"
    assert path in scope["test_files"]
    assert scope["docker"] == ["cli"]


def test_full_workspace_check_covers_cli_without_duplicate_container_startup():
    scope = ci_scope.select_scope(["Global/tests/test_docker_workflow.py"])
    assert "Global/tests/test_docker_cli.py" in scope["test_files"]
    assert scope["docker"] == ["workspace", "recovery"]
    assert "cli" not in ci_scope.full_scope()["docker"]


def test_registration_diff_is_proven_against_pr_merge_base_not_latest_commit(history):
    root, git, commit, _ = history
    project = '[project]\nname = "fixture"\n[project.scripts]\nold = "package:main"\n'
    baseline = {"Global/pyproject.toml": project}
    baseline.update({name: "def test_help():\n    assert True\n" for name in ci_scope.CLI_TESTS})
    base = commit(baseline)
    git("switch", "-c", "feature")
    commit({"Global/pyproject.toml": project.replace("old =", "new =")})
    head = commit({"README.md": "Follow-up documentation\n"})
    expected = ci_scope.select_scope(["Global/pyproject.toml"], root=root, scripts_only=True)
    assert expected["tests_mode"] == "focused"
    git("switch", "develop")
    advanced = commit(
        {"Global/pyproject.toml": project + '\n[dependency-groups]\ndev = ["pytest"]\n'}
    )
    # The checkout is the PR head/merge tree in CI; dependency changes on develop
    # do not change the proven scripts-only delta on this branch.
    git("switch", "feature")
    assert ci_scope.event_scope("pull_request", pr_event(base, head)) == expected
    assert ci_scope.event_scope("pull_request", pr_event(advanced, head)) == expected


@pytest.mark.parametrize("source", sorted(ci_scope.NATIVE_SOURCE_TESTS))
def test_platform_sensitive_sources_keep_focused_native_coverage(source):
    scope = ci_scope.select_scope([source])
    assert scope["tests_mode"] == "full"
    assert scope["native_mode"] == "focused"
    assert set(scope["native_tests"]) == ci_scope.NATIVE_SOURCE_TESTS[source]
    assert scope["native"] == ci_scope.NATIVE
