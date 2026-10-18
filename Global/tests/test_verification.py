"""Local verification includes branch history and retains every repository policy hook."""

import subprocess

import pytest

from scripts import verify_changes as verification
from scripts.ci_scope import select_scope


@pytest.fixture
def history(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-b", "develop")
    git("config", "user.name", "Verification Test")
    git("config", "user.email", "verification@example.invalid")
    git("config", "core.hooksPath", str(tmp_path / "no-hooks"))

    def write(name, text="value\n"):
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)
        return file

    write("README.md")
    write("Global/src/original.py", "pass\n")
    git("add", ".")
    git("commit", "--no-gpg-sign", "-m", "initial")
    git("update-ref", "refs/remotes/origin/develop", "HEAD")
    git("switch", "-c", "experiment/test")
    return root, git, write


def test_default_scope_includes_commits_staged_unstaged_and_untracked(history):
    root, git, write = history
    write("Members/연구자/experiments/one/train.py", "pass\n")
    git("add", ".")
    git("commit", "--no-gpg-sign", "-m", "personal")
    write("Members/연구자/requirements.in")
    git("add", ".")
    write("README.md", "new text\n")
    write("Members/연구자/experiments/two/config.toml")
    changes = verification.changed_files(root)
    assert changes.certain
    assert changes.base == "origin/develop"
    assert set(changes.paths) == {
        "Members/연구자/experiments/one/train.py",
        "Members/연구자/requirements.in",
        "Members/연구자/experiments/two/config.toml",
        "README.md",
    }
    assert select_scope(list(changes.paths))["tests_mode"] == "personal"


def test_staged_explicitly_excludes_committed_and_untracked_work(history):
    root, git, write = history
    write("Global/src/shared.py", "pass\n")
    git("add", ".")
    git("commit", "--no-gpg-sign", "-m", "shared")
    write("README.md", "staged\n")
    git("add", "README.md")
    write("Members/member/train.py", "pass\n")
    assert verification.changed_files(root, staged=True).paths == ("README.md",)
    assert "Global/src/shared.py" in verification.changed_files(root).paths


def test_deleted_and_renamed_common_code_still_select_full_tests(history):
    root, git, write = history
    write("Global/docs/copied.md", "pass\n")
    (root / "Global/src/original.py").unlink()
    git("add", ".")
    git("commit", "--no-gpg-sign", "-m", "move")
    changes = verification.changed_files(root)
    assert "Global/src/original.py" in changes.paths
    assert select_scope(list(changes.paths))["tests_mode"] == "full"


def test_missing_base_is_conservative_but_explicit_bad_base_is_an_error(history):
    root, git, _ = history
    git("update-ref", "-d", "refs/remotes/origin/develop")
    assert verification.changed_files(root).base == "develop"
    git("branch", "-D", "develop")
    assert not verification.changed_files(root).certain
    with pytest.raises(ValueError, match="기준 브랜치"):
        verification.changed_files(root, base="missing")
    with pytest.raises(ValueError, match="함께"):
        verification.changed_files(root, base="HEAD", staged=True)


@pytest.mark.parametrize("mode", ["none", "personal", "full"])
def test_local_and_ci_use_the_same_test_selection(mode):
    command = verification.test_command(mode)
    if mode == "none":
        assert command is None
    elif mode == "personal":
        assert command[-len(verification.PERSONAL_TESTS) :] == list(verification.PERSONAL_TESTS)
    else:
        assert not any(path.startswith("Global/tests/") for path in command)


@pytest.mark.parametrize("full", [False, True])
def test_verify_runs_all_policy_hooks_and_selected_tests(tmp_path, monkeypatch, full):
    target = tmp_path / "Members/member/notes.ipynb"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    monkeypatch.setattr(
        verification,
        "changed_files",
        lambda *args, **kwargs: verification.Changes(
            ("Members/member/notes.ipynb",), True, "develop"
        ),
    )
    monkeypatch.setenv("SKIP", "notebook-clean,tracked-files,no-commit-to-branch")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(verification.subprocess, "run", run)
    assert verification.verify(tmp_path, full=full) == 0
    assert calls[0][0][:4] == [verification.sys.executable, "-m", "pre_commit", "run"]
    assert "SKIP" not in calls[0][1]["env"]
    assert ("--all-files" in calls[0][0]) == full
    assert calls[1][0] == verification.test_command("full" if full else "personal")


def test_policy_failure_or_notebook_cleanup_stops_before_tests(tmp_path, monkeypatch):
    monkeypatch.setattr(
        verification, "changed_files", lambda *args, **kwargs: verification.Changes((), False, None)
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(verification.subprocess, "run", run)
    assert verification.verify(tmp_path) == 1
    assert len(calls) == 1
    assert "--all-files" in calls[0]


def test_metadata_only_changes_keep_policy_checks_without_pytest(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("documentation")
    monkeypatch.setattr(
        verification,
        "changed_files",
        lambda *args, **kwargs: verification.Changes(("README.md",), True, "develop"),
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(verification.subprocess, "run", run)
    assert verification.verify(tmp_path) == 0
    assert len(calls) == 1


def test_staged_policy_uses_precommit_index_inspection_not_working_files(tmp_path, monkeypatch):
    target = tmp_path / "Members/member/train.py"
    target.parent.mkdir(parents=True)
    target.write_text("pass\n")
    monkeypatch.setattr(
        verification,
        "changed_files",
        lambda *args, **kwargs: verification.Changes(("Members/member/train.py",), True, None),
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(verification.subprocess, "run", run)
    assert verification.verify(tmp_path, staged=True) == 0
    assert calls[0] == [verification.sys.executable, "-m", "pre_commit", "run"]


def test_focused_test_command_only_accepts_explicit_repository_files():
    files = ["Global/tests/test_personal_env.py", "Global/tests/test_experiments.py"]
    command = verification.test_command("focused", files + files)
    assert command[-2:] == sorted(files)
    for invalid in [
        [],
        ["-x"],
        ["Global/tests/../outside.py"],
        ["Global/tests/test_one.py::test_case"],
        "Global/tests/test_one.py",
        [3],
    ]:
        with pytest.raises(ValueError, match="repository test files"):
            verification.test_command("focused", invalid)
    with pytest.raises(ValueError, match="only accepted"):
        verification.test_command("full", files)


def test_focused_runner_rejects_missing_or_outside_files(tmp_path, monkeypatch):
    path = "Global/tests/test_selected.py"
    with pytest.raises(ValueError, match="검사 파일"):
        verification.run_tests(tmp_path, "focused", [path])
    selected = tmp_path / path
    selected.parent.mkdir(parents=True)
    selected.symlink_to(tmp_path / "outside.py")
    with pytest.raises(ValueError, match="검사 파일"):
        verification.run_tests(tmp_path, "focused", [path])
    selected.unlink()
    selected.write_text("def test_example():\n    assert True\n")
    calls = []
    monkeypatch.setattr(
        verification.subprocess,
        "run",
        lambda command, **kwargs: (
            calls.append(command) or subprocess.CompletedProcess(command, 0)
        ),
    )
    assert verification.run_tests(tmp_path, "focused", [path]) == 0
    assert calls[0][-1] == path


def test_local_registration_change_uses_same_semantic_selector_for_worktree_and_index(history):
    root, git, write = history
    project = '[project]\nname = "fixture"\n[project.scripts]\nold = "package:main"\n'
    write("Global/pyproject.toml", project)
    from scripts.ci_scope import CLI_TESTS

    for name in CLI_TESTS:
        write(name, "def test_help():\n    assert True\n")
    git("add", ".")
    git("commit", "--no-gpg-sign", "-m", "command baseline")
    git("update-ref", "refs/remotes/origin/develop", "HEAD")
    write("Global/pyproject.toml", project.replace("old =", "new ="))
    changes = verification.changed_files(root)
    assert verification.local_scope(root, changes)["tests_mode"] == "focused"
    git("add", "Global/pyproject.toml")
    staged = verification.changed_files(root, staged=True)
    assert verification.local_scope(root, staged, staged=True)["tests_mode"] == "focused"
    write(
        "Global/pyproject.toml",
        project.replace("old =", "new =") + '\n[dependency-groups]\ndev = ["pytest"]\n',
    )
    assert verification.local_scope(root, verification.changed_files(root))["tests_mode"] == "full"
