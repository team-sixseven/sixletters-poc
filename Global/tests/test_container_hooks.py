"""The host-side hook installer must preserve a member's existing Git hooks."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="The host launcher runs in WSL/Linux")


@pytest.mark.parametrize("existing_scope", [None, "global", "local"])
def test_hook_registration_respects_effective_configuration(tmp_path, existing_scope):
    root = tmp_path / "한글 저장소"
    root.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[2] / "dev", root / "dev")
    environment = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
        "GIT_CONFIG_NOSYSTEM": "1",
    }

    def git(*arguments, check=True):
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=check,
        )

    git("init")
    if existing_scope:
        git("config", f"--{existing_scope}", "core.hooksPath", "existing-hooks")
    installed = subprocess.run(
        ["sh", str(root / "dev"), "hooks"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if existing_scope:
        assert installed.returncode == 2
        assert git("config", "--get", "core.hooksPath").stdout.strip() == "existing-hooks"
        if existing_scope == "global":
            assert git("config", "--local", "--get", "core.hooksPath", check=False).returncode == 1
    else:
        assert installed.returncode == 0, installed.stderr
        assert git("config", "--local", "--get", "core.hooksPath").stdout.strip() == ".githooks"
