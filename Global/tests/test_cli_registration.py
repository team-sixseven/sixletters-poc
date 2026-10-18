"""The declared command must exist in the installed package and launch real help."""

import importlib.metadata
import os
import subprocess
import sysconfig
import tomllib
from pathlib import Path


def test_installed_commands_match_project_registration_and_launch_help(tmp_path):
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    declared = project["scripts"]
    distribution = importlib.metadata.distribution(project["name"])
    installed = {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }
    assert installed == declared, "Reinstall the project after changing command registration"
    for name in declared:
        launcher = Path(sysconfig.get_path("scripts")) / (
            name + (".exe" if os.name == "nt" else "")
        )
        assert launcher.is_file(), f"Installed command is missing: {name}"
        completed = subprocess.run(
            [str(launcher), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout
