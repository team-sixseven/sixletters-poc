"""Layout changes preserve saved run identity and existing personal training code."""

import json
import os
import subprocess
import sys

import pytest
from test_tracking import setup_tracking  # noqa: F401

from scripts.migrate_layout import migrate_layout
from sixletters import tracking
from sixletters.comparison import compare_runs
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.local_settings import project_root, resolve_workspace_path, write_settings
from sixletters.workspace import init_experiment


def test_migrated_runs_keep_scores_model_bytes_and_server_identity(request):
    artifact_root, client, make_run, enable = request.getfixturevalue("setup_tracking")
    root = artifact_root.parent
    first = make_run("팀원가")
    second = make_run("팀원나")
    # Represent the previous contributor layout rather than today's scaffold layout.
    for owner in ("팀원가", "팀원나"):
        folder = root / "experiments" / owner
        (folder / "experiments/exp001").rename(folder / "exp001")
        (folder / "experiments").rmdir()
        (folder / "notebooks").rmdir()
    before = {
        file.relative_to(artifact_root): file.read_bytes()
        for file in artifact_root.rglob("*")
        if file.is_file()
    }
    original = compare_runs([first, second], root / "before.csv")
    enable()
    assert tracking.auto_publish(first) == "uploaded"
    server_id = next(iter(client.runs))
    (root / "Global").mkdir()
    (root / "Global/pyproject.toml").touch()

    migrate_layout(root)
    moved_root = root / "Global/artifacts"
    moved = [moved_root / path.relative_to(artifact_root) for path in (first, second)]
    for relative, content in before.items():
        assert (moved_root / relative).read_bytes() == content
    compared = compare_runs(moved, root / "after.csv")
    assert original.to_dict("records") == compared.to_dict("records")
    synced = tracking.sync_runs(moved_root)
    assert synced["uploaded"] == 1 and synced["already_synced"] == 1
    assert client.create_count == 2 and server_id in client.runs
    assert tracking.archive_run(moved[0]) == "uploaded"
    assert client.create_count == 2
    assert not artifact_root.exists()


def test_migrated_existing_train_script_uses_new_data_and_result_locations(tmp_path):
    root = tmp_path / "팀 프로젝트"
    (root / "experiments/팀원").mkdir(parents=True)
    (root / "pyproject.toml").touch()
    save_prepared_dataset(make_demo_dataset(), root / "data/prepared/demo-v1", synthetic=True)
    config = init_experiment("팀원", "exp001", root=root / "scaffold")
    script = config.with_name("train.py")
    script.write_text(
        script.read_text(encoding="utf-8").replace(
            'root / "Global/artifacts"', 'root / "artifacts"'
        ),
        encoding="utf-8",
    )
    previous_code = script.read_bytes()
    config.parent.rename(root / "experiments/팀원/exp001")
    write_settings(root, "팀원", root / "data/prepared/demo-v1")
    (root / "Global").mkdir()
    (root / "Global/pyproject.toml").touch()
    (root / "pyproject.toml").unlink()
    migrate_layout(root)

    script = root / "Members/팀원/experiments/exp001/train.py"
    assert project_root(script.parent) == root
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parent,
        env={**os.environ, "MLFLOW_TRACKING_URI": ""},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert script.read_bytes() == previous_code
    markers = list((root / "Global/artifacts/팀원/exp001").glob("*/result.json"))
    assert len(markers) == 1
    assert json.loads(markers[0].read_text(encoding="utf-8"))["owner"] == "팀원"
    assert not (root / "artifacts").exists()


def test_shared_path_aliases_keep_custom_paths_and_refuse_unmigrated_data(tmp_path):
    (tmp_path / "Global").mkdir()
    (tmp_path / "Global/pyproject.toml").touch()
    for folder in ("data", "artifacts"):
        assert resolve_workspace_path(tmp_path, tmp_path / folder / "selected") == (
            tmp_path / "Global" / folder / "selected"
        )
    custom = tmp_path / "custom/results"
    assert resolve_workspace_path(tmp_path, custom) == custom
    legacy = tmp_path / "data/prepared"
    legacy.mkdir(parents=True)
    with pytest.raises(ValueError, match="migrate_layout"):
        resolve_workspace_path(tmp_path, legacy)


@pytest.mark.parametrize("legacy", ["data", "artifacts"])
def test_parent_traversal_is_normalized_before_legacy_path_aliasing(tmp_path, legacy):
    (tmp_path / "Global").mkdir()
    (tmp_path / "Global/pyproject.toml").touch()
    expected = tmp_path / "custom-location"
    assert (
        resolve_workspace_path(tmp_path, tmp_path / legacy / ".." / "custom-location") == expected
    )
    assert resolve_workspace_path(tmp_path, tmp_path / "Global" / ".." / legacy / "selected") == (
        tmp_path / "Global" / legacy / "selected"
    )


def test_old_explicit_experiment_root_creates_stable_member_paths(tmp_path):
    (tmp_path / "Global").mkdir()
    (tmp_path / "Global/pyproject.toml").touch()
    config = init_experiment("팀원", "exp001", root=tmp_path / "experiments")
    assert config == tmp_path / "Members/팀원/experiments/exp001/config.toml"
    assert (tmp_path / "Members/팀원/notebooks").is_dir()
    before = config.read_bytes()
    migrate_layout(tmp_path)
    assert config.read_bytes() == before
    assert not (tmp_path / "experiments").exists()
