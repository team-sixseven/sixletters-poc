import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sixletters.local_settings import SETTINGS_FILE, configured_data, read_settings, write_settings


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "한글 팀 프로젝트"
    (root / "Members").mkdir(parents=True)
    (root / "Global").mkdir()
    (root / "Global/pyproject.toml").write_text(
        '[project]\nname = "sixletters-poc"\n', encoding="utf-8"
    )
    return root


def run(root, *args, input_text=None):
    return subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=root,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def cli(root, *args, **kwargs):
    return run(root, "-m", "sixletters", *args, **kwargs)


def test_init_new_and_data_location_survive_ide_working_directory(checkout):
    assert cli(checkout, "demo").returncode == 0
    completed = cli(checkout, "init", input_text="팀원 가\n\n")
    assert completed.returncode == 0, completed.stderr
    assert read_settings(checkout) == {"owner": "팀원 가", "data": "Global/data/prepared/demo-v1"}
    assert cli(checkout, "new", "첫 실험").returncode == 0
    script = checkout / "Members" / "팀원 가" / "experiments" / "첫 실험" / "train.py"
    # IDEs often start train.py in its own directory; outputs still belong to the checkout.
    trained = run(script.parent, script)
    assert trained.returncode == 0, trained.stderr
    results = list((checkout / "Global/artifacts" / "팀원 가" / "첫 실험").glob("*/result.json"))
    assert len(results) == 1
    result = json.loads(results[0].read_text(encoding="utf-8"))
    assert result["synthetic"] is True
    assert not (script.parent / "artifacts").exists()
    before = script.read_bytes()
    duplicate = cli(checkout, "new", "첫 실험")
    assert duplicate.returncode == 2
    assert script.read_bytes() == before


def test_data_package_keeps_its_public_demo_factory():
    from sixletters.data import make_demo_dataset
    from sixletters.data.demo import make_demo_dataset as implementation

    assert make_demo_dataset is implementation
    assert len(make_demo_dataset().frame) == 180


def test_demo_never_replaces_real_data_selection_or_existing_demo(checkout):
    write_settings(checkout, "팀원", Path("Global/data/prepared/real-v1"))
    assert cli(checkout, "demo").returncode == 0
    bundle = checkout / "Global/data/prepared/demo-v1/dataset.json"
    before = bundle.read_bytes()
    assert cli(checkout, "demo").returncode == 2
    assert bundle.read_bytes() == before
    assert configured_data(checkout) == checkout / "Global/data/prepared/real-v1"


@pytest.mark.parametrize("data", ["..", "../outside", "."])
def test_local_settings_reject_unmounted_locations_without_changing_settings(checkout, data):
    saved = write_settings(checkout, "팀원", Path("Global/data/prepared/cohort-v1"))
    before = saved.read_bytes()
    with pytest.raises(ValueError, match="inside this repository"):
        write_settings(checkout, "다른 팀원", Path(data))
    assert saved.read_bytes() == before


def test_local_data_is_optional_when_personal_script_receives_explicit_arguments(checkout):
    assert cli(checkout, "demo").returncode == 0
    assert cli(checkout, "new", "exp001", "--owner", "팀원").returncode == 0
    script = checkout / "Members/팀원/experiments/exp001/train.py"
    missing = run(checkout, script)
    assert missing.returncode == 2
    assert "Specify --data" in missing.stderr
    assert not (checkout / "Global/artifacts").exists()
    trained = run(checkout, script, "--data", checkout / "Global/data/prepared/demo-v1")
    assert trained.returncode == 0, trained.stderr
    assert not (checkout / SETTINGS_FILE).exists()


def test_new_without_owner_and_init_without_input_fail_clearly(checkout):
    no_owner = cli(checkout, "new", "exp001")
    assert no_owner.returncode == 2
    assert "Register your name" in no_owner.stderr
    no_input = cli(checkout, "init", input_text="")
    assert no_input.returncode == 2
    assert "No input received" in no_input.stderr
    assert not (checkout / SETTINGS_FILE).exists()
