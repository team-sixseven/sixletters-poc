"""Short commands preserve local choices while discovering prepared MUSIC inputs."""

import sys
from pathlib import Path

import pytest

from sixletters import __main__ as cli
from sixletters.data.demo import make_demo_dataset
from sixletters.data.music import get_music_destination
from sixletters.data.prepared import save_prepared_dataset
from sixletters.local_settings import SETTINGS_FILE, read_settings, write_settings


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "팀 프로젝트"
    (root / "Members").mkdir(parents=True)
    (root / "Global").mkdir()
    (root / "Global/pyproject.toml").write_text('[project]\nname = "sixletters-poc"\n')
    monkeypatch.chdir(root)
    return root


def invoke(monkeypatch, *arguments, answers=()):
    monkeypatch.setattr(sys, "argv", ["poc", *arguments])
    prompts = []
    values = iter(answers)

    def answer(prompt):
        prompts.append(prompt)
        return next(values)

    monkeypatch.setattr("builtins.input", answer)
    assert cli.main() == 0
    return prompts


@pytest.mark.parametrize("cohort", ["basic", "hrv"])
def test_music_command_forwards_offline_cohort_and_never_changes_personal_settings(
    checkout, monkeypatch, cohort
):
    settings = write_settings(checkout, "팀원가", Path("Global/data/prepared/custom-reviewed-v1"))
    before = settings.read_bytes()
    calls = []

    def prepare(root, *, cohort="basic", offline=False):
        calls.append((root, cohort, offline))
        return get_music_destination(root, cohort) / "dataset.json"

    monkeypatch.setattr("sixletters.data.music.prepare_music", prepare)
    invoke(monkeypatch, "music", "--cohort", cohort, "--offline")
    assert calls == [(checkout, cohort, True)]
    assert settings.read_bytes() == before


def prepare_marker(checkout, cohort):
    # Only availability/default selection is under test; no clinical data is needed.
    destination = get_music_destination(checkout, cohort)
    save_prepared_dataset(make_demo_dataset(), destination, synthetic=True)
    return destination.relative_to(checkout).as_posix()


@pytest.mark.parametrize(
    ("cohorts", "expected"),
    [((), None), (("hrv",), "hrv"), (("basic",), "basic"), (("basic", "hrv"), "basic")],
)
def test_init_suggests_available_music_with_basic_preferred(
    checkout, monkeypatch, cohorts, expected
):
    locations = {cohort: prepare_marker(checkout, cohort) for cohort in cohorts}
    invoke(monkeypatch, "init", "--owner", "팀원가", answers=("",))
    selected = locations[expected] if expected else "Global/data/prepared/demo-v1"
    assert read_settings(checkout) == {"owner": "팀원가", "data": selected}


def test_init_keeps_existing_personal_data_and_explicit_data_takes_precedence(
    checkout, monkeypatch
):
    prepare_marker(checkout, "basic")
    write_settings(checkout, "기존팀원", Path("Global/data/prepared/custom-reviewed-v1"))
    invoke(monkeypatch, "init", "--owner", "팀원가", answers=("",))
    assert read_settings(checkout)["data"] == "Global/data/prepared/custom-reviewed-v1"
    invoke(monkeypatch, "init", "--owner", "팀원가", "--data", "Global/data/prepared/another-v1")
    assert read_settings(checkout)["data"] == "Global/data/prepared/another-v1"


def test_init_can_repair_invalid_settings_using_available_music(checkout, monkeypatch):
    expected = prepare_marker(checkout, "basic")
    (checkout / SETTINGS_FILE).write_text("invalid = [\n", encoding="utf-8")
    invoke(monkeypatch, "init", "--owner", "팀원가", answers=("",))
    assert read_settings(checkout) == {"owner": "팀원가", "data": expected}
