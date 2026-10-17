"""Folder copies and configuration typos cannot silently reassign team results."""

import shutil
import sys
import unicodedata
from pathlib import Path

import pytest

from sixletters import __main__ as cli
from sixletters import personal_env
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment
from sixletters.workspace import init_experiment, validate_experiment_identity


@pytest.mark.parametrize("field, wrong", [("owner", "Alicee"), ("name", "exp002")])
def test_identity_typo_is_rejected_before_loading_data(tmp_path, field, wrong):
    config = init_experiment("Alice", "exp001", root=tmp_path / "Members")
    correct = {"owner": "Alice", "name": "exp001"}[field]
    config.write_text(
        config.read_text().replace(f'{field} = "{correct}"', f'{field} = "{wrong}"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity does not match") as error:
        Experiment(tmp_path / "missing-data", config, tmp_path / "artifacts")
    message = str(error.value)
    assert str(config) in message
    assert "[experiment]" in message
    assert f'{field} = "{correct}"' in message
    assert not (tmp_path / "artifacts").exists()


def test_copying_an_experiment_identifies_both_fields_to_fix(tmp_path):
    original = init_experiment("Bob", "exp001", root=tmp_path / "Members")
    copied = tmp_path / "Members/Alice/experiments/exp002/config.toml"
    shutil.copytree(original.parent, copied.parent)
    with pytest.raises(ValueError, match="identity does not match") as error:
        validate_experiment_identity(copied)
    assert 'owner = "Alice"' in str(error.value)
    assert 'name = "exp002"' in str(error.value)
    copied.write_text(
        copied.read_text()
        .replace('owner = "Bob"', 'owner = "Alice"')
        .replace('name = "exp001"', 'name = "exp002"'),
        encoding="utf-8",
    )
    validate_experiment_identity(copied)


def test_unicode_directory_normalization_and_alternate_config_are_accepted(tmp_path):
    config = init_experiment("조성률", "실험", root=tmp_path / "Members")
    alternate = config.with_name("balanced.toml")
    alternate.write_bytes(config.read_bytes())
    validate_experiment_identity(alternate)
    decomposed = Path(unicodedata.normalize("NFD", str(config)))
    validate_experiment_identity(decomposed, settings={"owner": "조성률", "name": "실험"})


@pytest.mark.parametrize("folder", ["legacy-experiments", "standalone"])
def test_external_api_layout_keeps_configured_result_identity(tmp_path, folder):
    data = save_prepared_dataset(make_demo_dataset(), tmp_path / "data", synthetic=True)
    config = init_experiment("Alice", "exp001", root=tmp_path / folder)
    config.write_text(config.read_text().replace('owner = "Alice"', 'owner = "Bob"'))
    experiment = Experiment(data, config, tmp_path / "artifacts")
    assert experiment.settings["owner"] == "Bob"


def test_cli_rejects_copied_identity_before_creating_or_running_environment(
    tmp_path, monkeypatch, capsys
):
    config = init_experiment("Alice", "exp001", root=tmp_path / "Members")
    config.write_text(config.read_text().replace('owner = "Alice"', 'owner = "Bob"'))
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    calls = []
    monkeypatch.setattr(personal_env, "execute_script", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["poc", "exec", "exp001", "--owner", "Alice", "train.py"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert calls == []
    assert str(config) in capsys.readouterr().err
