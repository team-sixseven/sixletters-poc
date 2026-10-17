import json

import pandas as pd
import pytest

from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.execution import ENVIRONMENT_KEY, current_invocation, invocation
from sixletters.experiment import Experiment
from sixletters.workspace import init_experiment


def test_invocation_preserves_arguments_without_shell_interpolation(tmp_path):
    script = tmp_path / "nested/custom.py"
    record = invocation(tmp_path, script, ["--depth", "5", "path with spaces", "$(literal)"])
    assert record == {
        "script": "nested/custom.py",
        "args": ["--depth", "5", "path with spaces", "$(literal)"],
        "cwd": "workspace",
        "redacted": False,
    }


def test_sensitive_arguments_are_not_recorded(tmp_path):
    record = invocation(
        tmp_path,
        tmp_path / "train.py",
        [
            "--api-key=secret1",
            "--password",
            "secret2",
            "https://name:secret3@example.test",
            "--steps",
            "12",
        ],
    )
    serialized = json.dumps(record)
    assert all(secret not in serialized for secret in ("secret1", "secret2", "secret3"))
    assert record["redacted"] is True
    assert record["args"][-2:] == ["--steps", "12"]


def test_saved_run_records_actual_entrypoint_and_config_filename(tmp_path, monkeypatch):
    data = save_prepared_dataset(make_demo_dataset(), tmp_path / "data", synthetic=True)
    config = init_experiment("Alice", "example", root=tmp_path / "Members", script="custom.py")
    alternative = config.with_name("balanced.toml")
    alternative.write_text(config.read_text())
    record = invocation(config.parent, config.parent / "custom.py", ["--config", "balanced.toml"])
    monkeypatch.setenv(ENVIRONMENT_KEY, json.dumps(record))
    experiment = Experiment(data, alternative, output_root=tmp_path / "artifacts")
    model = tmp_path / "model.txt"
    model.write_text("opaque")
    predictions = pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5})
    run = experiment.save(
        predictions,
        model_files={"model.txt": model},
        training={"model": "Constant", "seed": 42, "parameters": {}},
    )
    result = json.loads((run / "result.json").read_text())
    assert result["execution"] == record
    assert result["config_filename"] == "balanced.toml"


def test_invocation_cannot_reference_another_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv(
        ENVIRONMENT_KEY,
        json.dumps({"script": "../other.py", "args": [], "cwd": "workspace", "redacted": False}),
    )
    with pytest.raises(ValueError, match="relative Python"):
        current_invocation(tmp_path)
