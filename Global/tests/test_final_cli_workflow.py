"""The documented baseline/final/restore commands with real sklearn inference."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from sixletters import __main__ as cli
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment
from sixletters.local_settings import write_settings
from sixletters.workspace import init_experiment


def test_six_baselines_freeze_predict_report_and_restore(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    (root / "Global").mkdir()
    (root / "Global/pyproject.toml").write_text('[project]\nname="sixletters-poc"\n')
    data = save_prepared_dataset(make_demo_dataset(), root / "Global/data/demo", synthetic=True)
    write_settings(root, "Alice", data)
    monkeypatch.setattr(cli, "project_root", lambda: root)

    def command(*arguments):
        monkeypatch.setattr(sys, "argv", ["poc", *arguments])
        return cli.main()

    assert command("baselines", "baseline") == 0
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("MLFLOW_", "SIXLETTERS_EXECUTION_"))
    }

    def script(path, *arguments):
        completed = subprocess.run(
            [sys.executable, str(path), *arguments],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    experiments = sorted((root / "Members/Alice/experiments").iterdir())
    assert len(experiments) == 6
    for experiment in experiments:
        script(experiment / "train.py", "--data", str(data))
    runs = sorted(path.parent for path in (root / "Global/artifacts/Alice").glob("*/*/result.json"))
    assert len(runs) == 6
    results = [json.loads((run / "result.json").read_text()) for run in runs]
    assert {result["training"]["model"] for result in results} == {
        "LogisticRegression",
        "RandomForestClassifier",
    }
    assert command("final", "freeze", "submission", "--runs", *map(str, runs)) == 0
    plan = root / "Global/artifacts/final/submission"
    for run in runs:
        experiment = root / "Members/Alice/experiments" / run.parent.name
        script(
            experiment / "predict.py",
            "--plan",
            str(plan),
            "--run-id",
            run.name,
            "--data",
            str(data),
        )
    assert command("final", "report", "submission", "--repeats", "100") == 0
    report = json.loads((plan / "report/final_metrics.json").read_text())
    assert len(report["models"]) == 6
    assert len(report["deltas"]) == 5
    assert command("restore", "reproduced", "--owner", "Bob", "--run", str(runs[0])) == 0
    restored = cli.personal_experiment(root, "Bob", "reproduced")
    script(restored / "train.py", "--data", str(data))
    new = next((root / "Global/artifacts/Bob/reproduced").glob("*/result.json")).parent
    pd.testing.assert_frame_equal(
        pd.read_csv(new / "predictions.csv"), pd.read_csv(runs[0] / "predictions.csv")
    )
    assert (restored / "outputs/.restore/models/model.pkl").is_file()


def test_baselines_does_not_partially_create_when_any_name_exists(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    (root / "Global").mkdir()
    (root / "Global/pyproject.toml").write_text('[project]\nname="sixletters-poc"\n')
    write_settings(root, "Alice", Path("Global/data/demo"))
    existing = root / "Members/Alice/experiments/base-rf-ecg"
    existing.mkdir(parents=True)
    monkeypatch.setattr(cli, "project_root", lambda: root)
    monkeypatch.setattr(sys, "argv", ["poc", "baselines", "base"])
    with pytest.raises(SystemExit):
        cli.main()
    assert list(existing.parent.iterdir()) == [existing]


def test_printed_baseline_commands_execute_the_explicit_owner_with_spaces(
    tmp_path, monkeypatch, capsys
):
    from sixletters import personal_env

    root = tmp_path.resolve()
    (root / "Global").mkdir()
    (root / "Global/pyproject.toml").write_text('[project]\nname="sixletters-poc"\n')
    write_settings(root, "Alice", Path("Global/data/demo"))
    monkeypatch.setattr(cli, "project_root", lambda: root)
    owner = "Bob O'Neil"
    monkeypatch.setattr(sys, "argv", ["poc", "baselines", "shared trial", "--owner", owner])
    assert cli.main() == 0
    commands = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("uv run --project Global poc ")
    ]
    executed = []

    def execute(directory, script, arguments):
        assert (directory / script).is_file()
        executed.append(directory)
        return 0

    monkeypatch.setattr(personal_env, "execute_script", execute)
    for command in commands:
        monkeypatch.setattr(sys, "argv", shlex.split(command)[4:])
        assert cli.main() == 0
    assert len(executed) == 6
    assert all(path.parent == root / "Members" / owner / "experiments" for path in executed)


@pytest.mark.parametrize("source", ["local", "remote"])
def test_interactive_freeze_accepts_one_candidate_without_weakening_comparison(
    tmp_path, monkeypatch, source
):
    from sixletters import reproduction, tracking
    from sixletters.run_catalog import local_entry

    root = tmp_path.resolve()
    (root / "Global").mkdir()
    (root / "Global/pyproject.toml").write_text('[project]\nname="sixletters-poc"\n')
    data = save_prepared_dataset(make_demo_dataset(), root / "Global/data/demo", synthetic=True)
    write_settings(root, "Alice", data)
    config = init_experiment("Alice", "single", root=root / "Members")
    experiment = Experiment(data, config, output_root=root / "Global/artifacts")
    model = root / "constant.txt"
    model.write_text("constant")
    run = experiment.save(
        pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5}),
        model_files={"constant.txt": model},
        training={"model": "ConstantScore", "seed": 42, "parameters": {}},
    )
    monkeypatch.setattr(cli, "project_root", lambda: root)
    monkeypatch.setattr(cli, "prompt", lambda *args: "1")
    entry = local_entry(json.loads((run / "result.json").read_text()), run / "result.json")
    entry["run_id"] = "b" * 32
    entry["archive_status"] = "complete"
    downloads = []

    def fetch(cache, remote_id):
        downloads.append((cache, remote_id))
        return run

    monkeypatch.setattr(tracking, "list_remote_runs", lambda **kwargs: [entry])
    monkeypatch.setattr(reproduction, "fetch_archive", fetch)
    arguments = ["poc", "final", "freeze", "single", "--limit", "1"]
    if source == "remote":
        arguments.append("--remote")
    monkeypatch.setattr(sys, "argv", arguments)
    assert cli.main() == 0
    plan = json.loads((root / "Global/artifacts/final/single/plan.json").read_text())
    assert list(plan["candidates"]) == [run.name]
    assert len(downloads) == (1 if source == "remote" else 0)
    with pytest.raises(ValueError, match="2개 이상"):
        if source == "remote":
            cli.choose_remote_runs([entry], limit=1)
        else:
            cli.choose_runs(root / "Global/artifacts", limit=1)
