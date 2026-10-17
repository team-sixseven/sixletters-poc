"""Final progress guides inference without choosing candidates, training, or exposing test scores."""

import json
import shlex

import pandas as pd
import pytest
from test_final_evaluation import candidates, predictions  # noqa: F401

from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment
from sixletters.final_evaluation import FinalEvaluation, freeze_candidates
from sixletters.reproduction import final_plan_status
from sixletters.run_catalog import show_final_status
from sixletters.workspace import init_experiment


def frozen(candidate_fixture):
    root, data, runs = candidate_fixture
    directory = root / "Global/artifacts/final/submission"
    freeze_candidates(runs, data, directory)
    return root, data, runs, directory


def test_status_shows_every_missing_candidate_without_scores_or_execution(candidates, capsys):  # noqa: F811
    root, data, runs, directory = frozen(candidates)
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    summary = final_plan_status(directory, data, root)
    assert summary["counts"] == {"complete": 0, "missing": 2, "invalid": 0}
    assert {item["run_id"] for item in summary["candidates"]} == {run.name for run in runs}
    assert not summary["next_commands"]
    for candidate in summary["candidates"]:
        assert all(
            "train.py" not in cmd and "predict.py" not in cmd for cmd in candidate["commands"]
        )
        assert "사용자 정의" in candidate["message"]
    assert before == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    show_final_status(summary)
    text = capsys.readouterr().out
    assert "미실행 2개" in text
    assert "AUROC" not in text and "patient_id" not in text


def test_status_reports_partial_completion_and_all_complete_next_command(candidates):  # noqa: F811
    root, data, runs, directory = frozen(candidates)
    session = FinalEvaluation(directory, runs[0].name, data)
    session.save(predictions(session.validation_data(), 0.4), predictions(session.test_data(), 0.4))
    partial = final_plan_status(directory, data, root)
    assert partial["counts"] == {"complete": 1, "missing": 1, "invalid": 0}
    assert partial["next_commands"] == []
    session = FinalEvaluation(directory, runs[1].name, data)
    session.save(predictions(session.validation_data(), 0.5), predictions(session.test_data(), 0.5))
    finished = final_plan_status(directory, data, root)
    assert finished["counts"] == {"complete": 2, "missing": 0, "invalid": 0}
    assert shlex.split(finished["next_commands"][0])[:4] == [
        "./dev",
        "final",
        "report",
        "submission",
    ]
    assert all(not item["commands"] for item in finished["candidates"])


def test_corrupt_receipt_is_invalid_without_blocking_other_candidates(candidates):  # noqa: F811
    root, data, runs, directory = frozen(candidates)
    session = FinalEvaluation(directory, runs[0].name, data)
    stored = session.save(
        predictions(session.validation_data(), 0.4), predictions(session.test_data(), 0.4)
    )
    receipt = stored / "receipt.json"
    value = json.loads(receipt.read_text())
    value["plan_hash"] = "changed"
    receipt.write_text(json.dumps(value))
    summary = final_plan_status(directory, data, root)
    assert summary["counts"] == {"complete": 0, "missing": 1, "invalid": 1}
    assert summary["candidates"][0]["commands"] == []


def test_status_rejects_wrong_data_without_changing_plan(candidates):  # noqa: F811
    root, data, _, directory = frozen(candidates)
    dataset = make_demo_dataset()
    dataset.dataset_id = "other-cohort"
    changed = save_prepared_dataset(dataset, root / "Global/data/other", synthetic=True)
    with pytest.raises(ValueError, match="같은 데이터"):
        final_plan_status(directory, changed, root)
    assert not (directory / "predictions").exists()


def test_standard_predictor_command_uses_existing_matching_source_and_shell_quotes(candidates):  # noqa: F811
    root, data, _ = candidates
    config = init_experiment("Member One", "quoted experiment", root=root / "Members")
    experiment = Experiment(data, config, output_root=root / "Global/artifacts")
    model = root / "not-executed.pkl"
    model.write_bytes(b"Not a pickle; status must never deserialize it")
    run = experiment.save(
        pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5}),
        model_files={"model.pkl": model},
        training={"model": "Constant", "seed": None, "parameters": {}},
    )
    directory = root / "Global/artifacts/final/quoted plan"
    freeze_candidates([run], data, directory)
    summary = final_plan_status(directory, data, root)
    commands = summary["candidates"][0]["commands"]
    assert len(commands) == 1
    assert shlex.split(commands[0])[:6] == [
        "./dev",
        "exec",
        "--owner",
        "Member One",
        "quoted experiment",
        "predict.py",
    ]
    assert "--run-id" in commands[0] and run.name in commands[0]
    config.with_name("predict.py").write_text("raise RuntimeError('Do not execute')")
    changed = final_plan_status(directory, data, root)
    restore, sync, predict = changed["candidates"][0]["commands"]
    assert shlex.split(restore)[:2] == ["./dev", "restore"]
    assert shlex.split(sync)[:3] == ["./dev", "env", "sync"]
    assert shlex.split(predict)[:2] == ["./dev", "exec"]
    assert "final-" in predict
    assert not list(config.parent.parent.glob("final-*"))
