import json

import pandas as pd
import pytest

from sixletters import final_evaluation
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment
from sixletters.final_evaluation import FinalEvaluation, freeze_candidates, generate_final_report
from sixletters.workspace import init_experiment


@pytest.fixture
def candidates(tmp_path):
    root = tmp_path.resolve()
    (root / "Global").mkdir()
    (root / "Global/pyproject.toml").write_text('[project]\nname="sixletters-poc"\n')
    data = save_prepared_dataset(make_demo_dataset(), root / "Global/data/demo", synthetic=True)
    runs = []
    for index, feature_set in enumerate(("clinical", "combined")):
        config = init_experiment(
            "Alice", f"exp{index}", root=root / "Members", feature_set=feature_set
        )
        experiment = Experiment(data, config, output_root=root / "Global/artifacts")
        model = root / f"constant{index}.txt"
        model.write_text(str(0.4 + index / 10))
        predictions = pd.DataFrame(
            {"patient_id": experiment.validation_data().index, "score": 0.4 + index / 10}
        )
        runs.append(
            experiment.save(
                predictions,
                model_files={"constant.txt": model},
                training={
                    "model": "ConstantScore",
                    "seed": 42,
                    "parameters": {"probability": 0.4 + index / 10},
                },
            )
        )
    return root, data, runs


def predictions(inputs, value):
    return pd.DataFrame({"patient_id": inputs.index, "score": value})


def test_final_freezes_models_replays_validation_and_reports_once(candidates):
    root, data, runs = candidates
    directory = root / "Global/artifacts/final/submission"
    freeze_candidates(runs, data, directory)
    plan = json.loads((directory / "plan.json").read_text())
    assert plan["synthetic"] is True
    assert plan["reference"] == runs[0].name
    assert len(plan["candidates"]) == 2
    for index, run in enumerate(runs):
        session = FinalEvaluation(directory, run.name, data)
        assert "target" not in session.test_data()
        assert set(session.test_data().index).isdisjoint(session.validation_data().index)
        session.save(
            predictions(session.validation_data(), 0.4 + index / 10),
            predictions(session.test_data(), 0.4 + index / 10),
        )
    report = generate_final_report(directory, data, repeats=100)
    for name in (
        "metrics.csv",
        "deltas.csv",
        "final_metrics.json",
        "roc.png",
        "pr.png",
        "calibration.png",
        "confusion.png",
        "protocol.json",
    ):
        assert (report / name).stat().st_size > 0
    assert "patient_id" not in (report / "metrics.csv").read_text()
    assert "patient_id" not in (report / "final_metrics.json").read_text()
    with pytest.raises(FileExistsError, match="already exists"):
        generate_final_report(directory, data, repeats=100)
    session = FinalEvaluation(directory, runs[0].name, data)
    with pytest.raises(ValueError, match="already complete"):
        session.save(
            predictions(session.validation_data(), 0.4), predictions(session.test_data(), 0.8)
        )


def test_test_predictions_cannot_be_saved_from_a_different_model(candidates):
    root, data, runs = candidates
    directory = root / "Global/artifacts/final/check"
    freeze_candidates([runs[0]], data, directory)
    session = FinalEvaluation(directory, runs[0].name, data)
    with pytest.raises(ValueError, match="Validation replay differs"):
        session.save(
            predictions(session.validation_data(), 0.9), predictions(session.test_data(), 0.9)
        )
    assert not (directory / "predictions" / runs[0].name).exists()


def test_freeze_requires_same_data_and_does_not_overwrite_plan(candidates):
    root, data, runs = candidates
    directory = root / "Global/artifacts/final/check"
    freeze_candidates(runs, data, directory, threshold=0.7)
    plan = json.loads((directory / "plan.json").read_text())
    assert all(value["threshold"]["threshold"] == 0.7 for value in plan["candidates"].values())
    with pytest.raises(FileExistsError):
        freeze_candidates(runs, data, directory)
    different = make_demo_dataset()
    different.dataset_id += "-different"
    other = save_prepared_dataset(different, root / "Global/data/other", synthetic=True)
    with pytest.raises(ValueError, match="common data"):
        freeze_candidates(runs, other, root / "another-plan")
    with pytest.raises(ValueError, match="exact frozen data"):
        FinalEvaluation(directory, runs[0].name, other)


def test_final_report_requires_every_frozen_candidate(candidates):
    root, data, runs = candidates
    directory = root / "Global/artifacts/final/check"
    freeze_candidates(runs, data, directory)
    session = FinalEvaluation(directory, runs[0].name, data)
    session.save(predictions(session.validation_data(), 0.4), predictions(session.test_data(), 0.4))
    with pytest.raises(ValueError, match="Missing test predictions"):
        generate_final_report(directory, data, repeats=100)
    assert not (directory / "report").exists()
    with pytest.raises(FileExistsError, match="cannot be replaced"):
        session.save(
            predictions(session.validation_data(), 0.4), predictions(session.test_data(), 0.7)
        )


def test_final_uses_common_test_patients_and_rejects_labels_from_caller(candidates):
    root, data, runs = candidates
    directory = root / "Global/artifacts/final/check"
    freeze_candidates([runs[0]], data, directory)
    session = FinalEvaluation(directory, runs[0].name, data)
    validation = predictions(session.validation_data(), 0.4)
    with pytest.raises(ValueError, match="common test patients"):
        session.save(validation, validation)
    supplied = predictions(session.test_data(), 0.4).assign(target=1)
    with pytest.raises(ValueError, match="labels come from common data"):
        session.save(validation, supplied)


@pytest.mark.parametrize("change", ["model", "plan", "prediction", "engine"])
def test_final_rejects_modified_frozen_evidence(candidates, monkeypatch, change):
    root, data, runs = candidates
    directory = root / "Global/artifacts/final/check"
    freeze_candidates([runs[0]], data, directory)
    session = FinalEvaluation(directory, runs[0].name, data)
    if change == "model":
        (session.run_directory / "models/constant.txt").write_text("modified")
    elif change == "plan":
        plan = json.loads((directory / "plan.json").read_text())
        plan["reference"] = "0" * 32
        (directory / "plan.json").write_text(json.dumps(plan))
    elif change == "prediction":
        destination = session.save(
            predictions(session.validation_data(), 0.4), predictions(session.test_data(), 0.4)
        )
        with (destination / "predictions.csv").open("a") as handle:
            handle.write("bogus,0,0.4\n")
    else:
        monkeypatch.setattr(final_evaluation, "_engine_identity", lambda: {})
    with pytest.raises(ValueError):
        if change == "prediction":
            generate_final_report(directory, data, repeats=100)
        else:
            session.test_data()


def test_freeze_rejects_duplicate_runs_and_invalid_reference(candidates):
    root, data, runs = candidates
    with pytest.raises(ValueError, match="distinct"):
        freeze_candidates([runs[0], runs[0]], data, root / "plan")
    with pytest.raises(ValueError, match="reference"):
        freeze_candidates(runs, data, root / "plan", reference="0" * 32)
    assert not (root / "plan").exists()


def test_freeze_cannot_recursively_copy_into_its_source_run(candidates):
    _, data, runs = candidates
    with pytest.raises(ValueError, match="must not overlap"):
        freeze_candidates([runs[0]], data, runs[0] / "final")
    assert not (runs[0] / "final").exists()


@pytest.mark.parametrize("linked_part", ["parent", "candidate"])
def test_final_predictions_never_write_through_symlinks(candidates, linked_part):
    root, data, runs = candidates
    directory = root / "Global/artifacts/final/check"
    freeze_candidates([runs[0]], data, directory)
    session = FinalEvaluation(directory, runs[0].name, data)
    outside = root / "outside-plan"
    outside.mkdir()
    link = directory / "predictions"
    if linked_part == "candidate":
        link.mkdir()
        link /= runs[0].name
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("This host cannot create directory symlinks")
    with pytest.raises(ValueError, match="must not contain symbolic links"):
        session.save(
            predictions(session.validation_data(), 0.4),
            predictions(session.test_data(), 0.4),
        )
    assert list(outside.iterdir()) == []
    assert link.is_symlink()
    assert not list(directory.glob(".predict-*"))
