"""Frozen validation replay accepts float32 rounding without relaxing other checks."""

import json

import numpy as np
import pandas as pd
import pytest

from sixletters.contracts import _hash_json
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment
from sixletters.final_evaluation import FinalEvaluation, freeze_candidates, generate_final_report
from sixletters.workspace import init_experiment


def _infer(inputs, weights, *, batch_size):
    values = inputs.fillna(0).to_numpy(dtype=np.float32)
    batches = []
    for start in range(0, len(values), batch_size):
        hidden = np.maximum(values[start : start + batch_size] @ weights["hidden"], np.float32(0))
        logits = hidden @ weights["output"]
        batches.append((np.float32(1) / (np.float32(1) + np.exp(-logits))).ravel())
    return pd.DataFrame({"patient_id": inputs.index, "score": np.concatenate(batches)})


@pytest.fixture
def float32_candidate(tmp_path):
    root = tmp_path.resolve()
    (root / "Global").mkdir()
    (root / "Global/pyproject.toml").write_text('[project]\nname="sixletters-poc"\n')
    data = save_prepared_dataset(make_demo_dataset(), root / "Global/data/demo", synthetic=True)
    config = init_experiment("Alice", "float32", root=root / "Members", feature_set="combined")
    experiment = Experiment(data, config)
    rng = np.random.default_rng(7)
    weights = {
        "hidden": rng.normal(0, 0.5, (4, 256)).astype(np.float32),
        "output": rng.normal(0, 0.15, (256, 1)).astype(np.float32),
    }
    model = root / "weights.npz"
    np.savez(model, **weights)
    saved = _infer(experiment.validation_data(), weights, batch_size=36)
    run = experiment.save(
        saved,
        model_files={"weights.npz": model},
        training={"model": "Float32Network", "seed": 7, "parameters": {"hidden_units": 256}},
    )
    directory = root / "Global/artifacts/final/check"
    freeze_candidates([run], data, directory)
    session = FinalEvaluation(directory, run.name, data)
    return directory, data, session, saved


def _frozen_predictions(session, *, batch_size=1):
    with np.load(session.run_directory / "models/weights.npz", allow_pickle=False) as weights:
        return (
            _infer(session.validation_data(), weights, batch_size=batch_size),
            _infer(session.test_data(), weights, batch_size=batch_size),
        )


def test_float32_same_weights_replay_with_different_batch_size(float32_candidate):
    directory, data, session, saved = float32_candidate
    validation, test = _frozen_predictions(session, batch_size=1)
    # BLAS implementations may round batched matrix multiplication differently
    # from one-row inference. Neither the model nor any weights have changed.
    np.testing.assert_allclose(validation.score, saved.score, atol=1e-7, rtol=1e-5)
    destination = session.save(validation, test)
    plan = json.loads((directory / "plan.json").read_text())
    receipt = json.loads((destination / "receipt.json").read_text())
    report = generate_final_report(directory, data, repeats=100)
    protocol = json.loads((report / "protocol.json").read_text())
    assert plan["schema_version"] == 2
    tolerance = {"atol": 1e-7, "rtol": 1e-5}
    for metadata in (plan, receipt, protocol):
        assert metadata["validation_replay_tolerance"] == tolerance
    diagnostics = receipt["validation_replay_diagnostics"]
    assert diagnostics["prediction_count"] == 36
    assert diagnostics["mismatch_count"] == 0
    assert diagnostics["max_absolute_delta"] <= 1.01e-5


def test_materially_different_replay_reports_aggregate_diagnostics(float32_candidate):
    directory, _, session, _ = float32_candidate
    validation, test = _frozen_predictions(session)
    validation["score"] = validation["score"] * 0.9
    with pytest.raises(ValueError, match="Validation replay differs") as failure:
        session.save(validation, test)
    message = str(failure.value)
    assert "Outside tolerance: 36/36" in message
    assert "max absolute delta=" in message
    assert "atol=1e-07" in message
    assert "rtol=1e-05" in message
    assert "DEMO-" not in message
    assert not (directory / "predictions" / session.run_id).exists()
    # A failed check does not consume the once-only result slot.
    validation, test = _frozen_predictions(session)
    session.save(validation, test)
    with pytest.raises(FileExistsError, match="cannot be replaced"):
        session.save(validation, test)


@pytest.mark.parametrize("rehash", [False, True])
def test_frozen_tolerance_cannot_be_relaxed(float32_candidate, rehash):
    directory, data, session, _ = float32_candidate
    path = directory / "plan.json"
    plan = json.loads(path.read_text())
    plan["validation_replay_tolerance"]["atol"] = 0.1
    if rehash:
        plan.pop("plan_hash")
        plan["plan_hash"] = _hash_json(plan)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="modified"):
        FinalEvaluation(directory, session.run_id, data)


def test_legacy_plan_keeps_original_strict_replay_policy(float32_candidate):
    directory, data, session, saved = float32_candidate
    path = directory / "plan.json"
    plan = json.loads(path.read_text())
    plan["schema_version"] = 1
    plan.pop("validation_replay_tolerance")
    plan.pop("plan_hash")
    plan["plan_hash"] = _hash_json(plan)
    path.write_text(json.dumps(plan))
    session = FinalEvaluation(directory, session.run_id, data)
    _, test = _frozen_predictions(session)
    validation = saved.copy()
    validation["score"] = validation["score"].astype(float) + 1e-7
    with pytest.raises(ValueError, match="atol=1e-10"):
        session.save(validation, test)
    destination = session.save(saved, test)
    receipt = json.loads((destination / "receipt.json").read_text())
    assert receipt["validation_replay_tolerance"] == {"atol": 1e-10, "rtol": 1e-8}


def test_final_report_rejects_receipt_with_different_tolerance(float32_candidate):
    directory, data, session, _ = float32_candidate
    destination = session.save(*_frozen_predictions(session))
    path = destination / "receipt.json"
    receipt = json.loads(path.read_text())
    receipt["validation_replay_tolerance"]["rtol"] = 0.1
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="incomplete or modified"):
        generate_final_report(directory, data, repeats=100)
    assert not (directory / "report").exists()


@pytest.mark.parametrize("score", [-1e-8, 1 + 1e-8, np.nan, np.inf])
def test_replay_tolerance_never_relaxes_probability_bounds(float32_candidate, score):
    _, _, session, _ = float32_candidate
    validation, test = _frozen_predictions(session)
    test["score"] = test["score"].astype(float)
    test.loc[test.index[0], "score"] = score
    with pytest.raises(ValueError, match="finite probabilities between 0 and 1"):
        session.save(validation, test)
