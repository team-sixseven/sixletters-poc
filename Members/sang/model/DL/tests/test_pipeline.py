"""Synthetic acceptance checks; these scores do not measure clinical performance."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from Members.sang.model.config import load_basic_config, load_yaml
from Members.sang.model.DL.composed import build_composed_model
from Members.sang.model.DL.dataloader.dataset import collate_patient_windows
from Members.sang.model.DL.dataloader.raw_waveform import (
    RawPatientSequenceDataset,
    load_raw_cohort,
)
from Members.sang.model.DL.dataloader.segmented_ecg import read_patient_split
from Members.sang.model.DL.raw_pipeline import load_raw_evaluation_cohort, train_raw_ecg
from Members.sang.model.evaluation.evaluation import evaluate_saved_model
from Members.sang.model.fusion import _aligned_predictions, run_fusion

MODEL = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_yaml(path, value):
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def test_drive_relative_config_paths(tmp_path):
    config = {
        "drive_root": str(tmp_path / "drive"),
        "data": {"clinical": {"path": "drive://clincal/clinical_clean.csv"}},
    }
    path = tmp_path / "basic.yaml"
    write_yaml(path, config)
    assert load_basic_config(path)["data"]["clinical"]["path"] == str(
        tmp_path / "drive/clincal/clinical_clean.csv"
    )
    config["data"]["clinical"]["path"] = "drive://../outside.csv"
    write_yaml(path, config)
    with pytest.raises(ValueError, match="Invalid drive-relative path"):
        load_basic_config(path)


@pytest.fixture
def raw_setup(tmp_path):
    wave = tmp_path / "wave"
    wave.mkdir()
    rows, quality = [], []
    for i, split in enumerate(("train",) * 4 + ("val",) * 2 + ("test",) * 2):
        pid, count = f"{i:04}", 4 + i % 2
        signal = np.random.default_rng(i).normal(size=(count * 60 + 7, 2 + i % 2))
        np.save(wave / f"{pid}.npy", signal.astype(np.float16))
        rows.append({"pid": pid, "split": split, "SCD": i % 2})
        quality.extend({"pid": pid, "window_index": w, "usable": int(w != 1)} for w in range(count))
    pd.DataFrame(rows).to_csv(tmp_path / "split.csv", index=False)
    pd.DataFrame(quality).to_csv(tmp_path / "quality.csv", index=False)
    basic = load_basic_config(MODEL / "basic.yaml")
    basic["data"]["raw_ecg"].update(
        split_path=str(tmp_path / "split.csv"),
        wave_directory=str(wave),
        sample_rate_hz=20,
        segment_seconds=3,
        source_metadata_confirmed=True,
        quality_index_path=str(tmp_path / "quality.csv"),
        quality_index_format="standard",
        lead_metadata_path=None,
        normalization="none",
    )
    basic_path = tmp_path / "basic.yaml"
    write_yaml(basic_path, basic)
    training = load_yaml(MODEL / "DL/training.yaml")
    training.update(epochs=1, num_workers=0, device="cpu")
    training.update(max_segments_per_patient=3, batch_size=2)
    dl = tmp_path / "DL"
    for subdirectory in ("CNN", "LSTM", "Transformer"):
        (dl / subdirectory).mkdir(parents=True, exist_ok=True)
    write_yaml(dl / "training.yaml", training)
    cnn = load_yaml(MODEL / "DL/CNN/model_set.yaml")
    cnn["models"]["ResNet"].update(width=4, blocks=[1, 1])
    cnn["models"]["InceptionTime"].update(filters=2, modules=1, kernels=[3, 5, 7])
    cnn["models"]["TCN"].update(hidden_dim=4, dilations=[1, 2])
    write_yaml(dl / "CNN/model_set.yaml", cnn)
    lstm = load_yaml(MODEL / "DL/LSTM/model_set.yaml")
    for spec in lstm["models"].values():
        spec["hidden_dim"] = 4
    write_yaml(dl / "LSTM/model_set.yaml", lstm)
    transformer = load_yaml(MODEL / "DL/Transformer/transformer.yaml")
    transformer["models"]["transformer_enc"].update(
        embedding_dim=8, heads=2, layers=1, feedforward_dim=16, dropout=0.0
    )
    write_yaml(dl / "Transformer/transformer.yaml", transformer)
    choice = load_yaml(MODEL / "DL/model_choice.yaml")
    choice_path = dl / "model_choice.yaml"
    write_yaml(choice_path, choice)
    return tmp_path, basic, basic_path, choice, choice_path


def test_raw_loader_quality_gaps_leads_tail_and_missing(raw_setup):
    root, basic, basic_path, _, _ = raw_setup
    cohorts, indices, audit = load_raw_cohort(basic, basic_path)
    assert {s: len(c) for s, c in cohorts.items()} == {
        "train": 4,
        "validation": 2,
        "test": 2,
    }
    assert cohorts["train"].iloc[0].patient_id == "0000"
    assert cohorts["train"].discarded_tail_samples.tolist() == [7] * 4
    dataset = RawPatientSequenceDataset(
        cohorts["train"], indices, basic["data"]["raw_ecg"], 3, "none"
    )
    _, signal, _, positions, lead_mask = dataset[0]
    assert signal.shape == (3, 2, 60)
    assert positions.tolist() == [0, 2, 3]
    assert lead_mask.tolist() == [True, True]
    assert dataset[1][-1].tolist() == [1, 1, 1]
    (root / "wave/0000.npy").unlink()
    cohorts, _, audit = load_raw_cohort(basic, basic_path)
    assert len(cohorts["train"]) == 3
    assert any(
        r["patient_id"] == "0000" and r["reason"] == "missing_wave_file"
        for r in audit["exclusions"]
    )


def test_quality_and_source_gates(raw_setup):
    root, basic, basic_path, _, _ = raw_setup
    basic["data"]["raw_ecg"]["source_metadata_confirmed"] = False
    with pytest.raises(RuntimeError, match="source metadata"):
        load_raw_cohort(basic, basic_path)
    basic["data"]["raw_ecg"]["source_metadata_confirmed"] = True
    quality = pd.read_csv(root / "quality.csv", dtype={"pid": str})
    quality.loc[quality.pid == "0000", "usable"] = 0
    quality.to_csv(root / "quality.csv", index=False)
    cohorts, _, audit = load_raw_cohort(basic, basic_path)
    assert "0000" not in set(cohorts["train"].patient_id)
    assert any(r["reason"] == "no_usable_30_second_segments" for r in audit["exclusions"])
    quality.iloc[:-1].to_csv(root / "quality.csv", index=False)
    with pytest.raises(ValueError, match="one quality row"):
        load_raw_cohort(basic, basic_path)


def test_drive_quality_and_missing_z_are_applied(raw_setup):
    root, basic, basic_path, _, _ = raw_setup
    split = pd.read_csv(root / "split.csv", dtype={"pid": str})
    for pid in split.pid:
        path = root / "wave" / f"{pid}.npy"
        signal = np.load(path)
        if signal.shape[1] == 2:
            signal = np.pad(signal, ((0, 0), (0, 1)))
            np.save(path, signal)
    pd.DataFrame({"file": split.pid + ".npy", "has_z": split.pid != "0000"}).to_csv(
        root / "leads.csv", index=False
    )
    quality = pd.read_csv(root / "quality.csv", dtype={"pid": str})
    drive = pd.DataFrame(
        {
            "pid": quality.pid,
            "file": quality.pid + ".npy",
            "window_idx": quality.window_index,
            "start_idx": quality.window_index * 60,
            "exclude": quality.usable == 0,
        }
    )
    drive.to_csv(root / "drive_quality.csv", index=False)
    basic["data"]["raw_ecg"].update(
        quality_index_format="drive_segment_rpeaks",
        quality_index_path=str(root / "drive_quality.csv"),
        lead_metadata_path=str(root / "leads.csv"),
    )
    cohorts, indices, _ = load_raw_cohort(basic, basic_path)
    dataset = RawPatientSequenceDataset(
        cohorts["train"], indices, basic["data"]["raw_ecg"], 3, "none"
    )
    assert dataset[0][-1].tolist() == [True, True, False]
    assert dataset[0][3].tolist() == [0, 2, 3]
    drive.loc[0, "start_idx"] = 1
    drive.to_csv(root / "drive_quality.csv", index=False)
    with pytest.raises(ValueError, match="alignment"):
        load_raw_cohort(basic, basic_path)
    split.loc[len(split)] = {"pid": "0000-1", "split": "test", "SCD": 0}
    split.to_csv(root / "split.csv", index=False)
    with pytest.raises(ValueError, match="patient split or label"):
        load_raw_cohort(basic, basic_path)


@pytest.mark.parametrize("duplicate", ["0000,train,0", "0000,test,1"])
def test_duplicate_split_or_conflicting_label_rejected(tmp_path, duplicate):
    path = tmp_path / "split.csv"
    path.write_text(f"pid,split,SCD\n0000,train,0\n{duplicate}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate pid"):
        read_patient_split(path)


@pytest.mark.parametrize("cnn", ["ResNet", "InceptionTime", "TCN"])
@pytest.mark.parametrize("sequence", ["BiLSTM", "Attention LSTM FCN", "transformer_enc"])
@pytest.mark.parametrize("pooling", ["attention", "mean", "max"])
def test_all_combinations_padding_and_gradient(raw_setup, cnn, sequence, pooling):
    _, basic, _, choice, choice_path = raw_setup
    set_pooling(choice_path, pooling)
    choice.update(
        cnn_model=cnn,
        sequence_model=sequence,
    )
    model = build_composed_model(choice, basic, choice_path=choice_path)
    x = torch.randn(2, 3, 3, 60)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    positions = torch.tensor([[0.0, 2.0, 5.0], [0.0, 0.0, 0.0]])
    leads = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    changed = x.clone()
    valid_samples = mask[..., None] & leads[:, None, :].bool()
    changed[~valid_samples] = float("nan")
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model(x, mask, positions, leads), model(changed, mask, positions, leads)
        )
    # Train-mode comparison also catches padding entering CNN batch-normalization.
    original, modified = copy.deepcopy(model).train(), copy.deepcopy(model).train()
    torch.manual_seed(42)
    expected = original(x, mask, positions, leads)
    torch.manual_seed(42)
    actual = modified(changed, mask, positions, leads)
    torch.testing.assert_close(expected, actual)
    for name, value in original.named_buffers():
        torch.testing.assert_close(value, dict(modified.named_buffers())[name])
    expected.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in original.encoder.parameters())
    # Adding an absent slot, changing its values, or permuting measured leads
    # must not alter output or create new parameters for another lead count.
    parameter_ids = {name: id(value) for name, value in model.named_parameters()}
    with torch.no_grad():
        reference = model(x, mask, positions, leads)
        extra = torch.cat((x, torch.full_like(x[:, :, :1], float("nan"))), dim=2)
        extra_leads = torch.cat((leads, torch.zeros(2, 1)), dim=1)
        torch.testing.assert_close(reference, model(extra, mask, positions, extra_leads))
        permutation = [2, 0, 1]
        torch.testing.assert_close(
            reference, model(x[:, :, permutation], mask, positions, leads[:, permutation])
        )
        # A fully excluded middle segment is equivalent to compacting it while
        # retaining original timestamps, including its time gap.
        hole_mask = torch.tensor([[True, False, True]])
        hole_x = x[:1].clone()
        hole_x[:, 1] = float("nan")
        torch.testing.assert_close(
            model(hole_x, hole_mask, positions[:1], leads[:1]),
            model(x[:1, [0, 2]], positions=positions[:1, [0, 2]], lead_presence=leads[:1]),
        )
    assert parameter_ids == {name: id(value) for name, value in model.named_parameters()}
    differentiable = x.clone().requires_grad_()
    model(differentiable, mask, positions, leads).sum().backward()
    assert differentiable.grad[0, :, 2].abs().sum() > 0  # The measured third lead is used.
    assert not differentiable.grad[1, :, 2].any()  # The absent third lead is ignored.


@pytest.mark.parametrize("pooling", ["attention", "mean", "max"])
def test_raw_train_restore_evaluation_config_integrity_and_fusion(raw_setup, monkeypatch, pooling):
    root, basic, basic_path, _, choice_path = raw_setup
    set_pooling(choice_path, pooling)
    run = train_raw_ecg(choice_path, limit_train_patients=2)
    assert len(pd.read_csv(run / "train_patients.csv")) == 2
    assert "smoke_train_patient_limit" in set(pd.read_csv(run / "excluded_patients.csv").reason)
    evaluation = {
        "basic_config": str(basic_path),
        "evaluation": {
            "target": "dl",
            "artifact_dir": str(run),
            "split": "test",
            "results_directory": str(root / "results"),
            "mlflow_logging": False,
            "metrics": ["auroc", "average_precision"],
        },
    }
    write_yaml(root / "eval.yaml", evaluation)
    result = evaluate_saved_model(root / "eval.yaml")
    predicted = pd.read_csv(result / "predictions.csv", dtype={"patient_id": str})
    assert set(predicted.patient_id) == {"0006", "0007"}
    assert predicted.p_positive.between(0, 1).all()
    assert all(1 not in json.loads(v) for v in predicted.used_window_indices)
    changed = copy.deepcopy(basic)
    changed["data"]["raw_ecg"]["segment_seconds"] = 30
    with pytest.raises(ValueError, match="segment_seconds"):
        load_raw_evaluation_cohort(run, changed, basic_path)
    split_path = root / "split.csv"
    old_split = split_path.read_bytes()
    split_path.write_bytes(old_split + b"\n")
    with pytest.raises(ValueError, match="split_path changed"):
        load_raw_evaluation_cohort(run, basic, basic_path)
    split_path.write_bytes(old_split)
    # Exercise the real raw-DL fusion path with controlled clinical predictions.
    import Members.sang.model.fusion as fusion

    cohorts, _, audit = load_raw_cohort(basic, basic_path)
    audit["available_patient_ids"]["ml"] = {s: f.patient_id.tolist() for s, f in cohorts.items()}
    monkeypatch.setattr(
        fusion,
        "load_cohorts",
        lambda *_: (None, {"ml": cohorts, "dl": cohorts}, audit, {}),
    )
    monkeypatch.setattr(fusion, "predict_ml_run", lambda _, f: (np.full(len(f), 0.4), {}))
    ml_dir = root / "ml"
    ml_dir.mkdir()
    metadata = json.loads((run / "metadata.json").read_text())
    (ml_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": "synthetic-clinical",
                "run_id": "clinical-stub",
                "target_column": "SCD",
                "target_semantics": metadata["target_semantics"],
                "class_mapping": {"negative": 0, "positive": 1},
            }
        )
    )
    config = {
        "basic_config": str(basic_path),
        "ml": {"artifact_dir": str(ml_dir)},
        "dl": {"artifact_dir": str(run)},
        "fusion": {"mode": "manual", "weights": {"ml": 0.5, "dl": 0.5}},
        "classification": {"threshold": 0.5},
        "missing_prediction_policy": "intersection_and_report",
        "results_directory": str(root / "fusion"),
    }
    write_yaml(root / "fusion.yaml", config)
    fused = run_fusion(root / "fusion.yaml", split_name="test")
    output = pd.read_csv(fused / "test_predictions.csv")
    np.testing.assert_allclose(output.p_final, 0.5 * output.p_ml + 0.5 * output.p_dl)
    assert len(output) == 2


def test_fusion_rejects_split_disagreement():
    ml = pd.DataFrame({"patient_id": ["p"], "split": ["train"], "y_true": [0]})
    dl = ml.assign(split="test")
    with pytest.raises(ValueError, match="split disagreement"):
        _aligned_predictions(ml, np.array([0.1]), dl, np.array([0.2]))


@pytest.mark.parametrize("owner", ["alice", "bob"])
@pytest.mark.parametrize("source", ["raw", "sqi"])
def test_personal_mlflow_training_evaluation_and_retry(raw_setup, monkeypatch, owner, source):
    """Exercise both DL APIs with the real bundle builder and a fake remote uploader."""
    import importlib

    import Members.sang.model.DL.training as legacy
    import Members.sang.model.evaluation.evaluation as evaluation_module
    import Members.sang.model.tracking.check_connection as preflight
    import Members.sang.model.tracking.log_dl as log_dl
    import Members.sang.model.tracking.log_evaluation as log_evaluation

    from sixletters.local_settings import connection, read_settings, write_settings

    uploader = importlib.import_module("sixletters.upload")
    root, basic, basic_path, _, choice_path = raw_setup
    tracking_root = root / "tracking"
    tracking_root.mkdir()
    for key in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_EXPERIMENT_NAME",
    ):
        monkeypatch.delenv(key, raising=False)
    write_settings(
        tracking_root,
        {
            "owner": owner,
            "username": owner,
            "password": "synthetic-private-password",
            "uri": "http://tracking.invalid:5000",
            "experiment": "team-dl",
        },
    )
    monkeypatch.setattr(log_dl, "REPO_ROOT", tracking_root)
    monkeypatch.setattr(log_evaluation, "REPO_ROOT", tracking_root)
    checks, captured = [], []
    monkeypatch.setattr(preflight, "check_mlflow_connection", lambda: checks.append(owner))

    def fake_upload(repo, directory, model=None):
        assert read_settings(repo)["owner"] == owner
        config = connection(repo)
        assert config["username"] == owner and config["experiment"] == "team-dl"
        assert directory.is_relative_to(repo / "Members" / owner)
        metrics, params, files = uploader.bundle(directory, model)
        text = "\n".join(
            path.read_text() for path in files.values() if path.suffix in {".json", ".yaml", ".csv"}
        )
        assert "synthetic-private-password" not in text
        assert "quality_window_counts" not in text
        assert '"0000"' not in text
        assert all("predictions" not in name and "patients.csv" not in name for name in files)
        if params["stage"] == "training":
            assert "models/best_model.pt" in files
            architecture = yaml.safe_load(files["models/model_config.yaml"].read_text())
            # The uploaded public configuration can actually restore the weights.
            from Members.sang.model.DL.composed import build_composed_model_from_specs

            restored = build_composed_model_from_specs(
                architecture["choice"],
                architecture["basic"],
                architecture["cnn_spec"],
                architecture["sequence_spec"],
                architecture["training"],
            )
            checkpoint = torch.load(files["models/best_model.pt"], weights_only=True)
            restored.load_state_dict(checkpoint["state_dict"])
        captured.append(
            {
                "directory": str(directory),
                "params": params,
                "hashes": {name: uploader.digest(path) for name, path in files.items()},
            }
        )
        return {
            "run_id": f"{owner}-{params['stage']}",
            "url": "http://tracking.invalid/run",
        }

    monkeypatch.setattr(uploader, "upload", fake_upload)
    if source == "raw":
        run = train_raw_ecg(choice_path, mlflow_logging=True)
    else:
        cohorts, _, audit = load_raw_cohort(basic, basic_path)
        indices = {}
        for frame in cohorts.values():
            for row in frame.itertuples():
                np.save(
                    row.ecg_path,
                    np.random.default_rng(10).normal(size=(3, 3, 60)).astype("float32"),
                )
                indices[row.patient_id] = np.arange(3)
        write_yaml(basic_path, basic)

        def loader(*_):
            return None, {"ml": cohorts, "dl": cohorts}, audit, indices

        monkeypatch.setattr(legacy, "load_cohorts", loader)
        monkeypatch.setattr(evaluation_module, "load_cohorts", loader)
        run = legacy.train_holter(choice_path, mlflow_logging=True)
    assert checks == [owner]
    assert json.loads((run / "mlflow_status.json").read_text())["status"] == "FINISHED"
    log_dl.log_dl_training_to_mlflow(run)
    assert captured[0] == captured[1]  # Exact same bundle/identity on retry.
    config = {
        "basic_config": str(basic_path),
        "evaluation": {
            "target": "dl",
            "artifact_dir": str(run),
            "split": "test",
            "mlflow_logging": True,
            "results_directory": str(root / "results"),
            "metrics": ["auroc", "average_precision"],
        },
    }
    write_yaml(root / "eval.yaml", config)
    result = evaluate_saved_model(root / "eval.yaml")
    assert json.loads((result / "mlflow_status.json").read_text())["status"] == "FINISHED"
    assert captured[-1]["params"]["source_mlflow_run_id"] == f"{owner}-training"

    # A failed retry cannot invalidate a completed local model or expose the error text.
    def fail(*args, **kwargs):
        raise ConnectionError("synthetic-private-password")

    monkeypatch.setattr(uploader, "upload", fail)
    with pytest.warns(RuntimeWarning, match="전송 미완료"):
        log_dl.finish_dl_training(run, mlflow_logging=True)
    pending = json.loads((run / "mlflow_status.json").read_text())
    assert pending == {"status": "PENDING", "error_type": "ConnectionError"}
    assert json.loads((run / "run_status.json").read_text())["status"] == "COMPLETE"
    assert (run / "best_model.pt").is_file()
    monkeypatch.setattr(uploader, "upload", fake_upload)
    log_dl.log_dl_training_to_mlflow(run)
    assert captured[0] == captured[-1]


def test_failed_personal_account_check_stops_before_training(raw_setup, monkeypatch):
    import Members.sang.model.tracking.check_connection as preflight

    root, _, _, _, choice_path = raw_setup

    def fail():
        raise ConnectionError("Personal account unavailable")

    monkeypatch.setattr(preflight, "check_mlflow_connection", fail)
    with pytest.raises(ConnectionError, match="Personal account"):
        train_raw_ecg(choice_path, mlflow_logging=True)
    assert not (root / "DL/models").exists()


@pytest.mark.parametrize(
    "notebook", ["DL/model_experiment_only_DL.ipynb", "ECG+clinical_experiment.ipynb"]
)
def test_notebook_automatically_logs_training_and_final_evaluation(
    raw_setup, monkeypatch, notebook
):
    """Execute notebook code with synthetic data and fake remote operations."""
    import Members.sang.model.config as config_module
    import Members.sang.model.DL.raw_pipeline as raw_module
    import Members.sang.model.evaluation.evaluation as evaluation_module
    import Members.sang.model.tracking.check_connection as preflight
    import nbformat

    root, basic, basic_path, choice, choice_path = raw_setup
    run = train_raw_ecg(choice_path)
    (run / "mlflow_status.json").write_text('{"status":"FINISHED"}')
    trained, evaluated = [], []
    monkeypatch.setattr(
        config_module,
        "load_experiment_config",
        lambda *_: (choice, basic, choice_path, basic_path),
    )
    monkeypatch.setattr(preflight, "check_mlflow_connection", lambda: None)

    def fake_train(path, **kwargs):
        assert kwargs["mlflow_logging"] is True
        trained.append(path)
        return run

    def fake_evaluate(path, *, split_override=None):
        settings = load_yaml(path)["evaluation"]
        assert settings["mlflow_logging"] is True
        split = split_override or settings["split"]
        if settings["target"] == "fusion":
            fusion = load_yaml(Path(settings["fusion_config"]))
            if split == "test":
                assert fusion["fusion"]["mode"] == "saved"
                assert Path(fusion["fusion"]["saved_run_dir"]) == evaluated[0][1]
        result = root / f"notebook_{split}"
        result.mkdir()
        (result / "metrics.json").write_text(json.dumps({"split": split}))
        predictions = pd.DataFrame({"patient_id": ["synthetic"], "p_positive": [0.3]})
        predictions.to_csv(result / "predictions.csv", index=False)
        predictions.to_csv(result / f"{split}_predictions.csv", index=False)
        evaluated.append((split, result))
        return result

    monkeypatch.setattr(raw_module, "train_raw_ecg", fake_train)
    monkeypatch.setattr(evaluation_module, "evaluate_saved_model", fake_evaluate)
    content = nbformat.read(MODEL / notebook, as_version=4)
    namespace = {"display": lambda *_: None}
    for cell in content.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, notebook, "exec"), namespace)
    if notebook.startswith("DL/"):
        assert trained == [choice_path]
        assert [split for split, _ in evaluated] == ["test"]
    else:
        assert not trained
        assert [split for split, _ in evaluated] == ["validation", "test"]


def test_mixed_lead_and_time_padding_are_independent(raw_setup):
    _, basic, basic_path, choice, choice_path = raw_setup
    cohorts, indices, _ = load_raw_cohort(basic, basic_path)
    dataset = RawPatientSequenceDataset(
        cohorts["train"], indices, basic["data"]["raw_ecg"], 10, "none"
    )
    first, second = dataset[0], dataset[1]
    ids, waves, _, time_mask, positions, lead_mask = collate_patient_windows([first, second])
    assert ids == ["0000", "0001"]
    assert waves.shape == (2, 4, 3, 60)
    assert time_mask.tolist() == [[True, True, True, False], [True] * 4]
    assert lead_mask.tolist() == [[True, True, False], [True, True, True]]
    assert positions[0, :3].tolist() == [0, 2, 3]
    torch.testing.assert_close(waves[0, :3, :2], first[1])
    assert not waves[0, :, 2].any() and not waves[0, 3].any()
    model = build_composed_model(choice, basic, choice_path=choice_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(waves, time_mask, positions, lead_mask).squeeze(1), torch.tensor([0.0, 1.0])
    )
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    shorter = (second[0], second[1][..., :-1], *second[2:])
    with pytest.raises(ValueError, match="no implicit cropping or resampling"):
        collate_patient_windows([first, shorter])


@pytest.mark.parametrize("activation", ["GELU", "ReLU", "LeakyReLU", "SiLU"])
@pytest.mark.parametrize("sequence", ["BiLSTM", "Attention LSTM FCN", "transformer_enc"])
def test_automatic_feature_dimensions_and_activations(raw_setup, activation, sequence):
    _, basic, _, choice, choice_path = raw_setup
    from Members.sang.model.config import dl_config_paths

    paths = dl_config_paths(choice_path)
    cnn = load_yaml(paths["cnn"])
    cnn["models"]["ResNet"].update(width=5, activation=activation)
    write_yaml(paths["cnn"], cnn)
    head_path = paths["transformer"] if sequence == "transformer_enc" else paths["lstm"]
    heads = load_yaml(head_path)
    if sequence == "transformer_enc":
        heads["models"][sequence].update(embedding_dim=12, heads=3, activation=activation)
    else:
        heads["models"][sequence].update(hidden_dim=7)
        if sequence == "Attention LSTM FCN":
            heads["models"][sequence]["activation"] = activation
    write_yaml(head_path, heads)
    training = load_yaml(paths["training"])
    training["mil"].update(hidden_dim=9, activation=activation)
    training["classifier"].update(hidden_dim=11, activation=activation)
    write_yaml(paths["training"], training)
    model = build_composed_model(
        choice | {"sequence_model": sequence}, basic, choice_path=choice_path
    ).eval()
    assert model.encoder.output_dim == 10
    assert model.sequence.feature_dim == (12 if sequence == "transformer_enc" else 14)
    assert model.mil.score[0].in_features == model.sequence.feature_dim
    assert model.classifier[0].in_features == model.sequence.feature_dim
    # The same model also accepts a different number of samples and leads.
    for samples, leads in ((60, 2), (137, 3), (91, 4)):
        output = model(torch.randn(2, 3, leads, samples))
        assert output.shape == (2, 1) and torch.isfinite(output).all()


def test_experiment_yaml_has_no_registry_or_shape_fields():
    paths = [
        "CNN/model_set.yaml",
        "LSTM/model_set.yaml",
        "Transformer/transformer.yaml",
        "model_choice.yaml",
        "training.yaml",
    ]
    forbidden = {"implementation", "feature_method", "input", "output", "channels", "parameters"}

    def inspect(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    for path in paths:
        inspect(load_yaml(MODEL / "DL" / path))
    assert set(load_yaml(MODEL / "DL/model_choice.yaml")) == {"cnn_model", "sequence_model"}
    basic = load_yaml(MODEL / "basic.yaml")
    assert "leads" not in basic["data"]["ecg"]
    assert "samples_per_window" not in basic["data"]["ecg"]


@pytest.mark.parametrize("pooling", ["attention", "mean", "max"])
def test_snapshot_restore_independent_of_changed_model_yaml(raw_setup, pooling):
    from Members.sang.model.config import dl_config_paths
    from Members.sang.model.DL.raw_pipeline import predict_raw_dl_run

    _, basic, basic_path, choice, choice_path = raw_setup
    paths = dl_config_paths(choice_path)
    set_pooling(choice_path, pooling)
    write_yaml(choice_path, choice | {"sequence_model": "transformer_enc"})
    training = load_yaml(paths["training"])
    if pooling == "attention":
        training["mil"]["hidden_dim"] = 9
    write_yaml(paths["training"], training)
    run = train_raw_ecg(choice_path)
    cohorts, indices, _ = load_raw_cohort(basic, basic_path)
    original, _, _ = predict_raw_dl_run(run, cohorts["test"], indices, device=torch.device("cpu"))
    changed = load_yaml(paths["cnn"])
    changed["models"]["ResNet"]["width"] = 17
    write_yaml(paths["cnn"], changed)
    training["mil"] = {"pooling": "max" if pooling != "max" else "mean"}
    write_yaml(paths["training"], training)
    restored, metadata, _ = predict_raw_dl_run(
        run, cohorts["test"], indices, device=torch.device("cpu")
    )
    np.testing.assert_allclose(original, restored)
    assert metadata["model_format_version"] == 2
    assert metadata["lead_strategy"] == "shared_single_lead_cnn_masked_mean"
    assert metadata["mil_pooling"] == pooling


def test_fixed_lead_checkpoint_is_not_silently_reinterpreted(tmp_path):
    from Members.sang.model.inference import load_dl_run

    (tmp_path / "metadata.json").write_text('{"model_format_version":1}')
    with pytest.raises(ValueError, match="predates shared-lead"):
        load_dl_run(tmp_path)


def test_legacy_window_loader_preserves_actual_sample_and_lead_shape(tmp_path):
    from Members.sang.model.data import load_patient_waveform

    signal = np.random.default_rng(42).normal(size=(4, 2, 77)).astype("float32")
    path = tmp_path / "patient.npy"
    np.save(path, signal)
    config = load_basic_config(MODEL / "basic.yaml")["data"]["ecg"]
    values, indices, lead_mask = load_patient_waveform(
        "patient", path, np.arange(4), config, return_lead_mask=True, max_windows=3
    )
    assert values.shape == (3, 2, 77)
    assert indices.tolist() == [0, 1, 3]
    assert lead_mask.tolist() == [True, True]


def set_pooling(choice_path, pooling):
    from Members.sang.model.config import dl_config_paths

    path = dl_config_paths(choice_path)["training"]
    training = load_yaml(path)
    training["mil"] = (
        (training["mil"] | {"pooling": pooling}) if pooling == "attention" else {"pooling": pooling}
    )
    write_yaml(path, training)


@pytest.mark.parametrize("pooling", ["attention", "mean", "max"])
def test_mil_pooling_values_masking_gradient_and_empty_bags(pooling):
    from Members.sang.model.DL.composed import build_mil

    mil = build_mil(2, {"pooling": pooling})
    if pooling == "attention":
        # Equal learned scores give a known, uniform valid-segment average.
        for parameter in mil.parameters():
            torch.nn.init.zeros_(parameter)
    features = torch.tensor(
        [
            [[-4.0, -1.0], [float("nan"), float("inf")], [-2.0, -6.0]],
            [[-10.0, -5.0], [-3.0, -8.0], [float("nan"), float("nan")]],
        ],
        requires_grad=True,
    )
    mask = torch.tensor([[True, False, True], [True, True, False]])
    result = mil(features, mask)
    expected = [[-2.0, -1.0], [-3.0, -5.0]] if pooling == "max" else [[-3.0, -3.5], [-6.5, -6.5]]
    torch.testing.assert_close(result, torch.tensor(expected))
    changed = features.detach().clone()
    changed[~mask] = 1e10
    torch.testing.assert_close(result, mil(changed, mask))
    result.sum().backward()
    assert torch.isfinite(features.grad).all()
    assert not features.grad[~mask].any()
    single = torch.tensor([[[7.0, -2.0], [float("nan"), float("nan")]]])
    torch.testing.assert_close(mil(single, torch.tensor([[True, False]])), single[:, 0])
    with pytest.raises(ValueError, match="at least one valid segment"):
        mil(features.detach(), torch.zeros_like(mask))
    with pytest.raises(ValueError, match="boolean"):
        mil(features.detach(), mask.float())


def test_mil_selection_validation_and_attention_checkpoint_default():
    from Members.sang.model.DL.composed import MaskedAttentionMIL, build_mil

    # Existing v2 attention snapshots without a pooling key still restore.
    assert isinstance(build_mil(4, {"hidden_dim": 9}), MaskedAttentionMIL)
    with pytest.raises(ValueError, match="Unknown MIL pooling"):
        build_mil(4, {"pooling": "unknown"})
    with pytest.raises(ValueError, match="attention-only"):
        build_mil(4, {"pooling": "mean", "hidden_dim": 9})


@pytest.mark.parametrize("name", ["Attention LSTM FCN", "BiLSTM"])
def test_lstm_yaml_uses_constructor_defaults(name):
    from Members.sang.model.DL.composed import SEQUENCE_MODELS

    parameters = load_yaml(MODEL / "DL/LSTM/model_set.yaml")["models"][name]
    assert set(parameters) <= {"activation", "dropout"}
    sequence = SEQUENCE_MODELS[name](8, **parameters)
    assert sequence.lstm.hidden_size == 32 and sequence.lstm.num_layers == 1
    assert sequence.feature_dim == 64
    if name == "Attention LSTM FCN":
        assert [conv.kernel_size for conv in sequence.convs] == [(3,), (5,)]


@pytest.mark.parametrize("name", ["Attention LSTM FCN", "BiLSTM"])
def test_single_layer_lstm_dropout_is_effective_and_disabled_in_eval(name):
    from Members.sang.model.DL.composed import SEQUENCE_MODELS

    sequence = SEQUENCE_MODELS[name](8, dropout=1.0)
    features = torch.randn(2, 3, 8)
    mask = torch.ones(2, 3, dtype=torch.bool)
    positions = torch.arange(3).expand(2, -1)
    sequence.train()
    training = sequence.forward_features(features, mask, positions)
    # Attention LSTM FCN applies dropout to its FCN branch; BiLSTM to its output.
    dropped = training[..., 32:] if name == "Attention LSTM FCN" else training
    assert not dropped.any()
    sequence.eval()
    prediction = sequence.forward_features(features, mask, positions)
    assert prediction.abs().sum() > 0
    torch.testing.assert_close(prediction, sequence.forward_features(features, mask, positions))


@pytest.mark.parametrize("name", ["Segment Attention MIL", "MIL-TCN"])
def test_legacy_segment_mil_uses_defaults_without_yaml(name, tmp_path):
    from Members.sang.model.DL.models import (
        ARCHITECTURE_CONFIGS,
        build_model,
        load_model_parameters,
        load_model_spec,
    )

    assert not (MODEL / "DL/Segment_MIL/model_set.yaml").exists()
    assert all("Segment_MIL" not in path.parts for path in ARCHITECTURE_CONFIGS)
    assert load_model_parameters(name) == {}
    assert load_model_spec(name, (tmp_path / "nonexistent.yaml",)) == {}
    model = build_model(name)
    layer = model.encoder[0] if name == "Segment Attention MIL" else model.temporal[0]
    assert (layer.out_features if name == "Segment Attention MIL" else layer.out_channels) == 32
    logits = model(torch.randn(2, 5, 16))
    assert logits.shape == (2, 1) and torch.isfinite(logits).all()
    torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([[0.0], [1.0]])
    ).backward()
    assert model.classifier.weight.grad is not None


def test_transformer_yaml_uses_former_values_as_defaults_and_restores(raw_setup):
    from Members.sang.model.config import dl_config_paths
    from Members.sang.model.inference import load_dl_run

    _, _, _, choice, choice_path = raw_setup
    parameters = load_yaml(MODEL / "DL/Transformer/transformer.yaml")
    assert set(parameters["models"]["transformer_enc"]) == {"dropout", "activation"}
    write_yaml(dl_config_paths(choice_path)["transformer"], parameters)
    write_yaml(choice_path, choice | {"sequence_model": "transformer_enc"})
    run = train_raw_ecg(choice_path)
    restored, _, resolved, _ = load_dl_run(run, torch.device("cpu"))
    sequence = restored.sequence
    assert sequence.feature_dim == 64
    assert len(sequence.encoder.layers) == 2
    for layer in sequence.encoder.layers:
        assert layer.self_attn.num_heads == 4
        assert layer.linear1.out_features == 128
        assert layer.dropout.p == 0.1
    assert resolved["sequence_spec"] == parameters["models"]["transformer_enc"]
    result = restored(torch.randn(2, 3, 3, 60))
    assert result.shape == (2, 1) and torch.isfinite(result).all()


@pytest.mark.parametrize("name", ["ResNet", "InceptionTime", "TCN"])
def test_cnn_yaml_retains_only_requested_options_and_defaults_restore(raw_setup, name):
    from Members.sang.model.config import dl_config_paths
    from Members.sang.model.inference import load_dl_run

    _, _, _, choice, choice_path = raw_setup
    configuration = load_yaml(MODEL / "DL/CNN/model_set.yaml")
    for model_name, parameters in configuration["models"].items():
        expected_keys = (
            {"activation", "modules"} if model_name == "InceptionTime" else {"activation"}
        )
        assert set(parameters) == expected_keys
    write_yaml(dl_config_paths(choice_path)["cnn"], configuration)
    write_yaml(choice_path, choice | {"cnn_model": name})
    run = train_raw_ecg(choice_path)
    restored, _, resolved, _ = load_dl_run(run, torch.device("cpu"))
    cnn = restored.encoder.model
    if name == "ResNet":
        assert cnn.stem[0].out_channels == 32
        assert len(cnn.layer1) == len(cnn.layer2) == 2
        assert cnn.feature_dim == 64
    elif name == "InceptionTime":
        assert len(cnn.features) == 3
        for module in cnn.features:
            assert [branch.kernel_size for branch in module.branches] == [(9,), (19,), (39,)]
            assert all(branch.out_channels == 16 for branch in module.branches)
        assert cnn.feature_dim == 64
    else:
        assert len(cnn.features) == 3
        assert [block.network[0].dilation for block in cnn.features] == [(1,), (2,), (4,)]
        assert all(block.network[0].kernel_size == (3,) for block in cnn.features)
        assert all(block.network[0].out_channels == 32 for block in cnn.features)
        assert cnn.feature_dim == 32
    assert resolved["cnn_spec"] == configuration["models"][name]
    result = restored(torch.randn(2, 3, 3, 60))
    assert result.shape == (2, 1) and torch.isfinite(result).all()


@pytest.mark.parametrize("pooling", ["attention", "mean", "max"])
@pytest.mark.parametrize("path_kind", ["checkpoint", "directory"])
def test_evaluation_loads_model_path_without_retraining(raw_setup, monkeypatch, pooling, path_kind):
    import hashlib

    import Members.sang.model.DL.raw_pipeline as raw_module
    import Members.sang.model.DL.training as training_module

    root, _, basic_path, _, choice_path = raw_setup
    set_pooling(choice_path, pooling)
    run = train_raw_ecg(choice_path)
    checkpoint = run / "best_model.pt"
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def no_training(*_args, **_kwargs):
        raise AssertionError("Saved-model evaluation must not call training")

    monkeypatch.setattr(raw_module, "train_raw_ecg", no_training)
    monkeypatch.setattr(training_module, "train_holter", no_training)
    reference = checkpoint if path_kind == "checkpoint" else run
    config = {
        "basic_config": str(basic_path),
        "evaluation": {
            "target": "dl",
            "model_path": reference.relative_to(root).as_posix(),
            "artifact_dir": str(run) if path_kind == "directory" else None,
            "split": "test",
            "threshold_override": None,
            "results_directory": "results",
            "mlflow_logging": False,
            "metrics": ["auroc", "average_precision"],
        },
    }
    config_path = root / "eval.yaml"
    write_yaml(config_path, config)
    other = root / "other_working_directory"
    other.mkdir()
    monkeypatch.chdir(other)
    output = evaluate_saved_model(config_path)
    predictions = pd.read_csv(output / "predictions.csv", dtype={"patient_id": str})
    assert set(predictions.patient_id) == {"0006", "0007"}
    assert predictions.p_positive.between(0, 1).all()
    resolved = load_yaml(output / "resolved_config.yaml")
    assert resolved["model_path"] == str(checkpoint)
    assert resolved["model"]["mil_pooling"] == pooling
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("case", ["missing", "incomplete", "conflicting", "wrong_filename"])
def test_evaluation_rejects_invalid_or_ambiguous_model_path(raw_setup, case):
    root, _, basic_path, _, _ = raw_setup
    first, second = root / "first_run", root / "second_run"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        (directory / "best_model.pt").write_bytes(b"not loaded by this test")
        (directory / "metadata.json").write_text("{}")
        (directory / "resolved_config.yaml").write_text("{}")
    evaluation = {
        "target": "dl",
        "model_path": str(first / "best_model.pt"),
        "artifact_dir": None,
        "split": "test",
        "results_directory": "results",
        "mlflow_logging": False,
        "metrics": ["auroc"],
    }
    if case == "missing":
        evaluation["model_path"] = str(root / "missing.pt")
        error, message = FileNotFoundError, "does not exist"
    elif case == "incomplete":
        (first / "metadata.json").unlink()
        error, message = FileNotFoundError, "companion files"
    elif case == "conflicting":
        evaluation["artifact_dir"] = str(second)
        error, message = ValueError, "different runs"
    else:
        (first / "different_weights.pt").write_bytes(b"must not silently use best_model.pt")
        evaluation["model_path"] = str(first / "different_weights.pt")
        error, message = ValueError, "best_model.pt"
    path = root / "eval.yaml"
    write_yaml(path, {"basic_config": str(basic_path), "evaluation": evaluation})
    with pytest.raises(error, match=message):
        evaluate_saved_model(path)
