"""Comparison uncertainty is derived from verified validation predictions only."""

import json

import numpy as np
import pandas as pd
import pytest
from test_experiments import completed_run, read_result

from sixletters.comparison import compare_runs
from sixletters.data.demo import make_demo_dataset
from sixletters.run_catalog import summary_from_run


def test_comparison_adds_counts_intervals_without_changing_saved_runs(tmp_path):
    runs = [completed_run(tmp_path / str(i), owner=f"member{i}") for i in range(2)]
    before = {
        run: {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
        for run in runs
    }
    table = compare_runs(runs, tmp_path / "comparison.csv")
    for row, run in zip(table.to_dict("records"), runs, strict=True):
        expected = summary_from_run(run)
        assert row["patient_count"] == expected["patients"]
        assert row["positive_count"] == expected["positive"]
        assert row["negative_count"] == expected["negative"]
        assert pd.isna(row["scd_count"])
        for metric, bounds in expected["intervals"].items():
            assert row[f"{metric}_ci_lower"] == bounds["lower"]
            assert row[f"{metric}_ci_upper"] == bounds["upper"]
        assert row["ci_level"] == 0.95
        assert row["ci_repeats"] == 2000
        assert row["ci_seed"] == 42
        assert row["ci_method"] == "stratified_patient_percentile_bootstrap"
        assert {
            p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()
        } == before[run]
    assert {"patient_id", "target", "score"}.isdisjoint(table.columns)
    csv = pd.read_csv(tmp_path / "comparison.csv")
    assert csv.patient_count.tolist() == table.patient_count.tolist()


def test_comparison_recomputes_instead_of_trusting_saved_display_metadata(tmp_path):
    runs = [completed_run(tmp_path / str(i)) for i in range(2)]
    result = read_result(runs[0])
    result["validation_summary"] = {"patients": 9999, "positive": 9999}
    (runs[0] / "result.json").write_text(json.dumps(result))
    table = compare_runs(runs, tmp_path / "comparison.csv")
    assert table.patient_count.iloc[0] == table.patient_count.iloc[1] < 9999
    predictions = pd.read_csv(runs[0] / "predictions.csv")
    predictions.loc[0, "score"] = 0.99
    predictions.to_csv(runs[0] / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="Saved metrics do not match"):
        compare_runs(runs, tmp_path / "tampered.csv")
    assert not (tmp_path / "tampered.csv").exists()


def test_scd_count_only_for_music_endpoint_and_small_class_interval_is_unavailable(tmp_path):
    dataset = make_demo_dataset()
    dataset.dataset_id = "music-1.0.1-scd-basic-v1"
    dataset.target_definition = (
        "music-scd-basic-v1: Cause of death=3 during observed follow-up is 1; others are 0"
    )
    validation = dataset.frame["split"].eq("validation")
    dataset.frame.loc[validation, "target"] = 0
    dataset.frame.loc[dataset.frame.index[validation][0], "target"] = 1
    runs = [completed_run(tmp_path / str(i), dataset=dataset, synthetic=False) for i in range(2)]
    table = compare_runs(runs, tmp_path / "small.csv")
    assert table.scd_count.tolist() == [1, 1]
    assert table.positive_count.tolist() == [1, 1]
    assert table.filter(regex="_ci_(lower|upper)$").isna().all().all()
    assert table.ci_unavailable_reason.eq("insufficient_class_counts").all()
    assert np.isfinite(table.roc_auc).all()


def _paired_frames():
    target = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    reference = pd.DataFrame(
        {
            "patient_id": [f"subject-{index}" for index in range(len(target))],
            "target": target,
            "score": [0.1, 0.7, 0.3, 0.6, 0.4, 0.6, 0.8, 0.9],
        }
    )
    candidate = reference.copy()
    candidate["score"] = [0.2, 0.6, 0.7, 0.3, 0.8, 0.2, 0.7, 0.8]
    return reference, candidate


def test_paired_interval_matches_independent_sklearn_resampling(monkeypatch):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    from sixletters import run_catalog

    # Keep this independent calculation small; production metadata/default tests use 2000.
    monkeypatch.setattr(run_catalog, "_SUMMARY_REPEATS", 200)
    reference, candidate = _paired_frames()
    summary = run_catalog.paired_validation_summary(reference, candidate)
    target = reference.target.to_numpy()
    positive, negative = np.flatnonzero(target == 1), np.flatnonzero(target == 0)
    rng = np.random.default_rng(42)
    metrics = (roc_auc_score, average_precision_score, brier_score_loss)
    draws = []
    for _ in range(200):
        indices = np.r_[
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ]
        draws.append(
            [
                metric(target[indices], candidate.score.to_numpy()[indices])
                - metric(target[indices], reference.score.to_numpy()[indices])
                for metric in metrics
            ]
        )
    expected = np.quantile(draws, [0.025, 0.975], axis=0, method="linear")
    for index, (key, metric) in enumerate(zip(summary["deltas"], metrics, strict=True)):
        assert summary["deltas"][key] == pytest.approx(
            metric(target, candidate.score) - metric(target, reference.score), abs=1e-15
        )
        assert summary["intervals"][key]["lower"] == pytest.approx(expected[0, index], abs=1e-15)
        assert summary["intervals"][key]["upper"] == pytest.approx(expected[1, index], abs=1e-15)
    assert summary["intervals"]["roc_auc"]["lower"] < 0
    assert summary["deltas"]["brier_score"] > 0


def test_paired_intervals_align_by_identifier_and_allow_negative_differences():
    from sixletters.run_catalog import paired_validation_summary

    reference, candidate = _paired_frames()
    rng = np.random.default_rng(19)
    reference["score"] = rng.uniform(0, 0.5, len(reference))
    reference.loc[reference.target.eq(1), "score"] += 0.5
    candidate["score"] = rng.uniform(0.5, 1, len(candidate))
    candidate.loc[candidate.target.eq(1), "score"] -= 0.5
    expected = paired_validation_summary(reference, candidate)
    reordered = paired_validation_summary(
        reference.sample(frac=1, random_state=8), candidate.sample(frac=1, random_state=2)
    )
    assert reordered == expected
    reverse = paired_validation_summary(candidate, reference)
    for key, bounds in expected["intervals"].items():
        assert reverse["deltas"][key] == -expected["deltas"][key]
        assert reverse["intervals"][key]["lower"] == pytest.approx(-bounds["upper"])
        assert reverse["intervals"][key]["upper"] == pytest.approx(-bounds["lower"])
    assert expected["deltas"]["roc_auc"] < 0
    assert expected["intervals"]["roc_auc"]["upper"] < 0


def test_identical_predictions_have_zero_paired_intervals():
    from sixletters.run_catalog import paired_validation_summary

    reference, _ = _paired_frames()
    summary = paired_validation_summary(reference, reference.sample(frac=1, random_state=2))
    assert summary["deltas"] == {key: 0 for key in summary["deltas"]}
    assert all(bounds == {"lower": 0, "upper": 0} for bounds in summary["intervals"].values())
    assert summary["patients"] == 8
    assert summary["positive"] == summary["negative"] == 4
    assert summary["confidence_level"] == 0.95
    assert summary["method"] == "paired_stratified_patient_percentile_bootstrap"
    assert summary["repeats"] == 2000
    assert summary["seed"] == 42


@pytest.mark.parametrize(
    "change", ["different_ids", "different_targets", "empty_id", "duplicate_id"]
)
def test_paired_summary_rejects_misaligned_or_invalid_patients(change):
    from sixletters.run_catalog import paired_validation_summary

    reference, candidate = _paired_frames()
    if change == "different_ids":
        candidate.loc[0, "patient_id"] = "other-subject"
    elif change == "different_targets":
        candidate.loc[[0, 1], "target"] = [1, 0]
    elif change == "empty_id":
        candidate.loc[0, "patient_id"] = " "
    else:
        candidate.loc[0, "patient_id"] = candidate.loc[1, "patient_id"]
    with pytest.raises(ValueError):
        paired_validation_summary(reference, candidate)


def test_wrapper_preserves_existing_columns_and_saved_bundles_and_selects_reference(tmp_path):
    from sixletters.run_catalog import compare_with_reference

    runs = [completed_run(tmp_path / str(i), owner=f"member{i}") for i in range(2)]
    before = {
        run: {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
        for run in runs
    }
    baseline = compare_runs(runs, tmp_path / "baseline.csv")
    reference = read_result(runs[1])["run_id"]
    output = tmp_path / "paired.csv"
    table = compare_with_reference(runs, output, reference=reference)
    pd.testing.assert_frame_equal(table[baseline.columns], baseline)
    assert table.reference_run_id.eq(reference).all()
    assert table.reference_owner.eq("member1").all()
    assert table.reference_experiment.eq("exp001").all()
    assert table.is_reference.tolist() == [False, True]
    assert table.filter(regex="_delta(?:_ci_(?:lower|upper))?$").eq(0).all().all()
    assert table.delta_ci_level.eq(0.95).all()
    assert table.delta_ci_repeats.eq(2000).all()
    assert table.delta_ci_seed.eq(42).all()
    assert table.delta_ci_method.eq("paired_stratified_patient_percentile_bootstrap").all()
    assert table.delta_ci_unavailable_reason.isna().all()
    assert {"patient_id", "target", "score"}.isdisjoint(table.columns)
    saved = pd.read_csv(output)
    assert saved.reference_run_id.tolist() == table.reference_run_id.tolist()
    for run in runs:
        assert {
            p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()
        } == before[run]


def test_wrapper_defaults_to_first_selection_and_supports_remote_minimal_bundles(tmp_path):
    import shutil

    from sixletters.run_catalog import compare_with_reference

    runs = [completed_run(tmp_path / str(i), owner=f"member{i}") for i in range(2)]
    for run in runs:
        # Ordinary remote comparison bundles need no model or archived source files.
        for path in run.iterdir():
            if path.name not in {"result.json", "predictions.csv", "config.toml"}:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    table = compare_with_reference(runs[::-1], tmp_path / "remote.csv")
    assert table.owner.tolist() == ["member1", "member0"]
    assert table.reference_run_id.eq(read_result(runs[1])["run_id"]).all()
    assert table.is_reference.tolist() == [True, False]


def test_wrapper_unknown_reference_or_invalid_result_leaves_no_output(tmp_path):
    from sixletters.run_catalog import compare_with_reference

    runs = [completed_run(tmp_path / str(i)) for i in range(2)]
    output = tmp_path / "paired.csv"
    with pytest.raises(ValueError, match="기준 실행.*찾을 수 없습니다"):
        compare_with_reference(runs, output, reference="unknown-or-mlflow-id")
    assert not output.exists()
    predictions = pd.read_csv(runs[1] / "predictions.csv")
    predictions.loc[0, "target"] = 1 - predictions.loc[0, "target"]
    predictions.to_csv(runs[1] / "predictions.csv", index=False)
    with pytest.raises(ValueError):
        compare_with_reference(runs, output)
    assert not output.exists()


def test_wrapper_preserves_existing_output_and_rejects_duplicate_paths(tmp_path):
    from sixletters.run_catalog import compare_with_reference

    runs = [completed_run(tmp_path / str(i)) for i in range(2)]
    output = tmp_path / "paired.csv"
    output.write_text("existing comparison")
    with pytest.raises(FileExistsError):
        compare_with_reference(runs, output)
    assert output.read_text() == "existing comparison"
    with pytest.raises(ValueError, match="distinct"):
        compare_with_reference([runs[0], runs[0]], tmp_path / "duplicate.csv")
    assert not (tmp_path / "duplicate.csv").exists()


def test_wrapper_small_class_differences_unavailable_except_explicit_reference(tmp_path):
    from sixletters.run_catalog import compare_with_reference

    dataset = make_demo_dataset()
    validation = dataset.frame["split"].eq("validation")
    dataset.frame.loc[validation, "target"] = 0
    dataset.frame.loc[dataset.frame.index[validation][0], "target"] = 1
    runs = [completed_run(tmp_path / str(i), dataset=dataset) for i in range(2)]
    table = compare_with_reference(runs, tmp_path / "small-paired.csv")
    assert table.is_reference.tolist() == [True, False]
    assert table.positive_count.tolist() == [1, 1]
    assert table.filter(regex="_delta_ci_(lower|upper)$").iloc[0].eq(0).all()
    assert table.filter(regex="_delta_ci_(lower|upper)$").iloc[1].isna().all()
    assert table.delta_ci_unavailable_reason.tolist() == [None, "insufficient_class_counts"]
    assert table.filter(regex="_delta$").eq(0).all().all()


def test_wrapper_keeps_existing_input_comparison_rules(tmp_path):
    from sixletters.run_catalog import compare_with_reference

    clinical = completed_run(tmp_path / "clinical", feature_set="clinical")
    ecg = completed_run(tmp_path / "ecg", feature_set="ecg")
    table = compare_with_reference([clinical, ecg], tmp_path / "inputs.csv", mode="inputs")
    assert table.feature_set.tolist() == ["clinical", "ecg"]
    with pytest.raises(ValueError, match="same feature_set"):
        compare_with_reference([clinical, ecg], tmp_path / "wrong-mode.csv")
    assert not (tmp_path / "wrong-mode.csv").exists()


@pytest.mark.parametrize("explicit_reference", [False, True])
def test_compare_cli_preserves_selection_order_and_displays_saved_reference(
    tmp_path, monkeypatch, capsys, explicit_reference
):
    import sys

    from sixletters import __main__ as cli

    runs = [completed_run(tmp_path / str(i), owner=f"member{i}") for i in range(2)]
    selected = runs[::-1]
    reference = runs[0] if explicit_reference else selected[0]
    reference_id = read_result(reference)["run_id"]
    output = tmp_path / "cli.csv"
    arguments = ["poc", "compare", "--runs", *map(str, selected), "--output", str(output)]
    if explicit_reference:
        arguments += ["--reference", reference_id]
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(cli, "choose_runs", lambda *_a, **_kw: pytest.fail("unexpected prompt"))
    assert cli.main() == 0
    displayed = capsys.readouterr().out
    table = pd.read_csv(output)
    assert table.owner.tolist() == ["member1", "member0"]
    assert table.reference_run_id.eq(reference_id).all()
    assert table.is_reference.tolist() == ([False, True] if explicit_reference else [True, False])
    assert f"기준 실행: {read_result(reference)['owner']} / exp001 / {reference_id}" in displayed
    assert "대상 − 기준" in displayed
    assert "AUROC·AP는 양수, Brier는 음수" in displayed
    assert "AUROC +0.0000 · 95% 구간 [+0.0000, +0.0000]" in displayed
    assert "학습·반복적인 모델 선택의 불확실성은 포함하지 않습니다" in displayed
    assert "Compared 2 validation runs:" in displayed
    for metric in ("roc_auc", "average_precision", "brier_score"):
        assert table[f"{metric}_delta"].eq(0).all()
        assert table[f"{metric}_delta_ci_lower"].eq(0).all()
        assert table[f"{metric}_delta_ci_upper"].eq(0).all()


@pytest.mark.parametrize("reference_kind", ["default", "saved", "server"])
def test_remote_compare_cli_fetches_in_selected_order_and_distinguishes_reference_ids(
    tmp_path, monkeypatch, capsys, reference_kind
):
    import sys

    from test_tracking import FakeClient

    from sixletters import __main__ as cli
    from sixletters import tracking

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    runs = [completed_run(tmp_path / str(i), owner=f"member{i}") for i in range(2)]
    client = FakeClient(tmp_path / "server")
    monkeypatch.setattr(tracking, "_client", lambda _: client)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.example:5000")
    for run in runs:
        assert tracking.auto_publish(run) == "uploaded"
    listed = tracking.list_remote_runs()
    by_owner = {entry["owner"]: entry for entry in listed}
    chosen = [by_owner[owner] for owner in ("member1", "member0")]
    identifiers = [entry["run_id"] for entry in chosen]
    assert all(entry["run_id"] != entry["local_run_id"] for entry in chosen)
    output = tmp_path / "remote-cli.csv"
    cache = tmp_path / "cache"
    arguments = [
        "poc",
        "compare",
        "--remote",
        "--run-ids",
        *identifiers,
        "--root",
        str(cache),
        "--output",
        str(output),
    ]
    if reference_kind != "default":
        field = "local_run_id" if reference_kind == "saved" else "run_id"
        arguments += ["--reference", chosen[1][field]]
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(tracking, "list_remote_runs", lambda **_kw: pytest.fail("unexpected list"))
    monkeypatch.setattr(cli, "prompt", lambda *_: pytest.fail("unexpected prompt"))
    if reference_kind == "server":
        with pytest.raises(SystemExit, match="2"):
            cli.main()
        assert not output.exists()
        message = capsys.readouterr().err
        assert "기준 실행을 선택한 결과에서 찾을 수 없습니다" in message
        assert "MLflow 서버 ID가 아닌" in message
    else:
        assert cli.main() == 0
        displayed = capsys.readouterr().out
        table = pd.read_csv(output)
        assert table.run_id.tolist() == [entry["local_run_id"] for entry in chosen]
        expected = chosen[1 if reference_kind == "saved" else 0]
        assert table.reference_run_id.eq(expected["local_run_id"]).all()
        assert f"기준 실행: {expected['owner']} / exp001 / {expected['local_run_id']}" in displayed
        assert table.delta_ci_method.eq("paired_stratified_patient_percentile_bootstrap").all()
    assert len(client.downloads) == 6
    assert set(client.downloads) == {
        "bundle/manifest.json",
        "bundle/result.json",
        "bundle/predictions.csv",
    }
    assert sorted(path.name for path in cache.glob(".remote/*/*") if path.is_dir()) == sorted(
        identifiers
    )
    # A comparison never publishes another run or requests an archived model.
    assert client.create_count == 2
