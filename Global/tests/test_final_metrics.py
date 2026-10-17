"""Final evaluation keeps validation threshold selection separate from test reports."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score

from sixletters import final_metrics

BASE = "a" * 32
OTHER = "b" * 32


def predictions(target=(0, 0, 1, 1), score=(0.1, 0.4, 0.35, 0.8)):
    return pd.DataFrame(
        {
            "patient_id": [f"PRIVATE-PATIENT-{index}" for index in range(len(target))],
            "target": target,
            "score": score,
        }
    )


def test_point_metrics_match_known_probabilities_and_confusion_counts():
    result = final_metrics.evaluate_final_predictions(predictions(), 0.5)
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}
    assert result["metrics"] == pytest.approx(
        {
            "roc_auc": 0.75,
            "average_precision": 5 / 6,
            "brier_score": 0.158125,
            "sensitivity": 0.5,
            "specificity": 1.0,
            "precision": 1.0,
            "f1": 2 / 3,
        }
    )
    assert result["undefined"] == {}


def test_youden_threshold_ties_use_largest_validation_threshold():
    # 0.35 and 0.8 both have J=0.5; the deterministic rule selects 0.8.
    selected = final_metrics.select_threshold(predictions())
    assert selected["threshold"] == 0.8
    assert selected["method"] == "youden_j"
    assert selected["validation_youden_j"] == 0.5
    assert selected["selection_split"] == "validation"
    assert selected["validation_counts"] == {"patients": 4, "positive": 2, "negative": 2}
    assert final_metrics.select_threshold(predictions().iloc[::-1]) == selected


def test_constant_validation_scores_use_finite_boundary_threshold():
    result = final_metrics.select_threshold(predictions(score=[0.5] * 4))
    assert result["threshold"] == 1.0
    assert result["validation_youden_j"] == 0.0
    result = final_metrics.select_threshold(predictions(score=[1.0] * 4))
    assert result["threshold"] == 1.0


@pytest.mark.parametrize("fixed", [0.0, 0.5, 1.0])
def test_fixed_threshold_is_recorded_without_optimization(fixed):
    selected = final_metrics.select_threshold(predictions(), fixed=fixed)
    assert selected["threshold"] == fixed
    assert selected["method"] == "fixed"
    assert selected["tie_break"] is None


@pytest.mark.parametrize(
    "invalid", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5", 1j, np.complex128(0.5 + 1j)]
)
def test_invalid_fixed_thresholds_fail_before_use(invalid):
    with pytest.raises(ValueError, match="Threshold"):
        final_metrics.select_threshold(predictions(), fixed=invalid)


@pytest.mark.parametrize("target", [[0] * 4, [1] * 4])
def test_threshold_selection_rejects_single_class_validation(target):
    with pytest.raises(ValueError, match="both validation target classes"):
        final_metrics.select_threshold(predictions(target=target))


def test_final_report_uses_frozen_threshold_and_writes_only_aggregate_outputs(
    tmp_path, monkeypatch
):
    selected = final_metrics.select_threshold(predictions(), fixed=0.5)

    def forbid_selection(*args, **kwargs):
        raise AssertionError("Test reporting must not select a threshold")

    monkeypatch.setattr(final_metrics, "select_threshold", forbid_selection)
    # Test ordering/performance differ from validation; the threshold stays fixed.
    test = predictions(score=(0.9, 0.7, 0.2, 0.1))
    report = final_metrics.build_final_report(
        {BASE: test},
        {BASE: selected["threshold"]},
        tmp_path,
        repeats=100,
        seed=2,
        labels={BASE: "Clinical / Logistic"},
    )
    assert report["models"][BASE]["threshold"] == 0.5
    assert report["models"][BASE]["confusion_matrix"] == {"tn": 0, "fp": 2, "fn": 2, "tp": 0}
    assert report["bootstrap"]["thresholds_refitted"] is False
    assert report["bootstrap"]["conditional_on_observed_class_counts"] is True
    assert json.loads((tmp_path / "final_metrics.json").read_text()) == report
    assert set(report["files"]) == {path.name for path in tmp_path.iterdir()}
    for filename in ("roc.png", "pr.png", "calibration.png", "confusion.png"):
        assert (tmp_path / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for path in tmp_path.iterdir():
        if path.suffix in {".json", ".csv"}:
            assert "PRIVATE-PATIENT-" not in path.read_text()
    assert len(pd.read_csv(tmp_path / "metrics.csv")) == len(final_metrics.METRICS)
    assert pd.read_csv(tmp_path / "deltas.csv").empty
    # Calibration binning matches the standard sklearn uniform-bin definition.
    fraction, mean_score = calibration_curve(test.target, test.score, n_bins=5, strategy="uniform")
    calibration = pd.read_csv(tmp_path / "calibration.csv")
    np.testing.assert_allclose(calibration.observed_fraction, fraction)
    np.testing.assert_allclose(calibration.mean_probability, mean_score)


def test_paired_bootstrap_zero_delta_for_identical_models_after_patient_alignment(tmp_path):
    first = predictions()
    report = final_metrics.build_final_report(
        {BASE: first, OTHER: first.sample(frac=1, random_state=4)},
        {BASE: 0.5, OTHER: 0.5},
        tmp_path / "first",
        repeats=100,
        seed=14,
    )
    repeated = final_metrics.build_final_report(
        {BASE: first, OTHER: first.iloc[::-1]},
        {BASE: 0.5, OTHER: 0.5},
        tmp_path / "second",
        repeats=100,
        seed=14,
    )
    assert report == repeated
    for metric, value in report["deltas"][OTHER]["metrics"].items():
        assert value == 0.0
        ci = report["deltas"][OTHER]["confidence_intervals"][metric]
        assert ci["lower"] == ci["upper"] == 0.0
        assert 0 < ci["valid_resamples"] <= 100
    assert report["models"][BASE]["confidence_intervals"]["roc_auc"]["valid_resamples"] == 100
    assert (
        report["models"][BASE]["confidence_intervals"]["average_precision"]["valid_resamples"]
        == 100
    )


def test_paired_delta_direction_and_confidence_intervals_for_perfect_vs_reversed_scores(tmp_path):
    report = final_metrics.build_final_report(
        {
            BASE: predictions(score=(0.9, 0.8, 0.2, 0.1)),
            OTHER: predictions(score=(0.1, 0.2, 0.8, 0.9)),
        },
        {BASE: 0.5, OTHER: 0.5},
        tmp_path,
        repeats=100,
    )
    delta = report["deltas"][OTHER]
    assert delta["metrics"]["roc_auc"] == 1.0
    assert delta["metrics"]["brier_score"] < 0
    assert delta["confidence_intervals"]["roc_auc"]["lower"] == 1.0
    assert delta["confidence_intervals"]["roc_auc"]["upper"] == 1.0


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_test_data_reports_undefined_metrics_and_no_valid_intervals(tmp_path, label):
    frame = predictions(target=[label] * 4, score=[float(label)] * 4)
    report = final_metrics.build_final_report(
        {BASE: frame, OTHER: frame},
        {BASE: 0.5, OTHER: 0.5},
        tmp_path,
        repeats=100,
    )
    model = report["models"][BASE]
    assert model["metrics"]["roc_auc"] is None
    assert model["confidence_intervals"]["roc_auc"] == {
        "lower": None,
        "upper": None,
        "valid_resamples": 0,
        "reason": "Test data contains only one target class",
    }
    denominator_metric = "sensitivity" if label == 0 else "specificity"
    assert model["metrics"][denominator_metric] is None
    assert model["confidence_intervals"][denominator_metric]["valid_resamples"] == 0
    assert model["undefined"][denominator_metric]
    if not label:
        for metric in ("average_precision", "precision", "f1"):
            assert model["metrics"][metric] is None
            assert model["confidence_intervals"][metric]["valid_resamples"] == 0
    assert report["deltas"][OTHER]["metrics"]["roc_auc"] is None
    assert report["deltas"][OTHER]["confidence_intervals"]["roc_auc"]["valid_resamples"] == 0
    assert model["metrics"]["brier_score"] == 0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("change", ["patient", "target"])
def test_comparison_rejects_unpaired_patients_or_targets_before_creating_report(tmp_path, change):
    frame = predictions()
    other = frame.copy()
    if change == "patient":
        other.loc[0, "patient_id"] = "PRIVATE-PATIENT-DIFFERENT"
    else:
        other.loc[0, "target"] = 1
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="same test patients and targets") as error:
        final_metrics.build_final_report(
            {BASE: frame, OTHER: other},
            {BASE: 0.5, OTHER: 0.5},
            output,
            repeats=100,
        )
    assert "PRIVATE-PATIENT-" not in str(error.value)
    assert not output.exists()


@pytest.mark.parametrize("column,value", [("score", np.nan), ("score", 1.1), ("target", 2)])
def test_malformed_prediction_values_fail(column, value):
    frame = predictions()
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        final_metrics.evaluate_final_predictions(frame, 0.5)


def test_duplicate_patient_scores_are_rejected():
    frame = predictions()
    frame.loc[1, "patient_id"] = frame.loc[0, "patient_id"]
    with pytest.raises(ValueError, match="unique"):
        final_metrics.evaluate_final_predictions(frame, 0.5)


@pytest.mark.parametrize("repeats", [99, 10_001, True, 100.5])
def test_bootstrap_repetition_bounds(tmp_path, repeats):
    with pytest.raises(ValueError, match="Bootstrap repeats"):
        final_metrics.build_final_report(
            {BASE: predictions()}, {BASE: 0.5}, tmp_path, repeats=repeats
        )


def test_invalid_run_identity_and_threshold_mapping_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="run IDs"):
        final_metrics.build_final_report({"../other": predictions()}, {"../other": 0.5}, tmp_path)
    with pytest.raises(ValueError, match="exactly one frozen threshold"):
        final_metrics.build_final_report({BASE: predictions()}, {}, tmp_path)


def test_existing_outputs_are_not_overwritten(tmp_path):
    destination = tmp_path / "roc.png"
    destination.write_bytes(b"keep existing result")
    with pytest.raises(FileExistsError, match="already exist"):
        final_metrics.build_final_report({BASE: predictions()}, {BASE: 0.5}, tmp_path, repeats=100)
    assert destination.read_bytes() == b"keep existing result"


def test_point_metrics_preserve_existing_sklearn_auroc_and_ap_definition():
    rng = np.random.default_rng(8)
    y = np.array([0] * 25 + [1] * 5)
    scores = np.round(rng.random(len(y)), 1)
    result = final_metrics.evaluate_final_predictions(predictions(y, scores), 0.5)
    assert result["metrics"]["roc_auc"] == roc_auc_score(y, scores)
    assert result["metrics"]["average_precision"] == average_precision_score(y, scores)
