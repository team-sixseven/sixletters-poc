"""Validation uncertainty matches patient bootstrap and remains honest for old results."""

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from sixletters.run_catalog import (
    is_scd_target,
    show_run_catalog,
    summary_from_run,
    validation_metadata,
    validation_summary,
)


def predictions(targets=(0, 0, 0, 0, 1, 1), scores=(0.1, 0.2, 0.6, 0.6, 0.6, 0.9)):
    return pd.DataFrame(
        {
            "patient_id": [f"patient-{index:03}" for index in range(len(targets))],
            "target": targets,
            "score": scores,
        }
    )


@pytest.mark.parametrize(
    ("targets", "scores"),
    [
        ((0, 0, 0, 0, 1, 1), (0.1, 0.2, 0.6, 0.6, 0.6, 0.9)),
        ((0,) * 18 + (1,) * 2, np.linspace(0.9, 0.1, 20)),
    ],
)
def test_intervals_match_sklearn_patient_resampling_including_ties_and_imbalance(targets, scores):
    frame = predictions(targets, scores)
    summary = validation_summary(frame)
    target = frame["target"].to_numpy()
    score = frame["score"].to_numpy()
    positive, negative = np.flatnonzero(target == 1), np.flatnonzero(target == 0)
    rng = np.random.default_rng(42)
    reference = []
    for _ in range(2000):
        selected = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        y, p = target[selected], score[selected]
        reference.append(
            [roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)]
        )
    bounds = np.quantile(reference, [0.025, 0.975], axis=0)
    for index, key in enumerate(("roc_auc", "average_precision", "brier_score")):
        assert summary["intervals"][key]["lower"] == pytest.approx(bounds[0, index], abs=1e-14)
        assert summary["intervals"][key]["upper"] == pytest.approx(bounds[1, index], abs=1e-14)
    assert summary["patients"] == len(frame)
    assert summary["positive"] == len(positive)
    assert summary["negative"] == len(negative)
    assert summary["reason"] is None
    assert validation_metadata(summary) == summary


def test_summary_is_deterministic_under_row_reordering_and_does_not_mutate_predictions():
    frame = predictions()
    original = frame.copy(deep=True)
    expected = validation_summary(frame)
    assert expected == validation_summary(frame.sample(frac=1, random_state=7))
    assert expected == validation_summary(frame)
    pd.testing.assert_frame_equal(original, frame)


@pytest.mark.parametrize("scores", [(0.5, 0.5, 0.5, 0.5), (0.0, 0.0, 1.0, 1.0)])
def test_constant_or_perfect_predictions_may_legitimately_have_zero_width(scores):
    summary = validation_summary(predictions((0, 0, 1, 1), scores))
    for interval in summary["intervals"].values():
        assert interval["lower"] == interval["upper"]
    assert summary["reason"] is None


def test_one_positive_keeps_counts_but_does_not_invent_a_precise_interval():
    summary = validation_summary(predictions((0, 0, 0, 1), (0.1, 0.4, 0.3, 0.8)))
    assert (summary["patients"], summary["positive"], summary["negative"]) == (4, 1, 3)
    assert summary["reason"] == "insufficient_class_counts"
    assert all(
        interval == {"lower": None, "upper": None} for interval in summary["intervals"].values()
    )
    assert validation_metadata(summary) == summary


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("patient_id", ["a"] * 6),
        ("patient_id", ["", "b", "c", "d", "e", "f"]),
        ("target", [0] * 6),
        ("score", [float("nan")] * 6),
        ("score", [2] * 6),
    ],
)
def test_invalid_predictions_do_not_produce_counts_or_intervals(column, values):
    frame = predictions()
    frame[column] = values
    with pytest.raises(ValueError):
        validation_summary(frame)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("version", True),
        ("method", "normal_approximation"),
        ("patients", 999),
        ("positive", True),
        ("confidence_level", 0.90),
        ("repeats", 20),
        ("seed", 17),
        ("reason", "insufficient_class_counts"),
    ],
)
def test_remote_metadata_rejects_unsupported_or_inconsistent_protocol(key, value):
    summary = validation_summary(predictions())
    summary[key] = value
    assert validation_metadata(summary) is None


@pytest.mark.parametrize(
    "interval",
    [
        {"lower": float("nan"), "upper": 1.0},
        {"lower": 0.8, "upper": 0.1},
        {"lower": -0.1, "upper": 0.8},
        {"lower": 0.1, "upper": 10**1000},
        {"lower": 0.1, "upper": True},
        {"lower": None, "upper": None},
        None,
    ],
)
def test_remote_metadata_rejects_invalid_intervals(interval):
    summary = validation_summary(predictions())
    summary["intervals"]["roc_auc"] = interval
    assert validation_metadata(summary) is None


def test_missing_protocol_fields_are_unknown_not_an_exception():
    summary = validation_summary(predictions())
    for key in summary:
        incomplete = {name: value for name, value in summary.items() if name != key}
        assert validation_metadata(incomplete) is None


def test_reading_saved_predictions_preserves_identifiers_precision_and_saved_files(tmp_path):
    frame = predictions()
    frame["patient_id"] = ["001", "002", "003", "004", "005", "006"]
    frame.loc[0, "score"] = np.nextafter(0.1234567891234567, 1.0)
    frame.to_csv(tmp_path / "predictions.csv", index=False)
    before = (tmp_path / "predictions.csv").read_bytes()
    assert summary_from_run(tmp_path) == validation_summary(frame)
    assert (tmp_path / "predictions.csv").read_bytes() == before
    assert list(tmp_path.iterdir()) == [tmp_path / "predictions.csv"]


def catalog_entry(**changes):
    return {
        "owner": "팀원",
        "experiment": "exp001",
        "local_run_id": "run-1",
        "model": "LogisticRegression",
        "parameters": {},
        "synthetic": False,
        "metrics": {"roc_auc": 0.81234, "average_precision": 0.43216, "brier_score": 0.12345},
        **changes,
    }


def test_catalog_recalculates_local_counts_and_intervals_without_replacing_point_scores(
    tmp_path, capsys
):
    predictions().to_csv(tmp_path / "predictions.csv", index=False)
    show_run_catalog([catalog_entry(path=str(tmp_path))])
    output = capsys.readouterr().out
    assert "검증 환자=6 | 양성(target=1)=2" in output
    assert "95% 신뢰구간: AUROC=[" in output and "AP=[" in output and "Brier=[" in output
    assert "AUROC=0.8123 | AP=0.4322 | Brier=0.1235" in output
    assert "학습·모델 선택의 변동을 포함하지 않으며" in output


def test_catalog_keeps_malformed_legacy_local_run_selectable(tmp_path, capsys):
    (tmp_path / "predictions.csv").touch()
    show_run_catalog([catalog_entry(path=str(tmp_path)), catalog_entry(local_run_id="run-2")])
    output = capsys.readouterr().out
    assert "1. 팀원 / exp001 / run-1" in output and "2. 팀원 / exp001 / run-2" in output
    assert "확인 불가 (예측 파일 확인 필요)" in output
    assert "미기록 · 비교 시 계산" in output


def test_remote_catalog_marks_invalid_metadata_and_small_counts_explicitly(capsys):
    summary = validation_summary(predictions((0, 0, 1), (0.1, 0.5, 0.8)))
    show_run_catalog(
        [
            catalog_entry(validation_summary_status="invalid"),
            catalog_entry(local_run_id="run-2", validation_summary=summary),
        ]
    )
    output = capsys.readouterr().out
    assert "확인 불가 · 비교 시 재계산" in output
    assert "검증 환자=3 | 양성(target=1)=1" in output
    assert "계산하지 않음 (양성·음성 각 2명 이상 필요)" in output


@pytest.mark.parametrize("synthetic", [False, True, None])
def test_catalog_names_scd_only_for_known_real_music_definition(synthetic, capsys):
    definition = (
        "music-scd-basic-v1: Cause of death=3 during observed follow-up is 1; survivors are 0"
    )
    entry = catalog_entry(
        synthetic=synthetic,
        target_definition=definition,
        validation_summary=validation_summary(predictions()),
    )
    original = copy.deepcopy(entry)
    show_run_catalog([entry])
    output = capsys.readouterr().out
    assert ("검증 환자=6 | SCD=2" in output) == (synthetic is False)
    assert ("양성(target=1)=2" in output) == (synthetic is not False)
    assert entry == original
    assert not is_scd_target(False, "music-scd-future: cause 3 is a negative class")
    assert not is_scd_target(False, None)
