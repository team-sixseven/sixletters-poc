"""Selection shows scores and meaningful settings without altering run identity or validation."""

import copy
import json
import os
from datetime import UTC, datetime

import pytest

from sixletters import __main__ as cli
from sixletters.run_catalog import display_text, show_run_catalog


def entry(index=1, **changes):
    return {
        "run_id": f"remote-{index}",
        "local_run_id": f"local-{index}",
        "owner": "팀원",
        "experiment": "exp001",
        "dataset_version": "cohort-v1",
        "feature_set": "clinical",
        "synthetic": False,
        "model": "LogisticRegression",
        "seed": 42,
        "seed_known": True,
        "parameters": {"max_iter": 200, "C": 1.0},
        "parameters_complete": True,
        "metrics": {"roc_auc": 0.81234, "average_precision": 0.43216, "brier_score": 0.12345},
        "created_at": "2026-10-17T09:00:00+09:00",
        "created_at_source": "tracking_start",
        **changes,
    }


def test_nested_settings_distinguish_missing_null_and_changed_values(capsys):
    first = entry(parameters={"preprocess": {"scale": True, "imputer": "median"}, "weight": None})
    second = entry(2, parameters={"preprocess": {"imputer": "mean", "scale": True}})
    show_run_catalog([first, second])
    output = capsys.readouterr().out
    common, displays = output.split("1. 팀원")
    assert "parameters/preprocess/scale=true" in common
    assert "parameters/preprocess/imputer" not in common
    first_display, second_display = displays.split("2. 팀원")
    assert 'parameters/preprocess/imputer="median"' in first_display
    assert 'parameters/preprocess/imputer="mean"' in second_display
    assert "parameters/weight=null" in first_display
    assert "parameters/weight=미기록" in second_display


def test_nested_parameter_paths_do_not_collide_with_literal_keys(capsys):
    show_run_catalog(
        [
            entry(parameters={"a/b": 1, "a": {"b": 2}, "a~b": 3}),
            entry(2, parameters={"a/b": 4, "a": {"b": 5}, "a~b": 6}),
        ]
    )
    output = capsys.readouterr().out
    assert "parameters/a~1b=1" in output
    assert "parameters/a/b=2" in output
    assert "parameters/a~0b=3" in output


def test_incomplete_remote_parameters_are_unknown_not_missing_or_common(capsys):
    first = entry(parameters={"C": 1, "class_weight": None})
    second = entry(2, parameters={"C": 1}, parameters_complete=False, seed=None, seed_known=False)
    show_run_catalog([first, second])
    output = capsys.readouterr().out
    common, displays = output.split("1. 팀원")
    assert "parameters/" not in common
    assert "seed=" not in common
    second_display = displays.split("2. 팀원")[1]
    assert "parameters/class_weight=확인 불가" in second_display
    assert "seed=확인 불가" in second_display
    assert "누락·축약된 값은 동일하다고 판단하지 않습니다" in second_display


def test_same_named_parameters_on_different_models_are_not_classified_as_common(capsys):
    show_run_catalog(
        [
            entry(model="a" * 150 + "1", parameters={"C": 1}),
            entry(2, model="a" * 150 + "2", parameters={"C": 1}),
        ]
    )
    output = capsys.readouterr().out
    assert "parameters/C" not in output.split("1. 팀원")[0]
    assert output.count("parameters/C=1") == 2


def test_legacy_minimal_metadata_and_invalid_metrics_do_not_invent_scores(capsys):
    minimal = {"owner": "old", "experiment": "exp", "local_run_id": "legacy", "model": "old-model"}
    bad = entry(metrics={"roc_auc": float("nan"), "average_precision": -1, "brier_score": True})
    show_run_catalog([minimal, bad])
    output = capsys.readouterr().out
    assert output.count("AUROC=확인 불가 | AP=확인 불가 | Brier=확인 불가") == 2
    assert "data type unknown" in output
    assert "시각: 확인 불가" in output
    assert "seed=확인 불가" in output


@pytest.mark.parametrize("timestamp", [None, "bad", "2026-10-17T00:00:00"])
def test_local_selection_labels_legacy_file_time_and_keeps_number_mapping(
    tmp_path, monkeypatch, capsys, timestamp
):
    directories = []
    for index in range(2):
        directory = tmp_path / "member" / "exp" / f"run-{index}"
        directory.mkdir(parents=True)
        (directory / "predictions.csv").touch()
        path = directory / "result.json"
        path.write_text(
            json.dumps(
                {
                    "owner": "member",
                    "experiment": "exp",
                    "run_id": f"run-{index}",
                    "feature_set": "clinical",
                    "synthetic": True,
                    "comparison_context": {"dataset": {"dataset_id": "demo-v1"}},
                    "training": {"model": "own", "seed": None, "parameters": {"C": index}},
                    "metrics": entry()["metrics"],
                    "created_at": timestamp,
                }
            )
        )
        file_time = datetime(2026, 10, 17, tzinfo=UTC).timestamp()
        os.utime(path, (file_time, file_time))
        directories.append(directory)
    monkeypatch.setattr("builtins.input", lambda _: "2 1")
    assert cli.choose_runs(tmp_path) == directories[::-1]
    output = capsys.readouterr().out
    assert "2026-10-17 00:00:00 UTC (파일 수정 시각·저장 시각 대체)" in output
    assert "Brier=0.1235" in output
    assert "seed=null" in output


def test_catalog_bounds_metadata_and_removes_terminal_controls(capsys):
    first = entry(owner="team\x1b[2J\nforged\r\u202e", parameters={"payload": "x" * 10000})
    original = copy.deepcopy(first)
    show_run_catalog([first])
    output = capsys.readouterr().out
    assert "\x1b" not in output and "\r" not in output and "\u202e" not in output
    assert "x" * 121 not in output
    assert "…" in output
    assert len(output) < 1400
    assert first == original
    assert display_text("a\n" * 300, 12) == "a a a a a a…"


@pytest.mark.parametrize(
    "values", [([1] * 100 + [2], [1] * 100 + [9]), ("x" * 300 + "a", "x" * 300 + "b")]
)
def test_truncated_long_settings_remain_distinguishable(values, capsys):
    show_run_catalog(
        [entry(parameters={"schedule": values[0]}), entry(2, parameters={"schedule": values[1]})]
    )
    output = capsys.readouterr().out
    settings = [line for line in output.splitlines() if "parameters/schedule=" in line]
    assert len(settings) == 2
    assert settings[0] != settings[1]
    assert all("sha256:" in line for line in settings)


def test_many_common_parameters_explicitly_report_omitted_details(capsys):
    parameters = {f"parameter-{index:02}": index for index in range(20)}
    show_run_catalog([entry(parameters=parameters), entry(2, parameters=parameters)])
    common = capsys.readouterr().out.split("1. 팀원")[0]
    assert 'model="LogisticRegression"' in common
    assert "seed=42" in common
    assert "공통 설정 17개 생략" in common


def test_nonfinite_parameter_values_never_become_confirmed_common_settings(capsys):
    show_run_catalog(
        [entry(parameters={"C": float("nan")}), entry(2, parameters={"C": float("nan")})]
    )
    output = capsys.readouterr().out
    assert "parameters/C" not in output.split("1. 팀원")[0]
    assert output.count("parameters/C=확인 불가") == 2
