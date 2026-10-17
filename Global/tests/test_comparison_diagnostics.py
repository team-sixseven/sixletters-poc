import json

import pytest
from test_experiments import completed_run, read_result

from sixletters.comparison import compare_runs


def update_result(run, callback):
    result = read_result(run)
    callback(result)
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")


def rejection(first, second, tmp_path, mode="inputs"):
    output = tmp_path / "comparison.csv"
    with pytest.raises(ValueError) as error:
        compare_runs([first, second], output, mode=mode)
    assert not output.exists()
    return str(error.value)


@pytest.mark.parametrize(
    ("section", "field", "changed"),
    [
        ("dataset", "dataset_id", "new-cohort"),
        ("dataset", "target_definition", "Different outcome"),
        ("dataset", "data_hash", "1" * 64),
        ("dataset", "cohort_hash", "1" * 64),
        ("dataset", "split_hash", "1" * 64),
        ("dataset", "target_hash", "1" * 64),
        ("dataset", "feature_schema_hash", "1" * 64),
        ("evaluation", "code_hash", "1" * 64),
    ],
)
def test_shared_mismatch_identifies_field_and_two_runs(tmp_path, section, field, changed):
    first = completed_run(tmp_path / "first")
    second = completed_run(tmp_path / "second")
    original = read_result(first)["comparison_context"][section][field]
    update_result(
        second, lambda result: result["comparison_context"][section].update({field: changed})
    )
    message = rejection(first, second, tmp_path, mode="experiments")
    assert f"comparison_context.{section}.{field}" in message
    assert original in message and changed in message
    assert read_result(first)["run_id"] in message
    assert read_result(second)["run_id"] in message
    assert "동일한 공통 데이터" in message


def test_input_mismatch_identifies_nested_settings_software_and_sources(tmp_path):
    first = completed_run(tmp_path / "first")
    second = completed_run(tmp_path / "second", feature_set="ecg")

    def change(result):
        result["training"]["parameters"]["learning_rate"] = 0.2
        result["training"]["source_hash"] = "1" * 64
        result["shared_code_scope"]["hash"] = "2" * 64
        result["software"]["packages"]["numpy"] = "different-version"

    update_result(second, change)
    message = rejection(first, second, tmp_path)
    for field in (
        "training.parameters.learning_rate",
        "training.source_hash",
        "training.shared_code_hash.hash",
        "software.packages.numpy",
    ):
        assert field in message
    assert "기준 0.1 / 대상 0.2" in message
    assert "different-version" in message
    assert "모델·설정·학습 코드·패키지 버전" in message


def test_missing_and_null_parameter_are_not_displayed_as_equal(tmp_path):
    first = completed_run(tmp_path / "first")
    second = completed_run(tmp_path / "second", feature_set="ecg")
    update_result(
        second, lambda result: result["training"]["parameters"].update({"max_depth": None})
    )
    message = rejection(first, second, tmp_path)
    assert "training.parameters.max_depth" in message
    assert "기준 [항목 없음] / 대상 null" in message


def test_diagnostics_hide_secrets_and_unknown_patient_metadata(tmp_path):
    first = completed_run(tmp_path / "first")
    second = completed_run(tmp_path / "second", feature_set="ecg")

    def change(result):
        result["training"]["parameters"].update(
            {
                "api_key": "DO-NOT-PRINT-KEY",
                "nested": {"password": "DO-NOT-PRINT-PASSWORD"},
                "service_url": "https://private-user:private-password@service.example/data?token=secret",
            }
        )

    update_result(second, change)
    message = rejection(first, second, tmp_path)
    for private in (
        "DO-NOT-PRINT",
        "private-user",
        "private-password",
        "service.example",
        "token=secret",
    ):
        assert private not in message
    assert "[민감한 값 숨김]" in message
    assert "[주소 숨김]" in message
    assert "training.parameters.api_key" in message

    update_result(
        second,
        lambda result: result["comparison_context"].update(
            {"individual_records": [{"id": "PRIVATE-PATIENT-ID"}]}
        ),
    )
    message = rejection(first, second, tmp_path)
    assert "PRIVATE-PATIENT-ID" not in message
    assert "comparison_context.individual_records" in message
    assert "내용 숨김" in message


def test_diagnostics_bound_differences_and_remove_control_characters(tmp_path):
    first = completed_run(tmp_path / "first")
    second = completed_run(tmp_path / "second", feature_set="ecg")

    def change(result):
        result["owner"] = "bad\x1b[2J\nowner" + "X" * 10000
        result["training"]["parameters"].update({f"parameter_{i}": "V" * 10000 for i in range(100)})

    update_result(second, change)
    message = rejection(first, second, tmp_path)
    assert "\x1b" not in message
    assert len(message) < 4000
    assert "처음 6개만 표시" in message
    assert "parameter_0" in message
    assert "parameter_99" not in message
    assert "식별값" in message


def test_feature_set_and_partial_column_errors_explain_the_required_selection(tmp_path):
    clinical = completed_run(tmp_path / "clinical")
    ecg = completed_run(tmp_path / "ecg", feature_set="ecg")
    message = rejection(clinical, ecg, tmp_path, mode="experiments")
    assert '기준 "clinical" / 대상 "ecg"' in message
    assert "같은 입력 그룹만 선택" in message
    subset = completed_run(tmp_path / "subset", features=["clinical_1"])
    message = rejection(subset, ecg, tmp_path)
    assert "clinical_1" in message and "clinical_2" in message
    assert "기준: 입력 그룹에 선언된 전체 컬럼" in message
    assert "features를 제거" in message


def test_duplicate_input_group_identifies_both_runs(tmp_path):
    first = completed_run(tmp_path / "first")
    second = completed_run(tmp_path / "second")
    message = rejection(first, second, tmp_path)
    assert "중복 입력 그룹: clinical" in message
    assert read_result(first)["run_id"] in message
    assert read_result(second)["run_id"] in message
    assert "하나씩만 선택" in message


def test_mixed_legacy_comparison_explains_full_source_check(tmp_path):
    first = completed_run(tmp_path / "first")
    second = completed_run(tmp_path / "second", feature_set="ecg")
    update_result(first, lambda result: result.pop("shared_code_scope"))
    update_result(second, lambda result: result["training"].update({"shared_code_hash": "1" * 64}))
    message = rejection(first, second, tmp_path)
    assert "training.shared_code_hash" in message
    assert "이전 형식의 실행이 포함되어 공통 코드 전체를 비교" in message


@pytest.mark.parametrize("field", ["patient_hash", "target_hash", "split"])
def test_invalid_recorded_evaluation_identifies_run_without_patient_values(tmp_path, field):
    first = completed_run(tmp_path / "first")
    second = completed_run(tmp_path / "second")
    changed = "1" * 64 if field.endswith("_hash") else "test"
    update_result(
        second, lambda result: result["comparison_context"]["evaluation"].update({field: changed})
    )
    message = rejection(first, second, tmp_path, mode="experiments")
    assert field in message and changed in message
    assert read_result(second)["run_id"] in message
    assert "predictions.csv나 result.json을 수정해 맞추지 말고" in message
