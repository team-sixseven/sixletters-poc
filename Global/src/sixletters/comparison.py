"""Compare completed validation results without loading or running any model."""

import hashlib
import json
import re
import unicodedata
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluation
from .experiment import METRICS, RESULT_SCHEMA_VERSION, validate_training
from .experiment import expected_features as _expected_features
from .experiment import prediction_identity as _prediction_identity
from .provenance import validate_resource_metadata, validate_shared_code_scope
from .run_catalog import is_scd_target, validation_summary

_MISSING = object()
_MAX_DIFFERENCES = 6
_SENSITIVE_FIELD = re.compile(
    r"password|passwd|secret|token|credential|authorization|api.?key|private.?key|patient",
    re.IGNORECASE,
)
_FIELD_LABELS = {
    "dataset_id": "데이터 버전",
    "target_definition": "예측 정답 정의",
    "data_hash": "데이터 내용 식별값",
    "cohort_hash": "전체 환자 구성 식별값",
    "split_hash": "환자 분할 식별값",
    "target_hash": "예측 정답 식별값",
    "feature_schema_hash": "입력 컬럼 정의 식별값",
    "patient_hash": "검증 환자 구성 식별값",
    "code_hash": "평가 코드 식별값",
    "source_hash": "개인 학습 코드 식별값",
    "shared_code_hash": "공통 코드 식별값",
    "synthetic": "합성 데이터 여부",
    "feature_set": "입력 그룹",
    "features": "입력 컬럼",
    "model": "모델 이름",
    "seed": "난수 시드",
    "split": "평가 데이터 구분",
    "metrics": "평가 지표",
}


def _bounded_text(value: str, limit: int = 140) -> str:
    # Result metadata can originate from another contributor's machine.
    value = re.sub(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"']+", "[주소 숨김]", value)
    text = "".join(" " if unicodedata.category(char).startswith("C") else char for char in value)
    if len(text) > limit:
        digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:8]
        return f"{text[:limit]}… [식별값 {digest}]"
    return text


def _display_value(value, path: tuple[str, ...]) -> str:
    if value is _MISSING:
        return "[항목 없음]"
    name = path[-1] if path else ""
    if any(_SENSITIVE_FIELD.search(part) for part in path) and not name.endswith("_hash"):
        return "[민감한 값 숨김]"
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    # Only the known shared identity fields are displayed as values. An extra
    # metadata field may contain individual records and is still compared strictly.
    if (
        isinstance(value, dict)
        or isinstance(value, list)
        and any(isinstance(item, dict | list) for item in value)
        or name.endswith("_hash")
        and (not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None)
        or path
        and path[0] == "comparison_context"
        and name not in {*_FIELD_LABELS, "numpy", "pandas", "scikit_learn"}
    ):
        digest = hashlib.sha256(serialized.encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"[내용 숨김, 식별값 {digest}]"
    return _bounded_text(serialized)


def _differences(left, right, path: tuple[str, ...] = ()):
    if left == right:
        return
    if isinstance(left, dict) and isinstance(right, dict) and len(path) < 10:
        for key in sorted(left.keys() | right.keys()):
            yield from _differences(left.get(key, _MISSING), right.get(key, _MISSING), (*path, key))
    else:
        yield path, left, right


def _run_label(result: dict) -> str:
    return "/".join(
        _bounded_text(str(result.get(key, "?")), 60) for key in ("owner", "experiment", "run_id")
    )


def _mismatch_message(
    message: str,
    first: dict | None,
    result: dict,
    left: dict,
    right: dict,
    action: str,
    *,
    reference_label: str = "입력 그룹에 선언된 전체 컬럼",
) -> str:
    differences = list(islice(_differences(left, right), _MAX_DIFFERENCES + 1))
    reference = f"기준 실행: {_run_label(first)}" if first else f"기준: {reference_label}"
    lines = [message, reference, f"대상 실행: {_run_label(result)}"]
    for path, expected, actual in differences[:_MAX_DIFFERENCES]:
        label = _FIELD_LABELS.get(path[-1], "설정 항목")
        field = _bounded_text(".".join(path), 180)
        lines.append(
            f"- {label} ({field}): 기준 {_display_value(expected, path)}"
            f" / 대상 {_display_value(actual, path)}"
        )
    if len(differences) > _MAX_DIFFERENCES:
        lines.append(f"- 차이가 더 있습니다. 처음 {_MAX_DIFFERENCES}개만 표시했습니다.")
    return "\n".join([*lines, f"해결: {action}"])


def _read_verified_result(run_dir: Path, *, with_uncertainty: bool = False) -> dict:
    try:
        return _validate_saved_result(run_dir, with_uncertainty=with_uncertainty)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("Run metadata is incomplete or malformed") from error


def _validate_saved_result(run_dir: Path, *, with_uncertainty: bool = False) -> dict:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported result schema; execute your training script and save a new result"
        )
    required = {
        "owner",
        "experiment",
        "run_id",
        "feature_set",
        "features",
        "feature_count",
        "feature_groups",
        "synthetic",
        "comparison_context",
        "training",
        "metrics",
        "model_files",
        "software",
    }
    if not required.issubset(result):
        raise ValueError("Result metadata is incomplete")
    if result["feature_set"] not in {"clinical", "ecg", "combined"}:
        raise ValueError("Unknown result feature_set")
    if not isinstance(result["synthetic"], bool):
        raise ValueError("Result must identify whether its data is synthetic")
    if not isinstance(result["model_files"], dict) or not result["model_files"]:
        raise ValueError("Result is missing exported model information")
    training = result["training"]
    if set(training) != {"model", "seed", "parameters", "source_hash", "shared_code_hash"}:
        raise ValueError("Result is missing training metadata")
    validate_training({key: training[key] for key in ("model", "seed", "parameters")}, stored=True)
    if "shared_code_scope" in result:
        validate_shared_code_scope(result["shared_code_scope"])
    validate_resource_metadata(result.get("resources", {}))
    available = _expected_features(result["feature_groups"], result["feature_set"])
    selected = result["features"]
    if (
        not isinstance(selected, list)
        or not selected
        or len(selected) != len(set(selected))
        or not set(selected).issubset(available)
        or result["feature_count"] != len(selected)
    ):
        raise ValueError("Result features are outside the permitted input group")
    predictions = pd.read_csv(
        run_dir / "predictions.csv",
        dtype={"patient_id": str},
        keep_default_na=False,
        float_precision="round_trip",
    )
    metrics = evaluation.evaluate_predictions(predictions)
    if set(result["metrics"]) != set(METRICS) or any(
        not isinstance(result["metrics"][key], int | float)
        or not np.isfinite(result["metrics"][key])
        or not np.isclose(result["metrics"][key], value, rtol=1e-12, atol=1e-12)
        for key, value in metrics.items()
    ):
        raise ValueError("Saved metrics do not match predictions; rerun evaluation")
    protocol = result["comparison_context"]["evaluation"]
    if (
        protocol["split"] != "validation"
        or protocol["metrics"] != list(METRICS)
        or any(
            protocol.get(key) != value for key, value in _prediction_identity(predictions).items()
        )
    ):
        expected = {
            "split": "validation",
            "metrics": list(METRICS),
            **_prediction_identity(predictions),
        }
        raise ValueError(
            _mismatch_message(
                "Prediction patients or targets differ from the recorded validation set",
                None,
                result,
                expected,
                {key: protocol.get(key, _MISSING) for key in expected},
                "이 실행의 예측 파일과 기록된 평가 정보가 일치하지 않습니다. "
                "predictions.csv나 result.json을 수정해 맞추지 말고 원래 준비 데이터로 "
                "학습·저장을 다시 실행하세요.",
                reference_label="공통 검증 규칙과 predictions.csv에서 계산한 식별값",
            )
        )
    if with_uncertainty:
        # Compute from the same verified frame, never a saved display summary.
        # This is an in-memory report field; the immutable result is not rewritten.
        result["validation_summary"] = validation_summary(predictions)
    return result


def compare_runs(run_dirs: list[Path], output: Path, *, mode: str = "experiments") -> pd.DataFrame:
    """Compare personal methods, or isolate input information under fixed training."""
    run_dirs = [Path(path) for path in run_dirs]
    output = Path(output)
    if mode not in {"experiments", "inputs"}:
        raise ValueError("mode must be experiments or inputs")
    if len(run_dirs) < 2 or len({path.resolve() for path in run_dirs}) != len(run_dirs):
        raise ValueError("Compare at least two distinct completed runs")
    results = [_read_verified_result(path, with_uncertainty=True) for path in run_dirs]
    if len({result["run_id"] for result in results}) != len(results):
        raise ValueError("The same run was included more than once")
    first = results[0]
    # Remote comparison bundles contain no source snapshot. A missing/different
    # scope cannot be safely converted to the newest one from its hash alone.
    scope_versions = {
        result["shared_code_scope"]["version"]
        for result in results
        if "shared_code_scope" in result
    }
    scoped = all("shared_code_scope" in result for result in results) and len(scope_versions) == 1
    if len(scope_versions) > 1:
        scope_guidance = (
            " 공통 코드 비교 범위 버전이 서로 달라 전체 공통 코드의 일치를 확인합니다. "
            "원격 기본 결과에는 소스 사본이 없어 이전 해시의 범위를 변환할 수 없습니다. "
            "같은 최신 환경에서 필요한 실험을 다시 학습·저장하세요."
        )
    elif not scoped:
        scope_guidance = (
            " 이전 형식의 실행이 포함되어 공통 코드 전체를 비교합니다. "
            "같은 최신 환경에서 다시 학습·저장하면 현재 비교 범위가 기록됩니다."
        )
    else:
        scope_guidance = ""

    def training_identity(result: dict) -> dict:
        identity = result["training"].copy()
        # The reserved selector chooses the package directory, not a learning setting.
        # Keep the actual package records and software versions in the comparison.
        identity["resources"] = {
            name: metadata
            for name, metadata in result.get("resources", {}).items()
            if name != "environment.toml"
        }
        if scoped:
            identity["shared_code_hash"] = result["shared_code_scope"]
        return identity

    def shared_identity(result: dict) -> dict:
        return {key: result[key] for key in ("comparison_context", "synthetic", "feature_groups")}

    for result in results:
        if (
            result["comparison_context"] != first["comparison_context"]
            or result["synthetic"] != first["synthetic"]
            or result["feature_groups"] != first["feature_groups"]
        ):
            raise ValueError(
                _mismatch_message(
                    "Incompatible comparison: shared data, patients, split or evaluation differs",
                    first,
                    result,
                    shared_identity(first),
                    shared_identity(result),
                    "동일한 공통 데이터·환자 분할·정답 정의·평가 버전으로 만든 실행을 선택하세요. "
                    "다른 버전의 실행은 같은 준비 데이터와 평가 환경에서 다시 학습·저장하세요.",
                )
            )
        if mode == "experiments" and result["feature_set"] != first["feature_set"]:
            raise ValueError(
                _mismatch_message(
                    "Personal experiments must use the same feature_set; "
                    "compare each input group separately",
                    first,
                    result,
                    {"feature_set": first["feature_set"]},
                    {"feature_set": result["feature_set"]},
                    "임상(clinical)·심전도(ecg)·결합(combined) 중 같은 입력 그룹만 선택하세요. "
                    "입력 그룹의 효과를 비교하려면 학습 방법을 같게 맞춘 뒤 inputs 비교를 사용하세요.",
                )
            )
        if mode == "inputs":
            if (
                training_identity(result) != training_identity(first)
                or result["software"] != first["software"]
            ):
                raise ValueError(
                    _mismatch_message(
                        "Input comparison requires the same training method, settings and environment",
                        first,
                        result,
                        {"training": training_identity(first), "software": first["software"]},
                        {"training": training_identity(result), "software": result["software"]},
                        "입력 그룹만 바꾸고 모델·설정·학습 코드·패키지 버전을 같게 맞춰 다시 실행하세요. "
                        "학습 방법 자체를 비교하려면 같은 입력 그룹에서 experiments 비교를 사용하세요."
                        + scope_guidance,
                    )
                )
            if tuple(result["features"]) != _expected_features(
                result["feature_groups"], result["feature_set"]
            ):
                raise ValueError(
                    _mismatch_message(
                        "Input comparison requires complete declared feature groups",
                        None,
                        result,
                        {
                            "features": list(
                                _expected_features(result["feature_groups"], result["feature_set"])
                            )
                        },
                        {"features": result["features"]},
                        "inputs 비교는 선택한 입력 그룹의 전체 컬럼과 선언 순서를 사용해야 합니다. "
                        "개인 설정의 features를 제거하고 다시 학습·저장하세요. "
                        "일부 컬럼을 선택한 방법의 성능은 experiments 비교에서 확인하세요.",
                    )
                )
    if mode == "inputs" and len({item["feature_set"] for item in results}) != len(results):
        selected = {}
        for result in results:
            group = result["feature_set"]
            if group in selected:
                raise ValueError(
                    "Choose one run per feature_set for an input comparison\n"
                    f"중복 입력 그룹: {group}\n"
                    f"첫 실행: {_run_label(selected[group])}\n"
                    f"중복 실행: {_run_label(result)}\n"
                    "해결: clinical·ecg·combined에서 입력 그룹마다 실행을 하나씩만 선택하세요."
                )
            selected[group] = result
    uncertainty = [item["validation_summary"] for item in results]
    summary = pd.DataFrame(
        [
            {
                "comparison_mode": mode,
                "owner": item["owner"],
                "experiment": item["experiment"],
                "run_id": item["run_id"],
                "feature_set": item["feature_set"],
                "feature_count": item["feature_count"],
                "features": json.dumps(item["features"], ensure_ascii=False),
                "model": item["training"]["model"],
                "seed": item["training"]["seed"],
                "parameters": json.dumps(item["training"]["parameters"], ensure_ascii=False),
                "source_hash": item["training"]["source_hash"],
                "synthetic": item["synthetic"],
                "evaluation_split": "validation",
                **item["metrics"],
                "patient_count": interval["patients"],
                "positive_count": interval["positive"],
                "negative_count": interval["negative"],
                "scd_count": (
                    interval["positive"]
                    if is_scd_target(
                        item["synthetic"],
                        item["comparison_context"]["dataset"].get("target_definition"),
                    )
                    else None
                ),
                **{
                    f"{metric}_ci_{bound}": interval["intervals"][metric][bound]
                    for metric in METRICS
                    for bound in ("lower", "upper")
                },
                "ci_level": interval["confidence_level"],
                "ci_method": interval["method"],
                "ci_repeats": interval["repeats"],
                "ci_seed": interval["seed"],
                "ci_unavailable_reason": interval["reason"],
            }
            for item, interval in zip(results, uncertainty, strict=True)
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        summary.to_csv(handle, index=False)
    return summary
