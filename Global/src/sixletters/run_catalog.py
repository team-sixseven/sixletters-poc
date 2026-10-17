"""Run selection displays and validation summaries, with numerical imports loaded on demand."""

import hashlib
import json
import math
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

_MISSING = object()
_UNKNOWN = object()
_MAX_SETTINGS = 60
_SUMMARY_METRICS = ("roc_auc", "average_precision", "brier_score")
_SUMMARY_REPEATS = 2000
_SUMMARY_SEED = 42
_SUMMARY_METHOD = "stratified_patient_percentile_bootstrap"
_PAIRED_METHOD = "paired_stratified_patient_percentile_bootstrap"


def validation_summary(predictions) -> dict:
    """Estimate fixed-model validation uncertainty, preserving the observed class counts.

    Draw patients with replacement within each class. Weighted score groups implement
    sklearn's tied-score AUROC and non-interpolated average precision exactly, avoiding
    thousands of calls to sklearn. The model is not retrained in these resamples.
    """
    import numpy as np

    from sixletters.evaluation import evaluate_predictions

    evaluate_predictions(predictions)
    frame = predictions.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    if frame["patient_id"].str.strip().eq("").any() or frame["patient_id"].duplicated().any():
        raise ValueError("Predictions require distinct, nonempty patient identifiers")
    frame = frame.sort_values("patient_id", kind="stable")
    targets = frame["target"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    positive = np.flatnonzero(targets == 1)
    negative = np.flatnonzero(targets == 0)
    summary = {
        "version": 1,
        "patients": len(frame),
        "positive": len(positive),
        "negative": len(negative),
        "confidence_level": 0.95,
        "method": _SUMMARY_METHOD,
        "repeats": _SUMMARY_REPEATS,
        "seed": _SUMMARY_SEED,
        "intervals": {key: {"lower": None, "upper": None} for key in _SUMMARY_METRICS},
        "reason": None,
    }
    if min(len(positive), len(negative)) < 2:
        summary["reason"] = "insufficient_class_counts"
        return summary

    grouping = _score_groups(scores)
    rng = np.random.default_rng(_SUMMARY_SEED)
    estimates = np.empty((_SUMMARY_REPEATS, 3))
    for repeat in range(_SUMMARY_REPEATS):
        estimates[repeat] = _weighted_metrics(
            grouping,
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        )
    bounds = np.clip(np.quantile(estimates, [0.025, 0.975], axis=0, method="linear"), 0, 1)
    summary["intervals"] = {
        key: {"lower": float(bounds[0, index]), "upper": float(bounds[1, index])}
        for index, key in enumerate(_SUMMARY_METRICS)
    }
    return summary


def _score_groups(scores):
    import numpy as np

    unique_scores, groups = np.unique(scores, return_inverse=True)
    return groups, (1 - unique_scores) ** 2, unique_scores**2


def _weighted_metrics(grouping, positive_draw, negative_draw):
    """Exact AUROC, non-interpolated AP and Brier on a shared patient resample."""
    import numpy as np

    groups, squared_positive, squared_negative = grouping
    positives = np.bincount(groups[positive_draw], minlength=len(squared_positive))
    negatives = np.bincount(groups[negative_draw], minlength=len(squared_negative))
    # Ascending score groups: negative scores below a positive score earn 1,
    # ties earn 1/2. AP uses descending cumulative precision at every tie group.
    auc = np.sum(positives * (np.cumsum(negatives) - negatives / 2))
    auc /= len(positive_draw) * len(negative_draw)
    true_positives = np.cumsum(positives[::-1])
    predicted_positives = true_positives + np.cumsum(negatives[::-1])
    precision = np.divide(
        true_positives,
        predicted_positives,
        out=np.zeros(len(squared_positive), dtype=float),
        where=predicted_positives > 0,
    )
    ap = np.sum(positives[::-1] * precision) / len(positive_draw)
    brier = (positives @ squared_positive + negatives @ squared_negative) / (
        len(positive_draw) + len(negative_draw)
    )
    return np.array((auc, ap, brier))


def paired_validation_summary(reference, candidate) -> dict:
    """Exploratory candidate-minus-reference intervals for already-fitted models.

    Align by patient ID and verify the target for every patient. Each class-stratified
    draw is applied to both models, preserving the paired observations. These intervals
    do not include training or model-selection uncertainty or multiple-comparison control.
    """
    import numpy as np

    from sixletters.evaluation import evaluate_predictions

    frames = []
    metrics = []
    for predictions in (reference, candidate):
        evaluate_predictions(predictions)
        frame = predictions.copy()
        frame["patient_id"] = frame["patient_id"].astype(str)
        if frame["patient_id"].str.strip().eq("").any() or frame["patient_id"].duplicated().any():
            raise ValueError("Predictions require distinct, nonempty patient identifiers")
        frame = frame.sort_values("patient_id", kind="stable").reset_index(drop=True)
        frames.append(frame)
        metrics.append(evaluate_predictions(frame))
    first, second = frames
    if not first["patient_id"].equals(second["patient_id"]):
        raise ValueError("차이 신뢰구간은 같은 검증 환자로 만든 실행끼리만 계산할 수 있습니다.")
    targets = first["target"].to_numpy(dtype=int)
    if not np.array_equal(targets, second["target"].to_numpy(dtype=int)):
        raise ValueError("차이 신뢰구간을 계산할 두 실행의 환자별 정답이 일치하지 않습니다.")
    positive = np.flatnonzero(targets == 1)
    negative = np.flatnonzero(targets == 0)
    summary = {
        "patients": len(first),
        "positive": len(positive),
        "negative": len(negative),
        "confidence_level": 0.95,
        "method": _PAIRED_METHOD,
        "repeats": _SUMMARY_REPEATS,
        "seed": _SUMMARY_SEED,
        "deltas": {key: metrics[1][key] - metrics[0][key] for key in _SUMMARY_METRICS},
        "intervals": {key: {"lower": None, "upper": None} for key in _SUMMARY_METRICS},
        "reason": None,
    }
    if min(len(positive), len(negative)) < 2:
        summary["reason"] = "insufficient_class_counts"
        return summary
    groupings = [_score_groups(frame["score"].to_numpy(dtype=float)) for frame in frames]
    rng = np.random.default_rng(_SUMMARY_SEED)
    differences = np.empty((_SUMMARY_REPEATS, 3))
    for repeat in range(_SUMMARY_REPEATS):
        positive_draw = rng.choice(positive, size=len(positive), replace=True)
        negative_draw = rng.choice(negative, size=len(negative), replace=True)
        differences[repeat] = _weighted_metrics(
            groupings[1], positive_draw, negative_draw
        ) - _weighted_metrics(groupings[0], positive_draw, negative_draw)
    bounds = np.quantile(differences, [0.025, 0.975], axis=0, method="linear")
    summary["intervals"] = {
        key: {"lower": float(bounds[0, index]), "upper": float(bounds[1, index])}
        for index, key in enumerate(_SUMMARY_METRICS)
    }
    return summary


def compare_with_reference(
    run_dirs: list[Path],
    output: Path,
    *,
    mode: str = "experiments",
    reference: str | None = None,
):
    """Keep saved-run compatibility checks and add paired validation differences.

    The default reference is the first selected run; an explicit reference is the exact
    saved run_id of one selected run. No model or saved result is altered or executed.
    Publish the comparison only after every compatibility and paired-data check succeeds.
    """
    import pandas as pd

    from sixletters.comparison import compare_runs

    run_dirs = [Path(path) for path in run_dirs]
    output = Path(output)
    if len(run_dirs) < 2 or len({path.resolve() for path in run_dirs}) != len(run_dirs):
        raise ValueError("Compare at least two distinct completed runs")
    with TemporaryDirectory(prefix="sixletters-compare-") as temporary:
        # Validate and calculate from the same bytes, even if a source bundle changes
        # while a comparison is running. Ordinary remote bundles need only these files.
        snapshots = []
        frames = []
        for index, path in enumerate(run_dirs):
            snapshot = Path(temporary) / str(index)
            snapshot.mkdir()
            for name in ("result.json", "predictions.csv"):
                (snapshot / name).write_bytes((path / name).read_bytes())
            snapshots.append(snapshot)
            frames.append(
                pd.read_csv(
                    snapshot / "predictions.csv",
                    dtype={"patient_id": str},
                    keep_default_na=False,
                    float_precision="round_trip",
                )
            )
        table = compare_runs(snapshots, Path(temporary) / "validated.csv", mode=mode)
    reference_id = table.iloc[0]["run_id"] if reference is None else reference
    if reference_id not in set(table["run_id"]):
        raise ValueError(
            "기준 실행을 선택한 결과에서 찾을 수 없습니다. "
            "--reference에는 MLflow 서버 ID가 아닌 목록의 이름/실험명 뒤에 있는 "
            "저장된 실행 ID를 입력하세요."
        )
    reference_index = table.index[table["run_id"].eq(reference_id)][0]
    summaries = [paired_validation_summary(frames[reference_index], frame) for frame in frames]
    table["reference_run_id"] = reference_id
    table["reference_owner"] = table.loc[reference_index, "owner"]
    table["reference_experiment"] = table.loc[reference_index, "experiment"]
    table["is_reference"] = table["run_id"].eq(reference_id)
    for index, is_reference in enumerate(table["is_reference"]):
        if is_reference:
            # A run minus itself is identically zero, not an estimated comparison.
            summaries[index]["intervals"] = {
                metric: {"lower": 0.0, "upper": 0.0} for metric in _SUMMARY_METRICS
            }
            summaries[index]["reason"] = None
    for metric in _SUMMARY_METRICS:
        table[f"{metric}_delta"] = [summary["deltas"][metric] for summary in summaries]
        for bound in ("lower", "upper"):
            table[f"{metric}_delta_ci_{bound}"] = [
                summary["intervals"][metric][bound] for summary in summaries
            ]
    for column, field in (
        ("delta_ci_level", "confidence_level"),
        ("delta_ci_method", "method"),
        ("delta_ci_repeats", "repeats"),
        ("delta_ci_seed", "seed"),
        ("delta_ci_unavailable_reason", "reason"),
    ):
        table[column] = [summary[field] for summary in summaries]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        table.to_csv(handle, index=False)
    return table


def summary_from_run(path: Path) -> dict:
    """Recalculate from saved validation predictions without altering the saved bundle."""
    import pandas as pd

    predictions = pd.read_csv(
        Path(path) / "predictions.csv",
        dtype={"patient_id": str},
        keep_default_na=False,
        float_precision="round_trip",
    )
    return validation_summary(predictions)


def validation_metadata(summary: object) -> dict | None:
    """Accept only the supported, internally consistent summary protocol from a server."""
    if (
        not isinstance(summary, dict)
        or "reason" not in summary
        or any(
            type(summary.get(key)) is not int or summary[key] != value
            for key, value in (
                ("version", 1),
                ("repeats", _SUMMARY_REPEATS),
                ("seed", _SUMMARY_SEED),
            )
        )
    ):
        return None
    if summary.get("confidence_level") != 0.95 or summary.get("method") != _SUMMARY_METHOD:
        return None
    if (
        any(
            type(summary.get(key)) is not int or summary[key] < 1
            for key in ("patients", "positive", "negative")
        )
        or summary["patients"] != summary["positive"] + summary["negative"]
    ):
        return None
    small = min(summary["positive"], summary["negative"]) < 2
    if summary.get("reason") != ("insufficient_class_counts" if small else None):
        return None
    intervals = summary.get("intervals")
    if not isinstance(intervals, dict) or set(intervals) != set(_SUMMARY_METRICS):
        return None
    for interval in intervals.values():
        if not isinstance(interval, dict) or set(interval) != {"lower", "upper"}:
            return None
        lower, upper = interval["lower"], interval["upper"]
        if small:
            if lower is not None or upper is not None:
                return None
        elif (
            any(type(value) not in (int, float) or not 0 <= value <= 1 for value in (lower, upper))
            or lower > upper
        ):
            return None
    return {
        key: summary[key]
        for key in (
            "version",
            "patients",
            "positive",
            "negative",
            "confidence_level",
            "method",
            "repeats",
            "seed",
            "intervals",
            "reason",
        )
    }


def is_scd_target(synthetic: object, target_definition: object) -> bool:
    """Name the positive class SCD only for the known, actual MUSIC endpoint."""
    return (
        synthetic is False
        and isinstance(target_definition, str)
        and target_definition.startswith("music-scd-")
        and "Cause of death=3 during observed follow-up is 1;" in target_definition
    )


def display_text(value: object, limit: int = 120) -> str:
    """Keep saved metadata on one bounded line without terminal control sequences."""
    text = "".join(
        " " if unicodedata.category(char).startswith("C") else char for char in str(value)
    )
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _value(value: object) -> str:
    if value is _MISSING:
        return "미기록"
    if value is _UNKNOWN:
        return "확인 불가"
    try:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError):
        return "확인 불가"
    if len(serialized) <= 120:
        return display_text(serialized)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:10]
    return f"{display_text(serialized, 100)} (sha256:{digest})"


def _flatten(parameters: dict) -> tuple[dict, bool]:
    """Use escaped JSON-pointer paths, retaining null and empty containers as values."""
    values = {}
    complete = True

    def visit(value, path, depth):
        nonlocal complete
        if len(values) >= _MAX_SETTINGS:
            complete = False
            return
        if isinstance(value, dict) and value and depth < 6:
            for key in sorted(value):
                visit(value[key], f"{path}/{key.replace('~', '~0').replace('/', '~1')}", depth + 1)
        else:
            values[path] = value

    for key in sorted(parameters):
        visit(parameters[key], f"parameters/{key.replace('~', '~0').replace('/', '~1')}", 0)
    return values, complete


def _settings(entry: dict) -> tuple[dict, bool]:
    values = {"model": entry.get("model", _UNKNOWN), "seed": entry.get("seed", _UNKNOWN)}
    if not entry.get("seed_known", "seed" in entry):
        values["seed"] = _UNKNOWN
    parameters = entry.get("parameters")
    if not isinstance(parameters, dict):
        return values, False
    flattened, unbounded = _flatten(parameters)
    values.update(flattened)
    return values, unbounded and entry.get("parameters_complete", True)


def local_entry(result: dict, path: Path) -> dict:
    """Adapt a local result without importing or running the evaluation pipeline."""
    training = result["training"]
    if not isinstance(training, dict):
        raise ValueError("Invalid training metadata")
    entry = {
        "owner": result["owner"],
        "experiment": result["experiment"],
        "local_run_id": result["run_id"],
        "dataset_version": result["comparison_context"]["dataset"]["dataset_id"],
        "target_definition": result["comparison_context"]["dataset"].get("target_definition"),
        "feature_set": result["feature_set"],
        "synthetic": result["synthetic"],
        "model": training["model"],
        "metrics": result.get("metrics", {}),
        "parameters": training.get("parameters"),
        "seed_known": "seed" in training,
        "seed": training.get("seed"),
        "created_at": result.get("created_at"),
        "created_at_source": "saved",
        "path": str(path.parent),
    }
    if _timestamp(entry["created_at"]) is None:
        entry["created_at"] = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
        entry["created_at_source"] = "file_modified"
    return entry


def _created_at(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(UTC)
    except (ValueError, OverflowError):
        return None


def _timestamp(value: object) -> str | None:
    parsed = _created_at(value)
    return parsed.strftime("%Y-%m-%d %H:%M:%S UTC") if parsed else None


def filter_run_catalog(
    entries: list[dict],
    *,
    owner: str | None = None,
    dataset: str | None = None,
    feature_set: str | None = None,
    experiment: str | None = None,
    archived_only: bool = False,
    limit: int = 20,
) -> tuple[list[dict], int]:
    """Filter before limiting; sort by recency, then stable identity, never by score."""
    if type(limit) is not int or limit < 1:
        raise ValueError("--limit에는 1 이상의 정수를 입력하세요.")
    filters = {
        "owner": owner,
        "dataset_version": dataset,
        "feature_set": feature_set,
        "experiment": experiment,
    }
    matches = [
        entry
        for entry in entries
        if all(value is None or entry.get(key) == value for key, value in filters.items())
        and (not archived_only or entry.get("archive_status") == "complete")
    ]
    # Sort ties by identity first; the stable recency sort keeps that order.
    matches.sort(
        key=lambda entry: tuple(
            str(entry.get(key, ""))
            for key in ("owner", "experiment", "local_run_id", "run_id", "path")
        )
    )
    matches.sort(
        key=lambda entry: _created_at(entry.get("created_at")) or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )
    return matches[:limit], len(matches)


def _metric(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
        return "확인 불가"
    return f"{value:.4f}"


def _print_settings(label: str, values: list[str]) -> None:
    line = label
    for value in values:
        separator = "" if line == label or line == "      " else " | "
        if len(line) + len(separator) + len(value) > 220 and line != label:
            print(line)
            line = "      " + value
        else:
            line += separator + value
    print(line)


def _show_validation_summary(entry: dict) -> None:
    if entry.get("path"):
        try:
            summary = summary_from_run(Path(entry["path"]))
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            print("   검증 환자·양성 수·95% 신뢰구간: 확인 불가 (예측 파일 확인 필요)")
            return
    else:
        summary = validation_metadata(entry.get("validation_summary"))
        if summary is None:
            status = entry.get("validation_summary_status")
            missing = entry.get("validation_summary") is None and status != "invalid"
            message = "미기록 · 비교 시 계산" if missing else "확인 불가 · 비교 시 재계산"
            print(f"   검증 환자·양성 수·95% 신뢰구간: {message}")
            return
    positive = (
        "SCD"
        if is_scd_target(entry.get("synthetic"), entry.get("target_definition"))
        else "양성(target=1)"
    )
    print(f"   검증 환자={summary['patients']} | {positive}={summary['positive']}")
    if summary["reason"]:
        print("   95% 신뢰구간: 계산하지 않음 (양성·음성 각 2명 이상 필요)")
        return
    intervals = summary["intervals"]
    print(
        "   95% 신뢰구간: "
        + " | ".join(
            f"{label}=[{intervals[key]['lower']:.4f}, {intervals[key]['upper']:.4f}]"
            for label, key in zip(("AUROC", "AP", "Brier"), _SUMMARY_METRICS, strict=True)
        )
    )


def show_run_catalog(entries: list[dict]) -> None:
    """Keep input order stable; displayed metadata never replaces result verification."""
    if not entries:
        return
    print("저장된 검증 점수·설정으로 선택합니다. 실제 비교 시 예측 파일을 다시 검증합니다.")
    print(
        "95% 신뢰구간은 이 검증 환자 구성에서의 표본 변동입니다. 학습·모델 선택의 변동을 포함하지 않으며, "
        "모델 간 차이의 유의성 검정이 아닙니다."
    )
    settings = [_settings(entry) for entry in entries]
    keys = sorted(
        set().union(*(values for values, _ in settings)),
        key=lambda key: (key not in ("model", "seed"), key),
    )
    models = [values["model"] for values, _ in settings]
    same_model = all(isinstance(value, str) and value == models[0] for value in models)
    common = {}
    if len(entries) > 1:
        for key in keys:
            if key.startswith("parameters/") and (
                not same_model or not all(c for _, c in settings)
            ):
                continue
            values = [items.get(key, _MISSING) for items, _ in settings]
            if all(value is not _MISSING and value is not _UNKNOWN for value in values):
                try:
                    encodings = [
                        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
                        for value in values
                    ]
                except (ValueError, TypeError):
                    continue
                if len(set(encodings)) == 1:
                    common[key] = values[0]
    if common:
        shown = list(common.items())[:5]
        summary = [f"{display_text(key, 70)}={_value(value)}" for key, value in shown]
        if len(common) > len(shown):
            summary.append(f"공통 설정 {len(common) - len(shown)}개 생략")
        _print_settings("공통 설정 (목록 기준): ", summary)
    for index, (entry, (values, complete)) in enumerate(
        zip(entries, settings, strict=True), start=1
    ):
        data_type = (
            "data type unknown"
            if entry.get("synthetic") is None
            else "synthetic"
            if entry["synthetic"]
            else "actual data"
        )
        labels = [entry.get(key, "확인 불가") for key in ("owner", "experiment", "local_run_id")]
        identity = " / ".join(display_text(value, 80) for value in labels)
        context = " | ".join(
            display_text(entry.get(key, "확인 불가"), 80)
            for key in ("dataset_version", "feature_set", "model")
        )
        print(f"{index}. {identity} | {context} | {data_type}")
        archived = {
            "complete": "완료",
            "pending": "전송 대기",
            "none": "미선택",
            "legacy": "이전 형식 전체 파일·별도 확인 필요",
        }.get(entry.get("archive_status"), "확인되지 않음")
        provenance = (
            " (로컬 전송 기록 기준)" if entry.get("archive_status_source") == "local_record" else ""
        )
        print(f"   모델 보관: {archived}{provenance}")
        metrics = entry.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        print(
            "   검증 점수: "
            + " | ".join(
                f"{label}={_metric(metrics, key)}"
                for label, key in (
                    ("AUROC", "roc_auc"),
                    ("AP", "average_precision"),
                    ("Brier", "brier_score"),
                )
            )
        )
        _show_validation_summary(entry)
        timestamp = _timestamp(entry.get("created_at"))
        source = {
            "saved": "결과 저장",
            "file_modified": "파일 수정 시각·저장 시각 대체",
            "tracking_start": "서버 등록",
        }.get(entry.get("created_at_source"), "기록 시각")
        print(f"   시각: {timestamp} ({source})" if timestamp else "   시각: 확인 불가")
        differences = []
        for key in keys:
            if key in common:
                continue
            missing = _UNKNOWN if key.startswith("parameters/") and not complete else _MISSING
            differences.append(f"{display_text(key, 70)}={_value(values.get(key, missing))}")
        if differences:
            _print_settings("   실행별 설정: ", differences)
        else:
            print("   실행별 설정: 표시된 설정이 모두 같습니다.")
        if not complete:
            print("   설정 일부 확인 불가: 누락·축약된 값은 동일하다고 판단하지 않습니다.")
        if entry.get("path"):
            print(f"   위치: {display_text(entry['path'], 240)}")
        elif entry.get("run_id"):
            print(f"   서버 실행 ID: {display_text(entry['run_id'], 80)}")


def show_sync_summary(summary: dict) -> None:
    """Render structured, sanitized issues supplied by tracking; never show exceptions."""
    labels = (
        ("uploaded", "전송 완료"),
        ("already_synced", "기존 전송 완료"),
        ("pending", "재전송 대기"),
        ("invalid", "확인 필요"),
    )
    print(" | ".join(f"{label}: {summary.get(key, 0)}" for key, label in labels))
    for issue in summary.get("issues", []):
        if issue.get("scope") == "scan":
            path = display_text(issue.get("path", "."), 240)
            print(f"- 탐색 실패: {path} [{display_text(issue.get('code', 'unknown'), 40)}]")
            print("  해당 경로의 실행 결과를 모두 확인하지 못했습니다.")
        else:
            identity = " / ".join(
                display_text(issue.get(key, "확인 불가"), 80)
                for key in ("owner", "experiment", "run_id")
            )
            print(f"- {identity} [{display_text(issue.get('code', 'unknown'), 40)}]")
        print(f"  원인: {display_text(issue.get('message', '확인 불가'), 240)}")
        print(
            f"  조치: {display_text(issue.get('action', 'uv run --project Global poc sync로 다시 확인하세요.'), 240)}"
        )


def show_remote_issues(issues: list[dict]) -> None:
    """Explain quarantined remote results without exposing server exceptions."""
    for issue in issues:
        identity = " / ".join(
            display_text(issue.get(key, "확인 불가"), 80)
            for key in ("owner", "experiment", "run_id")
        )
        print(f"- 비교 제외: {identity} [{display_text(issue.get('code', 'unknown'), 40)}]")
        for remote_id in issue.get("remote_run_ids", []):
            print(f"  MLflow 실행 ID: {display_text(remote_id, 80)}")
        print(f"  원인: {display_text(issue.get('message', '확인 불가'), 240)}")
        print(f"  조치: {display_text(issue.get('action', '서버 관리자에게 확인하세요.'), 240)}")


def show_final_status(summary: dict) -> None:
    """Show progress and reviewed next steps without exposing test scores or labels."""
    print(f"최종 평가: {display_text(summary['name'])}")
    counts = summary["counts"]
    print(
        f"예측 기록 완료 {counts['complete']}개 | 미실행 {counts['missing']}개 | 확인 필요 {counts['invalid']}개"
    )
    for candidate in summary["candidates"]:
        identity = " / ".join(
            display_text(candidate[key]) for key in ("owner", "experiment", "run_id")
        )
        label = {"complete": "예측 기록 완료", "missing": "미실행", "invalid": "확인 필요"}[
            candidate["status"]
        ]
        print(f"- {identity}: {label}")
        print(f"  {display_text(candidate['message'], 400)}")
        for command in candidate["commands"]:
            print(f"  {command}")
    if summary["report"]:
        print(f"기존 보고서: {display_text(summary['report'], 400)}")
    for command in summary["next_commands"]:
        print(command)
    print(
        "명령은 자동 실행하지 않습니다. 고정된 후보를 재학습하거나 테스트 결과로 교체하지 마세요."
    )
