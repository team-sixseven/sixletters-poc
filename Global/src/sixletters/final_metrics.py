"""Frozen-threshold test evaluation; no model fitting, loading, or test selection.

Intervals are percentile patient bootstraps. When both classes exist, sampling is
stratified by the observed target, so the intervals condition on those class
counts. Every model uses the same resampled patients; deltas are paired. These
intervals describe test-sample variation, not training or model-selection
uncertainty, and do not establish clinical validity.

AP uses sklearn's non-interpolated average precision, not trapezoidal PR area.
ROC and calibration follow sklearn.metrics.roc_curve and calibration_curve.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

METRICS = (
    "roc_auc",
    "average_precision",
    "brier_score",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
)
DEFAULT_BOOTSTRAP_REPEATS = 2000
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
_FILES = (
    "final_metrics.json",
    "metrics.csv",
    "deltas.csv",
    "calibration.csv",
    "roc.png",
    "pr.png",
    "calibration.png",
    "confusion.png",
)


def _predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if (
        not isinstance(frame, pd.DataFrame)
        or frame.empty
        or frame.columns.has_duplicates
        or set(frame.columns) != {"patient_id", "target", "score"}
    ):
        raise ValueError("Predictions require nonempty patient_id, target and score columns")
    ids = frame["patient_id"]
    if (
        ids.isna().any()
        or ids.duplicated().any()
        or not ids.map(lambda value: isinstance(value, str) and bool(value.strip())).all()
        or ids.ne(ids.str.strip()).any()
    ):
        raise ValueError("Predictions require unique, nonblank, unpadded string patient IDs")
    target, score = frame["target"], frame["score"]
    for name, values in (("target", target), ("score", score)):
        if (
            not is_numeric_dtype(values.dtype)
            or is_bool_dtype(values.dtype)
            or is_complex_dtype(values.dtype)
            or values.isna().any()
        ):
            raise ValueError(f"Prediction {name} must contain nonmissing real numbers")
    if not target.isin([0, 1]).all():
        raise ValueError("Prediction target must contain only 0 and 1")
    probabilities = score.to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Prediction score must be finite and between 0 and 1")
    return frame.sort_values("patient_id").reset_index(drop=True).copy()


def _threshold(value: float) -> float:
    if isinstance(value, bool | complex | np.complexfloating) or not isinstance(
        value, int | float | np.number
    ):
        raise ValueError("Threshold must be a finite probability between 0 and 1")
    try:
        threshold = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("Threshold must be a finite probability between 0 and 1") from error
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite probability between 0 and 1")
    return threshold


def select_threshold(validation: pd.DataFrame, *, fixed: float | None = None) -> dict:
    """Select a threshold on validation only, or record a prespecified fixed value.

    Youden J maximizes sensitivity + specificity - 1. Ties within 1e-12 use
    the largest threshold. Candidate thresholds are observed probabilities plus
    0 and 1; positive predictions use score >= threshold. No test data is accepted.
    """
    frame = _predictions(validation)
    y = frame["target"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    positives, negatives = int(y.sum()), int((y == 0).sum())
    if not positives or not negatives:
        raise ValueError("Threshold selection requires both validation target classes")
    if fixed is not None:
        threshold = _threshold(fixed)
        method, tie_break = "fixed", None
    else:
        fpr, tpr, candidates = roc_curve(y, scores, drop_intermediate=False)
        finite = np.isfinite(candidates)
        candidates, objective = candidates[finite], (tpr - fpr)[finite]
        for boundary in (0.0, 1.0):
            if boundary not in candidates:
                predicted = scores >= boundary
                j = float(predicted[y == 1].mean() - predicted[y == 0].mean())
                candidates = np.append(candidates, boundary)
                objective = np.append(objective, j)
        best = np.isclose(objective, objective.max(), rtol=0, atol=1e-12)
        threshold = float(candidates[best].max())
        method, tie_break = "youden_j", "largest_threshold_within_1e-12"
    predicted = scores >= threshold
    return {
        "threshold": threshold,
        "method": method,
        "selection_split": "validation",
        "positive_rule": "score >= threshold",
        "tie_break": tie_break,
        "validation_counts": {"patients": len(y), "positive": positives, "negative": negatives},
        "validation_youden_j": float(predicted[y == 1].mean() - predicted[y == 0].mean()),
    }


def _point_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[dict, dict, dict]:
    predicted = scores >= threshold
    positive = y == 1
    tn = int((~positive & ~predicted).sum())
    fp = int((~positive & predicted).sum())
    fn = int((positive & ~predicted).sum())
    tp = int((positive & predicted).sum())
    counts = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    undefined = {}

    def ratio(name: str, numerator: int, denominator: int, reason: str) -> float | None:
        if not denominator:
            undefined[name] = reason
            return None
        return float(numerator / denominator)

    both_classes = bool(tp + fn) and bool(tn + fp)
    if not both_classes:
        undefined["roc_auc"] = "Test data contains only one target class"
    if not tp + fn:
        undefined["average_precision"] = "Test data contains no positive patients"
    metrics = {
        "roc_auc": float(roc_auc_score(y, scores)) if both_classes else None,
        "average_precision": float(average_precision_score(y, scores)) if tp + fn else None,
        "brier_score": float(np.mean((scores - y) ** 2)),
        "sensitivity": ratio("sensitivity", tp, tp + fn, "No positive patients"),
        "specificity": ratio("specificity", tn, tn + fp, "No negative patients"),
        "precision": ratio("precision", tp, tp + fp, "No predicted positive patients"),
        "f1": ratio("f1", 2 * tp, 2 * tp + fp + fn, "No actual or predicted positive patients"),
    }
    return metrics, counts, undefined


def evaluate_final_predictions(predictions: pd.DataFrame, threshold: float) -> dict:
    """Evaluate a frozen threshold; undefined ratios are null with explicit reasons."""
    frame = _predictions(predictions)
    threshold = _threshold(threshold)
    metrics, confusion, undefined = _point_metrics(
        frame["target"].to_numpy(dtype=int), frame["score"].to_numpy(dtype=float), threshold
    )
    return {
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": confusion,
        "undefined": undefined,
    }


def _interval(values: np.ndarray, *, point: float | None, reason: str | None = None) -> dict:
    valid = values[np.isfinite(values)]
    result = {"lower": None, "upper": None, "valid_resamples": len(valid), "reason": None}
    if point is None:
        result["reason"] = reason or "Point estimate is undefined"
    elif len(valid) < 2:
        result["reason"] = "Fewer than two valid bootstrap resamples"
    else:
        lower, upper = np.quantile(valid, [0.025, 0.975], method="linear")
        result.update(lower=float(lower), upper=float(upper))
    return result


def _aligned(predictions: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not isinstance(predictions, Mapping) or not predictions:
        raise ValueError("Provide at least one run's test predictions")
    frames = {}
    identities = None
    for name, frame in predictions.items():
        if not isinstance(name, str) or not _RUN_ID.fullmatch(name):
            raise ValueError("Report keys must be immutable 32-character hexadecimal run IDs")
        frame = _predictions(frame)
        identity = frame[["patient_id", "target"]].copy()
        identity["target"] = identity["target"].astype(int)
        if identities is not None and not identities.equals(identity):
            raise ValueError("All final models must use the same test patients and targets")
        identities = identity
        frames[name] = frame
    return frames


def _label(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 120:
        raise ValueError("Report labels must be nonempty text of at most 120 characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("Report labels must not contain control characters")
    return value


def _bootstrap_report(frames: dict, thresholds: dict, *, reference: str, repeats: int, seed: int):
    y = next(iter(frames.values()))["target"].to_numpy(dtype=int)
    scores = {name: frame["score"].to_numpy(dtype=float) for name, frame in frames.items()}
    models = {
        name: evaluate_final_predictions(frame, thresholds[name]) for name, frame in frames.items()
    }
    samples = {name: np.full((repeats, len(METRICS)), np.nan) for name in frames}
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == label) for label in (0, 1)]
    for index in range(repeats):
        # A shared index array preserves paired patient/model relationships.
        selected = np.concatenate(
            [
                rng.choice(stratum, size=len(stratum), replace=True)
                for stratum in strata
                if len(stratum)
            ]
        )
        for name, probabilities in scores.items():
            metrics, _, _ = _point_metrics(y[selected], probabilities[selected], thresholds[name])
            samples[name][index] = [
                np.nan if metrics[metric] is None else metrics[metric] for metric in METRICS
            ]
    for name, model in models.items():
        model["confidence_intervals"] = {
            metric: _interval(
                samples[name][:, index],
                point=model["metrics"][metric],
                reason=model["undefined"].get(metric),
            )
            for index, metric in enumerate(METRICS)
        }
    deltas = {}
    for name, model in models.items():
        if name == reference:
            continue
        points, intervals = {}, {}
        for index, metric in enumerate(METRICS):
            candidate, baseline = model["metrics"][metric], models[reference]["metrics"][metric]
            points[metric] = None if candidate is None or baseline is None else candidate - baseline
            intervals[metric] = _interval(
                samples[name][:, index] - samples[reference][:, index],
                point=points[metric],
                reason="Candidate or reference point estimate is undefined",
            )
        deltas[name] = {
            "reference": reference,
            "metrics": points,
            "confidence_intervals": intervals,
        }
    return models, deltas


def _figures(frames: dict, models: dict, output_dir: Path) -> pd.DataFrame:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from sklearn.metrics import precision_recall_curve

    def figure(figsize=(8, 5)):
        fig = Figure(figsize=figsize, layout="constrained")
        FigureCanvasAgg(fig)
        return fig, fig.subplots()

    roc, roc_ax = figure()
    pr, pr_ax = figure()
    calibration, calibration_ax = figure()
    rows = []
    for name, frame in frames.items():
        y, score = frame["target"].to_numpy(dtype=int), frame["score"].to_numpy(dtype=float)
        model = models[name]
        label = model["label"].replace("$", r"\$")
        if model["metrics"]["roc_auc"] is not None:
            fpr, tpr, _ = roc_curve(y, score)
            roc_ax.plot(fpr, tpr, label=f"{label}: {model['metrics']['roc_auc']:.3f}")
        if model["metrics"]["average_precision"] is not None:
            precision, recall, _ = precision_recall_curve(y, score)
            pr_ax.step(
                recall,
                precision,
                where="post",
                label=f"{label}: AP={model['metrics']['average_precision']:.3f}",
            )
        # Five equal-width bins; empty bins are omitted. No recalibration is fitted.
        bins = np.searchsorted(np.linspace(0.0, 1.0, 6)[1:-1], score)
        model_rows = []
        for bin_index in range(5):
            selected = bins == bin_index
            if selected.any():
                row = {
                    "run_id": name,
                    "bin": bin_index + 1,
                    "patients": int(selected.sum()),
                    "mean_probability": float(score[selected].mean()),
                    "observed_fraction": float(y[selected].mean()),
                }
                rows.append(row)
                model_rows.append(row)
        calibration_ax.plot(
            [row["mean_probability"] for row in model_rows],
            [row["observed_fraction"] for row in model_rows],
            marker="o",
            label=label,
        )
    roc_ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    y = next(iter(frames.values()))["target"].to_numpy(dtype=int)
    pr_ax.axhline(float(y.mean()), linestyle="--", color="gray", label="Observed prevalence")
    calibration_ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal calibration")
    for fig, ax, filename, xlabel, ylabel, title in (
        (roc, roc_ax, "roc.png", "False positive rate", "Sensitivity", "Final test ROC"),
        (pr, pr_ax, "pr.png", "Recall", "Precision", "Final test precision-recall (AP)"),
        (
            calibration,
            calibration_ax,
            "calibration.png",
            "Mean predicted probability",
            "Observed positive fraction",
            "Final test calibration (5 uniform bins)",
        ),
    ):
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=(0, 1), ylim=(0, 1.02))
        if filename == "roc.png" and len(np.unique(y)) < 2:
            ax.text(0.5, 0.4, "AUROC undefined: one target class", ha="center")
        if filename == "pr.png" and not y.any():
            ax.text(0.5, 0.4, "AP undefined: no positive patients", ha="center")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.savefig(output_dir / filename, dpi=150)
        fig.clear()
    columns = min(3, len(models))
    nrows = math.ceil(len(models) / columns)
    confusion = Figure(figsize=(4 * columns, 3.8 * nrows), layout="constrained")
    FigureCanvasAgg(confusion)
    axes = np.asarray(confusion.subplots(nrows, columns, squeeze=False)).ravel()
    for ax, model in zip(axes, models.values(), strict=False):
        c = model["confusion_matrix"]
        matrix = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=len(y))
        for row, column in np.ndindex(2, 2):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > len(y) / 2 else "black",
            )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xlabel="Predicted",
            ylabel="Actual",
            title=f"{model['label'].replace('$', '')}\nthreshold={model['threshold']:.4g}",
        )
    for ax in axes[len(models) :]:
        ax.set_visible(False)
    confusion.savefig(output_dir / "confusion.png", dpi=150)
    confusion.clear()
    return pd.DataFrame(rows)


def build_final_report(
    predictions: Mapping[str, pd.DataFrame],
    thresholds: Mapping[str, float],
    output_dir: Path,
    *,
    reference: str | None = None,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = 42,
    labels: Mapping[str, str] | None = None,
) -> dict:
    """Write aggregate test results and figures using thresholds frozen elsewhere.

    The caller verifies the frozen plan, endpoint, common test identities, and
    exclusive final-evaluation lifecycle. This module neither selects models nor
    derives thresholds from test data. Existing report files are never replaced.
    """
    if type(repeats) is not int or not 100 <= repeats <= 10_000:
        raise ValueError("Bootstrap repeats must be an integer between 100 and 10000")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Bootstrap seed must be an integer between 0 and 2**32 - 1")
    frames = _aligned(predictions)
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(frames):
        raise ValueError("Provide exactly one frozen threshold for every run")
    thresholds = {name: _threshold(value) for name, value in thresholds.items()}
    reference = next(iter(frames)) if reference is None else reference
    if reference not in frames:
        raise ValueError("Reference must identify a provided run")
    if labels is not None and (not isinstance(labels, Mapping) or set(labels) != set(frames)):
        raise ValueError("Provide one display label for every run")
    labels = {name: _label(labels[name] if labels is not None else name[:8]) for name in frames}
    output_dir = Path(output_dir)
    if any((output_dir / filename).exists() for filename in _FILES):
        raise FileExistsError(
            "Final report files already exist; existing results were not replaced"
        )
    models, deltas = _bootstrap_report(
        frames, thresholds, reference=reference, repeats=repeats, seed=seed
    )
    y = next(iter(frames.values()))["target"].to_numpy(dtype=int)
    for name, model in models.items():
        model["label"] = labels[name]
    report = {
        "schema_version": 1,
        "evaluation_split": "test",
        "reference": reference,
        "cohort": {
            "patients": len(y),
            "positive": int(y.sum()),
            "negative": int((y == 0).sum()),
            "prevalence": float(y.mean()),
        },
        "bootstrap": {
            "method": "patient_stratified_percentile",
            "confidence_level": 0.95,
            "repeats": repeats,
            "seed": seed,
            "paired": True,
            "conditional_on_observed_class_counts": True,
            "thresholds_refitted": False,
            "limitations": [
                "Intervals condition on observed positive/negative counts and fixed models and thresholds.",
                "Training and model-selection uncertainty are not included.",
                "Undefined resampled metrics are omitted; valid_resamples reports the denominator.",
                "Small event counts can yield unstable or degenerate intervals; no clinical validity is implied.",
            ],
        },
        "metric_definitions": {
            "average_precision": "Non-interpolated sklearn AP; not trapezoidal PR area",
            "positive_rule": "score >= frozen threshold",
            "undefined_ratios": "Null with an explicit reason; never silently replaced with zero",
            "delta": "candidate minus reference; lower Brier is better, higher other metrics is better",
        },
        "models": models,
        "deltas": deltas,
        "files": list(_FILES),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_rows = _figures(frames, models, output_dir)
    metric_rows, delta_rows = [], []
    for name, model in models.items():
        for metric in METRICS:
            metric_rows.append(
                {
                    "run_id": name,
                    "label": labels[name],
                    "threshold": thresholds[name],
                    "metric": metric,
                    "estimate": model["metrics"][metric],
                    **model["confidence_intervals"][metric],
                    **model["confusion_matrix"],
                }
            )
    for name, delta in deltas.items():
        for metric in METRICS:
            delta_rows.append(
                {
                    "run_id": name,
                    "reference": reference,
                    "metric": metric,
                    "estimate": delta["metrics"][metric],
                    **delta["confidence_intervals"][metric],
                }
            )
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False, mode="x")
    pd.DataFrame(
        delta_rows,
        columns=[
            "run_id",
            "reference",
            "metric",
            "estimate",
            "lower",
            "upper",
            "valid_resamples",
            "reason",
        ],
    ).to_csv(output_dir / "deltas.csv", index=False, mode="x")
    calibration_rows.to_csv(output_dir / "calibration.csv", index=False, mode="x")
    with (output_dir / "final_metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return report
