"""Score saved patient-level probability files without retraining models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.evaluation.core import binary_metrics
from model.evaluation.DL.holter import choose_youden_threshold
from model.evaluation.reporting import create_report_dir, metric, write_report


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = {"patient_id", "target", "score"}
    if frame.columns.has_duplicates or set(frame.columns) != expected:
        raise ValueError(f"{path}: expected exactly patient_id,target,score columns")
    if frame["patient_id"].isna().any() or frame["patient_id"].duplicated().any():
        raise ValueError(f"{path}: patient_id values must be present and unique")
    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    targets = pd.to_numeric(frame["target"], errors="raise")
    if not targets.isin((0, 1)).all():
        raise ValueError(f"{path}: target must contain only 0 and 1")
    frame["target"] = targets.astype(int)
    frame["score"] = pd.to_numeric(frame["score"], errors="raise").astype(float)
    return frame


def evaluate_files(
    validation_path: Path,
    *,
    test_path: Path | None = None,
    input_group: str,
    threshold_mode: str = "auto",
    threshold: float = 0.5,
) -> dict:
    validation = load_predictions(validation_path)
    if threshold_mode == "auto":
        threshold_mode = "youden"
    if threshold_mode == "youden":
        threshold = choose_youden_threshold(
            validation["target"].to_numpy(), validation["score"].to_numpy()
        )["threshold"]
        validation_threshold = threshold
    elif threshold_mode == "fixed":
        validation_threshold = float(threshold)
    else:
        raise ValueError("threshold_mode must be auto, fixed or youden")

    output = {
        "input_group": input_group,
        "threshold_source": "validation",
        "threshold_strategy": threshold_mode,
        "threshold": validation_threshold,
        "validation": binary_metrics(
            validation["target"], validation["score"], validation_threshold
        ),
    }
    if test_path is not None:
        test = load_predictions(test_path)
        overlap = set(validation["patient_id"]) & set(test["patient_id"])
        if overlap:
            raise ValueError("Validation and test patient IDs overlap")
        output["test"] = binary_metrics(test["target"], test["score"], validation_threshold)
        output["test_used_for_threshold_selection"] = False
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument(
        "--input-group",
        choices=("clinical", "holter_raw_waveform", "holter_segment_statistics"),
        required=True,
    )
    parser.add_argument("--threshold-mode", choices=("auto", "fixed", "youden"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="Optional new JSON output file")
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs/evaluation",
        help="Parent directory for this run's mandatory report.md and results.json",
    )
    args = parser.parse_args()
    report_dir = create_report_dir(args.report_root, "prediction-review")

    result = evaluate_files(
        args.validation,
        test_path=args.test,
        input_group=args.input_group,
        threshold_mode=args.threshold_mode,
        threshold=args.threshold,
    )
    result["report_dir"] = str(report_dir)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        if args.output.exists():
            parser.error(f"Refusing to overwrite existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    write_report(
        report_dir,
        result,
        "\n".join(
            [
                "# Saved Prediction Evaluation",
                "",
                "- Status: COMPLETE",
                f"- Input group: {result['input_group']}",
                f"- Threshold strategy: {result['threshold_strategy']}",
                f"- Threshold (selected from validation): {metric(result['threshold'])}",
                "- Test data used for threshold selection: no",
                "",
                "## Validation",
                "",
                *[
                    f"- {key}: {metric(value) if isinstance(value, (int, float)) else value}"
                    for key, value in result["validation"].items()
                ],
                *(
                    [
                        "",
                        "## Test",
                        "",
                        *[
                            f"- {key}: {metric(value) if isinstance(value, (int, float)) else value}"
                            for key, value in result["test"].items()
                        ],
                    ]
                    if "test" in result
                    else []
                ),
            ]
        ),
    )
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
