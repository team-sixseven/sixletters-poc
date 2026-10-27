"""Backward-compatible entry point for the configured clinical model trainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.ML.training import train_clinical


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="Subset of configured model names")
    parser.add_argument("--run-id", help="Run ID prefix; each selected model gets its own run")
    parser.add_argument("--limit-train-patients", type=int)
    args = parser.parse_args()
    choice_path = Path(__file__).resolve().parents[1] / "model/ML/model_choice.yaml"
    run_dirs = train_clinical(
        choice_path,
        model_names=args.models,
        run_id=args.run_id,
        limit_train_patients=args.limit_train_patients,
    )
    print(json.dumps({"training_runs": [str(path) for path in run_dirs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
