"""Backward-compatible entry point for the YAML-selected Holter DL trainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.DL.training import train_holter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--limit-train-patients", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    choice_path = Path(__file__).resolve().parents[1] / "model/DL/model_choice.yaml"
    run_dir = train_holter(
        choice_path,
        run_id=args.run_id,
        max_epochs=args.max_epochs,
        limit_train_patients=args.limit_train_patients,
        device_override=args.device,
    )
    print(json.dumps({"training_run": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
