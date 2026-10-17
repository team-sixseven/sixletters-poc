"""Evaluate a trusted, frozen sklearn model; edit the loader for other model formats."""

import argparse
import pickle
from pathlib import Path

import pandas as pd

from sixletters.final_evaluation import FinalEvaluation
from sixletters.local_settings import configured_data, project_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict with a frozen model without refitting")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    root = project_root(Path(__file__).resolve().parent)
    plan = args.plan if args.plan.is_absolute() else root / args.plan
    session = FinalEvaluation(plan, args.run_id, args.data or configured_data(root))
    # Only load your team's reviewed model. Pickle can execute code when loaded.
    with (session.run_directory / "models/model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    positive = list(model.classes_).index(1)

    def predictions(inputs):
        return pd.DataFrame(
            {
                "patient_id": inputs.index,
                "score": model.predict_proba(inputs)[:, positive],
            }
        )

    validation = predictions(session.validation_data())
    test = predictions(session.test_data())
    destination = session.save(validation, test)
    print(f"Frozen candidate test predictions saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
