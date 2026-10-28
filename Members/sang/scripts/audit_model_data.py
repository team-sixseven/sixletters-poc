"""Write a data, split, modality-coverage, and exclusion audit from basic.yaml."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.config import load_basic_config, resolve_data_path
from model.data import load_cohorts, write_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--basic-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "model/basic.yaml",
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    basic_path = args.basic_config.resolve()
    basic = load_basic_config(basic_path)
    _, _, audit, _ = load_cohorts(basic, basic_path)
    run_id = args.run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    output = resolve_data_path(basic["outputs"]["audit_directory"], basic_path) / run_id
    if output.exists():
        raise FileExistsError(f"Audit output exists and will not be overwritten: {output}")
    write_audit(audit, output)
    print(
        json.dumps(
            {"audit_directory": str(output), **audit["counts"], "splits": audit["split_summary"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
