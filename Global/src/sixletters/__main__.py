"""Workspace commands for common preparation and comparison, never model training."""

import argparse
import sys
from pathlib import Path

from .local_settings import project_root, resolve_workspace_path


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Independent experiments with shared validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Prepare synthetic data for an execution check")
    demo.add_argument("--output", type=Path)
    music = subparsers.add_parser(
        "music", help="Download verified MUSIC CSVs and prepare shared labels, features and splits"
    )
    music.add_argument(
        "--cohort", choices=("basic", "hrv"), default="basic", help="Versioned patient selection"
    )
    music.add_argument(
        "--offline",
        action="store_true",
        help="Use the verified local CSV cache without downloading",
    )
    prepare = subparsers.add_parser(
        "prepare", help="Save reviewed common inputs and patient splits"
    )
    prepare_source = prepare.add_mutually_exclusive_group(required=True)
    prepare_source.add_argument("--demo", action="store_true", help="Use synthetic inputs")
    prepare_source.add_argument("--csv", type=Path, help="Use a normalized, reviewed patient CSV")
    prepare.add_argument("--schema", type=Path, help="JSON column and target definitions for --csv")
    prepare.add_argument("--output", type=Path, required=True, help="New common data directory")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.csv and args.schema is None:
            parser.error("prepare --csv requires --schema")
        if args.demo and args.schema is not None:
            parser.error("prepare --schema is only valid with --csv")
    try:
        root = project_root()
        for option in ("root", "data", "output", "run", "csv", "schema"):
            value = getattr(args, option, None)
            if value is not None:
                setattr(args, option, resolve_workspace_path(root, value))
        if getattr(args, "runs", None):
            args.runs = [resolve_workspace_path(root, path) for path in args.runs]
        if args.command == "demo":
            from .data.demo import make_demo_dataset
            from .data.prepared import save_prepared_dataset

            destination = args.output or root / "Global/data" / "prepared" / "demo-v1"
            path = save_prepared_dataset(make_demo_dataset(), destination, synthetic=True)
            print(f"Synthetic demo inputs (NOT MUSIC) prepared: {path}")
        elif args.command == "music":
            from .data.music import prepare_music

            path = prepare_music(root, cohort=args.cohort, offline=args.offline)
            print(f"MUSIC {args.cohort} common inputs verified: {path}")
            print(f"Preparation rules: {path.with_name('protocol.json')}")
            print(f"Patient counts and data checks: {path.with_name('report.json')}")
        elif args.command == "prepare":
            from .data.prepared import import_csv_dataset, save_prepared_dataset

            if args.demo:
                from .data.demo import make_demo_dataset

                dataset = make_demo_dataset()
            else:
                dataset = import_csv_dataset(args.csv, args.schema)
            path = save_prepared_dataset(dataset, args.output, synthetic=args.demo)
            label = "Synthetic demo inputs (NOT MUSIC)" if args.demo else "Common inputs"
            print(f"{label} prepared: {path}")
    except (ValueError, OSError, ImportError, TypeError) as error:
        parser.exit(2, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
