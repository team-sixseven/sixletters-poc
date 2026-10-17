"""Workspace commands for common preparation and comparison, never model training."""

import argparse
import shlex
import sys
from pathlib import Path

from .local_settings import project_root, read_settings, resolve_workspace_path, write_settings
from .workspace import init_experiment, validate_component, validate_experiment_identity


def prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{label}{suffix}: ").strip()
    except EOFError as error:
        raise ValueError("No input received; supply explicit command options instead") from error
    if not value and default is not None:
        return default
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def show_experiment(path: Path, script: str = "train.py") -> None:
    print(f"Experiment created: {path.parent}")
    print(f"Training script: {path.with_name(script)}")
    print(f"Configuration: {path}")


def personal_experiment(root: Path, owner: str | None, name: str) -> Path:
    owner = owner or read_settings(root).get("owner")
    if not owner:
        raise ValueError("Register your name with poc init or pass --owner")
    owner = validate_component(owner, "owner")
    name = validate_component(name, "experiment")
    directory = root / "Members" / owner / "experiments" / name
    if any(path.is_symlink() for path in (directory, *directory.parents) if path != root):
        raise ValueError("Personal experiment directories must not be symbolic links")
    if not directory.is_dir() or not (directory / "config.toml").is_file():
        raise ValueError("Experiment not found; create it with poc new first")
    validate_experiment_identity(directory / "config.toml")
    return directory


def default_data_location(root: Path) -> str:
    """Preserve a personal selection; suggest prepared MUSIC for a new checkout."""
    try:
        settings = read_settings(root)
    except (ValueError, OSError):
        settings = {}
    if settings.get("data"):
        return settings["data"]
    from .data.music import get_music_destination

    for cohort in ("basic", "hrv"):
        destination = get_music_destination(root, cohort)
        if (destination / "dataset.json").is_file():
            return destination.relative_to(root).as_posix()
    return "Global/data/prepared/demo-v1"


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Independent experiments with shared validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    local = subparsers.add_parser("init", help="Register local owner and common data location")
    local.add_argument("--owner")
    local.add_argument("--data", type=Path)
    new = subparsers.add_parser("new", help="Create a personal experiment using the local owner")
    new.add_argument("name")
    new.add_argument("--owner")
    new.add_argument("--script", help="Filename of the editable Python example")
    new.add_argument("--feature-set", choices=("clinical", "ecg", "combined"))
    new.add_argument("--model", choices=("logistic", "random-forest"))
    baselines = subparsers.add_parser(
        "baselines", help="Create six editable baseline experiments; does not train"
    )
    baselines.add_argument("prefix", nargs="?", default="baseline")
    baselines.add_argument("--owner")
    environment = subparsers.add_parser(
        "env", help="Manage member packages and optional isolated experiments"
    )
    environment_actions = environment.add_subparsers(dest="env_action", required=True)
    for action in ("add", "remove", "sync", "isolate", "share", "status"):
        operation = environment_actions.add_parser(action)
        operation.add_argument("name", help="Personal experiment name")
        operation.add_argument("--owner")
        if action in ("sync", "kernel"):
            operation.add_argument(
                "--relock", action="store_true", help="Refresh the dependency lock"
            )
        elif action in ("add", "remove"):
            operation.add_argument(
                "packages", nargs="+", help="Package names or version requirements"
            )
        if action == "sync":
            operation.add_argument(
                "--repair", action="store_true", help="Reinstall even when ready"
            )
    execute = subparsers.add_parser(
        "exec", help="Execute your Python file in its experiment environment"
    )
    execute.add_argument("name", help="Personal experiment name")
    execute.add_argument("--owner", help="Override owner; place before the script filename")
    execute.add_argument("script", type=Path, help="Python file relative to the experiment folder")
    execute.add_argument(
        "script_args", nargs=argparse.REMAINDER, help="Arguments for your Python file"
    )
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
    initialize = subparsers.add_parser("init-experiment", help="Create a personal experiment")
    initialize.add_argument("--owner", required=True, help="Team member's folder name")
    initialize.add_argument("--name", required=True, help="Experiment folder name, e.g. exp001")
    initialize.add_argument(
        "--feature-set", choices=("clinical", "ecg", "combined"), default="clinical"
    )
    initialize.add_argument("--root", type=Path, default=Path("Members"))
    initialize.add_argument("--script", default="train.py")
    initialize.add_argument("--model", choices=("logistic", "random-forest"), default="logistic")
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
        if args.command == "init":
            owner = args.owner or prompt("이름")
            data = args.data or Path(prompt("공통 데이터 경로", default_data_location(root)))
            path = write_settings(root, owner, data)
            print(f"Local settings saved: {path}")
            print("Data files are not created by init; prepare or obtain the selected version.")
        elif args.command == "new":
            owner = args.owner or read_settings(root).get("owner")
            if not owner:
                raise ValueError("Register your name with poc init or pass --owner")
            show_experiment(
                init_experiment(
                    owner,
                    args.name,
                    feature_set=args.feature_set or "clinical",
                    root=root / "Members",
                    script=args.script or "train.py",
                    model=args.model or "logistic",
                ),
                args.script or "train.py",
            )
        elif args.command == "baselines":
            owner = args.owner or read_settings(root).get("owner")
            if not owner:
                raise ValueError("Register your name with poc init or pass --owner")
            prefix = validate_component(args.prefix, "baseline prefix")
            specifications = [
                (f"{prefix}-{short}-{group}", model, group)
                for short, model in (("lr", "logistic"), ("rf", "random-forest"))
                for group in ("clinical", "ecg", "combined")
            ]
            for name, _, _ in specifications:
                validate_component(name, "experiment")
                if (root / "Members" / owner / "experiments" / name).exists():
                    raise ValueError("A baseline experiment already exists; use a new prefix")
            for name, model, group in specifications:
                init_experiment(owner, name, feature_set=group, model=model, root=root / "Members")
                command = ["uv", "run", "--project", "Global", "poc", "exec", name]
                if args.owner:
                    command.extend(["--owner", owner])
                command.append("train.py")
                print(shlex.join(command))
            print(
                "Created six editable experiments. Run each command to train and save validation results."
            )
        elif args.command in ("env", "exec"):
            from .personal_env import (
                env_add,
                env_isolate,
                env_remove,
                env_share,
                env_status,
                env_sync,
                environment_directory,
                execute_script,
            )

            directory = personal_experiment(root, args.owner, args.name)
            if args.command == "exec":
                return execute_script(directory, args.script, args.script_args)
            if args.env_action in ("isolate", "share"):
                operation = env_isolate if args.env_action == "isolate" else env_share
                operation(directory)
            selected = environment_directory(directory)
            scope = "실험 전용" if selected == directory else "개인 공통"
            print(f"패키지 환경: {scope} · {selected.relative_to(root).as_posix()}")
            if args.env_action in ("status", "isolate", "share"):
                from .workspace_status import show_environment_status

                show_environment_status(env_status(directory))
                return 0
            if args.env_action == "add":
                interpreter = env_add(directory, args.packages)
            elif args.env_action == "remove":
                interpreter = env_remove(directory, args.packages)
            else:
                options = {"relock": args.relock}
                if args.repair:
                    options["repair"] = True
                interpreter = env_sync(directory, **options)
            print(f"Python: {interpreter}")
        elif args.command == "demo":
            from .data.demo import make_demo_dataset
            from .data.prepared import save_prepared_dataset

            destination = args.output or root / "Global/data" / "prepared" / "demo-v1"
            path = save_prepared_dataset(make_demo_dataset(), destination, synthetic=True)
            print(f"Synthetic demo inputs (NOT MUSIC) prepared: {path}")
        elif args.command == "music":
            from .data.music import prepare_music

            path = prepare_music(root, cohort=args.cohort, offline=args.offline)
            relative = path.parent.relative_to(root).as_posix()
            print(f"MUSIC {args.cohort} common inputs verified: {path}")
            print(f"Preparation rules: {path.with_name('protocol.json')}")
            print(f"Patient counts and data checks: {path.with_name('report.json')}")
            print(f"처음 등록: poc init (준비된 {relative} 경로를 기본값으로 제시합니다.)")
            print("Personal settings and train.py are unchanged; training starts separately.")
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
        elif args.command == "init-experiment":
            path = init_experiment(
                args.owner,
                args.name,
                feature_set=args.feature_set,
                root=args.root,
                script=args.script,
                model=args.model,
            )
            show_experiment(path, args.script)
    except (ValueError, OSError, ImportError, TypeError) as error:
        parser.exit(2, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
