"""Workspace commands for common preparation and comparison, never model training."""

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .local_settings import project_root, read_settings, resolve_workspace_path, write_settings
from .run_catalog import (
    display_text,
    filter_run_catalog,
    local_entry,
    show_remote_issues,
    show_run_catalog,
    show_sync_summary,
)
from .workspace import (
    init_experiment,
    init_practice,
    validate_component,
    validate_experiment_identity,
)


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


class SelectionCancelled(Exception):
    """The person explicitly cancelled a catalog action."""


def _choose_indexes(
    count: int, *, archive: bool = False, minimum: int = 2, action: str | None = None
) -> list[int]:
    """Apply the same explicit-selection rules to local and shared result lists."""
    minimum = 1 if archive else minimum
    if count < minimum:
        if archive:
            raise ValueError("No completed local runs found; use --run for another location")
        raise ValueError(f"At least {minimum} completed runs are needed")
    label = action or (
        "모델과 재현 파일을 추가 업로드할 번호 하나를 입력"
        if archive
        else "최종 후보로 고정할 번호를 공백으로 구분해 입력"
        if minimum == 1
        else "비교할 번호를 공백으로 구분해 입력"
    )
    while True:
        # EOF/noninteractive mistakes fail once; only a live terminal offers retry.
        try:
            selected = prompt(label + " (취소: q)")
            if selected.lower() in {"q", "quit", "취소"}:
                raise SelectionCancelled
            try:
                indexes = [int(value) for value in selected.replace(",", " ").split()]
            except ValueError as error:
                raise ValueError("Choose run numbers, for example: 1 3") from error
            if (
                len(indexes) < minimum
                or (archive and len(indexes) != 1)
                or len(set(indexes)) != len(indexes)
                or any(index < 1 or index > count for index in indexes)
            ):
                if archive:
                    raise ValueError("Choose exactly one number from the displayed list")
                if minimum == 1:
                    raise ValueError(
                        "Choose at least one valid, non-repeated number from the displayed list"
                    )
                raise ValueError("Choose at least two different numbers from the displayed list")
            return [index - 1 for index in indexes]
        except ValueError as error:
            if not sys.stdin.isatty() or isinstance(error.__cause__, EOFError):
                raise
            print(f"{error}. 목록은 그대로입니다. 다시 입력하거나 q로 취소하세요.")


def _catalog_selection(
    entries: list[dict],
    *,
    archive: bool = False,
    minimum: int = 2,
    list_only: bool = False,
    action: str | None = None,
    **filters,
) -> list[dict]:
    shown, count = filter_run_catalog(entries, **filters)
    print(f"조건에 맞는 실행 {count}개 중 최근 {len(shown)}개 표시 (점수순 아님).")
    if list_only:
        show_run_catalog(shown)
        return []
    if not shown:
        raise ValueError(
            "조건에 맞는 완료된 실행이 없습니다. --owner, --dataset, --feature-set 값과 "
            "조회 위치를 확인하세요. 데이터는 경로가 아닌 dataset_id로 입력합니다."
        )
    if len(shown) < count:
        print("이전 실행을 보려면 --limit를 늘리거나 이름·데이터·입력 그룹으로 좁히세요.")
    show_run_catalog(shown)
    if not archive and len(shown) < minimum:
        raise ValueError(
            "비교에는 서로 다른 실행 2개 이상이 필요합니다. "
            "필터를 줄이거나 --limit를 2 이상으로 늘리세요."
        )
    return [
        shown[index]
        for index in _choose_indexes(len(shown), archive=archive, minimum=minimum, action=action)
    ]


def choose_runs(
    root: Path, run_dirs: list[Path] | None = None, *, archive: bool = False, **filters
) -> list[Path]:
    """Select completed local runs without traversing their saved model directories."""
    entries = []
    paths = (
        sorted(root.glob("*/*/*/result.json"))
        if run_dirs is None
        else [directory / "result.json" for directory in run_dirs]
    )
    for path in paths:
        if run_dirs is None and any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if not (path.parent / "predictions.csv").is_file():
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            entry = local_entry(result, path)
            from .tracking import local_archive_status

            entry.update(
                archive_status=local_archive_status(path.parent),
                archive_status_source="local_record",
            )
        except (ValueError, KeyError, TypeError, OSError):
            print(f"Skipping incomplete result: {display_text(path, 240)}", file=sys.stderr)
            continue
        entries.append(entry)
    selected = _catalog_selection(entries, archive=archive, **filters)
    return [Path(entry["path"]) for entry in selected]


def choose_remote_runs(runs: list[dict], **filters) -> list[str]:
    """Choose shared run IDs from metadata before downloading prediction files."""
    return [entry["run_id"] for entry in _catalog_selection(runs, **filters)]


def _positive_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--limit에는 1 이상의 정수를 입력하세요.") from error
    if count < 1:
        raise argparse.ArgumentTypeError("--limit에는 1 이상의 정수를 입력하세요.")
    return count


def _catalog_options(parser: argparse.ArgumentParser, *, destination_owner: bool = False) -> None:
    parser.add_argument(
        "--source-owner" if destination_owner else "--owner",
        help="Exact owner name in the run metadata",
    )
    parser.add_argument("--experiment", help="Exact experiment folder name in the run metadata")
    parser.add_argument("--dataset", help="Exact dataset_id, not a data directory")
    parser.add_argument("--feature-set", choices=("clinical", "ecg", "combined"))
    parser.add_argument(
        "--limit", type=_positive_count, help="Show this many recent matching runs (default: 20)"
    )
    parser.add_argument(
        "--list",
        dest="list_only",
        action="store_true",
        help="Show matching runs without selecting or changing anything",
    )


def show_experiment(path: Path, script: str = "train.py") -> None:
    print(f"Experiment created: {path.parent}")
    print(f"Training script: {path.with_name(script)}")
    print(f"Configuration: {path}")


def show_comparison_deltas(summary) -> None:
    reference = summary.loc[summary["is_reference"]].iloc[0]
    identity = " / ".join(display_text(reference[key]) for key in ("owner", "experiment", "run_id"))
    print(f"기준 실행: {identity}")
    print("차이 = 대상 − 기준. AUROC·AP는 양수, Brier는 음수일수록 개선입니다.")
    for _, row in summary.loc[~summary["is_reference"]].iterrows():
        label = " / ".join(display_text(row[key]) for key in ("owner", "experiment", "run_id"))
        values = []
        for metric, name in (
            ("roc_auc", "AUROC"),
            ("average_precision", "AP"),
            ("brier_score", "Brier"),
        ):
            lower, upper = row[f"{metric}_delta_ci_lower"], row[f"{metric}_delta_ci_upper"]
            interval = (
                f"[{lower:+.4f}, {upper:+.4f}]"
                if lower is not None
                and upper is not None
                and math.isfinite(lower)
                and math.isfinite(upper)
                else "계산 불가 (양성·음성 각 2명 이상 필요)"
            )
            values.append(f"{name} {row[f'{metric}_delta']:+.4f} · 95% 구간 {interval}")
        print(f"{label}\n  " + " | ".join(values))
    print(
        "검증 환자를 함께 재추출한 차이 구간입니다. 학습·반복적인 모델 선택의 불확실성은 포함하지 않습니다."
    )


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
        # init can repair a malformed local setting through a fresh user selection.
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
    subparsers.add_parser("status", help="Show local readiness and next steps without uploading")

    local = subparsers.add_parser("init", help="Register local owner and common data location")
    local.add_argument("--owner")
    local.add_argument("--data", type=Path)

    new = subparsers.add_parser("new", help="Create a personal experiment using the local owner")
    new.add_argument("name")
    new.add_argument("--from", dest="source", help="Copy existing experiment code and settings")
    new.add_argument("--owner")
    new.add_argument("--script", help="Filename of the editable Python example")
    new.add_argument("--feature-set", choices=("clinical", "ecg", "combined"))
    new.add_argument("--model", choices=("logistic", "random-forest"))

    baselines = subparsers.add_parser(
        "baselines", help="Create six editable baseline experiments; does not train"
    )
    baselines.add_argument("prefix", nargs="?", default="baseline")
    baselines.add_argument("--owner")

    restore = subparsers.add_parser(
        "restore", help="Restore selected reproduction files into a new personal experiment"
    )
    restore.add_argument("name", help="New experiment name")
    restore.add_argument("--owner")
    restore.add_argument("--data", type=Path)
    _catalog_options(restore, destination_owner=True)
    restore_source = restore.add_mutually_exclusive_group()
    restore_source.add_argument("--run", type=Path, help="Complete local run directory")
    restore_source.add_argument(
        "--run-id", help="Server run ID with completed reproduction archive"
    )
    restore_source.add_argument(
        "--remote", action="store_true", help="Select a complete archive from the server"
    )

    final = subparsers.add_parser(
        "final", help="Freeze candidates before test inference, then create the final report"
    )
    final_actions = final.add_subparsers(dest="final_action", required=True)
    freeze = final_actions.add_parser(
        "freeze", help="Snapshot selected models and validation-derived thresholds"
    )
    freeze.add_argument("name", help="New final evaluation plan name")
    freeze.add_argument("--runs", type=Path, nargs="+")
    freeze.add_argument(
        "--remote",
        action="store_true",
        help="Select archived candidates from the shared MLflow server",
    )
    freeze.add_argument(
        "--run-ids", nargs="+", help="Exact server run IDs with complete model archives"
    )
    _catalog_options(freeze)
    freeze.add_argument("--data", type=Path)
    freeze.add_argument(
        "--threshold",
        type=float,
        help="Predeclared cutoff; otherwise choose Youden J on validation",
    )
    freeze.add_argument(
        "--reference", help="Selected local run ID used for paired performance differences"
    )
    report = final_actions.add_parser(
        "report", help="Create test metrics, paired intervals and figures after all predictions"
    )
    report.add_argument("name")
    report.add_argument("--data", type=Path)
    report.add_argument("--repeats", type=int, default=2000)
    report.add_argument("--seed", type=int, default=42)
    final_status = final_actions.add_parser(
        "status", help="Show frozen candidates and remaining prediction steps"
    )
    final_status.add_argument("name")
    final_status.add_argument("--data", type=Path)

    notebook = subparsers.add_parser(
        "notebook", help="Prepare a personal notebook and package kernel"
    )
    notebook.add_argument("name", help="Existing experiment name")
    notebook.add_argument("--owner")
    notebook.add_argument(
        "--file", default="analysis.ipynb", help="Notebook filename inside the experiment"
    )
    notebook.add_argument("--relock", action="store_true")

    environment = subparsers.add_parser(
        "env", help="Manage member packages and optional isolated experiments"
    )
    environment_actions = environment.add_subparsers(dest="env_action", required=True)
    for action in ("add", "remove", "sync", "kernel", "isolate", "share", "status"):
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

    practice = subparsers.add_parser("practice", help="Create your MUSIC practice notebook")
    practice.add_argument("--owner", help="Override the name registered by poc init")

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

    sync = subparsers.add_parser("sync", help="Retry completed local results to the shared server")
    sync.add_argument("--root", type=Path, help="Local artifacts root")

    archive = subparsers.add_parser(
        "archive", help="Upload models and reproduction files for one selected local run"
    )
    archive.add_argument("--run", type=Path, help="Completed local run directory")
    archive.add_argument("--root", type=Path, help="Search local runs here when --run is omitted")
    _catalog_options(archive)

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

    compare = subparsers.add_parser("compare", help="Compare validation results")
    compare.add_argument(
        "--mode",
        choices=("experiments", "inputs"),
        default="experiments",
        help="experiments: methods on one input group; inputs: input groups with one fixed method",
    )
    compare.add_argument("--runs", type=Path, nargs="+")
    compare.add_argument(
        "--run-ids", nargs="+", help="Compare these exact server run IDs with --remote"
    )
    compare.add_argument("--output", type=Path)
    compare.add_argument(
        "--reference",
        help="기준으로 삼을 저장된 실행 ID(목록의 이름/실험명 뒤 ID). 생략하면 첫 선택 실행",
    )
    compare.add_argument(
        "--root", type=Path, help="Local result root; with --remote, cache downloaded results here"
    )
    compare.add_argument(
        "--remote", action="store_true", help="Select completed shared server runs"
    )
    _catalog_options(compare)
    args = parser.parse_args()
    if args.command == "new" and args.source and (args.script or args.feature_set or args.model):
        parser.error(
            "--from copies the source settings; edit the copy instead of using template options"
        )
    filters = {}
    if args.command in ("compare", "archive", "restore") or (
        args.command == "final" and args.final_action == "freeze"
    ):
        filters = {
            key: getattr(args, key)
            for key in ("owner", "dataset", "feature_set", "experiment", "limit")
            if getattr(args, key, None) is not None
        }
        if args.command == "restore":
            filters.pop("owner", None)
            if args.source_owner:
                filters["owner"] = args.source_owner
        if args.list_only:
            filters["list_only"] = True
        if args.command == "compare":
            if args.remote and args.runs:
                parser.error("compare --remote cannot be combined with local --runs")
            if args.runs and args.root:
                parser.error("compare accepts either --runs or --root, not both")
            if args.run_ids and not args.remote:
                parser.error("compare --run-ids requires --remote; use --runs for local paths")
            if args.run_ids and (
                len(args.run_ids) < 2
                or len(set(args.run_ids)) != len(args.run_ids)
                or any(not re.fullmatch(r"[0-9a-f]{32}", value) for value in args.run_ids)
            ):
                parser.error("--run-ids requires at least two distinct 32-character server run IDs")
            if filters and (args.runs or args.run_ids):
                parser.error("Use filters for the run list, or explicit --runs/--run-ids, not both")
        elif args.command == "archive" and args.run and (args.root or filters):
            parser.error("archive --run cannot be combined with --root or list filters")
        elif args.command == "restore" and (args.run or args.run_id) and filters:
            parser.error("Use a restore list or an explicit run, not both")
    if args.command == "prepare":
        if args.csv and args.schema is None:
            parser.error("prepare --csv requires --schema")
        if args.demo and args.schema is not None:
            parser.error("prepare --schema is only valid with --csv")
    if args.command == "final" and args.final_action == "freeze":
        if filters and (args.runs or args.run_ids):
            parser.error("Use a candidate list or explicit runs, not both")
        if args.runs and (args.remote or args.run_ids):
            parser.error("Choose local --runs or remote candidates, not both")
        if args.run_ids and (
            len(set(args.run_ids)) != len(args.run_ids)
            or any(not re.fullmatch(r"[0-9a-f]{32}", value) for value in args.run_ids)
        ):
            parser.error("--run-ids requires distinct 32-character server run IDs")
    try:
        root = project_root()
        migration = root / "Global/scripts/migrate_layout.py"
        read_only = (
            args.command == "status"
            or getattr(args, "list_only", False)
            or (args.command == "env" and args.env_action == "status")
            or (args.command == "final" and args.final_action == "status")
        )
        if migration.is_file() and not read_only:
            completed = subprocess.run([sys.executable, str(migration), "--quiet"], check=False)
            if completed.returncode:
                return completed.returncode
        for option in ("root", "data", "output", "run", "csv", "schema"):
            value = getattr(args, option, None)
            if value is not None:
                setattr(args, option, resolve_workspace_path(root, value))
        if getattr(args, "runs", None):
            args.runs = [resolve_workspace_path(root, path) for path in args.runs]
        if args.command == "status":
            from .workspace_status import show_workspace_status

            show_workspace_status(root)
        elif args.command == "init":
            owner = args.owner or prompt("이름")
            data = args.data or Path(prompt("공통 데이터 경로", default_data_location(root)))
            path = write_settings(root, owner, data)
            print(f"Local settings saved: {path}")
            print("Data files are not created by init; prepare or obtain the selected version.")
        elif args.command == "new":
            owner = args.owner or read_settings(root).get("owner")
            if not owner:
                raise ValueError("Register your name with poc init or pass --owner")
            if args.source:
                from .workspace import clone_experiment

                source = personal_experiment(root, owner, args.source)
                path = clone_experiment(source, owner, args.name, root=root / "Members")
                print(f"실험 복제 완료: {path.parent}")
                print("복사한 학습 파일·설정을 확인하고 poc exec로 실행하세요.")
                return 0
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
                command = ["./dev", "exec", name]
                if args.owner:
                    command.extend(["--owner", owner])
                command.append("train.py")
                print(shlex.join(command))
            print(
                "Created six editable experiments. Run each command to train and save validation results."
            )
        elif args.command == "restore":
            from .local_settings import configured_data
            from .reproduction import fetch_archive, restore_run

            owner = args.owner or read_settings(root).get("owner")
            if not owner:
                raise ValueError("Register your name with poc init or pass --owner")
            if args.run or args.run_id:
                source = args.run or fetch_archive(root / "Global/artifacts/.restored", args.run_id)
            elif args.remote:
                from .tracking import list_remote_runs

                issues = []
                catalog = list_remote_runs(issues=issues)
                show_remote_issues(issues)
                selected = choose_remote_runs(
                    catalog,
                    archive=True,
                    archived_only=True,
                    action="복원할 번호 하나를 입력",
                    **filters,
                )
                if args.list_only:
                    return 0
                source = fetch_archive(root / "Global/artifacts/.restored", selected[0])
            else:
                selected = choose_runs(
                    root / "Global/artifacts",
                    archive=True,
                    action="복원할 번호 하나를 입력",
                    **filters,
                )
                if args.list_only:
                    return 0
                source = selected[0]
            restored = restore_run(
                source,
                owner=owner,
                name=args.name,
                root=root / "Members",
                data=args.data or configured_data(root),
            )
            print(f"Verified original run: {source}")
            print(f"Restored experiment: {restored.parent}")
            print(
                "Model files: outputs/.restore/models/; original run details: outputs/.restore/trace.json"
            )
            print(
                "Review the saved execution arguments and source before running; restore does not execute code."
            )
        elif args.command == "final":
            from .final_evaluation import freeze_candidates, generate_final_report
            from .local_settings import configured_data
            from .reproduction import final_plan_status
            from .run_catalog import show_final_status

            name = validate_component(args.name, "final plan")
            directory = root / "Global/artifacts/final" / name
            data = args.data or configured_data(root)
            if args.final_action == "freeze":
                if args.remote or args.run_ids:
                    from .reproduction import fetch_archive
                    from .tracking import list_remote_runs

                    if args.run_ids:
                        selected = args.run_ids
                    else:
                        issues = []
                        catalog = list_remote_runs(issues=issues)
                        show_remote_issues(issues)
                        selected = choose_remote_runs(
                            catalog, minimum=1, archived_only=True, **filters
                        )
                    if args.list_only:
                        return 0
                    runs = [
                        fetch_archive(root / "Global/artifacts/.restored", run_id)
                        for run_id in selected
                    ]
                else:
                    runs = args.runs or choose_runs(root / "Global/artifacts", minimum=1, **filters)
                    if args.list_only:
                        return 0
                freeze_candidates(
                    runs, data, directory, threshold=args.threshold, reference=args.reference
                )
                print(f"Frozen final plan: {directory}")
                show_final_status(final_plan_status(directory, data, root))
                print("Do not fit on test data or change candidates after inspecting test results.")
            elif args.final_action == "status":
                show_final_status(final_plan_status(directory, data, root))
            else:
                output = generate_final_report(
                    directory, data, repeats=args.repeats, seed=args.seed
                )
                print(f"Final aggregate report: {output}")
        elif args.command == "notebook":
            from .notebook_workspace import prepare_notebook

            directory = personal_experiment(root, args.owner, args.name)
            prepared = prepare_notebook(directory, filename=args.file, relock=args.relock)
            print(f"노트북: {prepared['path'].relative_to(root).as_posix()}")
            print(f"커널 연결 완료: {prepared['kernel']}")
            print("JupyterLab에서 이 파일을 여세요. 기존에 열려 있으면 저장 후 다시 여세요.")
        elif args.command in ("env", "exec"):
            from .personal_env import (
                env_add,
                env_isolate,
                env_kernel,
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
            if args.env_action == "kernel":
                display, _ = env_kernel(directory, relock=args.relock)
                print(f"JupyterLab → Kernel → Change Kernel → {display}")
                print("패키지 변경 전에는 이 환경을 사용하는 모든 학습과 노트북 커널을 종료하세요.")
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
        elif args.command == "practice":
            owner = args.owner or read_settings(root).get("owner")
            if not owner:
                raise ValueError("Register your name with poc init or pass --owner")
            path = init_practice(owner, root=root / "Members")
            print(f"개인 실습 노트북: {path.relative_to(root).as_posix()}")
            print("MUSIC 데이터가 없으면 poc music을 실행하세요.")
            print(
                "JupyterLab에서 노트북을 열고 Kernel → Restart Kernel and Run All Cells를 선택하세요."
            )
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
        elif args.command == "sync":
            from .tracking import sync_runs

            summary = sync_runs(args.root or root / "Global/artifacts")
            show_sync_summary(summary)
            if summary.get("disabled"):
                print("Set the shared MLflow connection in .env and recreate the workspace.")
            elif summary.get("pending") or summary.get("invalid"):
                return 1
        elif args.command == "archive":
            from .tracking import archive_run

            selected = (
                [args.run]
                if args.run
                else choose_runs(args.root or root / "Global/artifacts", archive=True, **filters)
            )
            if args.list_only:
                return 0
            run_dir = selected[0]
            outcome = archive_run(run_dir)
            if outcome == "disabled":
                print("Set the shared MLflow connection in .env and recreate the workspace.")
                return 1
            if outcome == "pending":
                print(
                    "Model archive pending. Keep the local run; the workspace retries selected "
                    "archives. Run poc sync for an immediate retry."
                )
                return 1
            if outcome == "already_synced":
                print("Model and reproduction files are already archived on the shared server.")
            else:
                print("Model and reproduction files uploaded to the existing MLflow run.")
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
        else:
            from .run_catalog import compare_with_reference

            artifact_root = args.root or root / "Global/artifacts"
            if args.reference is None and not args.list_only:
                print("처음 선택한 실행을 기준으로 점수 차이를 계산합니다.")
            if args.remote:
                from .tracking import fetch_runs, list_remote_runs

                issues = []
                if args.run_ids:
                    selected = args.run_ids
                else:
                    catalog = list_remote_runs(issues=issues)
                    show_remote_issues(issues)
                    selected = choose_remote_runs(catalog, **filters)
                if args.list_only:
                    return 0
                shown_issues = list(issues)
                try:
                    runs = fetch_runs(artifact_root, run_ids=selected, issues=issues)
                finally:
                    show_remote_issues([issue for issue in issues if issue not in shown_issues])
            else:
                runs = args.runs or choose_runs(artifact_root, **filters)
                if args.list_only:
                    return 0
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            output = args.output or (
                root
                / "Global/artifacts"
                / "comparisons"
                / f"{args.mode}-{timestamp}-{uuid4().hex[:8]}.csv"
            )
            summary = compare_with_reference(runs, output, mode=args.mode, reference=args.reference)
            show_comparison_deltas(summary)
            print(f"Compared {len(summary)} validation runs: {output}")
    except SelectionCancelled:
        print("취소했습니다. 파일과 서버 결과는 변경하지 않았습니다.")
        return 0
    except (ValueError, OSError, ImportError, TypeError) as error:
        parser.exit(2, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
