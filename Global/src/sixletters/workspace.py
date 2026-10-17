"""Create independent experiment folders with portable, non-overwriting names."""

import json
import re
import shutil
import tomllib
import unicodedata
from pathlib import Path

WINDOWS_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)", re.I)
FEATURE_SETS = frozenset({"clinical", "ecg", "combined"})
STARTER_MODELS = frozenset({"logistic", "random-forest"})


def validate_component(value: str, label: str) -> str:
    """Validate one directory component on Windows and macOS, preserving Korean names."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty name")
    name = unicodedata.normalize("NFC", value)
    if (
        name != name.strip()
        or name.startswith(".")
        or name.endswith(".")
        or len(name.encode("utf-8", errors="replace")) > 100
        or any(char in '<>:"/\\|?*' or unicodedata.category(char).startswith("C") for char in name)
        or WINDOWS_RESERVED.match(name)
        or name.casefold() in {"_template", "__pycache__"}
    ):
        raise ValueError(f"{label} is not a portable folder name: {value!r}")
    return name


def _reject_ambiguous_name(parent: Path, name: str) -> None:
    """Avoid case or Unicode variants that collide on other team members' systems."""
    if parent.is_dir():
        for sibling in parent.iterdir():
            if (
                unicodedata.normalize("NFC", sibling.name).casefold() == name.casefold()
                and unicodedata.normalize("NFC", sibling.name) != name
            ):
                raise ValueError(f"Folder name collides with an existing name: {sibling.name}")


def validate_experiment_identity(config: Path, *, settings: dict | None = None) -> None:
    """Keep team results attached to their source folder; allow standalone API layouts."""
    config = Path(config).absolute()
    directory = config.parent
    if directory.parent.name != "experiments" or directory.parent.parent.parent.name != "Members":
        return
    if settings is None:
        settings = tomllib.loads(config.read_text(encoding="utf-8")).get("experiment")
    if not isinstance(settings, dict):
        raise ValueError(f"{config}: add [experiment] with owner and name matching its folder")
    expected = {
        "owner": unicodedata.normalize("NFC", directory.parent.parent.name),
        "name": unicodedata.normalize("NFC", directory.name),
    }
    corrections = []
    for field, value in expected.items():
        actual = settings.get(field)
        if not isinstance(actual, str) or unicodedata.normalize("NFC", actual) != value:
            corrections.append(f"{field} = {json.dumps(value, ensure_ascii=False)}")
    if corrections:
        raise ValueError(
            f"Experiment identity does not match its Members folder. "
            f"In {config}, set [experiment] " + "; ".join(corrections)
        )


def init_experiment(
    owner: str,
    name: str,
    *,
    feature_set: str = "clinical",
    root: Path = Path("Members"),
    script: str = "train.py",
    model: str = "logistic",
) -> Path:
    """Create a standalone train.py and config without overwriting existing work."""
    owner = validate_component(owner, "owner")
    name = validate_component(name, "name")
    script = validate_component(script, "script filename")
    if not script.endswith(".py"):
        raise ValueError("The example script filename must end with .py")
    if feature_set not in FEATURE_SETS:
        raise ValueError("feature_set must be clinical, ecg, or combined")
    if model not in STARTER_MODELS:
        raise ValueError("model must be logistic or random-forest")
    template = Path(__file__).resolve().parents[2] / "templates" / "experiment"
    config_template = "random_forest.toml" if model == "random-forest" else "config.toml"
    script_template = "random_forest.py" if model == "random-forest" else "train.py"
    if not all((template / filename).is_file() for filename in (config_template, script_template)):
        raise ValueError(
            "Experiment template is missing; run from an installed repository checkout"
        )
    root = Path(root)
    legacy_root = root.resolve()
    if (
        legacy_root.name == "experiments"
        and (legacy_root.parent / "Global/pyproject.toml").is_file()
    ):
        root = legacy_root.parent / "Members"
    root.mkdir(parents=True, exist_ok=True)
    _reject_ambiguous_name(root, owner)
    owner_dir = root / owner
    if owner_dir.is_symlink():
        raise ValueError("Owner folder must not be a symbolic link")
    owner_dir.mkdir(exist_ok=True)
    experiment_dir = owner_dir / "experiments"
    notebook_dir = owner_dir / "notebooks"
    if experiment_dir.is_symlink() or notebook_dir.is_symlink():
        raise ValueError("Personal work folders must not be symbolic links")
    experiment_dir.mkdir(exist_ok=True)
    notebook_dir.mkdir(exist_ok=True)
    _reject_ambiguous_name(experiment_dir, name)
    destination = experiment_dir / name
    destination.mkdir()  # Existing experiments, including empty folders, are never overwritten.
    try:
        config = (template / config_template).read_text(encoding="utf-8")
        for field, value in (("owner", owner), ("name", name), ("feature_set", feature_set)):
            replacement = f"{field} = {json.dumps(value, ensure_ascii=False)}"
            config = re.sub(
                rf"^{field} = .*$",
                lambda _, replacement=replacement: replacement,
                config,
                count=1,
                flags=re.MULTILINE,
            )
        (destination / "config.toml").write_text(config, encoding="utf-8")
        shutil.copyfile(template / script_template, destination / script)
        if (template / "predict.py").is_file():
            predictor = "final_eval.py" if script.casefold() == "predict.py" else "predict.py"
            shutil.copyfile(template / "predict.py", destination / predictor)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination / "config.toml"


def _clone_files(source: Path) -> dict[str, bytes]:
    """Reuse recorded source/settings rules; copy notes without generated notebook output."""
    from .personal_env import LOCK_NAMES, SCOPE_FILE
    from .provenance import (
        _SENSITIVE_RESOURCE_NAME,
        MAX_RESOURCE_BYTES,
        MAX_TOTAL_RESOURCE_BYTES,
        _check_unique_paths,
        _checked_snapshot_path,
        _snapshot_files,
        experiment_resources,
        python_sources,
    )

    excluded = {"data", "artifacts", "outputs", "output", "results", "reports", "checkpoints"}
    files = {
        name: raw
        for name, raw in python_sources(source).items()
        if not any(part.casefold() in excluded for part in Path(name).parts[:-1])
    }
    files.update(experiment_resources(source, exclude={*LOCK_NAMES, SCOPE_FILE}))
    for directory, path in _snapshot_files(source, resources=True):
        if (
            path.suffix.lower() not in {".md", ".ipynb"}
            or _SENSITIVE_RESOURCE_NAME.search(path.name) is not None
        ):
            continue
        name = _checked_snapshot_path(directory, path)
        if path.stat().st_size > MAX_TOTAL_RESOURCE_BYTES:
            raise ValueError(f"Clone note exceeds 8 MiB; clear its outputs first: {name}")
        raw = path.read_bytes()
        if path.suffix.lower() == ".ipynb":
            notebook = json.loads(raw)
            if (
                not isinstance(notebook, dict)
                or not isinstance(notebook.get("cells"), list)
                or not isinstance(notebook.get("metadata", {}), dict)
            ):
                raise ValueError(f"Invalid notebook: {name}")
            metadata = notebook.get("metadata", {})
            metadata.pop("widgets", None)
            metadata.pop("kernelspec", None)
            for cell in notebook["cells"]:
                if not isinstance(cell, dict) or not isinstance(cell.get("metadata", {}), dict):
                    raise ValueError(f"Invalid notebook cell: {name}")
                if cell.get("cell_type") == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
                    cell.get("metadata", {}).pop("execution", None)
            raw = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode()
        else:
            raw.decode("utf-8")
        if len(raw) > MAX_RESOURCE_BYTES:
            raise ValueError(f"Clone note exceeds 1 MiB: {name}")
        files[name] = raw
    config = source / "config.toml"
    _checked_snapshot_path(source, config)
    if config.stat().st_size > MAX_RESOURCE_BYTES:
        raise ValueError("config.toml exceeds 1 MiB")
    files["config.toml"] = config.read_bytes()
    _check_unique_paths(files)
    return files


def clone_experiment(
    source: Path,
    owner: str,
    name: str,
    *,
    root: Path = Path("Members"),
) -> Path:
    """Copy editable experiment inputs without copying models, data or virtual environments."""
    from .personal_env import (
        SCOPE_FILE,
        _directory,
        _locked,
        environment_directory,
        environment_resources,
    )
    from .provenance import _check_unique_paths
    from .reproduction import _restored_config

    owner = validate_component(owner, "owner")
    name = validate_component(name, "name")
    source = _directory(source)
    validate_experiment_identity(source / "config.toml")
    original_identity = tomllib.loads((source / "config.toml").read_text())["experiment"]
    root = Path(root).absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("Clone destination must not contain symbolic links")
    root.mkdir(parents=True, exist_ok=True)
    _reject_ambiguous_name(root, owner)
    member = root / owner
    if member.is_symlink():
        raise ValueError("Owner folder must not be a symbolic link")
    member.mkdir(exist_ok=True)
    target_parent = member / "experiments"
    if target_parent.is_symlink():
        raise ValueError("Experiment folder must not be a symbolic link")
    target_parent.mkdir(exist_ok=True)
    _reject_ambiguous_name(target_parent, name)
    destination = target_parent / name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Experiment already exists: {destination}")

    active = environment_directory(source)
    with _locked(active):
        files = _clone_files(source)
        if active == member.resolve():
            # Preserve shared selection without copying inactive local package records.
            files[SCOPE_FILE] = b'scope = "shared"\n'
        else:
            # An isolated source or another member must not change the recipient's packages.
            files.update(environment_resources(source))
            files[SCOPE_FILE] = b'scope = "isolated"\n'
        for filename, raw in list(files.items()):
            if Path(filename).suffix != ".toml" or filename == SCOPE_FILE:
                continue
            value = tomllib.loads(raw.decode("utf-8"))
            identity = value.get("experiment")
            if isinstance(identity, dict) and all(
                identity.get(key) == original_identity.get(key) for key in ("owner", "name")
            ):
                files[filename] = _restored_config(raw, owner=owner, name=name)
        _check_unique_paths(files)
        destination.mkdir()
        try:
            for filename, raw in files.items():
                target = destination / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
            validate_experiment_identity(destination / "config.toml")
        except BaseException:
            shutil.rmtree(destination)
            raise
    return destination / "config.toml"
