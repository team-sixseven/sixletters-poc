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
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination / "config.toml"
