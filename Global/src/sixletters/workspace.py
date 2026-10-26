"""Create independent experiment folders with portable, non-overwriting names."""

import re
import unicodedata
from pathlib import Path

WINDOWS_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)", re.I)


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
