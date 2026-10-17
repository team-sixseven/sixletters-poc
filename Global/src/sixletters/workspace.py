import re
import unicodedata

WINDOWS_RESERVED = re.compile("^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\\.|$)", re.I)


def validate_component(value: str, label: str) -> str:
    """Validate one directory component on Windows and macOS, preserving Korean names."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty name")
    name = unicodedata.normalize("NFC", value)
    if (
        name != name.strip()
        or name.startswith(".")
        or name.endswith(".")
        or (len(name.encode("utf-8", errors="replace")) > 100)
        or any(char in '<>:"/\\|?*' or unicodedata.category(char).startswith("C") for char in name)
        or WINDOWS_RESERVED.match(name)
        or (name.casefold() in {"_template", "__pycache__"})
    ):
        raise ValueError(f"{label} is not a portable folder name: {value!r}")
    return name
