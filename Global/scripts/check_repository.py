"""Reject accidentally tracked local data, model files and notebook outputs."""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import unicodedata
from pathlib import Path

BLOCKED_SUFFIXES = {
    ".dat",
    ".hea",
    ".edf",
    ".parquet",
    ".pkl",
    ".pickle",
    ".joblib",
    ".pt",
    ".pth",
    ".onnx",
    ".ubj",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pem",
    ".key",
}
MIB = 1024 * 1024
REPORT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".svg", ".webp", ".docx", ".pptx"}
PATIENT_ID_HEADERS = {"patientid", "subjectid", "participantid", "환자id", "대상자id"}
PATIENT_ID_VALUE = re.compile(r"P[0-9]{4}", re.IGNORECASE)
NUMERIC_VALUE = re.compile(r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?")
TEXT_EXPORT_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson"}


def _export_text(content: bytes) -> str:
    """Decode spreadsheet/text exports, including their common Unicode BOMs."""
    if content.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        encoding = "utf-32"
    elif content.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    else:
        encoding = "utf-8-sig"
    return content.decode(encoding, errors="replace")


def has_patient_identifier_header(content: bytes) -> bool:
    """Recognize patient-level CSV exports without blocking aggregate result tables."""
    text = _export_text(content)
    for delimiter in (",", ";", "\t"):
        rows = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
        try:
            first_row = True
            for header in rows:
                if not header or not any(field.strip() for field in header):
                    continue
                # Some spreadsheet exports declare their delimiter before the header.
                if "".join(header).strip().casefold() in {"sep=,", "sep=;", "sep=", "sep=\t"}:
                    continue
                normalized = {
                    "".join(
                        char
                        for char in unicodedata.normalize("NFKC", field).casefold()
                        if char.isalnum()
                    )
                    for field in header
                }
                if normalized & PATIENT_ID_HEADERS and (first_row or len(header) > 1):
                    return True
                first_row = False
                # A single-cell title may precede the actual delimited table.
                # Stop at its first table row so column-description values are not headers.
                if len(header) > 1:
                    break
        except csv.Error:
            continue
    return False


def _is_patient_identifier(value: object) -> bool:
    return isinstance(value, str) and PATIENT_ID_VALUE.fullmatch(value.strip()) is not None


def _json_has_patient_identifier(value: object) -> bool:
    pending = [value]
    while pending:
        item = pending.pop()
        if _is_patient_identifier(item):
            return True
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return False


def has_patient_identifier_value(path: Path, content: bytes) -> bool:
    """Catch known MUSIC identifiers as data values, not mentions in source or prose."""
    text = _export_text(content)
    # Parse JSON even when exported as .txt/.csv. This also decodes escaped IDs.
    # Per-line parsing supports JSONL and a title before a JSON record export.
    for candidate in (text, *text.splitlines()):
        try:
            value = json.loads(candidate)
        except (ValueError, RecursionError):
            continue
        if _json_has_patient_identifier(value):
            return True
    for delimiter in (",", ";", "\t"):
        try:
            rows = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
            if any(_is_patient_identifier(value) for row in rows for value in row):
                return True
        except csv.Error:
            continue
    if path.suffix.lower() == ".txt":
        # Also accept ordinary key=value / key: value text exports. A sentence
        # describing an ID format is deliberately not treated as a record.
        for line in text.splitlines():
            # NumPy/text-table exports may use spaces rather than a CSV delimiter.
            # Require an exact identifier and a numeric field on the same line.
            fields = [field.strip("\"'") for field in line.split()]
            if any(_is_patient_identifier(field) for field in fields) and any(
                NUMERIC_VALUE.fullmatch(field) for field in fields
            ):
                return True
            for separator in ("=", ":"):
                key, found, value = line.partition(separator)
                if found and key.strip() and _is_patient_identifier(value.strip().strip("\"'")):
                    return True
    return False


def experiment_relative_parts(path: Path) -> tuple[str, ...]:
    """Locate generated paths without classifying arbitrary JSON configuration."""
    if len(path.parts) > 4 and path.parts[0] == "Members" and path.parts[2] == "experiments":
        return path.parts[4:]
    if len(path.parts) > 3 and path.parts[0] == "experiments":
        return path.parts[3:]
    return ()


def file_size_limit(path: Path) -> int:
    """Allow reviewed report documents/images without relaxing data/model checks."""
    if path.parts[:2] == ("Global", "reports") and path.suffix.lower() in REPORT_SUFFIXES:
        return 20 * MIB
    return MIB


def violations(path: Path, content: bytes) -> list[str]:
    problems = []
    if (
        path.parts[0] in {"data", "artifacts"}
        or path.parts[:2] in {("Global", "data"), ("Global", "artifacts")}
    ) and path.as_posix() not in {
        "Global/data/README.md",
        "Global/artifacts/README.md",
    }:
        problems.append("local data/results must not be tracked")
    if path.as_posix() in {".env.example", ".env.tracking.example"}:
        for line in content.decode("utf-8").splitlines():
            key, separator, value = line.strip().partition("=")
            if (
                not key.startswith("#")
                and separator
                and any(marker in key.upper() for marker in ("PASSWORD", "SECRET", "TOKEN", "KEY"))
            ):
                if value.strip().strip("\"'"):
                    problems.append("example credentials must be empty")
                    break
    elif path.name == ".env" or path.name.startswith(".env."):
        problems.append("environment secrets must not be tracked")
    if path.name == ".poc.local.toml":
        problems.append("local owner and connection settings must not be tracked")
    if path.parts[0] == ".poc-cache":
        problems.append("local tool caches must not be tracked")
    personal_dirs = {part.casefold() for part in path.parts[2:-1]}
    if path.parts[0] == "Members" and (
        personal_dirs & {"results", "outputs"}
        or ("models" in personal_dirs and path.suffix.lower() == ".json")
    ):
        problems.append("personal results/models must not be tracked")
    experiment_directories = {part.casefold() for part in experiment_relative_parts(path)[:-1]}
    if "outputs" in experiment_directories:
        problems.append("generated experiment outputs must not be tracked")
    elif "models" in experiment_directories and path.suffix.lower() == ".json":
        problems.append("JSON model exports must not be tracked; keep settings in configs/")
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        problems.append("raw data/model/key file must not be tracked")
    if path.suffix.lower() in {".csv", ".tsv"} and has_patient_identifier_header(content):
        problems.append(
            "CSV/TSV with patient identifier columns must not be tracked; "
            "keep patient data and predictions in Global/data/ or Global/artifacts/"
        )
    elif path.suffix.lower() in TEXT_EXPORT_SUFFIXES and has_patient_identifier_value(
        path, content
    ):
        problems.append(
            "text export with MUSIC patient identifier values must not be tracked; "
            "keep patient data and predictions in Global/data/ or Global/artifacts/"
        )
    limit = file_size_limit(path)
    if len(content) > limit:
        problems.append(
            f"file exceeds the {limit // MIB} MiB review limit; "
            "see Global/reports/README.md for report formats and larger-file sharing"
        )
    if path.suffix == ".ipynb":
        notebook = json.loads(content)
        for cell in notebook.get("cells", []):
            if cell.get("outputs") or cell.get("execution_count") is not None:
                problems.append("clear notebook outputs and execution counts before commit")
                break
    return problems


def main() -> int:
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())
    tracked = (
        subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0")
    )
    failed = False
    for name in filter(None, tracked):
        path = Path(name)
        local = root / path
        if not local.is_file():
            continue
        for message in violations(path, local.read_bytes()):
            print(f"{name}: {message}")
            failed = True
    if not failed:
        print("Tracked file checks passed.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
