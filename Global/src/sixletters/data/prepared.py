"""Versioned common data bundles; no dataset-specific endpoint or split assumptions."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sixletters.contracts import RESERVED_COLUMNS, Dataset, _hash_json

BUNDLE_VERSION = 1
BUNDLE_FILENAME = "dataset.json"


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")


def _read_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8-sig") as handle:
        value = json.load(handle, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def _schema_fields(schema: dict) -> dict:
    required = {"dataset_id", "target_definition", "feature_groups"}
    if not isinstance(schema, dict) or not required.issubset(schema):
        raise ValueError("schema requires dataset_id, target_definition and feature_groups")
    if set(schema) - required - {"categorical_features"}:
        raise ValueError("schema contains unknown fields")
    groups = schema["feature_groups"]
    categories = schema.get("categorical_features", [])
    if not isinstance(groups, dict) or not all(isinstance(v, list) for v in groups.values()):
        raise ValueError("schema feature_groups must map group names to feature lists")
    if not isinstance(categories, list):
        raise ValueError("schema categorical_features must be a list")
    return {
        "dataset_id": schema["dataset_id"],
        "target_definition": schema["target_definition"],
        "feature_groups": {key: tuple(value) for key, value in groups.items()},
        "categorical_features": tuple(categories),
    }


def _columns(fields: dict) -> list[str]:
    groups = fields["feature_groups"]
    if set(groups) != {"clinical", "ecg"}:
        raise ValueError("feature_groups must define clinical and ecg groups")
    columns = [*RESERVED_COLUMNS, *groups["clinical"], *groups["ecg"]]
    if any(not isinstance(column, str) or not column.strip() for column in columns):
        raise ValueError("feature names must be non-empty strings")
    if len(columns) != len(set(columns)):
        raise ValueError("duplicate or reserved feature columns")
    return columns


def _csv_number(value: str) -> float:
    if value == "":
        return np.nan
    number = float(value)
    if np.isnan(number):
        raise ValueError("only an empty CSV cell denotes a missing numeric value")
    return number


def import_csv_dataset(csv_path: Path, schema_path: Path) -> Dataset:
    """Import reviewed labels/splits and declared features without fitting preprocessing.

    IDs and categorical strings are preserved verbatim; literal ``NA`` is not missing.
    Empty feature cells become missing values, while empty IDs/labels/splits are invalid.
    Numeric features accept real numbers or an empty cell, not string missing markers.
    Extra CSV columns are not imported. The schema explicitly allows model inputs.
    """
    fields = _schema_fields(_read_json(schema_path))
    columns = _columns(fields)
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or len(header) != len(set(header)):
            raise ValueError("CSV must have a header without duplicate column names")
        if not set(columns).issubset(header):
            raise ValueError("CSV is missing required or declared feature columns")
        positions = [header.index(column) for column in columns]
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"CSV row {row_number} has an incorrect number of fields")
            rows.append([row[position] for position in positions])
    frame = pd.DataFrame(rows, columns=columns)
    categories = fields["categorical_features"]
    for column in columns:
        if column in ("patient_id", "split"):
            continue
        if column in categories:
            frame[column] = frame[column].replace("", np.nan)
        else:
            try:
                frame[column] = [_csv_number(value) for value in frame[column]]
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"CSV column {column} must contain real numbers or empty cells"
                ) from error
    dataset = Dataset(frame=frame, **fields)
    dataset.validate()
    return dataset


def save_prepared_dataset(dataset: Dataset, output_dir: Path, *, synthetic: bool) -> Path:
    """Save declared inputs and integrity metadata in a new directory, never overwrite.

    JSON uses Python's float round-trip representation. Only reserved and declared
    feature columns are stored; unrelated metadata columns are excluded.
    """
    dataset.validate()
    if not isinstance(synthetic, bool):
        raise ValueError("synthetic must be a boolean")
    schema = {
        "dataset_id": dataset.dataset_id,
        "target_definition": dataset.target_definition,
        "feature_groups": {key: list(value) for key, value in dataset.feature_groups.items()},
        "categorical_features": list(dataset.categorical_features),
    }
    columns = _columns(_schema_fields(schema))
    rows = []
    for row in dataset.frame[columns].itertuples(index=False, name=None):
        normalized = []
        for column, value in zip(columns, row, strict=True):
            if pd.isna(value):
                normalized.append(None)
            elif column in ("patient_id", "split") or column in dataset.categorical_features:
                normalized.append(value)
            elif column == "target":
                normalized.append(int(value))
            else:
                normalized.append(float(value))
        rows.append(normalized)
    payload = {
        "format_version": BUNDLE_VERSION,
        "synthetic": synthetic,
        "schema": schema,
        "columns": columns,
        "rows": rows,
        "fingerprint": dataset.fingerprint(),
    }
    payload["bundle_hash"] = _hash_json(payload)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / BUNDLE_FILENAME
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
    return path


def load_prepared_dataset(path: Path) -> tuple[Dataset, bool]:
    """Load a bundle file or directory and verify its metadata and data fingerprint.

    Hashes detect accidental modification, not edits by a party that recomputes hashes.
    """
    path = Path(path)
    payload = _read_json(path / BUNDLE_FILENAME if path.is_dir() else path)
    expected = {
        "format_version",
        "synthetic",
        "schema",
        "columns",
        "rows",
        "fingerprint",
        "bundle_hash",
    }
    if set(payload) != expected:
        raise ValueError("prepared dataset contains missing or unknown fields")
    if type(payload["format_version"]) is not int or payload["format_version"] != BUNDLE_VERSION:
        raise ValueError("unsupported prepared dataset format_version")
    if not isinstance(payload["synthetic"], bool):
        raise ValueError("prepared dataset synthetic flag must be a boolean")
    bundle_hash = payload.pop("bundle_hash")
    if bundle_hash != _hash_json(payload):
        raise ValueError("prepared dataset integrity check failed: bundle hash mismatch")
    fields = _schema_fields(payload["schema"])
    columns = _columns(fields)
    if payload["columns"] != columns:
        raise ValueError("prepared dataset columns do not match its schema")
    rows = payload["rows"]
    if not isinstance(rows, list) or any(
        not isinstance(row, list) or len(row) != len(columns) for row in rows
    ):
        raise ValueError("prepared dataset rows must match its columns")
    frame = pd.DataFrame(rows, columns=columns)
    for column in fields["categorical_features"]:
        if column in frame:
            frame[column] = frame[column].map(lambda value: np.nan if value is None else value)
    # JSON null-only numeric columns need an explicit numeric dtype, not object.
    for column in columns:
        if column not in RESERVED_COLUMNS and column not in fields["categorical_features"]:
            if frame[column].isna().all():
                frame[column] = frame[column].astype(float)
    dataset = Dataset(frame=frame, **fields)
    dataset.validate()
    if dataset.fingerprint() != payload["fingerprint"]:
        raise ValueError("prepared dataset integrity check failed: fingerprint mismatch")
    return dataset, payload["synthetic"]
