"""Write durable Markdown and JSON artifacts for model evaluation runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def create_report_dir(base_dir: Path, run_name: str) -> Path:
    """Create a unique run directory and a report immediately, even before work starts."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report_dir = base_dir / f"{run_name}-{stamp}"
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / "report.md").write_text(
        f"# {run_name.replace('-', ' ').title()} Evaluation\n\n"
        f"- Status: RUNNING\n- Started (UTC): {stamp}\n",
        encoding="utf-8",
    )
    return report_dir


def write_report(report_dir: Path, payload: dict[str, Any], markdown: str) -> None:
    (report_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "report.md").write_text(markdown.rstrip() + "\n", encoding="utf-8")


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def metric(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"
