"""Identify CI build inputs and safely reuse images loaded by the build action."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

INPUT_LABEL = "io.sixseven.ci.input-sha256"


def build_fingerprint(root: Path, kind: str) -> str:
    """Hash the files used by the image, excluding local results and credentials."""
    files = {
        root / ".dockerignore",
        root / "Global/pyproject.toml",
        root / "Global/uv.lock",
        root / "Global/.python-version",
    }
    if kind == "workspace":
        files.add(root / "Global/Dockerfile")
        files.update((root / "Global/src").rglob("*.py"))
        files.update((root / "Global/scripts").glob("*.py"))
    elif kind == "tracking":
        files.update(
            {root / "Global/Dockerfile.tracking", root / "Global/scripts/start_tracking_server.py"}
        )
    else:
        raise ValueError(f"Unknown Docker image kind: {kind}")
    digest = hashlib.sha256(kind.encode())
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def prepare_prebuilt_image(
    root: Path, kind: str, tag: str, *, environment: Mapping[str, str] | None = None
) -> bool:
    """Use only the exact image built for this checkout; never fall back on a stale cache."""
    environment = os.environ if environment is None else environment
    if environment.get("POC_DOCKER_PREBUILT") != "1":
        return False
    variable = f"POC_DOCKER_{kind.upper()}_IMAGE"
    image = environment.get(variable, "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError(f"{variable} must contain the immutable ID from the Docker build step")
    completed = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )
    details = json.loads(completed.stdout)
    if len(details) != 1 or details[0].get("Id") != image:
        raise ValueError(f"The loaded {kind} image does not match the Docker build output")
    labels = details[0].get("Config", {}).get("Labels") or {}
    if labels.get(INPUT_LABEL) != build_fingerprint(root, kind):
        raise ValueError(f"The loaded {kind} image was not built from the current source files")
    subprocess.run(
        ["docker", "image", "tag", image, tag],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )
    return True


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = "".join(
        f"{kind}_fingerprint={build_fingerprint(root, kind)}\n"
        for kind in ("workspace", "tracking")
    )
    if path := os.environ.get("GITHUB_OUTPUT"):
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
