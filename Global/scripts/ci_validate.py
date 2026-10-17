"""Fail closed when required CI jobs or the selected scope are incomplete."""

import json
import os
import sys

NATIVE = ["windows-2025", "macos-26"]
DOCKER = {"workspace", "recovery", "tracking"}
JOBS = {"checks", "compatibility", "docker"}


def _output(outputs: dict, name: str):
    value = outputs.get(name)
    if not isinstance(value, str):
        raise ValueError(f"checks.outputs.{name} must be a JSON-encoded string.")
    try:
        return json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"checks.outputs.{name} contains invalid JSON.") from error


def validate(needs: dict) -> None:
    """Require successful selected jobs and skipped unselected jobs."""
    if not isinstance(needs, dict) or set(needs) != JOBS:
        raise ValueError("needs must contain exactly checks, compatibility, and docker.")
    if any(not isinstance(needs[name], dict) for name in JOBS):
        raise ValueError("Each required job must have a result object.")
    if needs["checks"].get("result") != "success":
        raise ValueError("checks must succeed.")
    outputs = needs["checks"].get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("checks.outputs is missing or invalid.")

    tests = _output(outputs, "tests")
    tests_mode = _output(outputs, "tests_mode")
    smoke = _output(outputs, "smoke")
    native = _output(outputs, "native")
    docker = _output(outputs, "docker")
    if type(tests) is not bool or type(smoke) is not bool:
        raise ValueError("checks.outputs.tests and smoke must be JSON booleans.")
    if tests_mode not in ("none", "personal", "full") or tests != (tests_mode != "none"):
        raise ValueError("checks.outputs.tests_mode must match the selected Linux tests.")
    if not isinstance(native, list) or native not in ([], NATIVE):
        raise ValueError("checks.outputs.native must be empty or the full native matrix.")
    if (
        not isinstance(docker, list)
        or any(not isinstance(item, str) or item not in DOCKER for item in docker)
        or len(set(docker)) != len(docker)
    ):
        raise ValueError("checks.outputs.docker must contain unique known Docker checks.")
    if not tests and (smoke or native or docker):
        raise ValueError("Smoke, native, and Docker checks require Linux tests.")
    if tests_mode == "personal" and (smoke or native or docker):
        raise ValueError("Integration checks require full regression tests.")

    for name, selected in (("compatibility", native), ("docker", docker)):
        expected = "success" if selected else "skipped"
        if needs[name].get("result") != expected:
            raise ValueError(f"{name} must report {expected} for the selected scope.")


def main() -> int:
    payload = os.environ.get("NEEDS_JSON")
    if payload is None:
        print("CI validation failed: NEEDS_JSON is missing.", file=sys.stderr)
        return 1
    try:
        needs = json.loads(payload)
    except ValueError:
        print("CI validation failed: NEEDS_JSON contains invalid JSON.", file=sys.stderr)
        return 1
    try:
        validate(needs)
    except ValueError as error:
        print(f"CI validation failed: {error}", file=sys.stderr)
        return 1
    print("All required CI checks passed; unselected jobs were skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
