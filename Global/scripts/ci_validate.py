"""Fail closed when required CI jobs or the selected scope are incomplete."""

import json
import os
import re
import sys

NATIVE = ["windows-2025", "macos-26"]
DOCKER = {"cli", "workspace", "recovery", "tracking"}
JOBS = {"scope", "checks", "linux", "compatibility", "docker"}


def _output(outputs: dict, name: str):
    value = outputs.get(name)
    if not isinstance(value, str):
        raise ValueError(f"scope.outputs.{name} must be a JSON-encoded string.")
    try:
        return json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"scope.outputs.{name} contains invalid JSON.") from error


def _test_files(value, *, focused: bool, name: str) -> None:
    if (
        not isinstance(value, list)
        or any(
            not isinstance(item, str)
            or re.fullmatch(r"Global/tests/test_[A-Za-z0-9_]+\.py", item) is None
            for item in value
        )
        or len(set(value)) != len(value)
        or bool(value) != focused
    ):
        raise ValueError(f"scope.outputs.{name} must match its focused test mode.")


def validate(needs: dict) -> None:
    """Require successful selected jobs and skipped unselected jobs."""
    if not isinstance(needs, dict) or set(needs) != JOBS:
        raise ValueError(
            "needs must contain exactly scope, checks, linux, compatibility, and docker."
        )
    if any(not isinstance(needs[name], dict) for name in JOBS):
        raise ValueError("Each required job must have a result object.")
    for name in ("scope", "checks"):
        if needs[name].get("result") != "success":
            raise ValueError(f"{name} must succeed.")
    outputs = needs["scope"].get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("scope.outputs is missing or invalid.")

    tests = _output(outputs, "tests")
    tests_mode = _output(outputs, "tests_mode")
    test_files = _output(outputs, "test_files")
    native_mode = _output(outputs, "native_mode")
    native_tests = _output(outputs, "native_tests")
    smoke = _output(outputs, "smoke")
    native = _output(outputs, "native")
    docker = _output(outputs, "docker")
    if type(tests) is not bool or type(smoke) is not bool:
        raise ValueError("scope.outputs.tests and smoke must be JSON booleans.")
    if tests_mode not in ("none", "personal", "focused", "full") or tests != (tests_mode != "none"):
        raise ValueError("scope.outputs.tests_mode must match the selected Linux tests.")
    _test_files(test_files, focused=tests_mode == "focused", name="test_files")
    if native_mode not in ("none", "focused", "full"):
        raise ValueError("scope.outputs.native_mode is invalid.")
    _test_files(native_tests, focused=native_mode == "focused", name="native_tests")
    if not isinstance(native, list) or native not in ([], NATIVE):
        raise ValueError("scope.outputs.native must be empty or the full native matrix.")
    if bool(native) != (native_mode != "none"):
        raise ValueError("scope.outputs.native_mode must match native selection.")
    if (
        not isinstance(docker, list)
        or any(not isinstance(item, str) or item not in DOCKER for item in docker)
        or len(set(docker)) != len(docker)
    ):
        raise ValueError("scope.outputs.docker must contain unique known Docker checks.")
    if not tests and (smoke or native or docker):
        raise ValueError("Smoke, native, and Docker checks require Linux tests.")
    if tests_mode == "personal" and (smoke or native or docker):
        raise ValueError("Integration checks require full regression tests.")

    for name, selected in (("linux", tests), ("compatibility", native), ("docker", docker)):
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
