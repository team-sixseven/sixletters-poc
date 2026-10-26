"""Fail closed for missing, skipped, failed or cancelled selected checks."""

import json
import os


def validate(needs: dict) -> None:
    if set(needs) != {"scope", "checks", "linux", "compatibility", "docker"}:
        raise ValueError("Required jobs missing")
    for key in ("scope", "checks"):
        if needs[key].get("result") != "success":
            raise ValueError(f"{key} must succeed")
    scope = {
        key: json.loads(needs["scope"]["outputs"][key]) for key in ("tests", "native", "docker")
    }
    if (
        type(scope["tests"]) is not bool
        or scope["native"] not in ([], ["windows-2025", "macos-26"])
        or not isinstance(scope["docker"], list)
        or set(scope["docker"]) - {"workspace", "tracking"}
        or len(scope["docker"]) != len(set(scope["docker"]))
    ):
        raise ValueError("Invalid scope")
    if not scope["tests"] and (scope["native"] or scope["docker"]):
        raise ValueError("Integration checks require Linux tests")
    for key, selected in [
        ("linux", scope["tests"]),
        ("compatibility", scope["native"]),
        ("docker", scope["docker"]),
    ]:
        if needs[key].get("result") != ("success" if selected else "skipped"):
            raise ValueError(f"{key} did not complete the selected checks")


if __name__ == "__main__":
    validate(json.loads(os.environ["NEEDS_JSON"]))
