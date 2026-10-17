"""Record a bounded script invocation, without persisting common credential arguments."""

import json
import os
import re
from pathlib import Path, PurePosixPath

ENVIRONMENT_KEY = "SIXLETTERS_EXECUTION_JSON"
_SECRET = re.compile(r"token|password|passwd|secret|credential|api[-_]?key|authorization", re.I)


def invocation(directory: Path, script: Path, arguments: list[str]) -> dict:
    redacted = False
    hide_next = False
    recorded = []
    for value in arguments:
        if hide_next:
            recorded.append("[REDACTED]")
            hide_next = False
            redacted = True
        elif value.startswith("-") and _SECRET.search(value.split("=", 1)[0]):
            option, separator, _ = value.partition("=")
            recorded.append(option + "=[REDACTED]" if separator else option)
            hide_next = not separator
            redacted = True
        elif "://" in value and ("@" in value or _SECRET.search(value)):
            recorded.append("[REDACTED_URL]")
            redacted = True
        else:
            recorded.append(value)
    result = {
        "script": script.relative_to(directory).as_posix(),
        "args": recorded,
        "cwd": "workspace",
        "redacted": redacted,
    }
    validate_invocation(result)
    return result


def validate_invocation(value: dict) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"script", "args", "cwd", "redacted"}
        or not isinstance(value["script"], str)
        or not isinstance(value["args"], list)
        or len(value["args"]) > 256
        or any(not isinstance(item, str) for item in value["args"])
        or value["cwd"] != "workspace"
        or type(value["redacted"]) is not bool
        or len(json.dumps(value).encode("utf-8")) > 32768
    ):
        raise ValueError("Execution metadata must be a bounded script/arguments record")
    path = PurePosixPath(value["script"])
    if (
        path.is_absolute()
        or not path.parts
        or any(part.startswith(".") or "\\" in part for part in path.parts)
        or path.suffix != ".py"
    ):
        raise ValueError("Execution metadata must reference a relative Python script")


def current_invocation(directory: Path) -> dict | None:
    raw = os.environ.get(ENVIRONMENT_KEY)
    if not raw:
        return None
    value = json.loads(raw)
    validate_invocation(value)
    if not (directory / value["script"]).is_file():
        raise ValueError("Recorded execution script is outside this experiment")
    return value
