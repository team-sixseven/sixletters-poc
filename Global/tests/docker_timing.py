"""Safe phase timings for real Docker tests; never record command arguments or output."""

from contextlib import contextmanager, nullcontext
from time import perf_counter

_enabled = False
_case = "session"
_depth = 0
_records = []


def reset(*, enabled=False):
    global _enabled, _case, _depth
    _enabled, _case, _depth = enabled, "session", 0
    _records.clear()


def set_case(nodeid):
    global _case
    # Parameter values need not appear in diagnostics.
    _case = nodeid.split("[")[0].rsplit("::", 1)[-1]


@contextmanager
def stage(name):
    global _depth
    if not _enabled:
        yield
        return
    started, depth = perf_counter(), _depth
    _depth += 1
    outcome = "passed"
    try:
        yield
    except BaseException:
        outcome = "failed"
        raise
    finally:
        _depth -= 1
        _records.append((_case, name, perf_counter() - started, outcome, depth))


def command_stage(service, arguments):
    """Measure only Compose lifecycle actions, without exposing credentials or paths."""
    if service not in {"workspace", "tracking"}:
        raise ValueError("Unknown Docker timing service")
    command = tuple(arguments)
    if service == "workspace":
        if command[:2] != ("docker", "compose"):
            return nullcontext()
        command = command[2:]
    if command and command[0] in {"up", "build", "restart", "stop", "down"}:
        return stage(f"{service}.{command[0]}")
    return nullcontext()


def record_report(report):
    if _enabled:
        name = report.nodeid.split("[")[0].rsplit("::", 1)[-1]
        _records.append((name, f"pytest.{report.when}", report.duration, report.outcome, 0))


def timing_lines():
    if not _enabled or not _records:
        return []
    lines = [
        "## Docker phase timings",
        "",
        "Phase rows can overlap; pytest.setup/call/teardown are the overall test timings.",
        "",
        "| Test | Phase | Seconds | Result |",
        "|---|---|---:|---|",
    ]
    for case, name, seconds, outcome, depth in _records:
        label = ("↳ " if depth else "") + name
        lines.append(f"| {case} | {label} | {seconds:.2f} | {outcome} |")
    return lines
