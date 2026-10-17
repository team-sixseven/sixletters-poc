"""Upload completed local results and retry failures while the workspace runs."""

import os
import signal
import sys
import threading
from pathlib import Path

from sixletters.tracking import sync_runs
from sixletters.tracking_diagnostics import failure_details


def run_worker(root: Path, interval: float, stop: threading.Event) -> None:
    previous = None
    while not stop.is_set():
        try:
            summary = sync_runs(root, retry_only=True)
            if summary.get("disabled"):
                return
            issues = tuple(
                (
                    issue["owner"],
                    issue["experiment"],
                    issue["run_id"],
                    issue["status"],
                    issue["code"],
                    issue.get("scope"),
                    issue.get("path"),
                )
                for issue in summary.get("issues", [])
            )
            status = (summary.get("pending", 0), summary.get("invalid", 0), issues)
            # Identity/cause changes matter even if the counts remain unchanged.
            # Also announce recovery once when the final pending issue disappears.
            if summary.get("uploaded") or (status != previous and (any(status) or previous)):
                print(f"MLflow synchronization: {summary}", flush=True)
            previous = status
        except Exception as error:
            # Network errors can include authorization details; never echo their text.
            details = failure_details(error)
            status = ("unavailable", details["code"])
            if previous != status:
                print(
                    "MLflow synchronization unavailable; local results retained. "
                    f"[{details['code']}] {details['message']} {details['action']}",
                    flush=True,
                )
            previous = status
        stop.wait(interval)


def main() -> int:
    try:
        interval = float(os.environ.get("POC_TRACKING_SYNC_INTERVAL", "60"))
        if not 5 <= interval <= 3600:
            raise ValueError
    except ValueError:
        print("POC_TRACKING_SYNC_INTERVAL must be between 5 and 3600 seconds.", file=sys.stderr)
        return 2
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    run_worker(Path("/workspace/Global/artifacts"), interval, stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
