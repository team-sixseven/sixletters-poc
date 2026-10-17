"""Check Jupyter readiness without printing or disabling its access token."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def main() -> int:
    runtime = Path(os.environ.get("JUPYTER_RUNTIME_DIR", "/tmp/poc-home/jupyter-runtime"))
    for server_file in runtime.glob("jpserver-*.json"):
        try:
            server = json.loads(server_file.read_text(encoding="utf-8"))
            token = server.get("token")
            if not token:
                continue
            base = server.get("base_url", "/").strip("/")
            prefix = f"/{base}" if base else ""
            url = f"http://127.0.0.1:{int(server['port'])}{prefix}/api/status"
            request = Request(url, headers={"Authorization": f"token {token}"})
            with urlopen(request, timeout=2) as response:
                if response.status == 200:
                    return 0
        except (OSError, URLError, ValueError, KeyError, TypeError):
            continue
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
