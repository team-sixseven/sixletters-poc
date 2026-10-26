"""Check the environment, then keep the authenticated workspace running."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    workspace = Path("/workspace")
    require_gpu = os.environ.get("POC_REQUIRE_GPU", "0") == "1"
    command = [
        sys.executable,
        "Global/scripts/check_environment.py",
        "--output-dir",
        f"Global/artifacts/environment-check/{'gpu' if require_gpu else 'cpu'}",
    ]
    if require_gpu:
        command.append("--gpu")
    try:
        subprocess.run(command, cwd=workspace, check=True)
        print("환경 준비 완료. JupyterLab에서 개인 코드와 노트북을 실행하세요.", flush=True)
        from sixletters.personal_env import jupyter_data_path

        # Registrations live in the bind-mounted ignored cache and survive recreation.
        # Runtime-specific directories keep host Python paths out of container kernels.
        kernel_data = str(jupyter_data_path(workspace))
        existing_paths = os.environ.get("JUPYTER_PATH", "").split(os.pathsep)
        os.environ["JUPYTER_PATH"] = os.pathsep.join(
            [kernel_data, *(value for value in existing_paths if value and value != kernel_data)]
        )
        os.chdir(workspace)
        os.execvp(
            "jupyter",
            [
                "jupyter",
                "lab",
                "--ServerApp.ip=0.0.0.0",
                "--ServerApp.port=8888",
                "--ServerApp.port_retries=0",
                "--ServerApp.root_dir=/workspace",
                # Login shells reset PATH and hide the image's shared Python environment.
                '--ServerApp.terminado_settings={"shell_command": ["/bin/bash"]}',
                "--no-browser",
            ],
        )
    except (subprocess.CalledProcessError, RuntimeError, OSError, ValueError) as exc:
        print(f"작업 환경 시작 실패: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
