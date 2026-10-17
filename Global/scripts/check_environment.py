"""Check the installed environment and optionally exercise the NVIDIA GPU."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from smoke_check import run_smoke


def check_gpu() -> dict:
    """Fail explicitly if CUDA cannot complete a small matrix multiplication."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch가 없습니다. GPU를 지원하는 Python 환경에 PyTorch를 설치하세요."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 사용할 수 없습니다. NVIDIA 드라이버와 Python 환경의 "
            "PyTorch CUDA 지원을 확인하세요. CPU 성공으로 대체하지 않습니다."
        )
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")
    result = left @ right
    torch.cuda.synchronize()
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    if not torch.allclose(result.cpu(), expected):
        raise RuntimeError("CUDA 행렬 연산 결과가 예상값과 다릅니다.")
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "matrix_multiplication": "passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Require an actual CUDA computation")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("Global/artifacts/environment-check")
    )
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        if platform.python_version() != "3.12.12":
            raise RuntimeError(f"Python 3.12.12가 필요합니다. 현재: {platform.python_version()}")
        gpu = check_gpu() if args.gpu else None
        run_smoke(args.output_dir, ecg=True)
        result = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn", "wfdb", "neurokit2", "jupyterlab")
            },
            "synthetic_data_and_ecg": "passed",
            "gpu": gpu if gpu else "not requested",
        }
        (args.output_dir / "environment.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"환경 점검 완료: {args.output_dir}")
    except (RuntimeError, OSError, ImportError) as exc:
        print(f"환경 점검 실패: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
