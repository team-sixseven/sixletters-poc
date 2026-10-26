"""Print versions and accelerator availability without accessing patient data."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess

print(f"OS: {platform.platform()}")
print(f"Architecture: {platform.machine()}")
print(f"Python: {platform.python_version()}")
print(f"uv: {shutil.which('uv') or 'not found'}")
for command in ("nvidia-smi", "nvcc"):
    path = shutil.which(command)
    if path:
        result = subprocess.run([command, "--version"], capture_output=True, text=True, check=False)
        print(f"{command}: {' '.join((result.stdout or result.stderr).split())}")
    else:
        print(f"{command}: not found")
for package in ("torch", "xgboost", "catboost", "wfdb"):
    try:
        print(f"{package}: {importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: not installed")
try:
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB")
except ImportError:
    print("PyTorch CUDA check: SKIP (PyTorch not installed)")
