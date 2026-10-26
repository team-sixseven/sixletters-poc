"""Run synthetic shape, loss, and gradient checks for segment-level prototypes."""

from __future__ import annotations

import json
import platform
import time

import torch
from model.DL.models import MILTCN, RawECGMIL, SegmentAttentionMIL
from torch import nn


def main() -> int:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = [
        ("Segment Attention MIL", SegmentAttentionMIL(), (8, 8, 16)),
        ("MIL-TCN", MILTCN(), (8, 8, 16)),
        ("Raw ECG MIL", RawECGMIL(), (2, 8, 3, 2000)),
    ]
    results = []
    for name, model, input_shape in cases:
        model = model.to(device)
        x = torch.randn(*input_shape, device=device)
        y = torch.randint(0, 2, (input_shape[0], 1), device=device).float()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = model(x)
        loss = nn.BCEWithLogitsLoss()(output, y)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        gradient_ok = all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        passed = (
            output.shape == (input_shape[0], 1)
            and torch.isfinite(output).all().item()
            and torch.isfinite(loss).item()
            and gradient_ok
        )
        result = {
            "model": name,
            "input_shape": list(input_shape),
            "output_shape": list(output.shape),
            "device": str(device),
            "forward": True,
            "loss": float(loss.item()),
            "backward": gradient_ok,
            "finite_outputs_loss_gradients": bool(passed),
            "seconds_forward_loss_backward": elapsed,
        }
        if device.type == "cuda":
            result["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 2)
        results.append(result)
        if not passed:
            raise RuntimeError(f"Smoke check failed: {name}: {result}")

    print(
        json.dumps(
            {
                "synthetic_data_only": True,
                "purpose": "shape/forward/loss/backward smoke check; not model performance",
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
