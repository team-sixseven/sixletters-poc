"""Run forward-shape smoke checks using only the real P0002 Holter NPY file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.config import load_yaml, resolve_repo_path  # noqa: E402
from model.DL.dataloader.holter_windows import load_patient_windows  # noqa: E402
from model.DL.models import build_model, load_model_spec  # noqa: E402


def main() -> int:
    model_root = Path(__file__).resolve().parents[1] / "model"
    choice = load_yaml(model_root / "DL/model_choice.yaml")
    argparse.ArgumentParser(description=__doc__).parse_args()
    window_file = resolve_repo_path("Members/sang/datas/Holter/windows/P0002.npy")

    waveform, segment_features, available_windows, constant_leads = load_patient_windows(
        window_file,
        windows_per_patient=choice["windows_per_patient"],
        expected_leads=choice["leads"],
        expected_samples=choice["samples_per_window"],
    )
    architecture_paths = tuple(resolve_repo_path(path) for path in choice["architecture_configs"])
    segment_models = set(choice["models"].get("segment_statistics", []))
    model_names = [name for names in choice["models"].values() for name in names]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
    feature_tensor = torch.from_numpy(segment_features).unsqueeze(0)
    results = []

    for name in model_names:
        spec = load_model_spec(name, architecture_paths)
        model = build_model(name, spec.get("parameters", {})).to(device).eval()
        model_input = (feature_tensor if name in segment_models else waveform_tensor).to(device)
        with torch.inference_mode():
            logits = model(model_input)
        passed = tuple(logits.shape) == (1, 1) and bool(torch.isfinite(logits).all().item())
        result = {
            "model": name,
            "input_shape": list(model_input.shape),
            "output_shape": list(logits.shape),
            "finite_logits": bool(torch.isfinite(logits).all().item()),
            "status": "PASS" if passed else "FAIL",
        }
        results.append(result)
        if not passed:
            raise RuntimeError(f"P0002 forward smoke failed: {result}")

    print(
        json.dumps(
            {
                "status": "FORWARD_SMOKE_ONLY",
                "patient_file": window_file.name,
                "available_windows": available_windows,
                "selected_windows": choice["windows_per_patient"],
                "constant_leads": constant_leads,
                "device": str(device),
                "weights_trained": False,
                "performance_metrics_computed": False,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
