"""Run real-input forward checks on P1030 without using or inventing a label."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.config import dl_config_paths, load_experiment_config, load_yaml
from model.data import load_patient_quality_indices, load_patient_waveform
from model.DL.composed import build_composed_model


def main() -> int:
    model_root = Path(__file__).resolve().parents[1] / "model"
    choice_path = model_root / "DL/model_choice.yaml"
    choice, basic, choice_path, basic_path = load_experiment_config(choice_path)
    paths = dl_config_paths(choice_path)
    training = load_yaml(paths["training"])
    patient_id = "P1030"
    ecg = basic["data"]["ecg"]
    patient_file = (
        basic_path.parent / ecg["directory"] / ecg["filename_pattern"].format(pid=patient_id)
    ).resolve()
    quality_indices = load_patient_quality_indices(patient_id, basic, basic_path)
    waveform, selected_rows = load_patient_waveform(
        patient_id,
        patient_file,
        quality_indices,
        ecg,
        max_windows=training["max_segments_per_patient"],
    )
    patient_tensor = torch.from_numpy(waveform).unsqueeze(0)
    cnn_config = load_yaml(paths["cnn"])
    lstm_config = load_yaml(paths["lstm"])
    results = []

    combinations = [
        ("CNN+LSTM", cnn_name, head_name)
        for cnn_name in cnn_config["models"]
        for head_name in lstm_config["models"]
    ] + [("CNN+Transformer", cnn_name, "transformer_enc") for cnn_name in cnn_config["models"]]
    torch.set_num_threads(1)
    for combination, cnn_name, sequence_name in combinations:
        configured = deepcopy(choice)
        configured["cnn_model"] = cnn_name
        configured["sequence_model"] = sequence_name
        model = build_composed_model(configured, basic, choice_path=choice_path).eval()
        with torch.inference_mode():
            logits = model(patient_tensor)
        result = {
            "combination": combination,
            "cnn_model": cnn_name,
            "sequence_model": sequence_name,
            "input_shape": list(patient_tensor.shape),
            "output_shape": list(logits.shape),
            "finite_logits": bool(torch.isfinite(logits).all()),
        }
        if result["output_shape"] != [1, 1] or not result["finite_logits"]:
            raise RuntimeError(f"P1030 forward validation failed: {result}")
        results.append(result)
    print(
        json.dumps(
            {
                "status": "FORWARD_CHECKS_PASS",
                "patient_id": patient_id,
                "patient_file": str(patient_file),
                "label_used": False,
                "training_performed": False,
                "array_shape": list(np.load(patient_file, mmap_mode="r").shape),
                "sqi_passing_windows": len(quality_indices),
                "selected_npy_rows": selected_rows.tolist(),
                "normalized_input_shape": list(patient_tensor.shape),
                "normalization": ecg["normalization"],
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
