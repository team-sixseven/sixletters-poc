"""Compare compact patient-level ECG architectures on the fixed clinical split."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.config import load_yaml, resolve_repo_path
from model.DL.dataloader.holter_windows import load_holter_windows
from model.DL.models import build_model, load_model_spec
from model.evaluation.DL.holter import (
    choose_youden_threshold,
    evaluate_holter_predictions,
    metrics,
)
from model.evaluation.reporting import create_report_dir, metric, write_report
from model.tracking.bundle import write_bundle


@torch.no_grad()
def predict(
    model: nn.Module, x: torch.Tensor, batch_size: int, device: torch.device
) -> torch.Tensor:
    model.eval()
    outputs = []
    for start in range(0, len(x), batch_size):
        logits = model(x[start : start + batch_size].to(device))
        outputs.append(torch.sigmoid(logits).squeeze(1).cpu())
    return torch.cat(outputs)


def train_one(name: str, datasets, args, device, model_parameters: dict):
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = build_model(name, model_parameters).to(device)
    train_x, train_y, _, train_features = datasets["train"]
    val_x, val_y, _, val_features = datasets["val"]
    positives = float(train_y.sum().item())
    negatives = len(train_y) - positives
    if positives == 0:
        raise RuntimeError("Training split has no ECG-matched SCD-positive patient")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_ap = -1.0
    best_state = None
    best_epoch = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_x))
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            raw_x = train_x[indices].to(device)
            feature_x = train_features[indices].to(device)
            x = feature_x if name in args.segment_statistics_models else raw_x
            y = train_y[indices].to(device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        val_input = val_features if name in args.segment_statistics_models else val_x
        val_probability = predict(model, val_input, args.batch_size, device)
        val_metrics = metrics(val_y, val_probability)
        ap = val_metrics["auprc_average_precision"]
        score = float(ap) if ap is not None else -1.0
        if score > best_ap:
            best_ap = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
    elapsed = time.perf_counter() - started
    peak_memory_mib = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
    )
    if best_state is None:
        raise RuntimeError(f"No validation checkpoint selected for {name}")
    model.load_state_dict(best_state)
    val_input = val_features if name in args.segment_statistics_models else val_x
    val_probability = predict(model, val_input, args.batch_size, device)
    val_metrics = metrics(val_y, val_probability)
    validation_threshold = choose_youden_threshold(val_y, val_probability)
    return model, {
        "model": name,
        "best_epoch_by_validation_ap": best_epoch,
        "training_seconds": elapsed,
        "cuda_peak_allocated_mib": peak_memory_mib,
        "validation": val_metrics,
        "validation_threshold_metrics": validation_threshold,
    }


def main() -> int:
    choice_path = Path(__file__).resolve().parents[1] / "model/DL/model_choice.yaml"
    choice = load_yaml(choice_path)
    training_path = resolve_repo_path(choice["training_config"])
    training = load_yaml(training_path)
    architecture_paths = tuple(resolve_repo_path(item) for item in choice["architecture_configs"])
    model_names = [name for names in choice["models"].values() for name in names]
    segment_statistics_models = set(choice["models"]["segment_statistics"])
    if len(model_names) != len(set(model_names)):
        raise ValueError("DL model_choice.yaml contains duplicate model names")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--splits-dir", type=Path)
    parser.add_argument("--windows-per-patient", type=int)
    parser.add_argument("--samples-per-window", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--tracking-output",
        type=Path,
        help="Optional local MLflow handoff bundle directory; does not upload to a server",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "model/outputs/evaluation",
        help="Parent directory for this run's mandatory report.md and results.json",
    )
    args = parser.parse_args()
    args.data_dir = args.data_dir or resolve_repo_path(choice["data_dir"])
    args.splits_dir = args.splits_dir or resolve_repo_path(choice["splits_dir"])
    args.windows_per_patient = (
        choice["windows_per_patient"]
        if args.windows_per_patient is None
        else args.windows_per_patient
    )
    args.samples_per_window = (
        choice["samples_per_window"] if args.samples_per_window is None else args.samples_per_window
    )
    args.epochs = training["epochs"] if args.epochs is None else args.epochs
    args.batch_size = training["batch_size"] if args.batch_size is None else args.batch_size
    args.learning_rate = (
        training["learning_rate"] if args.learning_rate is None else args.learning_rate
    )
    args.seed = training["seed"] if args.seed is None else args.seed
    args.segment_statistics_models = segment_statistics_models
    if training["optimizer"].lower() != "adam" or training["loss"] != "BCEWithLogitsLoss":
        raise ValueError("Only Adam with BCEWithLogitsLoss is currently supported")
    if training["model_selection_metric"] != choice["selection_metric"]:
        raise ValueError("DL model selection metric differs between choice and training YAML")
    if choice["threshold_strategy"] != "validation_youden_j":
        raise ValueError("Unsupported threshold strategy in DL model_choice.yaml")
    if choice["model_input_dtype"] != "float32":
        raise ValueError("Holter loader currently emits float32 model tensors")
    if training["positive_class_weight"] != "train_negatives_divided_by_train_positives":
        raise ValueError("Unsupported positive_class_weight setting")
    report_dir = create_report_dir(args.report_root, "holter")

    datasets, coverage = load_holter_windows(
        args.data_dir,
        args.splits_dir,
        windows_per_patient=args.windows_per_patient,
        leads=choice["leads"],
        samples_per_window=args.samples_per_window,
        window_selection=choice["window_selection"],
        normalization=choice["normalization"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_specs = {name: load_model_spec(name, architecture_paths) for name in model_names}
    for name, spec in model_specs.items():
        expected_shape = (
            ["batch", args.windows_per_patient, 16]
            if name in segment_statistics_models
            else ["batch", args.windows_per_patient, choice["leads"], args.samples_per_window]
        )
        configured_shape = spec.get("input")
        if (
            configured_shape is None
            or [str(part) if isinstance(part, str) else part for part in configured_shape]
            != expected_shape
        ):
            raise ValueError(
                f"{name} YAML input shape {configured_shape} does not match configured data shape {expected_shape}"
            )
        if spec.get("output") != ["batch", 1]:
            raise ValueError(f"{name} must declare a [batch, 1] output in its YAML")
        parameters = spec.get("parameters", {})
        if name in segment_statistics_models:
            if parameters.get("input_features") != 16:
                raise ValueError(
                    f"{name} input_features must match the 16 derived segment features"
                )
        elif parameters.get("channels") != choice["leads"]:
            raise ValueError(f"{name} channel count does not match the configured ECG lead count")
    models, results = [], []
    for name in model_names:
        model_params = model_specs[name].get("parameters", {})
        model, result = train_one(name, datasets, args, device, model_params)
        models.append((name, model))
        results.append(result)

    selected = max(
        results,
        key=lambda item: item["validation"]["auprc_average_precision"]
        if item["validation"]["auprc_average_precision"] is not None
        else -1,
    )["model"]
    test_x, test_y, _, test_features = datasets["test"]
    test_model = dict(models)[selected]
    test_input = test_features if selected in segment_statistics_models else test_x
    test_probability = predict(test_model, test_input, args.batch_size, device)
    selected_result = next(result for result in results if result["model"] == selected)
    selected_threshold = selected_result["validation_threshold_metrics"]["threshold"]
    test_metrics = evaluate_holter_predictions(
        test_y, test_probability, threshold=selected_threshold
    )

    if args.tracking_output is not None:
        run_records = []
        model_lookup = dict(models)
        test_ids = datasets["test"][2]
        test_prediction_frame = torch.cat(
            (
                torch.tensor(test_y.numpy(), dtype=torch.float32).unsqueeze(1),
                test_probability.unsqueeze(1),
            ),
            dim=1,
        ).numpy()
        with tempfile.TemporaryDirectory(prefix="scd-holter-models-") as temporary:
            temporary_dir = Path(temporary)
            for result in results:
                name = result["model"]
                val_ids = datasets["val"][2]
                val_y = datasets["val"][1]
                val_features = datasets["val"][3]
                val_x = datasets["val"][0]
                val_input = val_features if name in segment_statistics_models else val_x
                val_probability = predict(model_lookup[name], val_input, args.batch_size, device)
                val_frame = pd.DataFrame(
                    {
                        "patient_id": val_ids,
                        "target": val_y.numpy().astype(int),
                        "score": val_probability.numpy().astype(float),
                    }
                )
                model_path = (
                    temporary_dir / f"{name.lower().replace(' ', '_').replace('–', '-')}.pt"
                )
                portable_state = {
                    key: tensor.detach().cpu()
                    for key, tensor in model_lookup[name].state_dict().items()
                }
                torch.save(portable_state, model_path)
                run_record = {
                    "model": name,
                    "input_group": (
                        "holter_segment_statistics"
                        if name in segment_statistics_models
                        else "holter_raw_waveform"
                    ),
                    "parameters": {
                        "seed": args.seed,
                        "architecture": model_specs[name].get("parameters", {}),
                        "learning_rate": args.learning_rate,
                        "batch_size": args.batch_size,
                        "epochs_max": args.epochs,
                        "best_epoch": result["best_epoch_by_validation_ap"],
                        "positive_weight": training["positive_class_weight"],
                        "windows_per_patient": args.windows_per_patient,
                        "samples_per_window": args.samples_per_window,
                        "sampling_rate_hz_assumption": choice["sampling_rate_hz_assumption"],
                        "sampled_waveform_seconds_per_patient": args.windows_per_patient
                        * args.samples_per_window
                        / choice["sampling_rate_hz_assumption"],
                        "window_selection": choice["window_selection"],
                        "preprocessing": choice["normalization"],
                    },
                    "validation_metrics": result["validation"],
                    "validation_threshold_metrics": result["validation_threshold_metrics"],
                    "training_metrics": {
                        "training_seconds": result["training_seconds"],
                        "cuda_peak_allocated_mib": result["cuda_peak_allocated_mib"],
                    },
                    "validation_predictions": val_frame,
                    "artifacts": {"model_state_dict.pt": model_path},
                }
                if name == selected:
                    run_record["test_metrics"] = test_metrics
                    run_record["test_predictions"] = pd.DataFrame(
                        {
                            "patient_id": test_ids,
                            "target": test_prediction_frame[:, 0].astype(int),
                            "score": test_prediction_frame[:, 1].astype(float),
                        }
                    )
                run_records.append(run_record)

            saved_bundle = write_bundle(
                args.tracking_output,
                experiment="sang-holter-model-comparison",
                dataset="music-holter-windowed-spaced-8-v1",
                input_group="holter_raw_and_segment_statistics",
                split_summary={
                    split: {
                        "n": details["ecg_patients"],
                        "scd": details["ecg_scd_positive"],
                        "clinical_patients": details["clinical_patients"],
                        "missing_ecg": details["missing_ecg"],
                        "constant_selected_leads": details["constant_selected_leads"],
                    }
                    for split, details in coverage.items()
                },
                selected_model=selected,
                environment={
                    "python": platform.python_version(),
                    "pytorch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                    "seed": args.seed,
                },
                runs=run_records,
                source_files=[
                    Path(__file__),
                    choice_path,
                    training_path,
                    *architecture_paths,
                    Path(__file__).resolve().parents[1] / "model/DL/models.py",
                    Path(__file__).resolve().parents[1] / "model/DL/dataloader/holter_windows.py",
                    Path(__file__).resolve().parents[1] / "model/config.py",
                    Path(__file__).resolve().parents[1] / "model/tracking/bundle.py",
                ],
            )

    payload = {
        "purpose": "exploratory patient-level comparison on the fixed MUSIC clinical split; not external validation",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seed": args.seed,
        "input_shape_per_patient": [
            args.windows_per_patient,
            choice["leads"],
            args.samples_per_window,
        ],
        "selected_window_seconds": args.samples_per_window / choice["sampling_rate_hz_assumption"],
        "sampled_waveform_seconds_per_patient": args.windows_per_patient
        * args.samples_per_window
        / choice["sampling_rate_hz_assumption"],
        "sampling_rate_hz_assumption": choice["sampling_rate_hz_assumption"],
        "window_selection": choice["window_selection"],
        "preprocessing": choice["normalization"],
        "coverage": coverage,
        "selection_rule": "highest validation average precision; test evaluated once for selected model",
        "selected_model": selected,
        "selected_test_metrics": test_metrics,
        "models": results,
        "tracking_handoff_bundle": str(saved_bundle) if args.tracking_output is not None else None,
        "report_dir": str(report_dir),
    }
    table = [
        "| Model | Val AUROC | Val AP | Val Brier | Val Sens. | Val Spec. | Best epoch | Train (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        val = result["validation"]
        operating = result["validation_threshold_metrics"]
        table.append(
            f"| {result['model']} | {metric(val['auroc'])} | {metric(val['auprc_average_precision'])} | "
            f"{metric(val['brier'])} | {metric(operating['sensitivity'])} | "
            f"{metric(operating['specificity'])} | {result['best_epoch_by_validation_ap']} | "
            f"{result['training_seconds']:.2f} |"
        )
    split_lines = [
        f"| {name} | {details['ecg_patients']} | {details['ecg_scd_positive']} | {details['constant_selected_leads']} |"
        for name, details in coverage.items()
    ]
    report = "\n".join(
        [
            "# Holter ECG Model Evaluation",
            "",
            "- Status: COMPLETE",
            f"- ECG data: `{args.data_dir}`",
            f"- Clinical split files: `{args.splits_dir}`",
            f"- Device: {device}"
            + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
            f"- Input per patient: ({args.windows_per_patient}, {choice['leads']}, {args.samples_per_window}); at the assumed {choice['sampling_rate_hz_assumption']} Hz, each selected window is {payload['selected_window_seconds']:.1f} seconds.",
            f"- Window selection: {payload['window_selection']}",
            f"- Waveform used: {payload['sampled_waveform_seconds_per_patient']:.1f} seconds total, spread across the saved sequence; sampling rate is assumed from the source recording specification because NPY stores no rate metadata.",
            "- Segment Attention MIL and MIL-TCN use 16 simple window statistics (mean, standard deviation, RMS and range for three leads plus vector magnitude), not clinical HRV or rhythm annotations.",
            "- Training: each candidate fitted independently on the training split; best epoch and model selected using validation Average Precision.",
            "- Test: evaluated once for the selected model; threshold was selected from validation Youden J.",
            f"- Selected model: **{selected}**",
            "",
            "## Split coverage",
            "",
            "| Split | ECG patients | SCD-positive | Constant selected leads |",
            "|---|---:|---:|---:|",
            *split_lines,
            "",
            "## Validation comparison",
            "",
            *table,
            "",
            "## Selected model test result",
            "",
            f"- Threshold (validation Youden J): {metric(test_metrics['threshold'])}",
            f"- AUROC: {metric(test_metrics['auroc'])}",
            f"- Average Precision: {metric(test_metrics['auprc_average_precision'])}",
            f"- Brier: {metric(test_metrics['brier'])}",
            f"- Sensitivity / Specificity: {metric(test_metrics['sensitivity'])} / {metric(test_metrics['specificity'])}",
            "",
            "## Reproducibility and limitations",
            "",
            f"- Seed: {args.seed}; epochs: {args.epochs}; batch size: {args.batch_size}; learning rate: {args.learning_rate}",
            f"- Preprocessing: {payload['preprocessing']}",
            "- This is an exploratory comparison on the available ECG-matched subset, not external validation.",
            "- Only patients with NPY files are included; split coverage and positive counts are reported above. Sparse window sampling may miss transient arrhythmias.",
            "- Test metrics are reported only for the model selected on validation.",
        ]
    )
    write_report(report_dir, payload, report)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
