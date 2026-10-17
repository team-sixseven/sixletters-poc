"""Verify the local environment with generated data, never with patient data."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

FEATURES = ["feature_1", "feature_2", "feature_3"]
SEED = 42


def make_synthetic_data(seed: int = SEED) -> pd.DataFrame:
    """Create 120 fictitious patients, each with three rows and one label."""
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], 60)
    rng.shuffle(labels)
    patient_features = rng.normal(size=(len(labels), len(FEATURES)))
    patient_features[:, 0] += labels * 0.8
    values = np.repeat(patient_features, 3, axis=0)
    values += rng.normal(scale=0.15, size=values.shape)
    values[rng.random(values.shape) < 0.1] = np.nan
    frame = pd.DataFrame(values, columns=FEATURES)
    frame.insert(0, "patient_id", np.repeat([f"synthetic_{i:03d}" for i in range(120)], 3))
    frame["target"] = np.repeat(labels, 3)
    return frame


def split_patients(frame: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Use a stratified 60/20/20 split for this synthetic environment check."""
    if frame["patient_id"].isna().any() or frame["patient_id"].astype(str).str.strip().eq("").any():
        raise ValueError("patient_id must not be missing or blank")
    if frame["target"].isna().any() or not frame["target"].isin([0, 1]).all():
        raise ValueError("target must contain only 0 and 1")
    if frame.groupby("patient_id")["target"].nunique().gt(1).any():
        raise ValueError("Each patient must have one consistent target")

    patients = frame[["patient_id", "target"]].drop_duplicates()
    development, test = train_test_split(
        patients, test_size=0.2, stratify=patients["target"], random_state=seed
    )
    train, validation = train_test_split(
        development, test_size=0.25, stratify=development["target"], random_state=seed
    )
    return {
        name: frame.loc[frame["patient_id"].isin(part["patient_id"])].copy()
        for name, part in {
            "train": train,
            "validation": validation,
            "test": test,
        }.items()
    }


def fit_model(train: pd.DataFrame):
    """Learn missing-value replacements, scaling and model weights from train only."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(random_state=SEED, max_iter=500),
    )
    model.fit(train[FEATURES], train["target"])
    return model


def evaluate(model, frame: pd.DataFrame) -> dict[str, float]:
    predictions = frame[["patient_id", "target"]].copy()
    predictions["probability"] = model.predict_proba(frame[FEATURES])[:, 1]
    # Average the three synthetic rows so each fictitious patient counts once.
    patients = predictions.groupby("patient_id").agg(
        target=("target", "first"), probability=("probability", "mean")
    )
    return {
        "auroc": float(roc_auc_score(patients["target"], patients["probability"])),
        "accuracy_at_0_5": float(
            accuracy_score(patients["target"], patients["probability"] >= 0.5)
        ),
    }


def check_ecg(seed: int = SEED) -> dict[str, int | float]:
    """Check optional ECG packages with an in-memory generated signal."""
    try:
        import neurokit2 as nk
        import wfdb
    except ImportError as exc:
        raise RuntimeError("--ecg requires the optional wfdb and neurokit2 packages") from exc

    sampling_rate = 250
    signal = nk.ecg_simulate(
        duration=12, sampling_rate=sampling_rate, heart_rate=70, random_state=seed
    )
    with tempfile.TemporaryDirectory(prefix="sixletters-smoke-") as directory:
        wfdb.wrsamp(
            "synthetic_ecg",
            fs=sampling_rate,
            units=["mV"],
            sig_name=["ECG"],
            p_signal=signal[:, None],
            fmt=["16"],
            write_dir=directory,
        )
        restored = wfdb.rdrecord(str(Path(directory) / "synthetic_ecg"))
    if restored.fs != sampling_rate or restored.p_signal.shape != (len(signal), 1):
        raise RuntimeError("Synthetic WFDB roundtrip changed the signal shape or sampling rate")
    error = float(np.max(np.abs(restored.p_signal[:, 0] - signal)))
    if error > 0.001:
        raise RuntimeError("Synthetic WFDB roundtrip exceeded the expected quantization error")
    cleaned = nk.ecg_clean(restored.p_signal[:, 0], sampling_rate=sampling_rate)
    _, peaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
    count = len(peaks["ECG_R_Peaks"])
    if count < 2:
        raise RuntimeError("NeuroKit2 did not detect enough peaks in the generated signal")
    return {"samples": len(signal), "r_peaks": count, "max_roundtrip_error": error}


def run_smoke(output_dir: Path, ecg: bool = False) -> dict:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "synthetic_input.csv"
    make_synthetic_data().to_csv(csv_path, index=False)
    frame = pd.read_csv(csv_path, dtype={"patient_id": "string"})
    splits = split_patients(frame)
    model = fit_model(splits["train"])
    metrics = {name: evaluate(model, splits[name]) for name in ("validation", "test")}
    result = {
        "data_source": "generated synthetic data only",
        "purpose": "Environment check; NOT MUSIC data or clinical model performance",
        "seed": SEED,
        "split": "60/20/20 synthetic smoke test only; not a project evaluation decision",
        "patients": {name: part["patient_id"].nunique() for name, part in splits.items()},
        "rows": {name: len(part) for name, part in splits.items()},
        "missing_values_before_imputation": int(frame[FEATURES].isna().sum().sum()),
        "evaluation_unit": "synthetic patient, mean row probability; fixed threshold 0.5",
        "metrics": metrics,
    }
    if ecg:
        result["synthetic_ecg"] = check_ecg()

    figure, axes = plt.subplots(figsize=(7, 4))
    pd.DataFrame(metrics).T.plot.bar(ax=axes, ylim=(0, 1), rot=0)
    axes.set_title("Synthetic environment check — NOT clinical performance")
    axes.set_ylabel("Score on generated data")
    figure.tight_layout()
    figure.savefig(output_dir / "synthetic_metrics.png", dpi=120)
    plt.close(figure)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    # Redirected Windows output can default to cp1252 and reject Korean text.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Check Python/CSV/ML/plotting using synthetic data only."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "smoke",
        help="Output directory (default: Global/artifacts/smoke in the repository).",
    )
    parser.add_argument(
        "--ecg",
        action="store_true",
        help="Also check WFDB and NeuroKit2 using a generated ECG.",
    )
    args = parser.parse_args()
    result = run_smoke(args.output_dir, ecg=args.ecg)
    print("환경 점검 완료 — 합성 데이터 전용이며 실제 의료 성능이 아닙니다.")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
