"""Run API/shape checks with synthetic data only."""

from __future__ import annotations

import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_ml() -> list[tuple[str, bool, str]]:
    from model.config import load_yaml
    from model.ML.models import CATEGORICAL_FEATURES, build_models

    member_root = Path(__file__).resolve().parents[1]
    choice = load_yaml(member_root / "model/ML/model_choice.yaml")
    clinical = pd.read_csv(member_root / "datas/clincal/clinical_clean.csv", nrows=1)
    features = [name for name in clinical.columns if name not in {"Patient ID", "SCD"}]
    rng = np.random.default_rng(42)
    x = pd.DataFrame(index=range(256), columns=features)
    for feature in features:
        if feature in CATEGORICAL_FEATURES:
            x[feature] = rng.choice(["0", "1", "2"], size=256)
        else:
            x[feature] = rng.normal(size=256)
    y = rng.binomial(1, 0.2, size=256)
    if len(np.unique(y)) != 2:
        y[0], y[-1] = 0, 1
    models = build_models(features, choice["models"], choice["preprocessing"])
    rows = []
    for name, estimator in models.items():
        try:
            fit_x = x.copy()
            if name == "CatBoost":
                estimator.set_params(cat_features=list(CATEGORICAL_FEATURES))
            started = time.perf_counter()
            estimator.fit(fit_x, y)
            probability = estimator.predict_proba(fit_x)[:, 1]
            elapsed = time.perf_counter() - started
            assert probability.shape == (256,)
            assert (
                np.isfinite(probability).all() and ((0 <= probability) & (probability <= 1)).all()
            )
            rows.append(
                (
                    name,
                    True,
                    f"(256,{len(features)}) -> (256,); fit+predict={elapsed:.3f}s; "
                    f"p=[{probability.min():.3f},{probability.max():.3f}], mean={probability.mean():.3f}",
                )
            )
        except Exception as exc:
            rows.append((name, False, f"{type(exc).__name__}: {exc}"))
    return rows


def check_dl(device: str) -> list[tuple[str, bool, str]]:
    import torch
    from model.config import load_yaml, resolve_repo_path
    from model.DL.models import build_model, load_model_spec
    from torch import nn

    member_root = Path(__file__).resolve().parents[1]
    choice = load_yaml(member_root / "model/DL/model_choice.yaml")
    architecture_paths = tuple(resolve_repo_path(path) for path in choice["architecture_configs"])
    statistics_names = set(choice["models"]["segment_statistics"])
    model_names = [name for names in choice["models"].values() for name in names]

    rows = []
    for name in model_names:
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            spec = load_model_spec(name, architecture_paths)
            model = build_model(name, spec.get("parameters", {})).to(device)
            shape = (
                (2, choice["windows_per_patient"], 16)
                if name in statistics_names
                else (
                    2,
                    choice["windows_per_patient"],
                    choice["leads"],
                    choice["samples_per_window"],
                )
            )
            x = torch.randn(*shape, device=device)
            y = torch.randint(0, 2, (shape[0], 1), device=device).float()
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            output = model(x)
            loss = nn.BCEWithLogitsLoss()(output, y)
            loss.backward()
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert output.shape == (shape[0], 1)
            assert torch.isfinite(output).all() and torch.isfinite(loss)
            assert any(
                p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
            )
            rows.append(
                (
                    name,
                    True,
                    f"{shape} -> {tuple(output.shape)} on {device}; forward+backward={elapsed:.3f}s; "
                    f"BCE={loss.item():.4f}",
                )
            )
            if device == "cuda":
                torch.cuda.synchronize()
                print(
                    f"  {name} CUDA peak memory: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB"
                )
        except Exception as exc:
            rows.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
    return rows


def main() -> int:
    print(f"Python {platform.python_version()} ({sys.executable})")
    np.random.seed(42)
    results = check_ml()
    try:
        import torch

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        print(
            f"PyTorch {torch.__version__}; CUDA available={torch.cuda.is_available()}; "
            f"CUDA runtime={torch.version.cuda}"
        )
        devices = ["cpu"]
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.reset_peak_memory_stats()
            devices.insert(0, "cuda")
        for device in devices:
            results.extend(check_dl(device))
    except Exception as exc:
        from model.config import load_yaml

        choice = load_yaml(Path(__file__).resolve().parents[1] / "model/DL/model_choice.yaml")
        names = [name for group in choice["models"].values() for name in group]
        results.extend((name, False, f"PyTorch unavailable: {exc}") for name in names)
    for name, passed, details in results:
        print(f"{'PASS' if passed else 'FAIL'} {name}: {details}")
    return 0 if all(passed for _, passed, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
