"""Exercise installed packages without project data or model training."""

import argparse
import importlib
import json
from io import StringIO
from pathlib import Path


def run_smoke(output_dir: Path, ecg: bool = False) -> dict:
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    packages = ["sklearn", "jupyterlab", "ipykernel", "mlflow"]
    if ecg:
        packages += ["wfdb", "neurokit2"]
    for name in packages:
        importlib.import_module(name)
    frame = pd.read_csv(StringIO("x,y\n1,2\n2,4\n"))
    assert np.array_equal(frame.y.to_numpy(), [2, 4])
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots()
    axes.plot(frame.x, frame.y)
    figure.savefig(output_dir / "package-check.png")
    plt.close(figure)
    result = {"csv": "passed", "plot": "passed", "imports": packages}
    (output_dir / "package-check.json").write_text(json.dumps(result) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecg", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("Global/artifacts/environment-check")
    )
    args = parser.parse_args()
    print(run_smoke(args.output_dir, args.ecg))
