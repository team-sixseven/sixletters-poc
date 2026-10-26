"""Inspect CSV metadata and shapes; never print patient-level rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def inspect(path: Path) -> None:
    if not path.is_file():
        print(f"SKIP {path.name}: missing")
        return
    frame = pd.read_csv(path, sep=";", decimal=",", low_memory=False)
    print(f"{path.name}: {len(frame)} rows x {len(frame.columns)} columns")
    print(f"  columns: {list(frame.columns)}")
    id_col = next((c for c in frame.columns if c.strip().lower() == "patient_id"), None)
    if id_col:
        print(f"  patient_id duplicates: {int(frame[id_col].duplicated().sum())}")
    print(f"  missing cells: {int(frame.isna().sum().sum())}")
    print(f"  numeric columns: {int(frame.select_dtypes(include='number').shape[1])}")
    print(f"  nonnumeric columns: {int(frame.select_dtypes(exclude='number').shape[1])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "datas")
    args = parser.parse_args()
    for filename in ("clinical_features.csv", "ecg_features.csv", "labels.csv"):
        inspect(args.data / filename)
    holter = list(args.data.rglob("*.hea"))
    print(f"Holter headers: {len(holter)}")
    if holter:
        try:
            import wfdb

            for header in holter[:3]:
                record = wfdb.rdheader(str(header.with_suffix("")))
                print(
                    f"  {header.name}: channels={record.n_sig}, fs={record.fs}, "
                    f"samples={record.sig_len}, units={record.units}"
                )
        except ImportError:
            print("  SKIP header parsing: install wfdb")


if __name__ == "__main__":
    main()
