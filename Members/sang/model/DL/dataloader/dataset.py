"""Memory-mapped patient ECG dataset with fixed chronological window sampling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from ...data import load_patient_waveform


class PatientWindowDataset(Dataset):
    """Load only each patient's selected NPY rows per item; never split ECG windows by row."""

    def __init__(self, frame: pd.DataFrame, window_indices: dict, ecg_config: dict) -> None:
        self.frame = frame.reset_index(drop=True)
        self.window_indices = window_indices
        self.ecg_config = ecg_config

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        patient_id = str(row["patient_id"])
        if patient_id not in self.window_indices:
            raise KeyError(f"No validated SQI window index for patient {patient_id}")
        waveform, _ = load_patient_waveform(
            patient_id,
            Path(row["ecg_path"]),
            self.window_indices[patient_id],
            self.ecg_config,
        )
        return (
            patient_id,
            torch.from_numpy(waveform),
            torch.tensor(float(row["y_true"]), dtype=torch.float32),
        )
