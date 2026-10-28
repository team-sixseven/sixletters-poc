"""Memory-mapped patient ECG dataset with fixed chronological window sampling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from ...data import load_patient_waveform


class PatientWindowDataset(Dataset):
    """Load only each patient's selected NPY rows per item; never split ECG windows by row."""

    def __init__(
        self,
        frame: pd.DataFrame,
        window_indices: dict,
        ecg_config: dict,
        max_segments: int | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.window_indices = window_indices
        self.ecg_config = ecg_config
        self.max_segments = max_segments

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        patient_id = str(row["patient_id"])
        if patient_id not in self.window_indices:
            raise KeyError(f"No validated SQI window index for patient {patient_id}")
        if len(self.window_indices[patient_id]) == 0:
            raise ValueError(f"{patient_id}: no valid ECG windows; patient must be excluded")
        waveform, row_indices, lead_mask = load_patient_waveform(
            patient_id,
            Path(row["ecg_path"]),
            self.window_indices[patient_id],
            self.ecg_config,
            return_lead_mask=True,
            max_windows=self.max_segments,
        )
        return (
            patient_id,
            torch.from_numpy(waveform),
            torch.tensor(float(row["y_true"]), dtype=torch.float32),
            torch.from_numpy(row_indices.astype("float32")),
            torch.from_numpy(lead_mask),
        )


def collate_patient_windows(batch):
    """Right-pad patient sequences and return an explicit validity mask and source positions."""
    patient_ids, waveforms, labels, positions, lead_masks = zip(*batch, strict=True)
    if not waveforms or any(w.ndim != 3 or min(w.shape) < 1 for w in waveforms):
        raise ValueError("Expected nonempty patient arrays (segments, measured_leads, samples)")
    samples = waveforms[0].shape[-1]
    if any(w.shape[-1] != samples for w in waveforms):
        raise ValueError(
            "Segments in a batch have different sample counts; no implicit cropping or resampling"
        )
    batch_size = len(waveforms)
    max_windows = max(w.shape[0] for w in waveforms)
    max_leads = max(w.shape[1] for w in waveforms)
    padded = waveforms[0].new_zeros((batch_size, max_windows, max_leads, samples))
    mask = torch.zeros((batch_size, max_windows), dtype=torch.bool)
    padded_positions = positions[0].new_zeros((batch_size, max_windows))
    lead_presence = torch.zeros((batch_size, max_leads), dtype=torch.bool)
    for i, (waveform, source_positions, measured) in enumerate(
        zip(waveforms, positions, lead_masks, strict=True)
    ):
        windows, leads, _ = waveform.shape
        if measured.shape != (leads,) or source_positions.shape != (windows,):
            raise ValueError(
                "Patient lead mask or original time indices disagree with waveform dimensions"
            )
        padded[i, :windows, :leads] = waveform
        padded_positions[i, :windows] = source_positions
        mask[i, :windows] = True
        lead_presence[i, :leads] = measured.to(dtype=torch.bool)
    return (
        list(patient_ids),
        padded,
        torch.stack(labels),
        mask,
        padded_positions,
        lead_presence,
    )
