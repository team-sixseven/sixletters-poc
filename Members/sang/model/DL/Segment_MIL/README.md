# Segment feature MIL

Segment Attention MIL and MIL-TCN receive `(batch, 8, 16)` features computed from the same eight selected NPY windows used by raw-waveform models. Features are per-lead mean, standard deviation, RMS, and range for each of three leads, plus the same four statistics for the three-lead vector magnitude. Their architecture settings are loaded from [`model_set.yaml`](model_set.yaml).
