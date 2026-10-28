# Patient-level ECG loading

Raw files are memory-mapped sample-major `(sample, measured_leads)` arrays. The current real files have 3 columns and float16 dtype. `raw_waveform.py` reads patient splits and explicit original 30-second segment quality indices; rate/axis provenance and quality metadata remain required. It reads only deterministically selected usable segments and keeps their original positions. No lead identity is inferred from numeric column indices.

Datasets return natural `(segments, measured_leads, samples)` arrays and a measured-lead mask. `dataset.py:collate_patient_windows` pads the time and lead dimensions independently to each batch maximum and returns separate boolean window/lead masks. It rejects inconsistent sample counts instead of cropping/resampling. The composed model sends only valid (window, measured lead) pairs to its shared one-lead CNN, so both forms of padding are excluded from BatchNorm and feature pooling.

The legacy `PatientWindowDataset` reads the separate already segmented SQI source, preserves the actual NPY sample/lead dimensions, and normalizes each selected window per lead. Its cap comes from `training.yaml:max_segments_per_patient`. Legacy 2000-sample/10-second arrays are not used to define the raw 30-second input shape. Their `sqi_pass` index must not be reused for raw files.
