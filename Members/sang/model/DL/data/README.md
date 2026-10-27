# Holter data

The shared source of truth is `../../basic.yaml`. Waveforms are read from `../../datas/Holter/windows/{pid}.npy`; row order, `start_sec`, and `sqi_pass` are validated using `../../datas/Holter/index/windows.csv.gz`.

P1030 has shape `(8250, 3, 2000)`, dtype float16, and 6,991 SQI-passing windows. Its `pid` is in the train split, but it has no row in `clinical_clean.csv`; it is only valid for unlabeled loading/forward checks. Never create or infer its SCD label.

The dataloader filters to `sqi_pass=1`, selects eight windows evenly over the recording while preserving time order, converts the selected windows to float32, then applies per-window/per-lead z-score. It does not alter the NPY file. The NPY source's earlier filter/normalization operations are not documented locally.
