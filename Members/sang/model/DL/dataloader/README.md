# Patient Holter dataloader

`dataset.py` memory-maps one patient NPY at a time and returns one fixed patient sample `(8 windows, 3 leads, 2000 samples)`. The window dimension remains a sequence dimension. Windows are never randomly divided across splits; all split assignment comes from `basic.yaml`'s patient-level split CSV.

Before training, shared `model.data.load_cohorts()` validates clinical/split/ECG patient IDs, NPY headers and sidecar row alignment, then writes audit files. The selected eight windows must all pass `sqi_pass`; invalid/too-short records are excluded with a reason. Each selected window is normalized independently per lead in memory. Training and inference use the same loader.

Run the actual unlabeled P1030 input check and all supported CNN+LSTM/Transformer combinations with:

```bash
uv run --project Global --extra gpu python Members/sang/scripts/smoke_p1030_holter.py
```
