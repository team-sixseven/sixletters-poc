# Clinical dataloader

`model.data.load_cohorts()` is the common loader used by ML, DL, fusion, and evaluation. Clinical data and team split are joined by normalized string patient ID, not row order. It checks required columns, duplicate IDs, binary labels, split IDs/values, configured feature names, and modality coverage.

Clinical-only ML uses the 508 clinical/split intersection (train 337, validation 80, test 91). Missing split membership is reported; no replacement split is generated. Training preprocessing is fit on train and reused unchanged for validation, test, and inference.
