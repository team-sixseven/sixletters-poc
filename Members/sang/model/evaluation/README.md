# Common model evaluation

`evaluation.py` loads a previously saved ML, DL, or fusion run. It never refits model weights or preprocessing. Configure `evaluation/eval.yaml` with:

- `target`: `ml`, `dl`, or `fusion`
- the saved artifact path or `fusion.yaml`
- `split`: `validation` or `test`
- optional `threshold_override` (leave `null` to use the threshold saved with the model)
- `results_directory`

Run from repository root:

```bash
uv run --project Global python -m Members.sang.model.evaluation.evaluation
```

CLI overrides are available for a one-off run:

```bash
uv run --project Global python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml --split test --threshold-override 0.65
```

Threshold overrides use `p_positive >= threshold`; saved probabilities are not changed. Result directories are unique and contain `predictions.csv`, `metrics.json`, `resolved_config.yaml`, and data audit files. Metrics include AUROC, Average Precision (sklearn `average_precision_score`), Brier, accuracy, precision, recall/sensitivity, specificity, F1, and confusion matrix. If a metric cannot be computed for a one-class split, it is `null` with a reason rather than a fabricated zero.

Training and model/threshold selection use validation only. Test evaluation is a separate explicit step. A validation score is not an independent performance estimate when it selected the model or learned fusion weights.
