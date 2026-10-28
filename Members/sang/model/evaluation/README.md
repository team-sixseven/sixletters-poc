# Common model evaluation

`evaluation.py` loads a previously saved ML, DL, or fusion run. It never refits model weights or preprocessing. Configure `evaluation/eval.yaml` with:

- `target`: `ml`, `dl`, or `fusion`
- ML: `model_name` and `artifact_dir: null` to read the selected model parameter YAML `model_path`; an explicit `artifact_dir` overrides it
- DL: `model_path` selects `../DL/models/<run_id>/best_model.pt` or its saved run directory. The legacy `artifact_dir` remains supported; leave it null when using `model_path`. Fusion uses `fusion_config`.
- `split`: `validation` or `test`
- optional `threshold_override` (leave `null` to use the threshold saved with the model)
- `results_directory`
- `mlflow_logging`: `true` uploads aggregate evaluation results after local saving; `false` keeps evaluation offline. The provided YAML enables it.

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

## MLflow 평가 기록

공통 평가 진입점은 ML·DL·Fusion 평가 완료 후 별도 MLflow run에 `test.*` 또는 `validation.*` 지표를 기록합니다. `stage=evaluation`과 모델의 `source_run_id`를 기록하며, 학습 모델 폴더의 `summary.json`에 MLflow 기록이 있으면 `source_mlflow_url`로 학습 run을 연결합니다. 평가 run에는 집계 지표·실행 설정·Fusion 가중치만 업로드하며 환자별 예측 CSV와 data audit은 로컬에 보관합니다. 모델 파일은 재업로드하지 않습니다.

전송 성공 여부와 링크는 결과 폴더의 `mlflow_status.json`에서 확인합니다. 실패해도 로컬 평가는 완료되며 `PENDING` 상태를 기록합니다. 평가를 다시 실행하지 않고 전송만 재시도할 수 있습니다.

```python
from pathlib import Path
from Members.sang.model.tracking.log_evaluation import log_evaluation_to_mlflow

evaluation_dir = Path("Members/sang/model/evaluation/results/ml/<평가 폴더>")
result = log_evaluation_to_mlflow(evaluation_dir)
print(result["url"])
```

같은 평가 폴더를 그대로 재전송하면 기존 run을 재사용합니다. 학습된 모델·threshold를 확정한 뒤 test 평가를 실행하세요. Test 평가가 학습이나 Optuna 탐색을 실행하지는 않습니다.

For ML, `model_name` is one of `XGBoost`, `CatBoost`, or `Random Forest` from `ML/model_choice.yaml`. A configured native file path must retain the companion run files and fitted preprocessing. Optuna's best-result cell records this path automatically in the selected parameter YAML. Evaluation reports a missing model instead of training one. To use another model-choice YAML, optionally set `evaluation.model_choice`; relative paths resolve from `eval.yaml`.

ML input versions are selected with optional `evaluation.model_version: 1`–`4`. The model parameter YAML stores version-specific paths in `model_paths`; a null version selects the latest `model_path`. Stored metadata and the selected feature list must match the requested version. ML prediction CSV and metrics include the version; all versions use the same patient split. An explicit `artifact_dir` selects that saved run directly and its metadata determines the actual inputs.

## Raw waveform DL evaluation

Raw DL artifacts (`metadata.json:data_source=raw_wave`) use `basic.yaml:data.raw_ecg` and `final_split_v2_augmented.csv`; the legacy SQI-window cohort is not substituted. Evaluation rejects changed split/quality checksums or segment/lead/normalization settings. Prediction CSVs include `used_window_indices` plus the number and first/last indices of selected segments. The raw source requires confirmed sample-rate/channel-order provenance and an aligned quality CSV. See [`../DL/TRAINING.md`](../DL/TRAINING.md) for setup and offline commands.


## Saved DL weights in YAML

Set these fields in the existing `eval.yaml` (keep the other fields):

```yaml
evaluation:
  target: dl
  model_path: ../DL/models/<run_id>/best_model.pt
  artifact_dir: null
  split: test
  threshold_override: null
  mlflow_logging: true
```

Replace `<run_id>` with a completed training run. A run directory is also accepted. The checkpoint must be the run's `best_model.pt`, with `metadata.json` and `resolved_config.yaml` beside it. A bare weight file without its architecture/preprocessing snapshot is insufficient. If both DL model_path and artifact_dir are set, they must select the same run; conflicting paths fail rather than silently choosing a model. Paths resolve relative to eval.yaml.

The evaluator restores saved weights/configuration, enters eval mode, and predicts without calling training. It records the actual checkpoint path in the output resolved_config.yaml. The model_path field applies only to DL; the existing ML saved-model selection is unchanged.

Run the common evaluation CLI above, `evaluation/evaluation_experiment.ipynb`, or the DL-only `evaluation/DL/evaluation_only_DL.ipynb`. The DL training notebook also uses its just-trained best_model.pt for the final evaluation automatically.
