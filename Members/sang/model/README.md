# SCD model workflow

This directory provides one configuration-driven workflow for clinical ML, Holter DL, saved-model fusion, and evaluation. Notebooks call the same Python modules as the command line. Data files are read-only; generated models, audits, and predictions stay local.

```text
model/
├── basic.yaml                         # shared data paths, ID/label schema, features, ECG policy
├── fusion.yaml                        # selected saved ML/DL runs, weights, threshold
├── ECG+clinical_experiment.ipynb      # saved-model fusion only; does not train base models
├── config.py                          # YAML-relative path resolution
├── data.py                            # ID joins, validation, exclusions, cohort audit
├── inference.py                       # saved ML/DL artifact load and probability prediction
├── fusion.py                          # validation-only weight fit and patient-level probability join
├── ML/
│   ├── dataloader/                    # clinical loader entry points
│   ├── data/                          # data use notes; no duplicate CSVs
│   ├── models.py                      # model-specific preprocessing and estimators
│   ├── model_choice.yaml              # models, preprocessing, selection policy
│   ├── model_experiment_only_ML.ipynb
│   ├── xgboost/{xgboost.py,parameter.yaml}
│   ├── catboost/{catboost.py,parameter.yaml}
│   ├── random_forest/{random_forest.py,parameter.yaml}
│   └── models/<run_id>/               # generated best model, preprocessing, config, audit, metrics
├── DL/
│   ├── dataloader/                    # memory-mapped patient/window dataset
│   ├── data/                          # waveform use notes
│   ├── composed.py                    # CNN window encoder + LSTM/Transformer sequence head
│   ├── models.py                      # CNN feature encoders
│   ├── model_choice.yaml              # selected composition and threshold
│   ├── training.yaml                  # optimizer, epochs, early stopping, device/workers
│   ├── model_experiment_only_DL.ipynb
│   ├── CNN/model_set.yaml             # ResNet, InceptionTime, TCN parameters
│   ├── LSTM/model_set.yaml            # Attention LSTM FCN, BiLSTM parameters
│   ├── Transformer/transformer.yaml   # transformer_enc parameters
│   └── models/<run_id>/               # generated checkpoint, config, audit, metrics, history
└── evaluation/
    ├── eval.yaml                      # model, split, threshold override, result location
    ├── evaluation.py                  # common saved-model evaluation CLI/module
    └── results/<target>/<run_id>/     # generated predictions, audit, metrics
```

## Verified data and cohort policy

`basic.yaml` is the sole source for data paths and schema. Relative paths resolve from `basic.yaml` itself, regardless of the current working directory. Current paths are:

- Clinical: `../datas/clincal/clinical_clean.csv`
- Split: `../datas/clincal/split_team_v1.csv`
- ECG: `../datas/Holter/windows/{pid}.npy`
- ECG window metadata: `../datas/Holter/index/windows.csv.gz` (used to align array rows, timestamps, and `sqi_pass`)

The audit rechecked 552 clinical rows (486 label 0, 66 label 1), 708 unique split IDs (460 train, 114 validation, 134 test), and 265 NPY files. Clinical/split intersection is 508: train 337, validation 80, test 91. Of those, 229 also have valid ECG and clinical labels: train 148, validation 36, test 45. These are the only patients eligible for ML+DL fusion. P1030 is assigned to train and has a valid NPY but has no clinical row or SCD label, so it is used only for unlabeled waveform/forward checks and excluded from supervised fitting.

The audit writes exact modality patient lists and exclusion reasons. ML uses clinical∩split. DL uses clinical∩split∩valid ECG. Fusion uses the intersection of ML and DL predictions. No patient is moved to a new split. Duplicate IDs, conflicting labels, malformed split values, missing configured features, array shape/dtype errors, and invalid selected windows fail clearly. NPY arrays are memory-mapped; only the configured eight windows are read per patient per batch.

## ECG sequence and preprocessing

The sidecar has `pid`, `row`, `win_idx`, `start_sec`, and per-window SQI fields. For the matched cohort, array lengths are checked against sidecar row counts and rows are validated in chronological order without overlap. The inspected records (including P1030) have 10-second `start_sec` intervals; other matched records are checked for chronological non-overlap but are not asserted to have an exact 10-second interval. The source MUSIC Holter recording is documented at 200 Hz, while NPY files themselves carry no sampling-rate metadata. See the [MUSIC source paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0186152).

The chosen model sequence is **CNN each selected window → ordered window embeddings → LSTM or Transformer**. It does not flatten the whole NPY. Eight `sqi_pass=1` windows are selected evenly through the recording and kept in chronological order. The dataloader converts to float32 and applies per-window, per-lead z-score normalization. The upstream filtering/normalization history of the NPY values is not recorded in this repository; the local dataloader normalization is explicit and configurable in `basic.yaml`.

The SCD column is used as the supplied binary label without reassigning outcomes. MUSIC publications describe enrollment-time clinical predictors and SCD events observed during follow-up (median follow-up about 44 months); the supplied clean CSV has no event horizon/censoring columns, so these experiments do not claim a fixed-horizon probability or survival analysis. See the [MUSIC study description](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0186152) and [PhysioNet dataset page](https://physionet.org/content/music-sudden-cardiac-death/1.0.1/).

## Run

From the repository root, using the `Global` project environment:

```bash
uv run --project Global --extra ml python Members/sang/scripts/audit_model_data.py
uv run --project Global --extra ml python Members/sang/scripts/evaluate_preprocessed_clinical.py
uv run --project Global --extra gpu python Members/sang/scripts/evaluate_ecg_architectures.py
uv run --project Global python -m Members.sang.model.evaluation.evaluation
```

The compatibility-named clinical and ECG scripts now call the configured trainers. Training selects checkpoints using validation Average Precision and does not evaluate test. Small real-data smoke options are available as `--limit-train-patients 32` for ML and `--max-epochs 1 --limit-train-patients 16 --device cpu` for DL. These produce test-only local artifacts and are not full cohort models.

Run all nine supported encoder/head combinations on real, unlabeled P1030 windows:

```bash
uv run --project Global --extra gpu python Members/sang/scripts/smoke_p1030_holter.py
```

## YAML and artifacts

- Edit data paths, ID/label columns, split mapping, feature list, filename pattern, and ECG window policy in `basic.yaml`.
- Select XGBoost, CatBoost, and/or Random Forest in `ML/model_choice.yaml`. Change each model's hyperparameters and `classification.threshold` in its `parameter.yaml`. Random Forest uses train-fitted imputation/one-hot preprocessing and has no early stopping; its validation AP is recorded for comparison.
- ML별 `model`, `training`, `classification` parameter key의 역할은 [`ML/PARAMETER_GUIDE.md`](ML/PARAMETER_GUIDE.md)에 정리했습니다.
- Select `combination`, `cnn_model`, `sequence_model`, and the DL `classification.threshold` in `DL/model_choice.yaml`. Adjust CNN/LSTM/Transformer parameters in their referenced model YAMLs. Training settings live in `DL/training.yaml`.
- Each ML candidate writes a unique run to `ML/models/<run_id>/`; DL writes to `DL/models/<run_id>/`. Existing run directories are never overwritten. XGBoost and CatBoost select their best iteration by validation AP; Random Forest uses its configured tree count and records validation AP for candidate comparison. Test scoring is a separate evaluation step.
- `fusion.yaml` chooses saved model directories. `mode: manual` checks weights are nonnegative and sum to one; `mode: learn` fits only the scalar fusion weights on validation probabilities; `mode: saved` reuses weights saved by a prior fusion run. Base models and preprocessors are not refit during fusion.
- `evaluation/eval.yaml` selects `ml`, `dl`, or `fusion`, an explicit validation/test split, optional threshold override, and output path. Threshold overrides change binary decisions and threshold-dependent metrics without retraining or changing saved probabilities.

Evaluation outputs include patient ID, split, target, positive probability, actual threshold, binary decision, and model/run ID. Fusion additionally writes both input probabilities, final probability, and weights. Average Precision is named explicitly; AUROC is calculated from probabilities. Metrics undefined for a single-class split are recorded as unavailable with a reason.

**Do not select a model, epoch, fusion weight, or threshold after inspecting test performance.** A validation score used for checkpoint or fusion-weight selection is not an independent final estimate.

## MLflow handoff

MLflow 담당자용 절차와 로컬 bundle 생성 명령은 [`tracking/MLFLOW_HANDOFF.md`](tracking/MLFLOW_HANDOFF.md)에 있습니다. Bundle 필드, prediction CSV, artifact checksum 및 업로드 경계는 [`tracking/BUNDLE_CONTRACT.md`](tracking/BUNDLE_CONTRACT.md)를 참고하세요. Exporter는 완료된 전체 학습 run을 사용하며 자동 업로드하지 않습니다.

## 실험 순서도

임상 ML, Holter DL, 임상+Holter Fusion의 모델 분기와 tensor 흐름은 [`EXPERIMENT_FLOWCHARTS.md`](EXPERIMENT_FLOWCHARTS.md)의 SVG 구성도에서 확인할 수 있습니다.
