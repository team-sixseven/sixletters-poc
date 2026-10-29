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
│   ├── data/                          # clinical CSV and patient split CSV
│   ├── models.py                      # model-specific preprocessing and estimators
│   ├── feature_versions.py            # clinical input sets for model_version 1-4
│   ├── artifacts.py                   # native model loading and version-specific paths
│   ├── model_choice.yaml              # selected models, parameter paths, output directory
│   ├── model_experiment_only_ML.ipynb
│   ├── xgboost/{xgboost.py,parameter.yaml}
│   ├── catboost/{catboost.py,parameter.yaml}
│   ├── random_forest/{random_forest.py,parameter.yaml}
│   └── models/best/<algorithm>_vN/    # Optuna winner only; ordinary training uses models/<run_id>/
├── DL/
│   ├── dataloader/                    # memory-mapped patient/window dataset
│   ├── data/                          # waveform use notes
│   ├── composed.py                    # CNN window encoder + LSTM/Transformer sequence head
│   ├── models.py                      # CNN feature encoders
│   ├── model_choice.yaml              # selected CNN and temporal model only
│   ├── raw_pipeline.py                # raw waveform training, artifact checks and inference
│   ├── TRAINING.md                    # raw training/evaluation steps and verified limits
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

## Current DL configuration and lead handling

The default raw DL path uses the connected Drive `final_split_augmented.csv`, its 4-year SCD labels, sample-major NPY files, segment quality index, and measured-lead metadata. A single shared one-lead CNN processes every measured lead, then averages measured-lead features per segment before LSTM/Transformer → masked attention MIL → MLP. The Drive source documents X/Y/Z and identifies missing Z; the encoder pools measured leads without assuming lead identity is interchangeable across patients. Batch lead padding and time padding use independent masks and never enter CNN/BatchNorm. Record-length limits are separate from lead count.

Model selection YAML has two keys only. Each model YAML contains flat experiment hyperparameters; no implementation path, input/output shape, feature method or fixed channel count. All models expose structural feature dimensions and dimensions are connected automatically. See [`DL/TRAINING.md`](DL/TRAINING.md) for exact examples and checkpoint format v2 migration. Old fixed-lead DL checkpoints require their original code or retraining; ML model formats are unchanged.

The legacy SQI source described below is separate from raw 30-second waveforms. Its former 2000 samples correspond to the documented 200 Hz/10-second setup, not the raw input length. Current raw sample rate still requires provenance confirmation.

## Verified data and cohort policy

`basic.yaml` declares data paths and essential schema; `config.py:load_basic_config()` fills shared defaults for split/sidecar columns, filenames, ECG policies, and audit output paths. Relative paths resolve from `basic.yaml` itself, regardless of the current working directory. Current paths are:

- Clinical: connected Drive `DATA/clincal/clinical_clean.csv`
- Split: connected Drive `DATA/final_split.csv` (`val` maps to the internal `validation` split)
- ECG: `../datas/Holter/windows/{pid}.npy`
- ECG window metadata: `../datas/Holter/index/windows.csv.gz` (used to align array rows, timestamps, and `sqi_pass`)

The current `final_split.csv` contains 508 unique patient IDs: 325 train, 81 validation (`val` in the CSV), and 102 test. All 508 IDs match the clinical CSV and their split-file SCD labels agree with the clinical SCD labels. Older ECG cohort counts below this section were produced with a previous split and should be refreshed by rerunning the data audit before relying on modality-specific DL/fusion counts.

The audit writes exact modality patient lists and exclusion reasons. ML uses clinical∩split. DL uses clinical∩split∩valid ECG. Fusion uses the intersection of ML and DL predictions. No patient is moved to a new split. Duplicate IDs, conflicting labels, malformed split values, missing configured features, array shape/dtype errors, and invalid selected windows fail clearly. NPY arrays are memory-mapped; only the deterministically selected segments are read per patient per batch.

## ECG sequence and preprocessing

The sidecar has `pid`, `row`, `win_idx`, `start_sec`, and per-window SQI fields. For the matched cohort, array lengths are checked against sidecar row counts and rows are validated in chronological order without overlap. The inspected records (including P1030) have 10-second `start_sec` intervals; other matched records are checked for chronological non-overlap but are not asserted to have an exact 10-second interval. The source MUSIC Holter recording is documented at 200 Hz, while NPY files themselves carry no sampling-rate metadata. See the [MUSIC source paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0186152).

The chosen model sequence is **shared per-lead CNN → measured-lead mean per segment → LSTM or Transformer → masked MIL → MLP**. It does not flatten the whole NPY. `sqi_pass=1` windows are selected up to `training.yaml:max_segments_per_patient` evenly through the recording and kept in chronological order. The dataloader converts to float32 and applies per-window, per-lead z-score normalization. The upstream filtering/normalization history of the NPY values is not recorded in this repository; the local dataloader normalization is explicit in the shared defaults in `config.py`.

The SCD column is used as the supplied binary label without reassigning outcomes. MUSIC publications describe enrollment-time clinical predictors and SCD events observed during follow-up (median follow-up about 44 months); the supplied clean CSV has no event horizon/censoring columns, so these experiments do not claim a fixed-horizon probability or survival analysis. See the [MUSIC study description](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0186152) and [PhysioNet dataset page](https://physionet.org/content/music-sudden-cardiac-death/1.0.1/).

## Run

ECG + clinical Fusion의 환경 준비, 저장 모델 선택, validation 가중치 설정·학습, 최종 test와 MLflow 기록은 [fusion_manu.md](fusion_manu.md)를 순서대로 따라 실행합니다.

DL 학습 smoke, 전체 학습, checkpoint 확인, test 평가의 단계별 명령은 [`DL/TRAINING.md`](DL/TRAINING.md)에 정리했습니다. 기본 raw source는 연결된 Drive의 200Hz X/Y/Z 파형·구간 품질 표·유도 정보를 사용합니다. Drive 분할과 기존 임상 분할이 달라 결과를 결합하기 전에 대상·라벨·환자 분할의 일치를 확인해야 합니다. 기존 SQI-window trainer도 별도 경로로 유지합니다.

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

## ML Optuna 실행과 model_version 선택

사용할 노트북은 [`ML/model_experiment_only_ML.ipynb`](ML/model_experiment_only_ML.ipynb)입니다. 아래 순서로 **알고리즘 하나 + 입력 버전 하나**씩 실험합니다. 노트북을 처음 열거나 코드가 갱신된 경우 커널을 재시작한 뒤 공통 import 셀부터 실행합니다.

현재 호스트의 기본 uv와 프로젝트 요구 버전이 다르면 저장소 루트에서 다음 명령으로 실행합니다. 커널은 `Global/.venv`의 Python을 선택합니다.

```bash
uvx --from uv==0.9.26 uv run \
  --project Global --locked --extra ml --inexact jupyter lab
```

1. **첫 코드 셀**을 실행합니다. `Members import 준비 완료`가 표시되면 다음으로 진행합니다.
2. **입력 변수 버전 선택** 셀에서 `model_version`을 지정하고 실행합니다. 실제 입력 열 목록과 개수를 확인합니다.

   ```python
   model_version = 1  # 1, 2, 3, 4 중 선택; 노트북 기본값은 4
   ```

   | model_version | 입력 변수 수 | 구성 |
   |---|---:|---|
   | 1 | 4 | Gender, Age, LVEF, NYHA |
   | 2 | 8 | 버전 1 + eGFR, Ischemic etiology, Albumin, Diabetes |
   | 3 | 21 | 버전 2 + 제공된 심장 구조·기능, 병력, 혈압, Pro-BNP, Cardiothoracic ratio |
   | 4 | 23 | 버전 3 + cigarettes /year, alcohol consumption |

   정확한 열 이름과 순서는 [`ML/feature_versions.py`](ML/feature_versions.py)에 있습니다. `model_version`은 입력 묶음 번호이고, `model_name`은 학습 알고리즘 이름입니다.

3. 입력 버전과 함께 `training_mode="optuna"`, 알고리즘과 trial 횟수, MLflow/평가 플래그를 지정합니다. 일반 학습은 `training_mode="yaml"`로 선택합니다.

   ```python
   training_mode = "optuna"  # "yaml" 또는 "optuna"
   model_name = "XGBoost"  # Optuna 알고리즘
   n_trials = 100
   mlflow_logging = True
   run_final_test = True  # 후보 비교·튜닝 중이면 False
   ```

4. 위에서 아래로 전체 셀을 실행합니다. `training_mode`로 선택한 방식만 학습·기록하고, 마지막 셀은 이번에 선택한 모델을 직접 test 평가·기록합니다. 전송 재시도 셀은 기본 비활성입니다.
5. 결과 확인 셀에서 `best_run_dir`, `model_version`, `feature_names`, validation AP를 확인합니다. 저장과 YAML 경로 기록은 탐색 실행 셀이 성공적으로 끝날 때 자동 완료됩니다. trial 중에는 원본 YAML을 갱신하지 않습니다.

알고리즘이나 버전을 바꿀 때는 입력 버전 선택 셀과 Optuna 준비 셀부터 다시 실행합니다. **3개 알고리즘 × 4개 버전 = 12개 조합은 각각 별도 study로 실행**합니다. 한 번 실행해서 12개를 자동 탐색하는 노트북은 아닙니다. `n_trials`는 조합별 횟수이고 기본 100회는 시작값이며 최적값 발견을 보장하는 횟수는 아닙니다.

**Optuna는 조합별 최적 모델 한 개만 보관합니다.** 저장 위치는 `ML/models/best/<알고리즘>_v<버전>/`이며, 예를 들어 `best/xgboost_v4/`입니다. 같은 조합의 다음 탐색이 성공하면 해당 폴더를 새 study의 best 결과로 갱신합니다. 다른 알고리즘·버전은 유지합니다. 현재 study의 `trials.csv`에는 모든 trial의 파라미터·AP·상태를 기록하지만 trial 모델은 보관하지 않습니다.

`best_parameters.yaml`, `summary.json`, 평가에 필요한 전처리·metadata·설정·validation 예측·지표·완료 상태도 같은 폴더에 저장합니다. trial별 학습 이력과 반복되는 audit 파일은 영구 보관하지 않습니다. 학습 중 모델은 임시 폴더를 사용하며 탐색 완료·실패 시 정리됩니다. 실패하면 이전에 완료된 best 모델은 보존됩니다. 과거 코드로 이미 만든 날짜별 폴더와 검증용 폴더는 자동 삭제하지 않습니다.

이 정책은 Optuna 경로에 적용됩니다. 일반 `train_clinical()`의 고정값 학습은 기존 `models/<run_id>/` 저장 방식을 유지합니다. Optuna를 선택하면 일반 학습 셀은 자동으로 건너뜁니다. 지정한 버전과 입력 열이 다른 모델은 재사용하지 않습니다.

모든 조합은 같은 train/validation/test 환자 배정을 사용합니다. 노트북은 `mlflow_logging=True`로 학습을 자동 기록하고 `run_final_test=True`일 때 마지막 셀에서 test 평가도 기록합니다. 후보 비교 중에는 test를 끕니다. 별도 재평가 시 [`evaluation/eval.yaml`](evaluation/eval.yaml)의 기존 설정에서 아래 항목을 맞춥니다.

```yaml
evaluation:
  target: ml
  model_name: XGBoost
  model_version: 1
  artifact_dir: null
  split: test
```

`artifact_dir: null`이면 모델 YAML의 해당 버전 경로를 읽으며 평가 단계에서는 재학습하지 않습니다. 자세한 실행·재평가 방법은 [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md)를 참고하세요. 현재까지 검증한 범위는 실제 임상 데이터의 12개 조합 소규모 학습·저장·재로드, Optuna의 조건부 탐색·best 단일 저장·버전별 validation 평가입니다. 본격적인 12개 조합 Optuna 탐색과 최종 test 평가는 별도로 실행합니다.

## YAML and artifacts

- Edit data paths, clinical ID/label columns and verified sample rate in `basic.yaml`; actual array dimensions are inspected at runtime and segment cap is in `DL/training.yaml`. Categorical columns, fixed data conventions, and audit paths default in `config.py`; optional YAML overrides are supported.
- Select XGBoost, CatBoost, and/or Random Forest in `ML/model_choice.yaml`. Change each model's hyperparameters and `classification.threshold` in its `parameter.yaml`. Clinical features are required to be complete; ML does not impute missing values. XGBoost and Random Forest apply train-fitted scaling/one-hot encoding, while Random Forest has no early stopping and records validation AP for comparison.
- Model parameter YAMLs include `model_path`. Standard ML execution reuses an existing complete run or trains from YAML when absent. Optuna trains temporary trials, retains one winner per algorithm/version under `models/best/`, and records its path after successful study completion. ML evaluation can use `model_name` with `artifact_dir: null` to read that path without retraining.
- ML별 `model`, `training`, `classification` parameter key의 역할은 [`ML/ML_manu.md`](ML/ML_manu.md)에 정리했습니다.
- `DL/model_choice.yaml` contains only `cnn_model` and `sequence_model`. Flat model hyperparameters are in the existing CNN/LSTM/Transformer YAMLs; `classification_threshold`, MIL/MLP, batch size and segment cap are in `DL/training.yaml`. Data and `outputs.dl_model_directory` are in `basic.yaml`. Classes, feature interfaces and dimension wiring are managed by code.
- Ordinary ML training writes a unique run to `ML/models/<run_id>/`; DL writes to `DL/models/<run_id>/`. Optuna updates its managed `ML/models/best/<algorithm>_vN/` directory only after a successful search. XGBoost and CatBoost select their best iteration by validation AP; Random Forest uses its configured tree count and records validation AP for candidate comparison. Test scoring is a separate evaluation step.
- `fusion.yaml` chooses saved model directories. `mode: manual` checks weights are nonnegative and sum to one; `mode: learn` fits only the scalar fusion weights on validation probabilities; `mode: saved` reuses weights saved by a prior fusion run. Base models and preprocessors are not refit during fusion.
- `evaluation/eval.yaml` selects `ml`, `dl`, or `fusion`, an explicit validation/test split, optional threshold override, and output path. Threshold overrides change binary decisions and threshold-dependent metrics without retraining or changing saved probabilities.

Evaluation outputs include patient ID, split, target, positive probability, actual threshold, binary decision, and model/run ID. Fusion additionally writes both input probabilities, final probability, and weights. Average Precision is named explicitly; AUROC is calculated from probabilities. Metrics undefined for a single-class split are recorded as unavailable with a reason.

**Do not select a model, epoch, fusion weight, or threshold after inspecting test performance.** A validation score used for checkpoint or fusion-weight selection is not an independent final estimate.

## 이 PC의 MLflow 자동 기록

팀원 PC의 개인 계정과 Lab/AX Wi-Fi·유선 접속 설정은 [`tracking/TEAM_MLFLOW.md`](tracking/TEAM_MLFLOW.md)를 따릅니다. Optuna 준비 셀에서 실제 사용할 계정·서버 연결을 먼저 확인합니다.

저장 모델 평가도 `evaluation/eval.yaml`의 `evaluation.mlflow_logging: true`로 자동 기록합니다. 평가 노트북 또는 공통 평가 CLI를 명시적으로 실행하면 별도 run에 `test.*`/`validation.*` 지표를 저장하고 학습 run의 ID·링크를 남깁니다. 환자별 예측·audit은 로컬에만 보관하며, 전송 실패 시 재시도 방법은 [`tracking/LOCAL_MLFLOW.md`](tracking/LOCAL_MLFLOW.md)를 참고하세요.

`http://127.0.0.1:5000`에서 로컬 서버를 실행합니다. ML 노트북의 `mlflow_logging=True` 설정으로 Optuna 탐색 완료 후 최적 모델 하나와 파라미터·validation 지표를 `sang-scd-ml` 실험에 기록합니다. 로그인은 `sang` 계정이며 비밀번호·접속 정보는 Git 제외 파일 `.poc.local.toml`에 있습니다.

노트북 출력 링크 또는 best 폴더 `summary.json`의 `mlflow` 항목에서 결과를 확인합니다. 전송 실패 시 로컬 best를 보존하고 `PENDING`을 표시하며, **MLflow 전송 재시도** 셀로 재학습 없이 다시 전송할 수 있습니다. 각 trial의 모델이나 환자별 예측 CSV는 자동 전송하지 않습니다. 실행·재시도·서버 관리 방법은 [`tracking/LOCAL_MLFLOW.md`](tracking/LOCAL_MLFLOW.md)에 있습니다.

DL 노트북과 통합 fusion 노트북도 각 PC의 개인 계정 설정을 자동 사용합니다. 기본 `mlflow_logging=True`, `run_final_test=True`는 학습/validation 결과와 고정 모델의 test 평가를 별도 MLflow run으로 기록합니다. 모델 비교 중에는 test를 끄고, 오프라인 실행은 logging을 끕니다. 계정 등록·재시도·업로드 범위는 [`tracking/TEAM_MLFLOW.md`](tracking/TEAM_MLFLOW.md)와 [`DL/TRAINING.md`](DL/TRAINING.md)를 참고하세요. 실제 raw 데이터 학습 조건(metadata·품질 index)은 동일하게 적용됩니다.

## MLflow handoff

MLflow 담당자용 절차와 로컬 bundle 생성 명령은 [`tracking/MLFLOW_HANDOFF.md`](tracking/MLFLOW_HANDOFF.md)에 있습니다. Bundle 필드, prediction CSV, artifact checksum 및 업로드 경계는 [`tracking/BUNDLE_CONTRACT.md`](tracking/BUNDLE_CONTRACT.md)를 참고하세요. Exporter는 완료된 전체 학습 run을 사용하며 자동 업로드하지 않습니다.

## 실험 순서도

임상 ML, Holter DL, 임상+Holter Fusion의 모델 분기와 tensor 흐름은 [`EXPERIMENT_FLOWCHARTS.md`](EXPERIMENT_FLOWCHARTS.md)의 SVG 구성도에서 확인할 수 있습니다.
