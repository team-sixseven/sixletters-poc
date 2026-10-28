# 모델 학습·평가 가이드

이 문서는 현재 구현된 노트북, YAML, Python 진입점 기준으로 작성했습니다. 권장 순서는 **공통 준비 → clinical ML → ECG DL → ECG+clinical 결합 → 고정한 설정으로 test 평가**입니다. 이미 ML과 DL 모델 artifact가 있으면 결합부터 진행할 수 있습니다.

| 실험 | 노트북 | 설정할 YAML | 선행 학습 | 모델·결과 저장 위치 |
|---|---|---|---|---|
| Clinical ML | [`ML/model_experiment_only_ML.ipynb`](ML/model_experiment_only_ML.ipynb) | `basic.yaml`, `ML/model_choice.yaml`, 선택한 모델의 `parameter.yaml` | 없음 | `ML/models/best/<algorithm>_vN/` (Optuna), `ML/models/<run_id>/` (일반 학습) |
| ECG DL | [`DL/model_experiment_only_DL.ipynb`](DL/model_experiment_only_DL.ipynb) | `basic.yaml`, `DL/model_choice.yaml`, 선택한 architecture YAML, `DL/training.yaml` | 없음 | `DL/models/<run_id>/` |
| ECG + clinical 결합 | [`ECG+clinical_experiment.ipynb`](ECG+clinical_experiment.ipynb) | `basic.yaml`, `fusion.yaml` | ML·DL artifact 필요 | `evaluation/results/fusion/<run_id>/` |
| 저장 모델 평가 | [`evaluation/evaluation_experiment.ipynb`](evaluation/evaluation_experiment.ipynb) 또는 평가 CLI | `basic.yaml`, `evaluation/eval.yaml`, 결합 평가 시 `fusion.yaml` | 평가할 모델 artifact 필요 | `evaluation/results/<ml|dl|fusion>/<run_id>/` |

## 1. 공통 준비

### 1.1 저장소 루트와 Python 환경

모든 아래 명령은 `sixletters-poc/` 저장소 루트에서 실행합니다. `Global/pyproject.toml`과 `Global/uv.lock`이 Python 의존성 기준이며, 프로젝트는 Python 3.12와 `uv==0.9.26`을 요구합니다. ML extra에는 XGBoost·CatBoost·Optuna, GPU extra에는 Linux x86_64용 PyTorch가 선언되어 있습니다. JupyterLab과 ipykernel은 기본 의존성입니다.

현재 작업 환경의 `Global/.venv`가 이미 준비되어 있다면 패키지를 다시 설치할 필요는 없습니다. 새로 만들거나 lockfile과 동기화해야 하는 환경에서는 저장소가 고정한 uv 버전을 사용해 다음을 실행합니다.

```bash
uv self update 0.9.26
uv sync --project Global --locked --extra ml --extra gpu
```

`gpu` extra의 PyTorch 선언은 Linux x86_64에만 적용됩니다. 이 저장소 의존성 설정에는 macOS용 CPU PyTorch extra가 따로 없으므로, macOS에서 DL 노트북을 실행하려면 사용할 Python 환경에 호환 PyTorch가 이미 설치되어 있어야 합니다. ML에는 GPU extra가 필요하지 않습니다.

`uv run --project Global`은 `uv` 버전이 프로젝트 요구사항과 맞아야 동작합니다. 이 문서를 확인한 현재 호스트는 uv 0.12.3이어서 해당 명령이 버전 불일치로 중단됐습니다. 이 호스트에서 이미 설치된 환경을 재사용하려면 저장소 루트에서 `.venv` 실행 파일을 직접 사용하세요.

```bash
Global/.venv/bin/python -m ipykernel install --user \
  --name sixletters-poc-global \
  --display-name "Python (sixletters-poc Global)"
Global/.venv/bin/jupyter lab
```

Jupyter에서 위에 등록한 **Python (sixletters-poc Global)** 커널을 선택합니다. JupyterLab은 저장소 루트에서 시작하는 것을 권장합니다. ML 노트북의 공통 import 셀은 현재 작업 디렉터리에서 저장소 루트를 찾고 `sys.path`에 등록하므로, 커널 작업 디렉터리가 `ML/`이어도 `Members` import를 사용할 수 있습니다. 다른 노트북은 해당 import 경로 설정을 확인하세요.

uv 버전을 맞춘 환경에서는 같은 작업을 다음처럼 실행할 수 있습니다.

```bash
uv run --project Global jupyter lab
```

### 1.2 공통 데이터 경로와 스키마

데이터 경로와 주요 스키마는 [`basic.yaml`](basic.yaml)에 있고, 생략된 공통 규칙은 `config.py`의 `load_basic_config()`가 적용합니다. 상대 경로는 **해당 설정을 선언한 YAML 파일의 디렉터리** 기준입니다. 따라서 `basic.yaml`의 `../datas/...`는 `Members/sang/model/`이 아니라 `Members/sang/datas/` 아래를 가리킵니다. 노트북의 실행 위치에 따라 데이터 경로가 바뀌지 않습니다.

기본 데이터 설정은 다음과 같습니다.

```yaml
data:
  clinical:
    path: ML/data/clinical_clean.csv
    id_column: Patient ID
    label_column: SCD
  split:
    path: ML/data/final_split.csv
  ecg:
    directory: ../datas/Holter/windows
    metadata_path: ../datas/Holter/index/windows.csv.gz
    sample_rate_hz: 200
    leads: 3
    samples_per_window: 2000
    windows_per_patient: 8
```

경로를 옮겼다면 이 파일의 `path` 또는 `directory` 값을 실제 위치에 맞게 바꿉니다. 입력 feature는 임상 CSV에서 설정된 환자 ID와 라벨 열을 제외한 모든 열을 CSV 순서대로 사용합니다. CSV에는 입력 변수와 ID·라벨만 두고 split 정보는 별도 split CSV로 관리합니다. 범주형 열 7개는 `config.py`의 공통 기본 목록을 사용합니다. 임상 CSV 열 이름은 메모리에서 앞뒤 공백을 정리하며, 이 과정에서 중복 열 이름이 생기면 로딩을 중단합니다. 원본 데이터 파일은 수정하지 않습니다.

Split 열 기본값은 `pid`·`split`이며 CSV의 `val`을 내부 `validation`으로 매핑합니다. sidecar의 `pid`·`row`·`start_sec`·`sqi_pass`, 파일명 규칙, SQI 필터와 정규화는 `config.py`의 공통 기본값입니다.

ECG 파일은 `{pid}.npy` 형식이며 환자 ID 정규식은 `P` 다음 네 자리 숫자입니다. 현재 설정상 배열은 `[window, lead, sample]`, 3개 lead, window당 2000 samples입니다. 200 Hz는 원본 기록 사양에서 가져온 가정이며 NPY 자체에 sampling rate 메타데이터가 저장된 것은 아닙니다. 배열 행은 `windows.csv.gz` sidecar의 환자·row 정보로 맞춥니다. `sqi_pass=0` 구간은 DL 학습에서 제외합니다.

NPY에 dataloader 이전의 필터링·정규화가 적용되었는지는 현재 저장소에서 확인할 수 없습니다. 입력 시 dataloader가 `config.py` 기본값인 `per_window_per_lead_zscore`를 적용하는 것은 확인된 동작입니다.

현재 audit에서 확인된 규모는 임상 552명, split 708명, NPY 265개입니다. clinical∩split은 508명이고 ML 대상은 train 337명, validation 80명, test 91명입니다. 임상 라벨·split·ECG가 모두 연결된 DL 대상은 총 229명(train 148, validation 36, test 45)입니다. P1030은 split에는 있지만 임상 라벨이 없어 지도학습 대상에 포함되지 않습니다.

### 1.3 데이터 정합성 검사

저장소 루트에서 실행합니다. 이 audit는 기본 설정의 CSV, split, ECG 파일 헤더와 sidecar 정합성 및 modality별 환자 범위를 검사합니다. 학습은 수행하지 않습니다.

```bash
uv run --project Global python Members/sang/scripts/audit_model_data.py
```

이미 준비된 현재 호스트 환경을 그대로 쓰는 경우:

```bash
Global/.venv/bin/python Members/sang/scripts/audit_model_data.py
```

정상 실행 시 터미널에 `clinical_patients`, `split_patients`, `ecg_patients`, 교집합·제외 환자 수, split별 환자 및 라벨 수가 출력됩니다. 상세 audit 파일은 기본적으로 `model/reports/data_audit/` 아래에 run ID별로 생성됩니다. 경로를 바꾸려면 `basic.yaml`에 `outputs.audit_directory`를 지정합니다. 환자 ID가 포함된 local 산출물이므로 외부 공유에 주의하세요. 예상보다 환자 수가 적거나 필수 데이터 파일을 못 찾으면 학습으로 넘어가지 말고 경로와 ID·split 매핑부터 확인합니다.

## 2. Clinical 데이터만 사용하는 ML

### 2.1 모델과 파라미터 선택

기본 선택은 XGBoost, CatBoost, Random Forest입니다. 사용할 모델은 [`ML/model_choice.yaml`](ML/model_choice.yaml)의 `models`에서 선택하고, 같은 파일의 `model_configs`가 각 모델 parameter YAML 경로를 연결합니다.

XGBoost만 실행하는 예:

```yaml
models:
  - XGBoost
```

CatBoost만 실행하려면 다음처럼 바꿉니다.

```yaml
models:
  - CatBoost
```

XGBoost hyperparameter와 threshold는 [`ML/xgboost/parameter.yaml`](ML/xgboost/parameter.yaml), CatBoost 설정은 [`ML/catboost/parameter.yaml`](ML/catboost/parameter.yaml)에 있습니다. 예를 들어 XGBoost의 `model.n_estimators`, `model.max_depth`, `training.early_stopping_rounds`, `classification.threshold`를 바꿉니다. CatBoost에서는 `model.iterations`, `model.depth`, `training.early_stopping_rounds`, `classification.threshold`를 바꿉니다. Random Forest는 [`ML/random_forest/parameter.yaml`](ML/random_forest/parameter.yaml)을 사용하며 설정한 tree 수로 학습합니다. Random Forest에는 early stopping이 없습니다.

임상 입력 feature에 결측값이 있으면 audit에서 오류로 중단하며, 어떤 모델도 결측값을 대체하지 않습니다. 전처리는 train split에서만 fit됩니다. XGBoost와 Random Forest는 수치형 표준화와 범주형 one-hot encoding을 적용합니다. CatBoost는 공통 설정 로더가 제공하는 범주형 열을 문자열 범주형으로 전달합니다.

### 2.2 노트북 학습

Jupyter에서 `ML/model_experiment_only_ML.ipynb`를 열고 셀을 위에서 아래 순서대로 실행합니다.

1. 첫 코드 셀에서 저장소 루트, `choice_path`, trainer import를 준비합니다.
2. `## 입력 변수 버전 선택`에서 `model_version`을 1~4 중 하나로 설정합니다. 입력 개수는 각각 4·8·21·23개이고 기본값은 4입니다. 정확한 목록은 `ML/feature_versions.py`에 있습니다.
3. `## 학습 실행` 다음 코드 셀에서 `train_clinical(choice_path, model_version=model_version)`를 실행합니다. YAML의 `models`에 선택된 각 모델을 처리합니다. 선택한 버전과 입력 열이 일치하는 저장 run이 있으면 재사용하고, 없으면 해당 변수 목록과 YAML 파라미터로 학습합니다. 버전별 경로는 `model_paths`에 기록합니다.
4. `## 결과 확인` 다음 셀에서 각 run ID, validation prediction 일부와 validation metrics를 확인합니다.

학습 결과는 노트북 셀의 `run_dirs`에 표시되고 각 run의 `run_status.json`은 완료 시 `COMPLETE`가 됩니다. XGBoost·CatBoost는 validation Average Precision 기준 best iteration을 저장합니다. Random Forest는 설정한 tree 수를 사용하고 validation AP를 기록합니다. Test split은 학습에 사용되지 않습니다.

### 2.2.1 Optuna로 하이퍼파라미터 탐색

같은 ML 노트북에서 공통 import와 입력 변수 버전 선택 셀을 실행한 뒤 `## Optuna 실험 (선택)`으로 이동합니다. `model_name`은 알고리즘, `model_version`은 입력 변수 묶음입니다. 각 버전은 별도 Optuna study로 비교합니다. YAML 고정값 학습을 원하지 않으면 앞의 일반 학습·결과 확인 셀은 건너뜁니다.

1. `model_name`에 `XGBoost`, `CatBoost`, `Random Forest` 중 하나를 지정하고 `n_trials`를 정합니다. 확장 탐색의 기본값은 100회입니다. `mlflow_logging=True`이면 탐색 완료 시 best 하나를 로컬 MLflow에 전송합니다. 접속 정보는 `.poc.local.toml`에서 읽으며 노트북에 비밀번호를 적지 않습니다.
2. `## 탐색 범위와 학습 함수` 아래의 `objective(trial)` 셀에서 모델별 범위를 확인합니다.
3. `## 탐색 실행` 셀을 실행합니다. train으로 학습하고 validation AP를 최대화하며 trial 중에는 원본 YAML과 환자 split을 변경하지 않습니다. 기존 모델 경로가 있어도 `reuse_saved=False`로 새 trial을 학습합니다. Threshold와 조기 종료 횟수는 모델 YAML 값을 유지합니다.
4. `## 최적 결과 저장 및 확인` 셀에서 결과를 확인합니다. 저장은 직전 탐색 실행 셀에서 자동 완료되며, `best_run_dir`가 선택된 모델 경로입니다. 새 탐색은 Optuna 준비 셀부터 다시 실행합니다.

각 trial은 임시 폴더에서 학습하고 정리하며 현재 best 후보 하나만 유지합니다. 탐색 완료 시 최적 모델 하나와 `trials.csv`, `best_parameters.yaml`, `summary.json` 및 평가용 파일을 `ML/models/best/<algorithm>_vN/`에 저장합니다. 반복 실행 시 같은 조합의 저장 폴더를 새 study best로 갱신합니다. 실패하면 이전 모델은 보존합니다. 탐색 실행 셀에서 원본 YAML의 최신 `model_path`와 버전별 `model_paths`도 자동 갱신하고 기본 파라미터 값은 유지합니다. 선택이 끝나면 `evaluation/eval.yaml`에서 `target: ml`, 선택한 `model_name`과 `model_version`, `artifact_dir: null`, `split: test`로 평가합니다. 명시적인 `artifact_dir`를 사용해 `best_run_dir`를 지정할 수도 있습니다. Test 결과로 Optuna trial이나 threshold를 고르지 않습니다.

Optuna는 프로젝트의 `ml` extra에 포함되어 있습니다. 기존 환경에 없을 때는 저장소가 요구하는 uv 버전으로 `uv sync --project Global --locked --extra ml --inexact`를 실행하고 Jupyter 커널을 재시작합니다. 이 예시는 탐색 재개용 DB나 pruning 없이 순차 실행합니다.

### 2.3 같은 학습의 명령행 실행

명령행 학습은 노트북과 같은 `train_clinical`을 호출합니다. 저장소 루트에서 실행합니다.

```bash
uv run --project Global --extra ml python Members/sang/scripts/evaluate_preprocessed_clinical.py
```

특정 모델만 실행하거나 run 이름을 지정할 수 있습니다. 인자 이름과 지원 옵션은 아래 `--help`로 확인했습니다.

```bash
uv run --project Global --extra ml python Members/sang/scripts/evaluate_preprocessed_clinical.py \
  --models "Random Forest" --run-id clinical_rf
```

```bash
uv run --project Global --extra ml python Members/sang/scripts/evaluate_preprocessed_clinical.py --help
```

uv 버전 불일치 상태에서 기존 가상환경을 사용하는 명령:

```bash
Global/.venv/bin/python Members/sang/scripts/evaluate_preprocessed_clinical.py \
  --models "Random Forest" --run-id clinical_rf
```

### 2.4 ML 산출물 확인과 저장 모델 평가

Optuna 최적 모델은 [`ML/models/`](ML/models/) 아래 `best/<algorithm>_vN/`에 한 개씩 저장됩니다. 일반 YAML 고정값 학습은 같은 위치의 고유 run 폴더에 저장됩니다. 지정한 `--run-id clinical_rf`의 경우 예를 들어 `ML/models/clinical_rf_random_forest/`입니다. 기본 실행은 자동 생성 run ID를 사용합니다. XGBoost는 `best_model.json`과 `preprocessing.joblib`, CatBoost는 `best_model.cbm`, Random Forest는 `best_model.joblib`과 `preprocessing.joblib`을 저장합니다. 일반 학습 run에는 `metadata.json`, `resolved_config.yaml`, `metrics.json`, `validation_predictions.csv`, `training_history.csv`, `run_status.json`, `data_audit/`가 포함됩니다. Optuna 최적 모델은 평가에 필요한 앞의 metadata·설정·지표·validation 예측·상태와 전처리를 유지하고 `training_history.csv`, `data_audit/`는 보관하지 않습니다.

다음 단계에 전달할 정확한 run 경로는 학습 명령의 `training_runs` 출력, 노트북의 `run_dirs`, 또는 `ML/models/`의 생성 시각이 포함된 run 폴더명에서 확인합니다. 임의로 경로를 추측하지 말고 완료된 run의 `metadata.json`과 `run_status.json`을 확인합니다.

저장 모델 평가에는 [`evaluation/eval.yaml`](evaluation/eval.yaml)의 `evaluation.target`, `evaluation.artifact_dir`, `evaluation.split`을 지정합니다. 예를 들어 ML 평가라면:

```yaml
evaluation:
  target: ml
  model_name: Random Forest
  model_version: 4
  artifact_dir: null
  split: validation
  threshold_override: null
```

`artifact_dir: null`이면 `model_name`에 연결된 parameter YAML의 `model_path`를 읽습니다. 저장 모델이 없으면 먼저 일반 학습이나 Optuna 결과 저장을 실행해야 합니다. 특정 run을 지정하려면 `artifact_dir`를 `eval.yaml` 파일 위치인 `model/evaluation/` 기준 상대 경로로 설정합니다. 설정을 저장한 뒤 저장소 루트에서 평가 모듈을 실행합니다.

```bash
uv run --project Global --extra ml --extra gpu python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml
```

저장된 모델 재로드는 `metadata.json`의 `artifact_type`에 따라 [inference.py](inference.py)가 처리합니다. 재평가는 학습을 다시 하지 않습니다. 평가 출력은 `evaluation/results/ml/<evaluation_run_id>/` 아래의 `predictions.csv`, `metrics.json`, `resolved_config.yaml`, `data_audit/`입니다.

## 3. ECG만 사용하는 DL

### 3.1 ECG·라벨·split 연결

먼저 공통 audit를 완료합니다. DL 지도학습에는 `clinical_clean.csv`의 `Patient ID`·`SCD`, `ML/data/final_split.csv`의 `pid`·`split` (파일의 `val`은 내부 `validation`으로 매핑), 환자별 NPY와 `windows.csv.gz` sidecar의 환자·row가 모두 연결되어야 합니다. 누락 ECG, 임상 라벨, sidecar 또는 SQI 통과 window가 없는 경우 해당 modality 대상에서 빠지며 audit에서 사유를 확인합니다. P1030처럼 임상 라벨이 없는 환자는 파형 forward 점검에만 사용할 수 있습니다.

기본 dataloader는 SQI 통과 window 중 8개를 기록 전체에서 시간 순으로 고르게 선택하고, 각 window·lead에 z-score를 적용합니다. NPY 원본과 CSV는 수정하지 않습니다.

### 3.2 CNN과 sequence head 선택

[`DL/model_choice.yaml`](DL/model_choice.yaml)에서 `combination`, `cnn_model`, `sequence_model`을 선택합니다. 현재 구현이 지원하는 조합은 다음입니다.

| `combination` | 지원 CNN (`cnn_model`) | 지원 sequence head (`sequence_model`) |
|---|---|---|
| `CNN+LSTM` | `ResNet`, `InceptionTime`, `TCN` | `Attention LSTM FCN`, `BiLSTM` |
| `CNN+Transformer` | `ResNet`, `InceptionTime`, `TCN` | `transformer_enc` |

예를 들어 CNN+Transformer를 선택할 때는 실제 YAML에 다음 값을 씁니다.

```yaml
combination: CNN+Transformer
cnn_model: InceptionTime
sequence_model: transformer_enc
```

CNN parameter는 [`DL/CNN/model_set.yaml`](DL/CNN/model_set.yaml), LSTM parameter는 [`DL/LSTM/model_set.yaml`](DL/LSTM/model_set.yaml), Transformer parameter는 [`DL/Transformer/transformer.yaml`](DL/Transformer/transformer.yaml)에 있습니다. DL 모델의 최종 classification threshold는 `DL/model_choice.yaml`의 `classification.threshold`입니다.

Batch size, epoch, optimizer, learning rate, weight decay, early stopping patience, seed, device, workers는 [`DL/training.yaml`](DL/training.yaml)에서 설정합니다. 현재 값은 batch size 8, epochs 20, device `auto`입니다. `auto`는 CUDA가 사용 가능하면 GPU, 아니면 CPU를 선택합니다.

환자 단위 pooling은 모델 코드에 구현되어 있으며 별도의 aggregation YAML 키는 없습니다. Attention LSTM FCN은 attention 및 CNN feature pooling을 사용하고, BiLSTM은 sequence 출력 평균을 사용합니다. Transformer는 위치 정보를 더한 뒤 encoder 출력 평균을 사용합니다. 상세 구현은 [composed.py](DL/composed.py)의 `CNNLSTMClassifier`와 `CNNTransformerClassifier`입니다.

### 3.3 DL 노트북과 명령행 학습

`DL/model_experiment_only_DL.ipynb`를 열고 위에서 아래로 실행합니다.

1. 첫 코드 셀에서 저장소 루트, `choice_path`, `train_holter` import를 준비합니다.
2. `## 학습 실행` 다음 코드 셀에서 `train_holter(choice_path)`를 실행합니다.
3. `## Best run 확인` 다음 셀에서 validation metrics와 환자별 validation prediction을 확인합니다.

같은 실행은 저장소 루트에서 다음 CLI로 할 수 있습니다.

```bash
uv run --project Global --extra ml --extra gpu python Members/sang/scripts/evaluate_ecg_architectures.py
```

CLI는 `--run-id`, `--max-epochs`, `--limit-train-patients`, `--device {auto,cpu,cuda}`를 지원합니다. 실행 도움말은 다음과 같습니다.

```bash
uv run --project Global --extra ml --extra gpu python Members/sang/scripts/evaluate_ecg_architectures.py --help
```

GPU 메모리가 부족하면 `DL/training.yaml`의 `batch_size`를 낮추거나 CLI의 `--device cpu`를 지정해 다시 실행합니다. 이미 실패/완료된 run 디렉터리는 덮어쓰지 않으므로 다른 `--run-id`를 사용합니다.

### 3.4 Best checkpoint 재로드와 평가

DL run은 [`DL/models/`](DL/models/) 아래 생성됩니다. CLI의 `training_run` 또는 노트북 `run_dir` 출력에서 정확한 경로를 확인합니다. 완료된 run에는 best validation AP epoch의 `best_model.pt`, `metadata.json`, `resolved_config.yaml`, `metrics.json`, `validation_predictions.csv`, `training_history.csv`, `run_status.json`, `data_audit/`가 있습니다.

재평가 시 `evaluation/eval.yaml`을 다음처럼 지정합니다. 여기서 `artifact_dir`는 실제 DL run 이름으로 바꿉니다.

```yaml
evaluation:
  target: dl
  artifact_dir: ../DL/models/<실제_DL_run_id>
  split: validation
  threshold_override: null
```

위의 저장 모델 평가 CLI를 실행하면 `inference.py`가 `best_model.pt`와 `resolved_config.yaml`의 CNN/head 설정을 함께 읽어 모델을 재구성합니다. 모델 정의와 설정 중 하나가 달라졌거나 checkpoint가 다른 구조면 strict `load_state_dict` 단계에서 오류가 납니다. Test 평가는 선택을 끝낸 뒤 `split: test`로 실행합니다.

## 4. ECG + clinical 결합

### 4.1 선행 모델 지정

결합은 **저장된 ML 모델과 저장된 DL 모델을 각각 재로드해 환자별 확률을 산출한 뒤** 결합합니다. ML·DL 기본 모델을 다시 학습하지 않습니다. 먼저 각 학습 결과의 `training_runs`·`training_run` 또는 notebook `run_dirs`·`run_dir`에서 실제 run 폴더명을 확인합니다.

[`fusion.yaml`](fusion.yaml)의 artifact 경로는 이 파일이 있는 `model/` 기준입니다.

```yaml
ml:
  artifact_dir: ML/models/<실제_ML_run_id>
dl:
  artifact_dir: DL/models/<실제_DL_run_id>
```

기본값 `choose_ml_run`, `choose_dl_run`은 자리표시자이므로 실제 존재하는 run 디렉터리로 바꿉니다. 두 run의 `metadata.json`에서 target column, target semantics, class mapping이 호환되는지도 확인합니다.

### 4.2 가중치 선택

[`fusion.yaml`](fusion.yaml)의 `fusion.mode`는 세 가지입니다.

**수동 가중치 (`manual`)**는 모델 가중치를 직접 지정하며 두 값은 0 이상이고 합이 1이어야 합니다.

```yaml
fusion:
  mode: manual
  weights:
    ml: 0.6
    dl: 0.4
```

**가중치 학습 (`learn`)**은 ML·DL 기본 모델을 고정한 채 validation 환자 예측 확률에서 전역 가중치 두 개만 학습합니다. 구현은 binary cross-entropy 목적함수와 LBFGS를 사용하고 validation에 양·음성 라벨이 모두 있어야 합니다.

```yaml
fusion:
  mode: learn
  learning:
    split: validation
    objective: binary_cross_entropy
    parameterization: two_class_softmax
    optimizer: LBFGS
```

`learning.split`은 현재 설정값이 `validation`입니다. 코드도 validation split에서만 가중치를 학습합니다. Test로 바꾸어 가중치를 학습하는 설정은 지원하지 않습니다.

### 4.3 결합 노트북 실행 및 매칭 결과

`ECG+clinical_experiment.ipynb`를 열고 아래 순서로 실행합니다.

1. 첫 코드 셀에서 저장소 루트와 `fusion_config` 및 `run_fusion` import를 준비합니다.
2. 앞서 선택한 ML·DL artifact 경로와 mode·weights·threshold를 `fusion.yaml`에서 확인합니다.
3. `## Validation 또는 test 추론` 아래의 코드 셀을 실행합니다. 노트북의 코드는 `run_fusion(fusion_config, split_name="validation")`이므로 기본으로 validation 결과를 생성합니다.
4. `metrics.json`과 `validation_predictions.csv`를 확인합니다. 결과에는 두 모델 확률, 결합 확률, weights, 환자 ID, threshold 및 예측 정보가 들어갑니다. `excluded_patients.csv`에서 두 modality의 예측이 모두 있지 않은 환자와 제외 이유도 확인합니다.

환자 매칭은 환자 ID로 수행하며 ML/DL label 불일치 시 중단합니다. 현재 audit의 validation 대상은 ML 80명, DL 36명이고 공통 환자는 36명입니다. Test에서는 ML 91명, DL 45명이며 공통 환자는 45명입니다. 실제 run의 사용 환자 수는 결과 `metrics.json`의 `n` 및 `n_excluded_no_pair`, 제외 명단 CSV로 확인합니다.

단계 구분:

- **기본 모델 추론:** 저장된 ML·DL 모델과 preprocessing/checkpoint를 로드하여 확률을 계산합니다. 재학습하지 않습니다.
- **가중치 학습:** `mode: learn`일 때에만 validation의 두 기본 모델 확률을 이용해 가중치를 최적화합니다. 이것은 ML·DL 모델 학습이 아닙니다.
- **가중치 재사용:** 결합 run에는 `fusion_weights.json`과 `resolved_config.yaml`이 저장됩니다. 재사용하려면 `fusion.mode: saved`, `fusion.saved_run_dir`에 이전 결합 결과 run 경로를 지정합니다. 저장된 run의 base model artifact 경로와 weights를 다시 사용합니다.

수동 weights와 validation에서 학습한 weights 모두 같은 `run_fusion` 경로를 사용합니다. test 최종 평가에는 validation에서 고정한 weights를 적용하고 test 성능을 확인한 뒤 weights나 threshold를 다시 조정하지 않습니다.

### 4.4 결합 test 평가

결합 노트북 셀은 validation prediction CSV를 표시하도록 작성되어 있습니다. Test 평가는 공통 evaluation CLI를 사용하면 저장 파일명과 결과 위치가 올바르게 처리됩니다. `fusion.yaml`을 `mode: saved`로 지정해 validation 가중치 run을 재사용하거나, `manual`로 이미 결정한 weights를 유지합니다. 다음으로 `evaluation/eval.yaml`을 설정합니다.

```yaml
evaluation:
  target: fusion
  fusion_config: ../fusion.yaml
  split: test
  threshold_override: null
```

저장소 루트에서 실행합니다.

```bash
uv run --project Global --extra ml --extra gpu python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml --split test
```

평가 결과는 `evaluation/results/fusion/<evaluation_run_id>/test_predictions.csv`, `metrics.json`, `excluded_patients.csv`, `fusion_weights.json`, `resolved_config.yaml`, `data_audit/`에 저장됩니다. Test로 최종 성능을 확인하기 전에 모델·weights·threshold를 고정합니다.

## 5. Threshold 조정과 저장 예측 재평가

| 종류 | 새 학습 run의 기본 threshold | 저장된 모델 확률 재평가 시 |
|---|---|---|
| ML | 각 모델 `parameter.yaml`의 `classification.threshold` | `evaluation/eval.yaml`에서 `target: ml`, 해당 `artifact_dir` 지정 후 평가 CLI `--threshold-override 0.4` |
| DL | `DL/model_choice.yaml`의 `classification.threshold` | `evaluation/eval.yaml`에서 `target: dl`, 해당 `artifact_dir` 지정 후 동일 CLI 옵션 사용 |
| 결합 | `fusion.yaml`의 `classification.threshold` | 평가 YAML `target: fusion`으로 설정 후 동일 CLI 옵션 사용 |

예를 들어 저장된 확률을 validation split에서 threshold 0.4로 다시 평가하는 명령은 다음과 같습니다. 먼저 `evaluation/eval.yaml`에서 대상과 실제 artifact 경로를 설정해야 합니다.

```bash
uv run --project Global --extra ml --extra gpu python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml \
  --split validation --threshold-override 0.4
```

threshold override는 확률값을 바꾸지 않고 분류 결과와 threshold 의존 metrics만 다시 계산합니다. 따라서 **저장된 모델 확률을 재평가하기 위해 재학습할 필요는 없습니다.** 모델별 parameter YAML 또는 DL model_choice의 threshold를 바꿔도 이미 생성된 artifact의 metadata와 결과는 자동 변경되지 않습니다. 기존 artifact를 평가할 때는 `--threshold-override`를 사용합니다. Fusion은 weights를 YAML에 설정하고 threshold는 별도 `classification.threshold`에 설정합니다.

Threshold 비교는 validation으로 끝내고 선택을 고정한 뒤 test는 최종 평가에 한 번 사용합니다. Test 성능을 보고 threshold, fusion weight, 모델 또는 epoch을 반복 조정하지 않습니다.

저장된 환자 확률 CSV만 평가하는 [`evaluation/evaluate_predictions.py`](evaluation/evaluate_predictions.py)도 있습니다. 이 스크립트는 다음 세 열만 포함하는 CSV를 입력으로 받습니다: `patient_id,target,score`. 현재 ML/DL trainer와 평가기가 생성하는 CSV에는 추가 열과 다른 이름(`y_true`, `p_positive`)이 있어 그대로 넣을 수 없습니다. 그 파일을 별도 호환 형식으로 준비한 경우에만 다음 명령을 사용합니다.

```bash
uv run --project Global python Members/sang/model/evaluation/evaluate_predictions.py \
  --validation /path/to/validation_predictions.csv \
  --test /path/to/test_predictions.csv \
  --input-group clinical \
  --threshold-mode fixed --threshold 0.4
```

이전 예측 파일로 validation에서 Youden threshold를 계산하려면 `--threshold-mode youden`을 사용합니다. `auto`도 현재 구현에서 `youden`으로 바뀝니다. test 예측을 함께 넣으면 validation에서 선택된 threshold를 test에 그대로 적용하고, validation/test 환자 ID 중복이 있으면 중단합니다. 결과 report 기본 위치는 `model/outputs/evaluation/`입니다. 현재 run artifact의 prediction을 바로 재평가하려면 CSV 변환을 추측해 이 레거시 도구에 넣지 말고, 위 `evaluation.py`가 저장 모델에서 확률을 다시 계산하도록 사용하세요.

## 6. 결과 위치와 실행 완료 확인

| 작업 | 완료 확인 파일 |
|---|---|
| ML 학습 | `ML/models/<run_id>/run_status.json`의 `COMPLETE`, `metadata.json`, `metrics.json`, `validation_predictions.csv` |
| DL 학습 | `DL/models/<run_id>/run_status.json`의 `COMPLETE`, `best_model.pt`, `metadata.json`, `metrics.json`, `validation_predictions.csv` |
| 평가 | `evaluation/results/<target>/<run_id>/metrics.json`, `predictions.csv` (fusion은 split이 포함된 prediction 파일명) |
| 결합 | `evaluation/results/fusion/<run_id>/fusion_weights.json`, `resolved_config.yaml`, split별 prediction 및 `excluded_patients.csv` |

각 config에 기록되는 상대 artifact 경로는 config를 담은 YAML 파일 디렉터리를 기준으로 합니다. 결과 디렉터리는 생성된 run ID를 덮어쓰지 않습니다. 지정한 run ID가 이미 있으면 다른 ID를 사용하세요.

## 7. 자주 발생하는 문제

- **데이터 경로 오류:** `basic.yaml` 상대 경로가 `basic.yaml` 파일 기준인지 확인하고, clinical/split 파일과 ECG 디렉터리가 실제 존재하는지 확인합니다. 임의의 대체 데이터나 split은 자동 탐색하지 않습니다.
- **Jupyter import 오류 또는 다른 Python 환경:** 저장소 루트에서 Jupyter를 시작했는지, 커널이 `Python (sixletters-poc Global)`인지 확인합니다. `Members.sang...` import는 저장소 루트를 Python 경로로 필요로 합니다.
- **환자·라벨·split 불일치:** ID는 문자열로 읽고 환자 ID로 조인합니다. duplicate ID, 알 수 없는 split 값, 라벨 불일치는 audit 또는 학습 중 오류로 처리되므로 ID 열과 `values` 매핑을 확인합니다.
- **ECG/임상 데이터 누락:** 학습 modality에 필요한 환자 교집합을 audit에서 봅니다. ECG만 있고 임상 라벨이 없는 환자는 지도학습·결합 대상이 아닙니다.
- **GPU 메모리 부족:** `DL/training.yaml`의 `batch_size`를 낮추거나 `--device cpu`로 실행합니다.
- **저장 모델 경로 오류:** 해당 run의 `metadata.json` 또는 `best_model.pt`가 있는지 확인하고, 경로를 참조하는 YAML 기준 상대 경로인지 확인합니다. 완료되지 않은 `RUNNING` run은 사용할 수 없습니다.
- **모델과 설정 불일치:** DL checkpoint는 학습 run의 `resolved_config.yaml`에 저장된 CNN·head 설정으로 재구성됩니다. 학습 때 사용한 run 폴더를 통째로 지정하고 checkpoint 파일만 다른 run으로 교체하지 않습니다.
- **uv 실행 오류:** `Global/pyproject.toml`의 required uv 0.9.26과 실행 중인 버전을 확인합니다. 현재 검토 환경의 uv 0.12.3에서는 `uv run --project Global`이 버전 불일치로 실패했습니다.

## 8. 현재 문서 범위의 제약

공통 평가 진입점(`evaluation/evaluation.py` 및 이를 호출하는 평가 노트북)은 `evaluation/eval.yaml`의 `evaluation.mlflow_logging: true`일 때 로컬 저장 후 집계 지표와 실행 설정을 별도 MLflow run으로 전송합니다. Test는 모델·threshold 확정 후 명시적으로 평가하며 Optuna가 자동 실행하지 않습니다. `test.*`/`validation.*` 지표와 `source_run_id`, 가능한 경우 `source_mlflow_url`로 평가와 학습을 연결합니다. 실패 시 로컬 결과 폴더의 `mlflow_status.json`을 확인하고 [`tracking/LOCAL_MLFLOW.md`](tracking/LOCAL_MLFLOW.md)의 전송 재시도를 사용하세요. 환자별 예측과 audit은 업로드하지 않습니다. ML·DL·Fusion 공통 평가 경로를 지원하지만 이번 연동의 실제 실행 검증은 Random Forest ML test 평가로 수행했습니다.

전체 학습 또는 장시간 실험은 이 가이드를 작성하면서 실행하지 않았습니다. CLI `--help`, notebook 셀 흐름, YAML 키와 저장 경로를 코드와 대조했습니다. 새 ML/DL run을 기존 MLflow handoff bundle로 자동 변환하는 연결은 현재 별도 상태이며, 자세한 내용은 [`tracking/MLFLOW_HANDOFF.md`](tracking/MLFLOW_HANDOFF.md)를 확인하세요.
