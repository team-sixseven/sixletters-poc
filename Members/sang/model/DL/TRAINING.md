# ECG DL 학습·평가 실행 순서

기본 raw pipeline은 `basic.yaml:data.raw_ecg`의 Google Drive 경로를 사용합니다. Drive의 4년 SCD 라벨과 `final_split_augmented.csv`를 사용하며 재분할하지 않습니다. 이 PC의 WSL 마운트 경로는 `/mnt/g`이고, PC별 Drive 경로가 다르면 설정을 조정해야 합니다. 아래 명령은 저장소 루트 기준입니다.

`basic.yaml`의 `drive_root`는 Drive `DATA` 폴더이며, `drive://`로 시작하는 데이터 경로는 이 폴더 기준입니다. 로딩 시 절대경로로 변환되고 학습 설정 snapshot에는 변환된 실제 경로가 기록됩니다. 200Hz와 X/Y/Z의 출처는 Drive `ECG_tem/README.md`이며, YAML의 `source_metadata_confirmed`는 이 출처를 확인했다는 학습 실행 조건입니다.

WSL의 `/mnt/g`가 없으면 Windows에서 연결된 G:가 이 배포판에서 접근 가능한지 확인한 뒤 마운트합니다. 마운트는 현재 WSL 세션에 적용되며 재시작 후 다시 필요할 수 있습니다.

```bash
wsl.exe -d Ubuntu-24.04 -u root --exec mkdir -p /mnt/g
wsl.exe -d Ubuntu-24.04 -u root --exec mount -t drvfs G: /mnt/g
findmnt /mnt/g
```

Drive 경로의 실행 파일은 `/mnt/c/Windows/System32/wsl.exe`일 수 있습니다. 마운트와 Drive 로그인·공유 권한은 각각 필요합니다. 학습 중 필요한 데이터가 Drive 캐시에 전송될 수 있습니다.

## 1. Python 환경

```bash
cd /home/sang/sixletters-poc
Global/.venv/bin/python --version
Global/.venv/bin/python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

새 환경을 준비할 때만 프로젝트가 고정한 uv로 동기화합니다.

```bash
uvx --from uv==0.9.26 uv sync --project Global --locked --extra gpu --extra ml
```

노트북 커널도 `Global/.venv` Python을 선택합니다.

### 개인 계정 자동 기록

각 PC에서 `dev setup`으로 본인 계정을 한 번 등록합니다. `tracking/TEAM_MLFLOW.md`의 계정·서버 설정 절차를 따릅니다. 설정 파일 `.poc.local.toml`은 Git에서 제외되며, 브라우저 로그인 계정 대신 이 실행 환경의 계정을 사용합니다. 비어 있지 않은 `MLFLOW_TRACKING_*`, `MLFLOW_EXPERIMENT_NAME` 환경 변수가 파일보다 우선합니다.

DL 노트북의 실행 설정 기본값은 다음과 같습니다.

```python
mlflow_logging = True
run_final_test = True
```

계정 확인 → 학습·validation → best checkpoint 업로드 → 고정 checkpoint의 test 평가·업로드 순서로 실행합니다. 설정을 비교하는 중에는 `run_final_test=False`, 오프라인 실행은 `mlflow_logging=False`로 바꿉니다. 계정명을 노트북에 하드코딩하지 않습니다.

MLflow에는 집계 validation/test 지표, epoch 이력, checkpoint 및 환자 정보 없는 architecture 설정을 기록합니다. 예측 CSV, PID audit, 품질 인덱스, PID별 quality count는 로컬에만 둡니다. 업로드한 `model_config.yaml`과 `best_model.pt`는 `build_composed_model_from_specs`로 network를 복원할 수 있고, raw 데이터·품질 정보는 로컬 source에서 별도로 제공합니다.

학습 전 계정 확인 실패는 학습을 중단합니다. 학습 완료 후 업로드 실패는 로컬 모델을 보존하고 `mlflow_status.json`을 `PENDING`으로 남깁니다. 전송만 재시도하려면:

```python
from Members.sang.model.tracking.log_dl import log_dl_training_to_mlflow
from Members.sang.model.tracking.log_evaluation import log_evaluation_to_mlflow

log_dl_training_to_mlflow(run_dir)
log_evaluation_to_mlflow(evaluation_dir)
```

동일 run의 재전송은 기존 공통 업로더의 중복 방지 기능을 사용합니다. 성공 시 `FINISHED`와 MLflow 링크를 기록합니다. 학습과 평가는 별도 MLflow run이며 평가 run에 원래 학습 run ID와 링크를 연결합니다.

## 2. 파일·split 감사

```bash
Global/.venv/bin/python -m Members.sang.model.DL.dataloader.segmented_ecg
```

CSV schema는 `pid,split,SCD`이고, PID는 문자열, SCD는 0/1입니다. `val`은 내부에서 `validation`으로 처리합니다. 중복 PID는 같은 split/라벨이어도 거부하며, 환자를 다른 split으로 옮기지 않습니다.

Drive가 제공한 현재 split은 원본 445명과 train에만 포함된 증강 기록 64개로 총 509개 기록입니다. 기록 수 기준 train 349개(양성 96), validation 71개(양성 8), test 89개(양성 10)입니다. 증강 기록과 원본은 환자 단위 split과 라벨이 같아야 하며 로더가 이를 확인합니다. 데이터 또는 Drive 파일이 바뀌면 실제 감사 결과를 다시 확인해야 합니다.

Drive의 `README.md`는 이 NPY가 200Hz, `(샘플, X/Y/Z)` float16이며 SCD 라벨은 Holter 이후 4년 기준이라고 설명합니다. `ECG_tem_clean.csv`의 `has_z`가 False이면 저장된 Z의 0열은 실측 유도가 아니므로 유도 mask에서 제외합니다. 다른 source에 이 설정을 사용할 때는 해당 source의 provenance를 다시 확인하세요.

## 3. Raw metadata와 품질 인덱스 설정

`Members/sang/model/basic.yaml`의 `data.raw_ecg`에서 경로·sample rate·30초 segment 길이·정규화를 관리합니다.

- `source_metadata_confirmed: true`와 200Hz는 연결된 Drive의 `ECG_tem/README.md`에 근거합니다.
- `quality_index_format: drive_segment_rpeaks`는 Drive의 `segment_rpeaks.csv`를 직접 읽습니다. 파일명으로 각 원본·증강 기록을 식별하고 `window_idx`, `start_idx`가 30초 간격과 일치하는지 확인하며 `exclude=True`인 구간은 사용하지 않습니다. 선택된 파일의 전체 30초 구간에 품질 행이 하나씩 있어야 합니다.
- `lead_metadata_path`는 Drive의 `ECG_tem_clean.csv`입니다. `has_z=False`이면 Z를 제외하고 X/Y만 사용합니다. 이 파일과 품질·분할 파일의 해시는 저장 후 평가 때 확인합니다.
- `target_semantics`는 Drive의 4년 SCD 라벨을 명시합니다. 기존 프로젝트의 임상 라벨 및 분할과 다르므로 검증 없이 결과를 결합하지 않습니다.
- `normalization: none`은 Drive가 제공한 정제 데이터를 다시 정규화하지 않는 설정입니다.

기존 별도 원본에 `pid,window_index,usable` 형식을 쓸 때는 `quality_index_format: standard`를 지정하고 해당 CSV를 설정합니다. `lead_metadata_path`를 생략하면 저장된 모든 열을 측정 유도로 취급하므로 source에서 이를 확인해야 합니다. 품질 정보와 확인된 source metadata가 없으면 학습이 중단됩니다.

### 가변 유도와 가변 시간 길이

각 환자는 실제 `(segments, measured_leads, samples)` 크기를 유지합니다. 배치를 만들 때만 유도 축과 시간 축을 각각 배치 최대치까지 0으로 패딩하고, `lead_mask (batch, leads)`와 `window_mask (batch, segments)`를 분리합니다. 각 실제 유도에 하나의 공유 1유도 CNN을 적용하고, 유도 특징을 평균 통합해 구간 벡터를 만듭니다. 패딩 유도와 제외 시간 구간은 CNN·BatchNorm에 입력하지 않습니다. 없는 유도의 값·패딩 길이·유도 순서를 바꿔도 유효 출력은 같아야 합니다.

이 방식은 유도 이름의 환자 간 대응을 모델에 하드코딩하지 않습니다. 고정 슬롯 패딩 방식은 2열이 3열의 어떤 유도와 대응하는지 가정해야 해서 채택하지 않았습니다. 평균 통합은 모든 측정 유도를 사용하고 유도 순서에 의존하지 않는 대신, 이름에 따른 유도별 차이나 유도 사이 방향 관계를 구별하지 못합니다. Drive source의 `has_z`는 없는 유도를 mask하는 데만 사용합니다.

기록 길이는 유도 수와 별도로 유효 구간 수·시간 mask로 처리합니다. 제외 구간을 압축해 전달하되 원래 segment index와 gap embedding을 유지합니다. LSTM은 packed lengths, Transformer는 key-padding mask, MIL은 masked attention을 적용합니다. Batch 안의 sample 수가 다르면 임의로 자르거나 resampling하지 않고 오류를 냅니다.

기존 `2000`은 별도 SQI-window 경로의 200 Hz·10초 구간에 대응하던 값입니다. 실제 SQI sidecar `start_sec`는 10초 간격이고, 현재 그 source의 NPY들은 없어 헤더 재확인은 수행하지 못했습니다. Raw 경로에 2000을 적용하지 않습니다. Raw는 확인된 sample rate × 30초로 구간을 자르며, 200 Hz가 확인되면 6000 samples입니다. CNN은 adaptive pooling으로 실제 sample 길이를 받아들이고, legacy loader도 실제 NPY의 sample 축 길이를 보존합니다.

## 4. 모델·학습 설정

모델 선택은 `DL/model_choice.yaml`의 두 항목만 사용합니다.

```yaml
cnn_model: ResNet
sequence_model: BiLSTM
```

시간 모델은 `Attention LSTM FCN`, `BiLSTM`, `transformer_enc` 중 선택합니다. CNN+LSTM/Transformer 조합 판별, 클래스 등록, 특징 메서드, 차원 연결은 코드에서 처리합니다. YAML에 `implementation`, `feature_method`, `input`, `output`, `channels`나 중복 입력/출력 차원을 적지 않습니다.

`DL/CNN/model_set.yaml`:

```yaml
models:
  ResNet:
    activation: GELU
  InceptionTime:
    modules: 3
    activation: GELU
  TCN:
    activation: GELU
```

CNN은 각 모델의 activation과 InceptionTime의 modules만 YAML에 둡니다. 나머지는 기존 값과 같은 코드 기본값입니다: ResNet width 32·blocks (2, 2), InceptionTime filters 16·kernels (9, 19, 39), TCN hidden_dim 32·dilations (1, 2, 4)·kernel_size 3.

LSTM YAML은 Attention LSTM FCN의 `activation`·`dropout`, BiLSTM의 `dropout`만 둡니다. Hidden width 32, recurrent layer 1개, FCN kernel (3, 5)는 코드 기본값을 사용합니다. BiLSTM의 gate는 고정 sigmoid/tanh이며, dropout은 1-layer에서도 sequence 출력에 적용합니다. Transformer YAML도 `dropout`, `activation`만 설정합니다. `embedding_dim=64`, `heads=4`, `layers=2`, `feedforward_dim=128`은 이전 YAML 값과 같은 코드 기본값이며, CNN 출력에서 embedding 차원으로 자동 projection합니다. 각 CNN/시간 모델의 `feature_dim`과 `forward_features`가 다음 계층 차원을 결정합니다.

`DL/training.yaml`에는 optimizer·학습률·epoch·seed·early stopping과 다음 설정을 함께 둡니다.

```yaml
batch_size: 1
max_segments_per_patient: 128
classification_threshold: 0.5
mil:
  pooling: attention
  hidden_dim: 32
  activation: GELU
  dropout: 0.0
classifier:
  hidden_dim: 32
  activation: GELU
  dropout: 0.1
```

### MIL pooling 선택

`training.yaml:mil.pooling`은 `attention`, `mean`, `max` 중 선택합니다. 기본은 기존 `attention`입니다. 이 설정은 LSTM/Transformer 뒤의 환자 단위 시간 구간 통합에 적용합니다. CNN 단계의 실측 유도 특징 평균은 그대로입니다.

평균 pooling은 **mil 블록 전체**를 다음으로 교체합니다.

```yaml
mil:
  pooling: mean
```

최댓값 pooling:

```yaml
mil:
  pooling: max
```

Attention은 위 예시처럼 `pooling: attention`과 `hidden_dim`, `activation`, `dropout`을 설정합니다. Mean/max에는 pooling 자체의 학습 파라미터가 없으므로 이 attention 전용 항목들을 제거해야 합니다. 남겨두면 사용하지 않는 설정을 조용히 무시하지 않고 오류로 알립니다. 뒤의 MLP 설정 `classifier`는 세 방식 모두 적용됩니다.

세 방식 모두 time mask를 반영합니다. Mean은 유효 구간 개수로만 나누고, max는 패딩을 음의 무한대로 처리해 유효 특징이 모두 음수여도 패딩 0이 선택되지 않게 합니다. NaN/Inf가 있는 제외 구간은 통합 전에 마스킹합니다. 유효 구간이 하나도 없는 환자는 오류로 처리합니다.

선택한 pooling은 `resolved_config.yaml`, `metadata.json:mil_pooling` 및 MLflow 파라미터에 기록됩니다. 복원·평가는 현재 YAML 대신 학습 run의 snapshot을 사용합니다. 기존 format v2 attention run에 `pooling` 항목이 없으면 attention으로 복원하며, 다른 pooling으로 바꿔 실험할 때는 새 run을 학습합니다.

CNN, Transformer FFN, Attention LSTM FCN의 FCN, MIL과 MLP의 중간 activation은 GELU/ReLU/LeakyReLU/SiLU 중 선택합니다. LSTM 내부 sigmoid/tanh gate는 교체하지 않습니다. 추론에서 sigmoid를 한 번 적용합니다.

설정 파일 경로는 기존 디렉터리 구조로 코드가 결정합니다. 선택 파일과 같은 DL 디렉터리의 `training.yaml`, `CNN/model_set.yaml`, `LSTM/model_set.yaml`, `Transformer/transformer.yaml`, 상위의 `basic.yaml`을 사용합니다. 데이터 경로와 모델 출력 경로 `outputs.dl_model_directory`는 `basic.yaml`에 있습니다. 파일 구조를 유지하면 `--choice`로 다른 이름의 모델 선택 파일도 사용할 수 있습니다.

기본 batch size는 1, 환자별 cap은 128개입니다. 품질 통과 segment 전체에서 균등 간격으로 선택하고 시간 순서·원래 위치를 보존합니다. 전체 기록을 모두 쓰는 설정은 아니며 평가에서도 같은 결정적 선택을 사용합니다. 공유 CNN은 실제 유도별 특징을 계산하므로 GPU 메모리에 맞게 cap/batch size를 조절한 뒤 새 run을 학습합니다.

## 5. 짧은 학습 후 전체 학습

실제 파일과 metadata가 준비된 뒤 먼저 smoke run을 실행합니다. Train은 최대 16명이며 validation은 전체 validation cohort입니다. Smoke 성능은 전체 성능으로 해석하지 않습니다.

```bash
Global/.venv/bin/python -m Members.sang.model.DL.raw_pipeline \
  --max-epochs 1 --limit-train-patients 16 --device cpu
```

개인 계정 MLflow 자동 기록을 포함한 전체 학습:

```bash
Global/.venv/bin/python -m Members.sang.model.DL.raw_pipeline --device auto --mlflow-logging
```

Python API와 CLI는 오프라인 호환성을 위해 MLflow 기록이 기본 비활성입니다. CLI에서 `--mlflow-logging`, API에서 `mlflow_logging=True`를 지정합니다. 노트북은 이를 기본 활성화합니다. CLI 학습 명령은 test 평가를 자동 실행하지 않으며, 노트북이 학습 뒤 평가를 연결합니다.

CUDA를 강제하려면 `--device cuda`를 사용합니다. CUDA가 없으면 오류가 발생합니다. 다른 모델 선택 파일은 `--choice <path>`로 지정합니다.

Train만 가중치와 클래스 불균형 보정값 학습에 사용합니다. Validation AP로 best checkpoint와 early stopping을 결정합니다. Train/validation이 비거나 한 클래스이면 학습을 거부합니다. 학습 중 test 예측은 수행하지 않습니다. 파일 shape·품질 인덱스 감사는 모든 split에 적용합니다.

노트북은 `DL/model_experiment_only_DL.ipynb`에서 설정 → 파일 감사 → 로더·모델 생성 → raw 학습 → 복원 → 선택적 최종 평가 순서로 실행합니다.

## 6. 저장 결과 확인

CLI가 출력한 `DL/models/<run_id>/`에 다음 파일이 저장됩니다.

- `best_model.pt`: validation AP 기준 checkpoint
- `resolved_config.yaml`, `metadata.json`: 복원 설정, 전처리, segment cap, split/quality SHA-256
- `train_patients.csv`, `excluded_patients.csv`, `data_audit.json`: 실제 사용 PID·split·라벨, 누락/품질 제외/smoke cap 사유
- `validation_predictions.csv`: PID별 라벨·확률·threshold와 사용한 segment 인덱스
- `metrics.json`, `training_history.csv`, `run_status.json`: validation 지표, epoch 이력(train/validation loss 포함), 완료 상태. `training_history.csv`는 매 epoch 갱신됩니다.
- MLflow 활성 시 `mlflow_status.json`, `summary.json`: 계정별 전송 상태와 학습 run 링크

`run_status.json`의 `COMPLETE`를 확인합니다. `used_window_indices`에는 실제 선택 인덱스 전체가 JSON list로 기록됩니다. `windows_used`, `first_window_index`, `last_window_index`도 함께 저장됩니다.

## 7. 최종 test 평가

모델과 threshold를 확정한 뒤 기존 `evaluation/eval.yaml`의 다음 필드를 바꿉니다. 다른 기존 필드는 유지합니다.

```yaml
evaluation:
  target: dl
  model_path: ../DL/models/<run_id>/best_model.pt
  artifact_dir: null
  split: test
  threshold_override: null
  mlflow_logging: true
```

```bash
Global/.venv/bin/python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml --split test
```

Raw artifact는 자동 인식됩니다. `model_path`에는 `best_model.pt` 또는 run 폴더를 지정하며, 같은 폴더의 `metadata.json`과 `resolved_config.yaml`이 필요합니다. 기존 `artifact_dir`도 지원하지만 두 필드가 서로 다른 run을 지정하면 오류를 냅니다. 평가 중 재학습하지 않고 저장된 weights·설정·threshold를 복원합니다. 학습 때와 split CSV·quality CSV의 checksum, sample rate·segment 길이·정규화가 다른 경우 중단합니다. 파일을 뒤늦게 추가해 새 환자가 생기면 새 학습 run을 만들어 사용합니다. 상대 경로는 해당 YAML 위치를 기준으로 해석합니다.

오프라인 평가는 `mlflow_logging: false`로 바꿉니다.

결과는 `evaluation/results/dl/<evaluation_run_id>/`에 저장됩니다. `predictions.csv`는 PID별 확률과 사용 segment를 포함하고, `metrics.json`은 AUROC, Average Precision, sensitivity, specificity, precision, F1, confusion matrix 등을 기록합니다. AP는 sklearn average precision이며 사다리꼴 PR-AUC가 아닙니다. 계산 불가능한 지표는 `null`과 이유를 기록합니다.

## 8. 임상 ML과 확률 결합

`fusion.yaml`에서 준비된 ML·DL artifact 경로를 설정하고, `evaluation/eval.yaml`의 `target`을 `fusion`으로 바꾼 뒤 같은 평가 명령을 실행합니다. Raw DL artifact도 지원합니다.

- `mode: manual`: 음수가 아니며 합이 1인 가중치 사용
- `mode: learn`: validation에서만 가중치 학습
- `mode: saved`: 이전 run의 `fusion_weights.json`을 재사용

ML·DL의 양성 클래스·예측 대상 의미를 확인하고, 동일 PID의 split 또는 라벨이 다르면 중단합니다. 두 모델의 PID 교집합으로 결합하며 한쪽에만 있는 PID는 제외 사유로 남깁니다. 통합 노트북은 `ECG+clinical_experiment.ipynb`입니다. 기본적으로 개인 계정 확인 후 validation 평가와 최종 test 평가를 공통 evaluation 경로로 실행·기록하며, test에서는 validation run에 저장된 가중치를 재사용합니다. 모델 비교 중에는 `run_final_test=False`로 설정합니다.

## 9. 구현 검증과 한계

재현 가능한 합성 데이터 acceptance test:

```bash
Global/.venv/bin/python -m pytest Members/sang/model/DL/tests/test_pipeline.py -q
```

테스트는 가변 길이, 2·3 lead 혼합 배치와 별도 유도/시간 mask, 패딩 NaN·유도 순서 불변성, 실제 세 번째 유도 gradient, 품질 제외 gap·tail, 누락 파일, 유효 segment 없음, 중복/상충 PID, 9개 모델 조합 forward·padding 불변성·CNN gradient, 짧은 raw 학습·checkpoint 복원·환자 평가, 설정/split 변경 차단, 실제 raw 추론과 모의 ML 확률의 fusion을 확인합니다. Fusion 테스트의 ML 예측은 stub이며 실제 임상 모델의 성능 검증이 아닙니다. 두 모의 계정의 raw/SQI 학습·평가 업로드 경로, 전송 실패·재시도, 업로드 설정으로 network 복원, 두 노트북의 자동 학습/평가 연결도 검증합니다. 이 검증은 실제 서버에 데이터를 전송하지 않습니다.

이전 별도 source의 합성 데이터로 590개 기록의 407/81/102 split 로딩을 확인했습니다. 이후 Drive source로 소규모 실제 학습·평가 및 MLflow 기록을 실행했습니다. 임상 성능은 검증하지 않았습니다.

CNN은 기존 구현을 모델별 `CNN/resnet.py`, `inception_time.py`, `TCN.py`로 분리해 재사용합니다. 시간 특징 구현도 `LSTM/Attention_LSTM_FCN.py`, `BiLSTM.py`, `Transformer/transformer_enc.py`에서 관리하며 공통 조립·MIL은 `composed.py`에 있습니다. Padding을 CNN/BatchNorm에서 제외하도록 조립 경로를 수정했습니다. Packed LSTM, Transformer key-padding mask, masked MIL, measured-lead 표시와 gap 입력을 추가했습니다. Attention LSTM FCN은 CNN 구간 벡터에 packed LSTM과 masked temporal FCN을 병렬 적용하고 구간별 특징을 결합한 뒤 환자 MIL을 적용하는 변형입니다. FCN은 위치별 LayerNorm을 사용하고 각 convolution 뒤 시간 padding을 지워 패딩 통계/값이 유효 구간에 영향을 주지 않게 합니다. 독립 segment 분류를 수행하지 않습니다. Raw loader는 기존 10초 SQI source의 format·mask를 재사용하면 정렬을 보장할 수 없어 별도로 구현했습니다.

DL 학습의 양성 클래스 가중치는 `basic.yaml:dl.positive_class_weight`에서 설정합니다. 기본값은 실제 학습 환자의 음성 수 / 양성 수이며, `none`을 지정하면 보정하지 않습니다. Raw와 legacy DL에 적용되며 ML에는 적용되지 않습니다.

## 기존 SQI-window 경로

기존 `basic.yaml:data.ecg`의 10초 window source가 필요한 경우 별도로 실행합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.DL.training \
  --max-epochs 1 --limit-train-patients 16 --device cpu
Global/.venv/bin/python -m Members.sang.model.DL.training
```

이 명령은 `DL/data/wave` raw 파일을 학습하지 않습니다. SQI run도 공통 평가 artifact 인터페이스를 사용합니다.


## Checkpoint 형식 변경

새 가변 유도 모델의 format version은 2이며 `lead_strategy=shared_single_lead_cnn_masked_mean`을 기록합니다. Feature 차원과 MIL/MLP 설정을 snapshot에 저장하고 복원 시 현재 실험 YAML을 다시 읽지 않습니다. 기존 고정 3유도 checkpoint는 첫 convolution과 유도 통합 방식이 달라 자동 변환하지 않습니다. 예전 코드로 읽거나 새 구조로 재학습해야 하며, 새 로더는 이를 명확한 오류로 알립니다. 기존 ML 평가/모델 형식에는 이 DL version 검사를 적용하지 않습니다.


### 이번 변경의 실행 검증

`DL/tests/test_pipeline.py`와 변경하지 않은 `ML/tests/test_workflow.py`를 함께 실행해 **46개 테스트가 통과**했습니다. 2·3유도 혼합/가변 시간 길이의 실제 collator 배치로 BCE 학습·optimizer step, 9개 CNN×시간 조합의 padding NaN 불변성과 BatchNorm buffer 일치, 유도 순서 변경, 네 번째 absent slot 추가, 실제 세 번째 유도 gradient, 4개 activation과 자동 차원 연결, snapshot 복원, 환자 단위 평가와 MLflow bundle network 복원을 합성 데이터로 확인했습니다.

이전 별도 source 검증 당시 실제 raw `P0008.npy` `(17271800, 3)` float16에서 읽은 두 개의 6000-sample block으로 9개 조합의 라벨 없는 forward를 확인했습니다. 이는 tensor 형식 검증이며 sample rate/품질이 확인된 30초 구간이라고 주장하지 않습니다. 현재 실제 2유도 파일은 확인되지 않았습니다.

이전 별도 source 검증 당시 기존 저장 Random Forest v4를 새 공통 로더/평가 경로로 validation 81명에 적용해 PID 유일성과 0~1 환자 확률 출력을 확인했습니다. 재학습·test 평가·MLflow 전송·원본 파일 수정은 수행하지 않았습니다. 실제 raw 학습은 기존 rate/quality provenance 조건이 충족되지 않아 수행하지 않았습니다.
