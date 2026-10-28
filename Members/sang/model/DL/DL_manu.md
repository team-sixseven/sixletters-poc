# ECG DL 사용 매뉴얼 — 준비부터 학습·평가·MLflow까지

이 문서는 `Members/sang/model/DL`의 **현재 raw ECG 파이프라인**을 사용하는 순서입니다.

```text
환자별 NPY·split·품질 인덱스 준비
→ 모델·학습 YAML 설정
→ train 학습 / validation으로 checkpoint 선택
→ 저장 가중치로 최종 test 평가
→ 학습·평가 결과를 개인 계정 MLflow에 기록
```

아래 터미널 명령은 **Linux/WSL의 저장소 루트**에서 실행합니다. Windows 사용자는 WSL Ubuntu 터미널을 사용하고, PowerShell 명령은 별도로 표시한 곳에서만 실행합니다. 저장소가 다른 위치에 있으면 `cd` 경로를 본인 위치로 바꿉니다.

**현재 기본 `basic.yaml`은 `source_metadata_confirmed: false`, `quality_index_path: null`입니다.** 원본의 sample rate·축 의미를 확인하고 실제 전처리 품질 인덱스를 준비하기 전에는 raw 학습을 중단하도록 되어 있습니다. 실행을 통과시키려고 확인 없이 값을 바꾸거나 품질 표시를 만들어 넣지 않습니다.

## 1. 실행 환경 준비

### 1-1. 저장소 루트로 이동

```bash
cd ~/sixletters-poc
```

루트에는 `Global/`, `Members/`가 있어야 합니다. `DL/` 안에서 아래 학습 CLI를 직접 실행하지 않습니다.

### 1-2. 기존 Python 환경 확인

```bash
Global/.venv/bin/python --version
Global/.venv/bin/python -c "import torch, numpy, pandas, sklearn, yaml; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

프로젝트는 Python 3.12와 고정한 uv 버전을 사용합니다. 기존 환경이 정상이라면 바로 다음 단계로 갑니다. 환경이 없거나 패키지가 빠져 있다면 저장소가 고정한 uv로 설치합니다.

```bash
uvx --from uv==0.9.26 uv sync --project Global --locked --extra gpu --extra ml
```

공통 평가 코드가 ML 모델 로더도 import하므로 이 환경에는 `ml` extra도 설치합니다. 위 GPU 환경 설정은 프로젝트의 Linux/WSL 환경 기준입니다.

### 1-3. 노트북을 사용할 경우 Jupyter 실행

커널 등록은 처음 한 번만 합니다.

```bash
Global/.venv/bin/python -m ipykernel install --user \
  --name sixletters-poc-global \
  --display-name "Python (sixletters-poc Global)"
Global/.venv/bin/python -m jupyterlab
```

브라우저에서 Jupyter를 열고 커널을 **Python (sixletters-poc Global)**로 선택합니다. 터미널의 Python과 다른 커널을 사용하면 설치 패키지가 달라질 수 있습니다.

## 2. 개인 MLflow 계정 연결

MLflow에 기록할 경우에 진행합니다. 오프라인 실행은 이후 노트북에서 `mlflow_logging=False`로 설정하거나, CLI에서 `--mlflow-logging`을 생략합니다.

### 2-1. 본인 계정 한 번 등록

다음은 `euna` 계정 예시입니다. **owner·username은 본인 계정, uri는 실제 MLflow 서버 주소**로 바꿉니다. 비밀번호는 프롬프트에서 입력합니다.

```bash
Global/.venv/bin/python -m sixletters setup \
  --owner euna \
  --username euna \
  --uri http://192.168.10.88:5000 \
  --experiment sang-scd-ml \
  --password
```

서버 PC 자체에서는 주소가 `http://127.0.0.1:5000`일 수 있습니다. 원격 PC는 서버의 실제 LAN 주소를 사용합니다. 주소·계정·Wi-Fi/유선 접속 설명은 [TEAM_MLFLOW.md](../tracking/TEAM_MLFLOW.md)에 있습니다.

설정은 저장소의 Git 제외 파일 `.poc.local.toml`에 보관합니다. 비밀번호를 YAML·노트북에 적지 않습니다. 브라우저 로그인만으로 Python 업로드 계정이 설정되지는 않습니다.

### 2-2. 접속 주소와 인증 확인

저장소 루트의 터미널에서 실행합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.tracking.check_connection
```

출력된 `owner`, `username`, `experiment`, 서버 주소가 맞는지 확인합니다. 이 명령은 비밀번호를 출력하거나 MLflow run을 생성하지 않고 인증·실험 조회를 확인합니다. 학습 코드도 MLflow를 켰을 때 학습 전에 이 검사를 호출합니다.

`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_EXPERIMENT_NAME` 환경 변수가 설정돼 있으면 파일보다 우선합니다. 다른 사람의 계정이 보이면 현재 커널과 환경 변수 설정을 확인합니다.

접속 자체가 안 되면 **팀원 PC의 Windows PowerShell**에서 먼저 확인할 수 있습니다.

```powershell
Test-NetConnection 192.168.10.88 -Port 5000
```

`False`이면 서버 동작·포트 전달·방화벽·대역 간 통신을 확인해야 합니다. 서버 설정 절차는 [팀 접속 문서](../tracking/TEAM_MLFLOW.md)를 따릅니다. TCP 접속 성공과 개인 계정 인증 성공은 각각 확인합니다.

## 3. 학습 데이터 준비

### 3-1. 파일 위치 확인

기본 배치는 다음과 같습니다.

```text
model/
├── basic.yaml
└── DL/
    └── data/
        ├── final_split_v2_augmented.csv
        ├── wave/
        │   ├── <pid>.npy
        │   └── ...
        └── <전처리에서 생성한 품질 인덱스>.csv
```

- Split CSV: `pid,split,SCD`. PID는 문자열, SCD는 0/1, split은 train/val/test입니다. `validation`도 지원합니다.
- Waveform: 현재 확인된 파일은 `(samples, measured_leads)` float16 NPY입니다. 환자별 `{pid}.npy`를 memory mapping으로 읽습니다.
- 실제 파일에서는 3유도가 확인됐고, 2·3유도 혼합 처리는 합성 데이터로 검증했습니다. 유도 이름을 열 번호로 추측하지 않습니다.
- 환자를 다른 split에서 가져오거나 ECG 구간을 임의 분할해 환자 split을 바꾸지 않습니다.

### 3-2. `basic.yaml`의 경로 확인

[basic.yaml](../basic.yaml)의 관련 항목입니다. **기존 파일 전체를 아래 발췌로 덮어쓰지 말고 해당 값만 수정**합니다.

```yaml
data:
  raw_ecg:
    split_path: DL/data/final_split_v2_augmented.csv
    wave_directory: DL/data/wave
    sample_rate_hz: 200
    source_metadata_confirmed: false
    segment_seconds: 30
    normalization: none
    quality_index_path: null
outputs:
  dl_model_directory: DL/models
```

경로는 실행 위치가 아니라 **해당 YAML 위치 기준**입니다. 여기서는 `basic.yaml`이 있는 `model/` 기준입니다.

1. NPY 생성 과정에서 실제 sample rate와 sample/lead 축 의미를 확인합니다.
2. 확인한 rate를 설정하고, 확인 완료 후에만 `source_metadata_confirmed: true`로 바꿉니다.
3. 실제 전처리 품질 CSV를 준비하고 `quality_index_path`에 경로를 넣습니다. 예를 들어 실제 파일이 그 위치에 있을 때 `DL/data/quality_index.csv`로 설정합니다.

NPY에는 sample rate가 내장돼 있지 않습니다. 현재 200 Hz는 관련 source 문서에 근거한 값이며 해당 raw 파일의 provenance와 대조해야 합니다. 200 Hz가 확인되면 30초는 6000 samples입니다. 예전 SQI-window 경로의 `2000`을 raw 입력 길이로 사용하지 않습니다.

`normalization: none`은 로더에서 추가 정규화를 하지 않는 설정입니다. 기존 전처리에 정규화가 포함됐는지 확인하지 않고 중복 적용하지 않습니다.

### 3-3. 품질 CSV 확인

품질 CSV의 schema 예시입니다. 실제 내용은 전처리 결과에서 받아야 합니다.

```csv
pid,window_index,usable
P0001,0,1
P0001,1,0
P0001,2,1
```

- `window_index`: 파일 시작 기준 0-based 30초 구간 번호
- `usable`: 명시적 0/1, 1인 구간만 사용
- 파일이 있는 환자의 완전한 30초 구간마다 한 행 필요
- 마지막 30초 미만 꼬리는 제외하고 sample 수를 audit에 기록
- 중복·누락·범위 초과·split에 없는 PID는 오류
- 파일이 아직 없는 환자는 누락으로 보고하고 제외

0으로 된 파형을 임의로 잡음으로 판정하거나 전체 구간을 확인 없이 `usable=1`로 만들지 않습니다. 별도 `datas/Holter/windows`의 `sqi_pass` 인덱스를 raw 파일에 재사용하지 않습니다.

### 3-4. 파일·split 감사 실행

```bash
Global/.venv/bin/python -m Members.sang.model.DL.dataloader.segmented_ecg
```

출력에서 split별 환자·클래스 수, 파일 존재 여부, 누락/잘못된 파일과 split에 없는 파일을 확인합니다. 이 명령은 학습하지 않으며 품질 CSV까지 승인해주는 명령도 아닙니다.

실제 raw 학습은 train과 validation에 모두 사용 가능한 환자가 있고, 두 split 각각에 양성·음성이 있어야 합니다. 유효 구간이 없는 환자는 제외 사유로 기록됩니다. 정보가 부족하면 라벨·품질 표시를 임의로 만들어 진행하지 않습니다.

## 4. 모델과 실험값 설정

### 4-1. CNN·시간 모델 선택

[model_choice.yaml](model_choice.yaml)에는 모델 이름 두 개만 둡니다.

```yaml
cnn_model: ResNet
sequence_model: BiLSTM
```

| 항목 | 가능한 값 |
|---|---|
| `cnn_model` | `ResNet`, `InceptionTime`, `TCN` |
| `sequence_model` | `Attention LSTM FCN`, `BiLSTM`, `transformer_enc` |

예를 들어 TCN + Transformer는 각각 `TCN`, `transformer_enc`로 바꿉니다. `combination`, 구현 경로, 입력/출력 shape, channels는 추가하지 않습니다.

### 4-2. 모델별 activation·dropout 수정

현재 [CNN 설정](CNN/model_set.yaml):

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

`modules: 3`은 Inception 모듈 3개를 연속으로 쌓는 값입니다. 탐색 후보 목록이 아닙니다. 그 외 CNN 크기·kernel·dilation은 코드 기본값을 사용합니다.

현재 [LSTM 설정](LSTM/model_set.yaml):

```yaml
models:
  Attention LSTM FCN:
    activation: GELU
    dropout: 0.0
  BiLSTM:
    dropout: 0.0
```

현재 [Transformer 설정](Transformer/transformer.yaml):

```yaml
models:
  transformer_enc:
    dropout: 0.1
    activation: GELU
```

중간 activation은 `GELU`, `ReLU`, `LeakyReLU`, `SiLU` 중 선택합니다. LSTM 내부 gate의 sigmoid/tanh는 바꾸지 않습니다. BiLSTM dropout은 기본 1-layer에서도 sequence 출력에 적용됩니다. 평가에서는 dropout이 비활성화됩니다.

### 4-3. 학습값·MLP 설정

[training.yaml](training.yaml)에서 자주 바꿀 학습값을 수정합니다.

```yaml
epochs: 20
batch_size: 1
learning_rate: 0.001
optimizer: Adam
weight_decay: 0.0001
positive_class_weight: train_negatives_divided_by_train_positives
early_stopping:
  patience: 5
  min_delta: 0.0001
seed: 42
device: auto
num_workers: 0
pin_memory: true
max_segments_per_patient: 128
classification_threshold: 0.5
classifier:
  hidden_dim: 32
  activation: GELU
  dropout: 0.1
mil:
  pooling: attention
  hidden_dim: 32
  activation: GELU
  dropout: 0.0
```

| 설정 | 의미 |
|---|---|
| `learning_rate` | 학습률 |
| `weight_decay` | 가중치 감쇠 |
| `batch_size` | 한 배치의 환자 수 |
| `epochs` | 최대 epoch; early stopping으로 먼저 끝날 수 있음 |
| `early_stopping` | validation AP 개선 여부를 기다리는 조건 |
| `max_segments_per_patient` | 환자당 사용할 최대 유효 구간 수 |
| `device` | auto / cpu / cuda |
| `classifier.activation` | 최종 MLP의 히든 레이어 활성화 함수 |
| `classification_threshold` | 확률을 이진 예측으로 바꾸는 기준 |

모델은 환자 logit과 `BCEWithLogitsLoss`로 학습하고, 추론·평가에서 sigmoid를 한 번 적용합니다. 클래스 보정값은 train label만으로 계산하며 best checkpoint는 validation Average Precision으로 선택합니다.

구간 cap 128은 유도 수나 구간 내 sample 수가 아닙니다. 유효 구간 전체에서 최대 128개를 균등 간격으로 선택하고 원래 순서·위치를 보존합니다. 전체 기록을 항상 모두 사용하는 것은 아니며 실제 사용 구간은 결과 CSV에 남습니다.

### 4-4. MIL 선택

Attention은 위 설정을 사용합니다. Mean 또는 max로 바꾸려면 **`mil` 블록 전체를 교체**합니다.

```yaml
mil:
  pooling: mean
```

```yaml
mil:
  pooling: max
```

Mean/max에는 attention용 `hidden_dim`, `activation`, `dropout`을 남기지 않습니다. 사용하지 않는 설정이 있으면 오류를 냅니다. 뒤의 `classifier` MLP는 계속 사용합니다.

유도 특징 통합과 MIL은 별개입니다. 실제 유도들은 공유 CNN 뒤에서 평균 통합하고, 여기서 선택하는 MIL은 **LSTM/Transformer 뒤의 여러 시간 구간을 환자 벡터 하나로 통합**합니다. 유도 mask와 시간 mask가 분리되며, 패딩 유도·제외 구간은 CNN 통계나 최종 pooling에 들어가지 않습니다.

## 5. 실행 방법 선택

| 방법 | 학습 MLflow | 최종 test |
|---|---|---|
| DL 노트북 | 기본 True | `run_final_test=True`이면 자동 실행 |
| Raw CLI | `--mlflow-logging`을 줘야 활성화 | 별도 평가 명령 필요 |
| 저장 모델 평가 노트북/CLI | YAML의 `mlflow_logging` | 저장 가중치를 평가하며 학습 없음 |

**노트북 학습과 CLI 학습은 같은 raw 학습 함수를 호출합니다. 둘을 연달아 실행하면 별도 학습 run이 두 개 생깁니다.** 원하는 방법 하나를 선택합니다.

## 6. 방법 A — 노트북으로 학습부터 평가까지

### 6-1. 노트북 열기

Jupyter에서 [model_experiment_only_DL.ipynb](model_experiment_only_DL.ipynb)를 엽니다. 처음 시작하거나 코드·환경이 갱신됐으면 커널을 재시작하고 맨 위부터 실행합니다.

### 6-2. 처음에는 짧은 smoke 실행

데이터 준비 조건을 충족한 뒤 실행 설정 셀을 다음처럼 바꿉니다.

```python
mlflow_logging = True
run_final_test = False
max_epochs = 1
limit_train_patients = 16
device_override = "cpu"
```

위에서 아래로 실행해 파일 감사, 로더 생성, 짧은 학습, checkpoint 저장·복원까지 확인합니다. Train은 최대 16명만 사용하지만 validation은 전체 사용 가능한 validation cohort입니다. Smoke에서는 최종 test를 실행하지 않습니다. MLflow를 켜면 축소 학습 여부도 기록합니다.

### 6-3. 전체 학습 설정

최종 확정한 설정으로 학습·최종 평가까지 진행할 경우 실행 설정을 다음처럼 둡니다.

```python
mlflow_logging = True
run_final_test = True
max_epochs = None
limit_train_patients = None
device_override = "auto"
```

- `max_epochs=None`: `training.yaml`의 epoch 사용
- `limit_train_patients=None`: 사용 가능한 전체 train cohort 사용
- `device_override`: YAML의 device보다 우선. YAML 값을 따르려면 `None` 사용
- 후보 비교·튜닝 중에는 `run_final_test=False`로 설정
- 오프라인 학습은 `mlflow_logging=False`로 설정

### 6-4. 셀 실행과 확인할 내용

1. **공통 import·설정 로딩:** 선택한 CNN과 시간 모델이 맞는지 확인합니다.
2. **파일 감사:** split별 실제 파일, 누락 사유를 확인합니다.
3. **품질 인덱스·로더·모델:** metadata/품질 검증이 통과하고 첫 배치 shape가 출력되는지 봅니다. 실패하면 원본 정보와 경로를 먼저 해결합니다.
4. **Train·validation:** epoch별 train loss와 validation AP를 확인합니다. 학습 후 `run_dir`가 출력됩니다.
5. **Checkpoint 복원:** best epoch와 validation 결과가 출력되는지 확인합니다.
6. **최종 test:** 플래그가 True이면 방금 학습한 `best_model.pt`를 다시 불러와 평가합니다. 학습과 평가를 별도 MLflow run으로 기록합니다.

최종 평가 셀은 이번 `run_dir`를 직접 지정한 임시 설정을 사용합니다. 이 흐름에서는 공통 `eval.yaml`의 target이 fusion이어도 DL로 덮어써 평가하며 원본 YAML은 수정하지 않습니다.

**이미 저장한 모델만 평가하려면 학습 셀을 다시 실행하지 말고 9절로 이동**합니다.

## 7. 방법 B — CLI로 학습

### 7-1. 짧은 학습

```bash
Global/.venv/bin/python -m Members.sang.model.DL.raw_pipeline \
  --max-epochs 1 \
  --limit-train-patients 16 \
  --device cpu \
  --mlflow-logging
```

오프라인이면 마지막 `--mlflow-logging`을 빼고 실행합니다. CLI smoke도 실제 run 폴더를 만들지만 전체 성능 실험으로 해석하지 않습니다.

### 7-2. 전체 학습

```bash
Global/.venv/bin/python -m Members.sang.model.DL.raw_pipeline \
  --device auto \
  --mlflow-logging
```

선택 파일·run 이름을 직접 정할 수도 있습니다. 기존 파일 구조는 유지합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.DL.raw_pipeline \
  --choice Members/sang/model/DL/model_choice.yaml \
  --run-id ecg_resnet_bilstm_001 \
  --device auto \
  --mlflow-logging
```

같은 run 이름의 폴더가 이미 있으면 덮어쓰지 않고 오류를 냅니다. 새 이름을 사용합니다. CLI의 마지막 `training_run` 경로를 복사해두고, **CLI는 최종 test를 자동 실행하지 않으므로 9절의 평가를 별도로 진행**합니다.

옵션 확인:

```bash
Global/.venv/bin/python -m Members.sang.model.DL.raw_pipeline --help
```

## 8. 학습 완료 결과 확인

기본 저장 위치는 `Members/sang/model/DL/models/<run_id>/`입니다. `basic.yaml:outputs.dl_model_directory`를 바꿨다면 출력 경로를 따릅니다.

| 파일 | 확인 내용 |
|---|---|
| `run_status.json` | `COMPLETE`인지 확인 |
| `best_model.pt` | validation AP 기준 best weights |
| `metadata.json` | 모델·run ID·threshold·MIL 방식·format version·전처리 정보 |
| `resolved_config.yaml` | 학습 당시 모델·학습 설정 snapshot |
| `training_history.csv` | epoch별 loss와 validation AP |
| `metrics.json` | validation 지표 |
| `validation_predictions.csv` | 환자별 확률·실제 라벨·threshold·사용 구간 |
| `train_patients.csv` | 실제 학습한 환자 목록 |
| `excluded_patients.csv`, `data_audit.json` | 누락·품질 제외·smoke cap 사유 |
| `mlflow_status.json`, `summary.json` | MLflow 활성 시 전송 상태와 링크 |

`run_status`는 학습 완료 상태이고 `mlflow_status`는 전송 상태입니다. 학습이 COMPLETE여도 전송은 PENDING일 수 있습니다. MLflow를 끈 실행에는 전송 상태 파일이 없을 수 있습니다.

원본 데이터·환자별 예측·PID audit은 자동 업로드하지 않습니다. MLflow에는 집계 지표, epoch 이력, checkpoint와 환자 정보 없는 network 설정을 기록합니다.

## 9. 저장된 DL 가중치만 불러와 평가

### 9-1. 평가할 run 선택

`best_model.pt`, `metadata.json`, `resolved_config.yaml`이 같은 run 폴더에 있어야 합니다. 가중치 파일 하나만 임의의 폴더로 복사하면 구조·전처리를 복원할 수 없습니다.

### 9-2. 평가 YAML 설정

[evaluation/eval.yaml](../evaluation/eval.yaml)은 ML·DL·fusion의 공통 평가 설정입니다. 기존 파일의 **아래 항목을 수정하고 metrics 등 다른 항목은 유지**합니다.

```yaml
evaluation:
  target: dl
  fusion_config: null
  model_path: ../DL/models/<run_id>/best_model.pt
  artifact_dir: null
  split: test
  threshold_override: null
  mlflow_logging: true
```

`<run_id>`를 실제 폴더명으로 바꿉니다. 다른 위치면 올바른 상대 경로나 절대 경로를 사용합니다. 이 상대 경로는 **eval.yaml이 있는 evaluation 디렉터리 기준**입니다.

- `model_path`: 저장된 `best_model.pt` 또는 해당 run 폴더
- `artifact_dir`: 기존 DL 폴더 지정 방식도 지원하지만 model_path 사용 시 null로 두는 것이 명확함
- 두 경로를 모두 주면 동일 run을 가리켜야 함
- `split: validation`: 후보 비교용 validation 재평가
- `split: test`: 선택을 마친 최종 평가
- `threshold_override: null`: 모델에 저장된 threshold 사용
- 오프라인 평가는 `mlflow_logging: false`

### 9-3. 평가 실행 — 노트북 또는 CLI 중 선택

평가 전용 노트북:

1. [evaluation/DL/evaluation_only_DL.ipynb](../evaluation/DL/evaluation_only_DL.ipynb)를 엽니다.
2. 커널을 동일한 프로젝트 환경으로 선택합니다.
3. 위에서 아래로 실행합니다. YAML target이 dl인지 확인한 뒤 저장 가중치로 평가합니다.

CLI:

```bash
Global/.venv/bin/python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml \
  --split test
```

두 경로 모두 **학습 함수를 호출하지 않고**, saved configuration으로 모델을 만들고 weights를 로드한 뒤 eval 모드에서 추론합니다. 현재 모델 YAML을 바꿨더라도 저장된 run의 설정을 사용합니다.

Raw run은 학습 때와 split/품질 CSV checksum, sample rate·segment 설정·정규화가 일치하는지도 확인합니다. 설정이나 품질 인덱스를 임의로 바꿔 기존 결과와 섞지 않습니다.

### 9-4. 평가 결과 확인

기본 출력은 `Members/sang/model/evaluation/results/dl/<evaluation_run_id>/`입니다.

- `predictions.csv`: PID, split, 실제 label, 환자 확률, threshold, 이진 예측, 모델/run ID
- `windows_used`, `first_window_index`, `last_window_index`, `used_window_indices`: 실제 사용한 구간 범위와 전체 인덱스
- `metrics.json`: 환자 수·클래스별 수, AUROC, Average Precision, sensitivity/recall, specificity, precision, F1, confusion matrix 등
- `resolved_config.yaml`: 실제 checkpoint 경로, 모델 및 평가 설정
- `data_audit/`: 사용 가능 환자와 누락/제외 사유
- `mlflow_status.json`: MLflow 활성 시 전송 결과

Average Precision은 sklearn AP이며 사다리꼴 적분 PR-AUC와 구분합니다. 단일 클래스나 분모 0으로 계산할 수 없는 지표는 `null`과 이유를 기록합니다.

Test 결과를 보고 모델·threshold·MIL을 반복 선택하지 않습니다. 비교는 validation에서 하고 최종 설정을 고정한 뒤 test를 사용합니다.

## 10. MLflow 전송 실패 시 재학습 없이 재시도

`PENDING`이면 연결 문제를 해결한 후 **학습·평가를 반복하지 않고 전송 함수만 실행**합니다. 같은 결과 재전송은 공통 업로더의 중복 방지 기능을 사용합니다.

저장소 루트에서 Python을 엽니다.

```bash
Global/.venv/bin/python
```

학습 결과 전송 예시입니다. `<run_id>`를 바꿉니다.

```python
from pathlib import Path
from Members.sang.model.tracking.log_dl import log_dl_training_to_mlflow

run_dir = Path("Members/sang/model/DL/models/<run_id>")
result = log_dl_training_to_mlflow(run_dir)
print(result["url"])
```

평가 결과 전송 예시입니다. `<evaluation_run_id>`를 바꿉니다.

```python
from pathlib import Path
from Members.sang.model.tracking.log_evaluation import log_evaluation_to_mlflow

evaluation_dir = Path("Members/sang/model/evaluation/results/dl/<evaluation_run_id>")
result = log_evaluation_to_mlflow(evaluation_dir)
print(result["url"])
```

노트북 커널이 살아 있고 해당 변수가 있으면 import 후 같은 함수를 바로 호출할 수 있습니다. 커널을 재시작했다면 위처럼 실제 경로로 변수를 다시 정의합니다. 전송 성공 시 FINISHED와 MLflow 링크를 확인합니다.

## 11. Clinical + ECG 결합이 필요한 경우

ECG 단독 평가는 `target: dl`입니다. Clinical까지 결합하려면 다음을 별도로 준비합니다.

1. 저장된 ML run과 DL run의 실제 경로를 [fusion.yaml](../fusion.yaml)에 지정합니다.
2. ML/DL의 동일 SCD 양성 클래스와 PID를 맞춰 확률을 결합합니다.
3. `mode: manual`의 가중치는 음수가 아니고 합이 1이어야 합니다.
4. `mode: learn`이면 가중치는 validation에서만 학습합니다. Test에서는 저장 가중치를 사용합니다.
5. [ECG+clinical_experiment.ipynb](../ECG+clinical_experiment.ipynb)를 실행하거나 공통 평가 YAML을 `target: fusion`, `fusion_config: ../fusion.yaml`로 바꿉니다.

결합 노트북은 저장된 기반 모델을 재학습하지 않습니다. 두 예측의 split/label이 충돌하면 중단하고, 한쪽에만 있는 환자는 제외 사유로 보고합니다. 자세한 공통 절차는 [TRAINING_GUIDE.md](../TRAINING_GUIDE.md)를 참고합니다.

## 12. 자주 만나는 오류와 다음 행동

| 증상 | 확인할 내용·행동 |
|---|---|
| `ModuleNotFoundError: Members` | 저장소 루트에서 모듈 명령을 실행하고 Jupyter 커널을 확인 |
| torch/xgboost/catboost import 실패 | Global 환경과 gpu/ml extra 설치 확인 |
| source metadata 확인 요구 | 원본 rate/축 provenance 확인 후 설정; 확인 없이 true로 우회하지 않음 |
| quality index 없음/누락/중복 | 실제 전처리 CSV 경로·전체 구간 coverage·PID·index 확인 |
| train/validation 단일 클래스/비어 있음 | 누락 파일과 품질 제외 현황 확인; 다른 split에서 환자를 이동하지 않음 |
| mean/max에 attention-only 설정 오류 | mil 블록을 pooling 한 항목만 남도록 교체 |
| CUDA unavailable | auto 또는 cpu 선택; GPU 환경과 실제 GPU 사용 가능 여부 확인 |
| GPU 메모리 부족 | batch size와 max_segments_per_patient를 줄여 새 run으로 실행 |
| MLflow HTTP/접속 오류 | 서버·주소·본인 인증·실험 권한·환경 변수 우선순위 확인 |
| MLflow PENDING | 로컬 결과 보존 상태 확인 후 전송만 재시도 |
| 모델 경로 없음/companion 파일 없음 | 실제 best_model.pt와 metadata/config가 같은 run 폴더에 있는지 확인 |
| model_path/artifact_dir 충돌 | 한쪽을 null로 두거나 동일 run으로 맞춤 |
| split/품질 checksum 변경 | 학습 때 source를 복원하거나 변경 데이터로 새 학습 run 생성 |
| `predates shared-lead format v2` | 기존 고정 유도 checkpoint는 원래 코드로 읽거나 현재 구조로 재학습 |
| run 폴더 이미 존재 | 새 run-id 사용; 완료/실패 폴더를 덮어쓰지 않음 |

## 13. 코드 수정 후 검증과 팀 공유

`tests/`는 평소 학습 실행에 필수인 폴더가 아닙니다. 코드·패키지·환경을 바꿨다면 다음 자동 검증을 실행할 수 있습니다.

```bash
Global/.venv/bin/python -m pytest Members/sang/model/DL/tests/test_pipeline.py -q
```

혼합 유도·가변 시간 길이, 마스크, 세 MIL 방식, gradient, 기본값, checkpoint 복원·환자 평가·모의 MLflow 연동을 검증합니다. 합성 검증을 임상 성능 검증으로 해석하지 않습니다.

팀에는 코드·YAML·매뉴얼·테스트를 공유하고, 개인 계정 설정·비밀번호·실제 환자 데이터·생성 checkpoint/결과·캐시는 Git에 넣지 않습니다. 현재 폴더 역할·구현 선택·검증 근거는 [TRAINING.md](TRAINING.md)에 더 자세히 정리돼 있습니다.

### 기존 SQI-window 명령과의 구분

```bash
Global/.venv/bin/python -m Members.sang.model.DL.training
```

이 명령은 별도 `basic.yaml:data.ecg`의 과거 SQI-window source를 사용합니다. 이 매뉴얼의 raw waveform 학습 명령은 **`DL.raw_pipeline`**입니다. 두 source의 입력·품질 인덱스를 섞지 않습니다.
