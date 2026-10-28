# 모델 학습·평가 가이드

이 문서는 각 팀원이 **자기 PC·자기 MLflow 계정으로 학습하고 저장된 모델을 평가**하는 통합 실행 가이드입니다. 상세 모델 파라미터는 [ML 매뉴얼](ML/ML_manu.md), [DL 매뉴얼](DL/DL_manu.md), [Fusion 매뉴얼](fusion_manu.md)을 함께 참고합니다. **ML·DL 각각의 노트북은 개인 계정 MLflow 자동 기록과 마지막 test 평가를 포함합니다.** 기본 `mlflow_logging=True`, `run_final_test=True`이며, 후보 비교 중에는 test를 끕니다. Raw DL 실행 절차의 기준은 [`DL/TRAINING.md`](DL/TRAINING.md)이며 아래의 기존 SQI-window source 설명과 구분합니다. 권장 순서는 **공통 준비 → clinical ML → ECG DL → ECG+clinical 결합 → 고정한 설정으로 test 평가**입니다. 이미 ML과 DL 모델 artifact가 있으면 결합부터 진행할 수 있습니다.

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
cd ~/sixletters-poc
uvx --from uv==0.9.26 uv sync --project Global --locked --extra ml --extra gpu
```

위 설치는 Linux/WSL x86_64 기준입니다. 공통 평가 모듈은 ML·DL 로더를 함께 import하므로 두 extra를 모두 준비합니다. Windows 사용자는 WSL Ubuntu 터미널에서 실행합니다. 기존 환경이 있다면 다음으로 먼저 확인합니다.

```bash
cd ~/sixletters-poc
Global/.venv/bin/python --version
Global/.venv/bin/python -c "import torch, pandas, sklearn, yaml, xgboost, catboost; print('환경 확인 완료'); print('CUDA:', torch.cuda.is_available())"
```

이 문서의 명령은 **저장소 루트에서 `Global/.venv/bin/python`**으로 실행합니다. `~/sixletters-poc`는 실제 설치 경로로 바꿉니다. 시스템 uv 버전을 바꾸지 않고도 위 고정 버전 명령으로 설치할 수 있습니다.

노트북 사용 시 처음 한 번 커널을 등록하고 Jupyter를 시작합니다.

```bash
Global/.venv/bin/python -m ipykernel install --user \
  --name sixletters-poc-global \
  --display-name "Python (sixletters-poc Global)"
Global/.venv/bin/python -m jupyterlab
```

브라우저에서 **Python (sixletters-poc Global)** 커널을 선택합니다. 이미 노트북을 열었다면 계정·환경 변경 후 커널을 재시작하고 위에서부터 실행합니다.

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
```

경로를 옮겼다면 이 파일의 `path` 또는 `directory` 값을 실제 위치에 맞게 바꿉니다. ML 입력 feature는 `model_version`별 목록으로 선택합니다. CSV에는 입력 변수와 ID·라벨을 두고 split 정보는 별도 split CSV로 관리합니다. 범주형 열은 공통 설정과 선택한 입력 버전에 따라 처리합니다. 임상 CSV 열 이름은 메모리에서 앞뒤 공백을 정리하며, 이 과정에서 중복 열 이름이 생기면 로딩을 중단합니다. 원본 데이터 파일은 수정하지 않습니다.

Split 열 기본값은 `pid`·`split`이며 CSV의 `val`을 내부 `validation`으로 매핑합니다. sidecar의 `pid`·`row`·`start_sec`·`sqi_pass`, 파일명 규칙, SQI 필터와 정규화는 `config.py`의 공통 기본값입니다.

현재 주력 raw DL 설정은 같은 `basic.yaml`의 다음 블록입니다. 위 `data.ecg`는 기존 SQI-window 모델용입니다.

```yaml
  raw_ecg:
    split_path: DL/data/final_split_v2_augmented.csv
    wave_directory: DL/data/wave
    sample_rate_hz: 200
    sample_rate_source: MUSIC_recording_headers_documented_in_members_sang_docs
    source_metadata_confirmed: false
    segment_seconds: 30
    normalization: none
    quality_index_path: null
```

위 블록은 `data:` 아래에 둡니다. Raw NPY는 sample-major `[sample, lead]`이며 실제 유도 수를 읽습니다. 기존 SQI-window 배열은 `[window, lead, sample]`이며 과거의 2000 samples는 200 Hz 기준 10초 구간입니다. 이를 raw 30초 구간 길이로 사용하지 않습니다. Raw DL은 유도 수와 기록 길이를 별도로 처리합니다.

**현재 기본 설정에서는 raw 원본 metadata 확인과 실제 품질 인덱스 준비가 남아 있어 DL·fusion 실데이터 실행이 차단됩니다.** `source_metadata_confirmed`는 원본 sampling rate·축 의미를 실제 확인한 뒤 설정하고, `quality_index_path`에는 실제 전처리에서 생성한 `pid,window_index,usable` CSV 경로를 지정합니다. 실행을 위해 모든 구간을 임의로 정상 처리하지 않습니다. [DL 매뉴얼](DL/DL_manu.md)의 준비 절차를 먼저 따릅니다.

ML은 선택한 `model_version`의 feature 목록으로 입력을 구성합니다. 현재 split 경로는 ML과 raw DL이 다르므로 fusion에서는 동일 환자의 split·라벨 일치 여부도 확인합니다. 이전 실험의 고정 환자 수를 현재 실행에 적용하지 말고 각 run의 감사 자료와 metrics를 확인합니다.

### 1.3 데이터 정합성 검사

저장소 루트에서 실행합니다. 이 audit는 기본 설정의 CSV, split, ECG 파일 헤더와 sidecar 정합성 및 modality별 환자 범위를 검사합니다. 학습은 수행하지 않습니다.

```bash
Global/.venv/bin/python Members/sang/scripts/audit_model_data.py
```

이 audit는 기존 공통 clinical/SQI source 기준입니다. Raw waveform의 split·파일·품질 검증은 raw 학습/평가 로더에서도 별도로 수행합니다.

정상 실행 시 터미널에 `clinical_patients`, `split_patients`, `ecg_patients`, 교집합·제외 환자 수, split별 환자 및 라벨 수가 출력됩니다. 상세 audit 파일은 기본적으로 `model/reports/data_audit/` 아래에 run ID별로 생성됩니다. 경로를 바꾸려면 `basic.yaml`에 `outputs.audit_directory`를 지정합니다. 환자 ID가 포함된 local 산출물이므로 외부 공유에 주의하세요. 예상보다 환자 수가 적거나 필수 데이터 파일을 못 찾으면 학습으로 넘어가지 말고 경로와 ID·split 매핑부터 확인합니다.

### 1.4 각자 자기 MLflow 계정 등록

#### 계정·owner·experiment·run의 차이

**개인 계정으로 로그인하는 것과 개인 experiment를 사용하는 것은 서로 다른 선택입니다.** MLflow experiment는 여러 실행 결과를 묶는 공간이고, 계정은 그 공간에 접근할 때 사용하는 인증 정보입니다. 계정마다 experiment 이름이 자동으로 달라지는 구현은 아닙니다.

| 항목 | 예시 | 현재 코드에서의 역할 |
| --- | --- | --- |
| 서버 주소 `--uri` | `http://192.168.10.88:5000` | 결과를 보낼 MLflow 서버 |
| 인증 계정 `--username` | `euna` | 서버 로그인과 접근 권한 확인 |
| 작성자 `--owner` | `euna` | 개인 로컬 전송 폴더와 run의 `sixseven.owner` 태그 |
| 실험 이름 `--experiment` | `sang-scd-ml` 또는 `euna-scd-ml` | 해당 서버에서 결과를 모을 experiment 선택 |
| 개별 실행 run | 한 번의 ML 학습, DL 학습, test 평가 | experiment 안에 만들어지는 실행별 기록 |

따라서 `--username euna --experiment sang-scd-ml`은 **은아 계정으로 인증해 `sang-scd-ml`이라는 experiment에 기록**한다는 뜻입니다. Experiment 이름에 `sang`이 들어가도 인증 계정이 sang으로 바뀌지 않습니다. 반대로 `--experiment euna-scd-ml`만 바꾸고 username을 sang으로 두면 은아 계정으로 인증되는 것도 아닙니다. 보통 username과 owner를 본인 ID로 동일하게 설정하고 experiment 이름은 아래 두 운영 방식 중에서 선택합니다.

#### 방식 A: 팀 공용 experiment에 모으기

기존 가이드의 `sang-scd-ml`은 **팀 공용으로 사용하도록 안내한 experiment 이름**입니다. 이름에 ML이 들어가지만 현재 코드는 같은 experiment에 ML·DL 학습과 각각의 평가·fusion 평가도 기록할 수 있습니다. 이름 자체가 기록 가능한 모델 종류를 제한하지 않습니다.

```text
공용 experiment: sang-scd-ml
├── euna의 ML 학습 run        [sixseven.owner=euna]
├── minjeong의 DL 학습 run    [sixseven.owner=minjeong]
├── euna의 ML test 평가 run   [sixseven.owner=euna]
└── mingyu의 fusion 평가 run  [sixseven.owner=mingyu]
```

팀 전체의 모델과 성능을 같은 experiment에서 비교하기 편한 방식입니다. 여러 사람이 같은 이름을 사용해도 매 실행은 별도 run으로 기록됩니다. MLflow 화면에서 `sixseven.owner` 태그를 열로 추가하거나, 다음 검색 조건으로 본인 결과를 고를 수 있습니다.

```text
tags.`sixseven.owner` = 'euna'
```

owner 태그는 작성자를 구분하기 위한 정보이며 접근 권한을 설정하는 기능은 아닙니다. 조회·기록 가능 여부는 서버의 계정과 experiment 권한을 따릅니다.

#### 방식 B: 개인별 experiment로 분리하기

개인별 실험 공간을 원하면 experiment 이름도 각자 다르게 지정합니다. 다음은 이름 예시이며 서버에 이미 존재한다는 뜻은 아닙니다.

| 사용자 | username / owner | 개인 experiment 예시 |
| --- | --- | --- |
| 은아 | `euna` | `euna-scd-ml` |
| 민정 | `minjeong` | `minjeong-scd-ml` |
| 민규 | `mingyu` | `mingyu-scd-ml` |
| 성률 | `seongryul` | `seongryul-scd-ml` |

개인 결과를 찾기 쉬운 대신 팀 전체 결과가 여러 experiment로 나뉩니다. 이름을 개인별로 나누는 것만으로 다른 사람의 접근이 차단되지는 않으며, 비공개 공간이 필요하면 서버 권한도 별도로 설정해야 합니다.

개인 experiment를 쓰는 순서는 다음과 같습니다.

1. 본인 서버 계정이 준비됐는지 확인합니다.
2. MLflow 웹 화면에서 본인 계정으로 로그인합니다. 생성 권한이 있으면 experiment 생성 기능으로 정확한 이름(예: `euna-scd-ml`)을 만듭니다. 권한이 없으면 서버 관리자에게 생성과 본인 계정의 조회·기록 권한을 요청합니다.
3. 아래처럼 로컬 연결 설정을 저장합니다. `euna`와 서버 주소는 본인 환경에 맞춥니다.

```bash
Global/.venv/bin/python -m sixletters setup \
  --owner euna \
  --username euna \
  --uri http://192.168.10.88:5000 \
  --experiment euna-scd-ml \
  --password
```

4. 연결 확인 명령을 실행하고 출력의 `experiment`가 `euna-scd-ml`인지 확인합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.tracking.check_connection
```

5. 열려 있던 노트북 커널을 재시작하고 공통 설정 셀부터 실행합니다. 이후 이 가이드의 학습·평가 명령을 동일하게 사용합니다. 모델 YAML이나 Python import 경로를 개인 experiment 이름에 맞춰 바꿀 필요는 없습니다.

**`setup --experiment`는 로컬에서 사용할 이름을 저장하는 명령입니다.** 현재 학습 노트북과 ML·DL 학습 경로는 학습 전에 experiment가 조회되는지 확인하므로, 새 이름만 적고 실행하면 사전 확인에서 중단될 수 있습니다. 먼저 서버에 experiment와 접근 권한을 준비합니다. 공통 업로더 자체에는 없는 experiment를 생성하는 처리가 있지만, 이것이 학습 전 조회 확인을 대신하지는 않습니다.

#### 공용 experiment로 설정하는 실행 예시

계정은 **코드의 `Members/sang/` 경로와 별개**입니다. 팀원이 이 코드를 그대로 실행해도 업로드 계정은 각 PC의 로컬 설정을 사용합니다. 모델 코드를 `Members/본인/`으로 복사하거나 Python import를 바꿀 필요는 없습니다. owner는 개인 결과 위치·태그, username은 서버 인증 계정입니다. 같은 PC·같은 checkout을 여러 사람이 쓰면 로컬 설정을 공유하므로 실행 전에 항상 계정을 확인합니다.

1. 서버 관리자에게 개인 username·비밀번호와 사용 가능한 공용 experiment 이름을 받습니다. 계정 등록 명령은 서버 계정을 새로 만드는 기능이 아닙니다.
2. 서버 PC 자체는 `http://127.0.0.1:5000`, 팀원 PC는 서버에 도달 가능한 실제 주소를 사용합니다.
3. 아래는 **방식 A의 공용 experiment**를 쓰는 예시입니다. 개인별로 분리하려면 방식 B의 명령을 대신 사용합니다. `euna`와 주소는 본인 값으로 바꿉니다.

```bash
Global/.venv/bin/python -m sixletters setup \
  --owner euna \
  --username euna \
  --uri http://192.168.10.88:5000 \
  --experiment sang-scd-ml \
  --password
```

비밀번호는 프롬프트에서 입력합니다. `.poc.local.toml`에 저장되며 Git에서 제외됩니다. 브라우저 로그인만으로 Python 업로드 계정이 설정되지는 않습니다. 다른 사람의 설정 파일을 복사하지 않습니다.

환경 변수 `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_EXPERIMENT_NAME`이 비어 있지 않으면 파일보다 우선합니다. 설정 후 확인합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.tracking.check_connection
```

출력의 `owner`, `username`, `experiment`, `uri`, `environment_overrides`를 확인합니다. 비밀번호를 출력하지 않고 인증·experiment 조회만 확인하며 run을 만들지 않습니다. 다른 계정이 나오면 터미널·Jupyter 커널의 환경 변수와 로컬 설정을 확인합니다. Experiment 조회 실패는 이름과 접근 권한을 관리자에게 확인합니다.

#### 학습과 evaluation은 어느 experiment에 기록되는가?

현재 ML·DL 학습과 공통 evaluation은 실행 시점의 같은 개인 연결 설정을 사용합니다. 예를 들어 은아가 `euna-scd-ml`을 설정해 두고 ML 학습과 test 평가를 실행하면, 두 실행은 **같은 `euna-scd-ml` 안의 별도 run**으로 기록됩니다. Fusion 평가도 해당 설정을 사용합니다. 학습용·평가용 experiment를 YAML에 중복 지정하지 않습니다.

계정 이름에서 experiment를 자동 생성하거나 ML·DL·fusion별 experiment로 자동 분리하지 않습니다. 실행 종류는 run의 모델 정보와 단계, 평가 split 등으로 구분합니다. MLflow에 기록하려면 1.5절의 logging 옵션도 켜야 합니다.

Experiment를 바꾸려면 원하는 이름으로 setup을 다시 실행하고 연결 확인을 합니다. `MLFLOW_EXPERIMENT_NAME` 환경 변수가 설정돼 있으면 setup에서 저장한 이름보다 우선하므로 출력의 `environment_overrides`도 확인합니다. 브라우저에서 다른 experiment를 선택하는 것만으로 Python 전송 대상이 바뀌지는 않습니다.

학습 이후 평가 전에 experiment를 바꾸면 평가 run은 변경된 experiment에 기록되며, 학습 run은 원래 위치에 남습니다. 기존 run이 이동하지 않습니다. 같은 로컬 결과를 다른 experiment로 재전송하면 별도 기록이 생길 수 있으므로, 재시도할 때는 원래 사용한 서버·계정·owner·experiment를 유지합니다.

Wi-Fi ↔ 유선 연결이 의심되면 **팀원 PC의 Windows PowerShell**에서 확인합니다.

```powershell
Test-NetConnection 192.168.10.88 -Port 5000
```

`TcpTestSucceeded: False`이면 계정 설정 전에 네트워크 경로부터 해결해야 합니다. 서버 주소·TCP 5000·라우팅 설명은 [TEAM_MLFLOW.md](tracking/TEAM_MLFLOW.md)를 따릅니다. 서버 자기 자신의 연결 성공은 팀원 PC 연결 확인을 대신하지 않습니다.

### 1.5 자동 기록을 켜는 위치

| 실행 경로 | 학습/평가 MLflow 설정 | 최종 test |
| --- | --- | --- |
| ML·DL 학습 노트북 | `mlflow_logging=True` | `run_final_test=True`일 때 저장 모델로 별도 평가 |
| Fusion 노트북 | `mlflow_logging=True` | validation 가중치 저장 후 `run_final_test=True`일 때 평가 |
| ML 학습 Python API | `train_clinical(..., mlflow_logging=True)` | 별도 공통 평가 실행 |
| Raw DL CLI | `--mlflow-logging` | 별도 공통 평가 실행 |
| 저장 모델 공통 평가 CLI | `evaluation/eval.yaml`의 `evaluation.mlflow_logging: true` | `--split test`로 명시 |

학습과 평가는 별도 MLflow run입니다. 학습 run에는 모델·설정·validation 집계, 평가 run에는 validation/test 집계를 기록합니다. 각 노트북의 기본 `run_final_test=True`는 **후보 비교 중에는 False로 바꿉니다.** 모델·가중치·임계값을 validation으로 고른 뒤 최종 test를 수행합니다.

오프라인은 Python/노트북의 logging을 False로, DL CLI는 옵션 생략, 평가 YAML은 false로 설정합니다. 이후 저장된 결과만 업로드할 수 있으며 6.2절에 명령이 있습니다.

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

### 2.2 노트북 학습·MLflow·최종 평가

`ML/model_experiment_only_ML.ipynb`를 열고 실행 설정에서 다음을 선택한 뒤 전체 셀을 실행합니다.

```python
model_version = 4
training_mode = "optuna"  # "yaml"이면 YAML 고정값 학습
model_name = "XGBoost"   # Optuna: 알고리즘 하나
n_trials = 100
reuse_saved = False      # yaml: 새로 학습; True이면 동일 버전 저장 모델 재사용
mlflow_logging = True
run_final_test = True    # 후보 비교·튜닝 중이면 False
retry_mlflow = False
```

`yaml`은 `ML/model_choice.yaml`의 모델 목록을 처리하고 각 모델의 native weights·전처리·설정·validation 결과를 자동 기록합니다. `optuna`는 선택한 알고리즘·입력 버전으로 study 하나만 실행하고 best 하나만 기록합니다. Trial 모델은 임시 폴더에서 학습하며 기록하지 않습니다. 기존 objective의 조건부 탐색 범위와 best-only 저장 정책은 유지합니다. 전체 셀 실행이 두 경로를 중복 학습하거나 전송 재시도를 자동 반복하지 않습니다.

학습 전에는 각 PC의 `.poc.local.toml` 및 환경 변수로 개인 계정 연결을 확인합니다. 학습 후 전송 실패는 모델을 보존하고 `summary.json`/`mlflow_status.json`에 `PENDING`을 남깁니다. 재학습 없는 재전송은 `Members.sang.model.tracking.log_best.log_ml_training_to_mlflow(run_dir)`로 실행합니다. 계정 설정은 [`tracking/TEAM_MLFLOW.md`](tracking/TEAM_MLFLOW.md)를 참고하세요.

마지막 최종 평가 셀은 YAML 학습의 `run_dirs` 각각 또는 Optuna의 `best_run_dir` 하나를 공통 `evaluation`에 직접 전달합니다. 저장된 전처리·best 모델·threshold를 그대로 사용하고, test 결과와 학습 MLflow run 연결을 별도 평가 run으로 기록합니다. 원본 `eval.yaml`을 바꾸거나 평가 중 학습하지 않습니다. Test 결과를 보고 모델·입력 버전·threshold를 다시 고르지 않습니다.

아래 CLI 경로는 노트북과 별도입니다. Python API는 오프라인 호환성을 위해 logging 기본값이 False이므로, 직접 호출할 때 자동 기록은 `mlflow_logging=True`를 지정합니다.

### 2.3 같은 학습의 명령행 실행

각 모델의 parameter YAML을 먼저 저장한 뒤 아래 명령 전체를 실행합니다. 예시는 XGBoost 입력 버전 4를 **새로 학습하고 개인 계정 MLflow에 기록**합니다.

```bash
Global/.venv/bin/python - <<'PY'
from pathlib import Path
from Members.sang.model.ML.training import train_clinical

runs = train_clinical(
    Path("Members/sang/model/ML/model_choice.yaml"),
    model_names=["XGBoost"],
    model_version=4,
    reuse_saved=False,
    mlflow_logging=True,
)
for run in runs:
    print("학습 결과:", run.resolve())
PY
```

`model_names`는 `XGBoost`, `CatBoost`, `Random Forest` 중 선택합니다. `model_version`은 1~4 입력 feature 버전입니다. `reuse_saved=False`는 YAML 설정으로 새로 학습합니다. True이면 해당 버전의 등록된 모델이 유효할 때 재사용하고, 없으면 학습 후 등록합니다. 새 실험에는 False를 사용합니다.

이 명령은 test를 자동 실행하지 않습니다. 출력된 run 폴더를 평가 YAML에 지정합니다. API 기본값은 `mlflow_logging=False`이므로 위 True를 생략하지 않습니다. `reuse_saved=False`로 만든 새 run은 parameter YAML의 `model_path`를 자동 갱신하지 않으므로 평가할 때 실제 폴더를 명시합니다.

기존 `Members/sang/scripts/evaluate_preprocessed_clinical.py`는 호환 학습/재사용 진입점이며 `--models`, `--run-id`, `--limit-train-patients`만 제공합니다. 이 스크립트만 실행하면 MLflow 기록이나 test가 자동 실행되지 않습니다. 개인 계정 자동 기록은 위 API나 노트북을 사용합니다. Optuna 탐색은 노트북에서 `training_mode="optuna"`를 선택하고 [ML 매뉴얼](ML/ML_manu.md)을 따릅니다.

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

`artifact_dir: null`이면 `model_name`·`model_version`에 연결된 parameter YAML의 등록 경로를 읽습니다. 새 run을 평가할 때는 실제 `artifact_dir`를 명시합니다. 저장 모델이 없으면 먼저 일반 학습이나 Optuna 결과 저장을 실행해야 합니다. 특정 run을 지정하려면 `artifact_dir`를 `eval.yaml` 파일 위치인 `model/evaluation/` 기준 상대 경로로 설정합니다. 설정을 저장한 뒤 저장소 루트에서 평가 모듈을 실행합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml
```

저장된 모델 재로드는 `metadata.json`의 `artifact_type`에 따라 [inference.py](inference.py)가 처리합니다. 재평가는 학습을 다시 하지 않습니다. 평가 출력은 `evaluation/results/ml/<evaluation_run_id>/` 아래의 `predictions.csv`, `metrics.json`, `resolved_config.yaml`, `data_audit/`입니다.

## 3. ECG만 사용하는 DL

상세 절차와 검증 근거는 [`DL/TRAINING.md`](DL/TRAINING.md)에 있습니다. Raw source는 `basic.yaml:data.raw_ecg`의 split·waveform·확인된 sample rate·30초 품질 인덱스를 사용합니다. 별도 기존 SQI source의 2000-sample/10-second window를 raw 입력 규격으로 사용하지 않습니다.

`DL/model_choice.yaml`에는 모델 이름 두 개만 둡니다.

```yaml
cnn_model: ResNet
sequence_model: BiLSTM
```

CNN은 ResNet/InceptionTime/TCN, 시간 모델은 Attention LSTM FCN/BiLSTM/transformer_enc 중 선택합니다. 클래스 등록과 특징 차원 연결은 코드가 처리합니다. 각 모델 YAML에는 flat 실험 하이퍼파라미터만 두며 `implementation`, `feature_method`, `input`, `output`, `channels`는 사용하지 않습니다. Transformer YAML에는 `dropout`·`activation`만 두고, embedding 64·head 4·layer 2·FFN 128은 코드 기본값을 사용합니다. CNN 출력에서 embedding 차원으로 자동 projection합니다. Threshold는 `DL/training.yaml:classification_threshold`, MIL/MLP는 같은 파일의 `mil`/`classifier`에서 수정합니다.

환자는 실제 유도 수를 유지하고, 배치에서만 유도/시간 축을 각각 패딩합니다. 별도 measured-lead mask와 time mask로 유효 유도·구간만 공유 1유도 CNN에 입력하고, 유도 특징 평균을 구간 벡터로 통합합니다. 유도 이름은 가정하지 않으며 실제 세 번째 유도도 사용합니다. CNN의 `feature_dim`으로 시간 모델 입력 차원을, 시간 모델의 `feature_dim`으로 MIL/MLP 차원을 결정합니다. Packed LSTM, Transformer key-padding mask, masked MIL이 시간 padding을 처리하며 원래 구간 위치와 gap을 보존합니다.

`DL/model_experiment_only_DL.ipynb`를 열고 기본 `mlflow_logging=True`, `run_final_test=True`로 실행하면 파일 감사 → 로더·모델 생성 → 학습/validation → MLflow → best checkpoint 복원 → 최종 test 평가/MLflow 순서로 진행합니다. 후보 비교 중에는 test를 끕니다. Raw metadata/품질 CSV가 없으면 원래 요구대로 학습을 중단합니다.

CLI는 저장소 루트에서 다음과 같습니다. CLI 학습 후 평가는 별도 공통 evaluation 명령으로 실행합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.DL.raw_pipeline --device auto --mlflow-logging
```

결과는 `basic.yaml:outputs.dl_model_directory` 아래 저장합니다. 새 checkpoint는 format version 2이며 기존 고정 유도 checkpoint와 자동 호환되지 않습니다. 원본 코드로 읽거나 새 구조로 재학습해야 합니다. 새 checkpoint는 saved configuration으로 복원하므로 실험 YAML이 나중에 바뀌어도 원래 모델을 재구성합니다.

학습 완료 시 출력된 `training_run` 폴더를 기록합니다. `--run-id ecg_resnet_bilstm_01`로 이름을 지정할 수 있습니다. 작은 실행 확인은 다음과 같습니다. 실제 원본 정보·품질 인덱스 준비 조건은 동일합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.DL.raw_pipeline \
  --run-id ecg_smoke_01 \
  --max-epochs 1 \
  --limit-train-patients 16 \
  --device cpu \
  --mlflow-logging
```

위 결과는 실행 경로 확인용입니다. 본 학습에서는 `--max-epochs`, `--limit-train-patients`를 빼고 `DL/training.yaml`을 사용합니다. CLI는 test를 자동 실행하지 않으므로 저장 가중치 평가를 5.1절에서 별도로 진행합니다.

기존 SQI-window 실행이 필요한 경우에만 `python -m Members.sang.model.DL.training`을 별도 사용합니다. 이 경로도 실제 NPY 유도/sample 축 길이를 그대로 읽고 같은 공유 유도 모델과 평가 인터페이스를 사용합니다.

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
    max_iterations: 100
    tolerance_grad: 1.0e-9
```

`learning.split`은 현재 설정값이 `validation`입니다. 코드도 validation split에서만 가중치를 학습합니다. Test로 바꾸어 가중치를 학습하는 설정은 지원하지 않습니다.

### 4.3 결합 노트북 실행 및 매칭 결과

[ECG+clinical_experiment.ipynb](ECG+clinical_experiment.ipynb)를 열고 아래 순서로 실행합니다. 전체 명령은 [fusion_manu.md](fusion_manu.md)를 참고합니다.

1. `fusion.yaml`의 실제 ML·DL 폴더와 mode·weights·threshold를 확인하고 저장합니다.
2. Global 커널에서 공통 import 셀부터 실행합니다.
3. 후보 비교 중에는 `mlflow_logging=True`, `run_final_test=False`로 설정합니다.
4. 평가 셀은 공통 평가기를 통해 validation 결과를 생성하고 MLflow에 기록합니다.
5. 출력된 `Validation:` 폴더의 metrics·예측·가중치·제외 명단을 확인합니다.
6. 모델·가중치·임계값을 확정하면 `run_final_test=True`로 바꾸고 설정 셀과 평가 셀을 실행합니다. 해당 validation 결과를 saved 모드로 연결해 같은 가중치로 test를 평가·기록합니다.

노트북은 임시 설정을 사용하므로 원본 평가 YAML을 바꾸지 않습니다. 환자 ID로 매칭하며 라벨·split 불일치 시 중단합니다. 사용 환자 수와 제외 수는 실행 결과의 `n`, `n_excluded_no_pair`, 감사 자료에서 확인합니다.

단계 구분:

- **기본 모델 추론:** 저장된 ML·DL 모델과 preprocessing/checkpoint를 로드하여 확률을 계산합니다. 재학습하지 않습니다.
- **가중치 학습:** `mode: learn`일 때에만 validation의 두 기본 모델 확률을 이용해 가중치를 최적화합니다. 이것은 ML·DL 모델 학습이 아닙니다.
- **가중치 재사용:** 결합 run에는 `fusion_weights.json`과 `resolved_config.yaml`이 저장됩니다. 재사용하려면 `fusion.mode: saved`, `fusion.saved_run_dir`에 이전 결합 결과 run 경로를 지정합니다. 저장된 run의 base model artifact 경로와 weights를 다시 사용합니다.

수동 weights와 validation에서 학습한 weights 모두 같은 `run_fusion` 경로를 사용합니다. test 최종 평가에는 validation에서 고정한 weights를 적용하고 test 성능을 확인한 뒤 weights나 threshold를 다시 조정하지 않습니다.

### 4.4 결합 test 평가

결합 노트북도 개인 계정으로 validation을 기록하고 `run_final_test=True`이면 저장 가중치를 재사용해 test 평가까지 기록합니다. 별도 Test 재평가는 공통 evaluation CLI를 사용하면 됩니다. `fusion.yaml`을 `mode: saved`로 지정해 validation 가중치 run을 재사용하거나, `manual`로 이미 결정한 weights를 유지합니다. 다음으로 `evaluation/eval.yaml`을 설정합니다.

```yaml
evaluation:
  target: fusion
  fusion_config: ../fusion.yaml
  split: test
  threshold_override: null
```

저장소 루트에서 실행합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml --split test
```

평가 결과는 `evaluation/results/fusion/<evaluation_run_id>/test_predictions.csv`, `metrics.json`, `excluded_patients.csv`, `fusion_weights.json`, `resolved_config.yaml`, `data_audit/`에 저장됩니다. Test로 최종 성능을 확인하기 전에 모델·weights·threshold를 고정합니다.

## 5. 저장 모델 evaluation 실행

학습한 모델을 재학습하지 않고 불러옵니다. CLI는 [`evaluation/eval.yaml`](evaluation/eval.yaml)을 수정하고 저장한 뒤 실행합니다. 공통 평가 노트북은 [`evaluation/evaluation_experiment.ipynb`](evaluation/evaluation_experiment.ipynb), DL 전용은 [`evaluation/DL/evaluation_only_DL.ipynb`](evaluation/DL/evaluation_only_DL.ipynb)입니다. 노트북도 같은 공통 평가기를 사용하며 셀을 위에서부터 실행합니다.

### 5.1 평가 대상·저장 경로·MLflow 설정

다음 공통 틀에서 `실제_ML_run`을 학습에서 출력된 폴더로 바꿉니다. 경로는 **eval.yaml이 있는 `model/evaluation/` 기준**입니다.

```yaml
basic_config: ../basic.yaml
evaluation:
  target: ml
  fusion_config: ../fusion.yaml
  model_name: XGBoost
  model_version: 4
  model_path: null
  artifact_dir: ../ML/models/실제_ML_run
  split: validation
  threshold_override: null
  results_directory: results
  mlflow_logging: true
  metrics:
    - auroc
    - average_precision
    - accuracy
    - precision
    - recall
    - sensitivity
    - specificity
    - f1
    - confusion_matrix
```

대상에 따라 **기존 evaluation 블록의 해당 키만 수정**하고 나머지 공통 설정을 유지합니다.

| 대상 | target | model_path | artifact_dir | 추가 선택 |
| --- | --- | --- | --- | --- |
| Clinical ML | `ml` | `null` | `../ML/models/실제_ML_run` | 실제 run 폴더를 명시해 기존 등록 모델 대신 평가 |
| ECG DL | `dl` | `../DL/models/실제_DL_run/best_model.pt` | `null` | checkpoint가 있는 run 폴더 경로도 가능 |
| Fusion | `fusion` | `null` | `null` | `fusion_config: ../fusion.yaml`에서 두 모델·가중치 선택 |

DL 설정 예:

```yaml
evaluation:
  target: dl
  model_path: ../DL/models/실제_DL_run/best_model.pt
  artifact_dir: null
```

DL은 checkpoint와 같은 폴더의 `metadata.json`, `resolved_config.yaml` 등이 필요합니다. 폴더 전체를 유지합니다. `model_path`와 `artifact_dir`를 함께 지정하면 같은 run을 가리켜야 하므로 하나만 사용합니다. 현재 모델 YAML 대신 저장된 설정으로 구조를 복원합니다.

ML은 `artifact_dir: null`일 때 `model_name`·`model_version`으로 등록된 모델을 선택합니다. 새 모델이 등록되지 않았을 수 있으므로 실제 run 폴더를 명시하는 방법을 권장합니다. DL용 `model_path`로 ML 모델을 선택하지 않습니다.

### 5.2 validation 평가

```bash
Global/.venv/bin/python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml \
  --split validation
```

모델·threshold 후보 비교는 validation으로 수행합니다. 저장 모델에서 확률을 다시 계산하며 기반 모델을 학습하지 않습니다. Fusion `learn` 모드는 validation에서 결합 가중치만 학습합니다.

### 5.3 최종 test 평가

모델·임계값을 확정한 뒤 실행합니다. Fusion은 먼저 `fusion.mode: saved`, `fusion.saved_run_dir: evaluation/results/fusion/선택한_validation_run`으로 설정합니다. Saved 모드는 기반 모델 경로와 가중치를 재사용하지만 임계값은 현재 YAML 또는 override를 읽으므로 같은 확정 값을 유지합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml \
  --split test
```

`--run-id my_final_test_01`로 이름을 지정할 수 있습니다. 이미 있는 ID는 덮어쓰지 않으므로 새 이름을 사용합니다. 결과는 `evaluation/results/<ml|dl|fusion>/<run_id>/`이며 MLflow를 켰으면 전송 URL 또는 전송 실패 상태를 확인합니다. 환자별 예측·audit은 로컬에 남고 집계 평가 결과를 업로드합니다.

### 5.4 Threshold 조정과 저장 예측 재평가

| 종류 | 새 학습 run의 기본 threshold | 저장된 모델 확률 재평가 시 |
|---|---|---|
| ML | 각 모델 `parameter.yaml`의 `classification.threshold` | `evaluation/eval.yaml`에서 `target: ml`, 해당 `artifact_dir` 지정 후 평가 CLI `--threshold-override 0.4` |
| DL | `DL/training.yaml`의 `classification_threshold` | `evaluation/eval.yaml`에서 `target: dl`, 해당 `model_path` 지정 후 동일 CLI 옵션 사용 |
| 결합 | `fusion.yaml`의 `classification.threshold` | 평가 YAML `target: fusion`으로 설정 후 동일 CLI 옵션 사용 |

예를 들어 저장된 확률을 validation split에서 threshold 0.4로 다시 평가하는 명령은 다음과 같습니다. 먼저 `evaluation/eval.yaml`에서 대상과 실제 artifact 경로를 설정해야 합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml \
  --split validation --threshold-override 0.4
```

threshold override는 확률값을 바꾸지 않고 분류 결과와 threshold 의존 metrics만 다시 계산합니다. 따라서 **저장된 모델 확률을 재평가하기 위해 재학습할 필요는 없습니다.** 모델별 parameter YAML 또는 DL training YAML의 threshold를 바꿔도 이미 생성된 artifact의 metadata와 결과는 자동 변경되지 않습니다. 기존 artifact를 평가할 때는 `--threshold-override`를 사용합니다. Fusion은 weights를 YAML에 설정하고 threshold는 별도 `classification.threshold`에 설정합니다.

Threshold 비교는 validation으로 끝내고 선택을 고정한 뒤 test는 최종 평가에 한 번 사용합니다. Test 성능을 보고 threshold, fusion weight, 모델 또는 epoch을 반복 조정하지 않습니다.

저장된 환자 확률 CSV만 평가하는 [`evaluation/evaluate_predictions.py`](evaluation/evaluate_predictions.py)도 있습니다. 이 스크립트는 다음 세 열만 포함하는 CSV를 입력으로 받습니다: `patient_id,target,score`. 현재 ML/DL trainer와 평가기가 생성하는 CSV에는 추가 열과 다른 이름(`y_true`, `p_positive`)이 있어 그대로 넣을 수 없습니다. 그 파일을 별도 호환 형식으로 준비한 경우에만 다음 명령을 사용합니다.

```bash
Global/.venv/bin/python Members/sang/model/evaluation/evaluate_predictions.py \
  --validation /path/to/validation_predictions.csv \
  --test /path/to/test_predictions.csv \
  --input-group clinical \
  --threshold-mode fixed --threshold 0.4
```

이전 예측 파일로 validation에서 Youden threshold를 계산하려면 `--threshold-mode youden`을 사용합니다. `auto`도 현재 구현에서 `youden`으로 바뀝니다. test 예측을 함께 넣으면 validation에서 선택된 threshold를 test에 그대로 적용하고, validation/test 환자 ID 중복이 있으면 중단합니다. 결과 report 기본 위치는 `model/outputs/evaluation/`입니다. 현재 run artifact의 prediction을 바로 재평가하려면 CSV 변환을 추측해 이 레거시 도구에 넣지 말고, 위 `evaluation.py`가 저장 모델에서 확률을 다시 계산하도록 사용하세요.

## 6. 결과 위치·MLflow 완료 확인과 재전송

### 6.1 로컬 완료와 원격 기록 구분

| 작업 | 완료 확인 파일 |
|---|---|
| ML 학습 | `ML/models/<run_id>/run_status.json`의 `COMPLETE`, `metadata.json`, `metrics.json`, `validation_predictions.csv` |
| DL 학습 | `DL/models/<run_id>/run_status.json`의 `COMPLETE`, `best_model.pt`, `metadata.json`, `metrics.json`, `validation_predictions.csv` |
| 평가 | `evaluation/results/<target>/<run_id>/metrics.json`, `predictions.csv` (fusion은 split이 포함된 prediction 파일명) |
| 결합 | `evaluation/results/fusion/<run_id>/fusion_weights.json`, `resolved_config.yaml`, split별 prediction 및 `excluded_patients.csv` |

각 config에 기록되는 상대 artifact 경로는 config를 담은 YAML 파일 디렉터리를 기준으로 합니다. 결과 디렉터리는 생성된 run ID를 덮어쓰지 않습니다. 지정한 run ID가 이미 있으면 다른 ID를 사용하세요.

`run_status.json: COMPLETE`는 학습이 로컬에서 완료됐다는 뜻입니다. 원격 전송 완료는 `mlflow_status.json`의 `FINISHED`와 URL로 별도 확인합니다. `PENDING`이면 모델·평가 결과는 저장됐지만 전송이 남았습니다. Logging을 끈 실행은 상태 파일이 없을 수 있습니다.

MLflow 브라우저에서 본인의 experiment를 열고 owner 태그·모델명·단계를 확인합니다. 공통 평가 지표는 `validation.*`, `test.*`입니다. ML/DL 평가에는 `source_run_id`, 학습 원격 기록이 확인되는 경우 `source_mlflow_run_id`·`source_mlflow_url`이 연결됩니다. Fusion은 `ml_source_run_id`, `dl_source_run_id`와 가중치를 기록합니다. 로컬 run ID와 MLflow run ID는 다른 식별자입니다.

### 6.2 재학습 없이 업로드만 재시도

먼저 1.4절의 연결 확인을 실행합니다. 아래 중 필요한 것 하나를 선택해 **실제 폴더로 경로를 바꾸고** 실행합니다. 새 학습·재평가 없이 저장된 결과를 전송합니다. 현재 PC의 계정을 사용하므로 다른 PC·계정에서 재전송할 때도 owner를 먼저 확인합니다.

ML 학습 결과:

```bash
Global/.venv/bin/python - <<'PY'
from pathlib import Path
from Members.sang.model.tracking.log_best import log_ml_training_to_mlflow

run = Path("Members/sang/model/ML/models/실제_ML_run")
print(log_ml_training_to_mlflow(run))
PY
```

DL 학습 결과:

```bash
Global/.venv/bin/python - <<'PY'
from pathlib import Path
from Members.sang.model.tracking.log_dl import log_dl_training_to_mlflow

run = Path("Members/sang/model/DL/models/실제_DL_run")
print(log_dl_training_to_mlflow(run))
PY
```

ML·DL·Fusion 평가 결과 공통:

```bash
Global/.venv/bin/python - <<'PY'
from pathlib import Path
from Members.sang.model.tracking.log_evaluation import log_evaluation_to_mlflow

run = Path("Members/sang/model/evaluation/results/dl/실제_평가_run")
print(log_evaluation_to_mlflow(run))
PY
```

ML 평가는 경로의 `dl`을 `ml`, Fusion은 `fusion`으로 바꿉니다. 같은 결과의 재시도는 유지된 캐시를 통해 기존 원격 run을 재사용합니다. 캐시나 결과 폴더를 임의 삭제해 재전송하지 않습니다.

## 7. 자주 발생하는 문제

- **데이터 경로 오류:** `basic.yaml` 상대 경로가 `basic.yaml` 파일 기준인지 확인하고, clinical/split 파일과 ECG 디렉터리가 실제 존재하는지 확인합니다. 임의의 대체 데이터나 split은 자동 탐색하지 않습니다.
- **Jupyter import 오류 또는 다른 Python 환경:** 저장소 루트에서 Jupyter를 시작했는지, 커널이 `Python (sixletters-poc Global)`인지 확인합니다. `Members.sang...` import는 저장소 루트를 Python 경로로 필요로 합니다.
- **환자·라벨·split 불일치:** ID는 문자열로 읽고 환자 ID로 조인합니다. duplicate ID, 알 수 없는 split 값, 라벨 불일치는 audit 또는 학습 중 오류로 처리되므로 ID 열과 `values` 매핑을 확인합니다.
- **ECG/임상 데이터 누락:** Raw DL은 raw split 라벨, ML은 clinical/split을 사용합니다. Fusion은 양쪽 예측의 교집합을 사용합니다. 각 감사 자료와 split·라벨 일치를 확인합니다.
- **GPU 메모리 부족:** `DL/training.yaml`의 `batch_size`를 낮추거나 `--device cpu`로 실행합니다.
- **저장 모델 경로 오류:** 해당 run의 `metadata.json` 또는 `best_model.pt`가 있는지 확인하고, 경로를 참조하는 YAML 기준 상대 경로인지 확인합니다. 완료되지 않은 `RUNNING` run은 사용할 수 없습니다.
- **모델과 설정 불일치:** DL checkpoint는 학습 run의 `resolved_config.yaml`에 저장된 CNN·head 설정으로 재구성됩니다. 학습 때 사용한 run 폴더를 통째로 지정하고 checkpoint 파일만 다른 run으로 교체하지 않습니다.
- **uv 실행 오류:** 1.1절의 고정 uv 버전 명령을 사용합니다.
- **계정이 다름:** 현재 커널·터미널의 환경 변수 override와 `.poc.local.toml`을 확인합니다.
- **MLflow PENDING:** 네트워크·인증·experiment 권한을 확인하고 6.2절로 재전송합니다.
- **Raw metadata·품질 인덱스 오류:** 실제 원본·전처리 자료를 준비하고 학습과 같은 조건을 유지합니다.

## 8. 검증 범위와 실행 전 남은 조건

이 가이드는 현재 API·CLI·노트북 기준입니다. 최근 관련 회귀 검증에서 다음 명령으로 **92개 테스트 통과**를 확인했습니다. 합성 입력과 대체한 원격 업로드를 포함하므로 실제 전체 데이터 학습·원격 전송 성공과 구분합니다.

```bash
Global/.venv/bin/python -m pytest \
  Members/sang/model/DL/tests \
  Members/sang/model/ML/tests -q
```

현재 PC의 로컬 MLflow 주소에서 개인 계정 인증·experiment 조회도 확인했습니다. 각 팀원 PC의 접속, 전체 raw DL/fusion 학습·평가·실제 업로드 성공은 별도 확인이 필요합니다. 테스트 폴더는 일상 학습에 필수는 아니며 코드 수정 후 회귀 확인에 사용합니다.

실험 전에 **개인 계정 연결, 실제 데이터, raw 원본 정보 확인과 품질 인덱스, 완료된 모델 폴더**를 준비합니다. 각 환경의 실제 학습·test·업로드는 이 가이드 순서로 실행해 확인합니다. 상세 내용은 [ML_manu.md](ML/ML_manu.md), [DL_manu.md](DL/DL_manu.md), [fusion_manu.md](fusion_manu.md), [TEAM_MLFLOW.md](tracking/TEAM_MLFLOW.md)에 있습니다.
