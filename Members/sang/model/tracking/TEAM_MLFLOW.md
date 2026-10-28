# 팀원별 MLflow 사용

ML·DL 학습과 최종 평가는 각자의 학습 PC에서 실행하고, 로컬에 설정한 개인 계정으로 공용 `sang-scd-ml` 실험에 기록합니다. ML Optuna는 완료된 study의 best 하나만 기록합니다. 각 PC에서 개인 계정을 한 번 설정합니다. 브라우저에 로그인한 계정이 아니라 Python 실행 환경의 접속 설정으로 업로드합니다.

## 1. 접속 주소 선택

| 접속하는 PC | 사용할 주소 |
|---|---|
| MLflow 서버 PC 자체 | `http://127.0.0.1:5000` |
| AI_LAB Wi-Fi (`192.168.50.x`) | `http://192.168.10.88:5000` (Wi-Fi ↔ 유선 라우팅 필요) |
| `192.168.10.x` 대역의 유선 LAN | `http://192.168.10.88:5000` |

SSID 이름보다 실제 IPv4 주소와 네트워크 간 통신 정책을 기준으로 선택합니다. 위 주소는 현재 서버에 설정된 주소이며 DHCP로 바뀌면 갱신해야 합니다. 현재 검증 범위는 유선 LAN과 AI_LAB입니다. AI_AX는 IPv4 대역과 연결이 확인되지 않아 이번 범위에서 제외합니다. 공유기에서 기기 간 통신을 차단하면 별도 확인이 필요합니다. 서버 PC는 인터넷 연결 유지를 위해 유선만 사용하며 Wi-Fi에는 연결하지 않습니다. Wi-Fi 팀원이 유선 서버로 접속하려면 공유기/사내 네트워크가 `192.168.50.0/24` ↔ `192.168.10.0/24` 통신을 허용해야 합니다. `Get-NetIPAddress`에 주소가 보여도 Wi-Fi가 꺼져 있으면 다른 PC는 접속할 수 없습니다.

서버의 Windows PowerShell에서 `Get-NetAdapter`와 `netsh wlan show interfaces`로 확인합니다. 서버 자기 자신에게 `/health`가 응답하는 것만으로 원격 접속을 보장하지 않습니다. 접속하는 팀원 PC에서도 아래를 실행합니다. Wi-Fi와 유선 모두 같은 서버 주소를 사용합니다.

```powershell
Test-NetConnection 192.168.10.88 -Port 5000
```

`TcpTestSucceeded: True`인지 확인하고 브라우저에서 같은 주소에 본인 계정으로 로그인합니다. 서버 WSL 포트 바인딩·Windows 포트 전달·방화벽 설정은 [서버 운영 문서](../../../../Global/docs/tracking-server.md)를 참고하세요. 다른 LAN에 대한 라우팅·공유기 설정은 MLflow 계정으로 해결되지 않습니다.

### 유선 서버에 Wi-Fi 클라이언트 허용

서버의 현재 Windows 방화벽 규칙이 `LocalSubnet`이면 다른 대역의 Wi-Fi 클라이언트를 허용하는 규칙이 필요합니다. 서버 **관리자 PowerShell**에서 다음을 실행합니다. Wi-Fi 연결이나 기본 경로를 변경하지 않습니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "\\wsl.localhost\Ubuntu-24.04\home\sang\sixletters-poc\Global\scripts\configure_mlflow_lan.ps1" -ListenAddress 192.168.10.88 -RemoteAddress "192.168.10.0/24","192.168.50.0/24"
```

이는 Windows의 TCP/5000 수신 범위만 변경합니다. 공유기의 VLAN/대역 간 차단을 해제하거나 라우팅을 추가하지는 않습니다. 이 명령 이후에도 Wi-Fi PC에서 `TcpTestSucceeded: False`이면 네트워크 관리자에게 두 대역 간 해당 서버의 TCP/5000 허용을 요청해야 합니다. 서버 자기 자신의 접속 성공은 팀원 PC 접속 검증을 대신하지 않습니다. 현재 원격 팀원 PC의 연결 성공은 아직 확인되지 않았습니다.

## 2. 개인 계정 등록

| 이름 | 계정 / 권장 owner |
|---|---|
| 은아 | `euna` |
| 민정 | `minjeong` |
| 민규 | `mingyu` |
| 성률 | `seongryul` |

서버 관리자로부터 **자기 계정의 비밀번호만** 전달받습니다. 전체 계정 JSON이나 서버 관리자 비밀번호를 팀원에게 배포하지 않습니다.

각 PC에서 저장소 루트로 이동한 뒤 기존 프로젝트 실행 환경을 사용합니다. 다음은 은아가 Wi-Fi에서 설정하는 예시이며 `owner`, `username`, `uri`를 본인에게 맞게 바꿉니다. 비밀번호는 명령 인자에 적지 않고 프롬프트에 입력합니다.

```bash
cd ~/sixletters-poc
./dev setup --owner euna --username euna --uri http://192.168.10.88:5000 --experiment sang-scd-ml --password
```

Docker 없이 이미 준비된 `Global/.venv` Python으로 ML 노트북을 사용하는 경우에는 같은 설정을 아래 명령으로 할 수 있습니다. 두 가지 중 자신의 실행 환경에 맞는 하나만 사용합니다.

```bash
cd ~/sixletters-poc
Global/.venv/bin/python -m sixletters setup --owner euna --username euna --uri http://192.168.10.88:5000 --experiment sang-scd-ml --password
```

`.poc.local.toml`은 PC마다 독립적이며 Git에서 제외됩니다. `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_EXPERIMENT_NAME` 환경 변수가 비어 있지 않으면 파일보다 우선합니다. 다른 사람의 계정으로 업로드되면 환경 변수와 실제 노트북 커널을 확인하세요. 다른 사람의 `.poc.local.toml`을 복사하지 않습니다.

## 3. 연결 확인 후 학습

ML 노트북 공통 import 셀 실행 후 아래로 접속 주소·계정을 확인할 수 있습니다. 비밀번호는 출력하지 않고 run도 생성하지 않습니다.

```python
from Members.sang.model.tracking.check_connection import check_mlflow_connection

check_mlflow_connection()
```

이 검사는 개인 계정 인증과 공용 실험 조회를 확인합니다. 읽기 검사가 성공했다고 모델 업로드 권한·용량까지 검증한 것은 아닙니다. 처음 사용할 때 완료된 실험 한 개의 업로드까지 확인하세요.

ML 일반 학습·Optuna와 DL 학습 모두 같은 검사를 호출합니다. `mlflow_logging=True`이면 학습을 시작하기 전에 연결을 확인하며, 실패하면 먼저 주소·계정을 고칩니다. 오프라인 학습에는 `mlflow_logging=False`를 사용합니다.

1. `ML/model_experiment_only_ML.ipynb`의 공통 import와 `model_version` 선택 셀을 실행합니다.
2. 실행 설정에서 `training_mode`(yaml/optuna), Optuna의 `model_name`·`n_trials`, `mlflow_logging=True`, `run_final_test`를 설정합니다.
3. 위에서 아래로 셀을 실행합니다. 선택한 학습 방식만 실행됩니다.
4. 출력된 MLflow 링크에서 본인의 `sixseven.owner` 태그, 모델 버전·파라미터·validation 점수·모델 파일을 확인합니다.
5. `run_final_test=True`이면 노트북 마지막 셀이 이번에 선택한 모델을 공통 `evaluation`으로 test 평가하고 별도 MLflow run에 기록합니다. 후보 비교·튜닝 중에는 `False`로 둡니다.

**자동 전송은 ML 일반 학습·Optuna, DL 학습 노트북, 공통 evaluation 및 통합 fusion 노트북에 적용됩니다.** ML·DL 각각의 노트북은 기본적으로 학습 결과를 기록하고 마지막 단계에서 고정 모델의 test 평가도 기록합니다. `fusion.py` 직접 호출은 자동 업로드하지 않습니다. 학습 중 서버 연결이 끊기면 로컬 best는 유지하며 전송 상태·재시도 방법은 [LOCAL_MLFLOW.md](LOCAL_MLFLOW.md)를 따릅니다.


## 4. DL·통합 노트북 자동 실행

개인 계정을 위 절차로 한 번 설정하면 DL 노트북도 같은 `.poc.local.toml`과 환경 변수를 사용합니다. `DL/model_experiment_only_DL.ipynb`의 기본 `mlflow_logging=True`, `run_final_test=True`는 학습·validation 기록 후 고정 checkpoint의 test 평가를 별도 MLflow run으로 기록합니다. 비교 실험 중에는 `run_final_test=False`, 오프라인 실행은 `mlflow_logging=False`로 바꿉니다. Raw source metadata와 품질 인덱스의 기존 학습 조건은 그대로 필요합니다.

`ECG+clinical_experiment.ipynb`도 같은 계정으로 validation·test 평가를 기록하며, test 단계는 validation run의 저장된 가중치를 재사용합니다. 각자 계정명을 노트북 코드에 넣지 않습니다. 업로드 경로와 MLflow owner tag는 로컬 `owner`, 인증 계정·서버·실험은 공통 connection 설정으로 결정됩니다. 같은 PC를 다른 사람이 쓰면 `dev setup`을 본인 계정으로 다시 실행하고 환경 변수의 이전 계정 설정도 확인합니다.

학습 전 계정 확인이 실패하면 중단합니다. 학습 후 전송이 실패해도 모델을 보존하며, `log_dl_training_to_mlflow(run_dir)`로 재학습 없이 전송합니다. 평가는 `log_evaluation_to_mlflow(evaluation_dir)`로 재평가 없이 전송합니다. 두 함수의 import와 실행 예시는 [`../DL/TRAINING.md`](../DL/TRAINING.md)에 있습니다.


## 5. ML·DL 각각의 노트북에서 학습부터 평가까지

두 노트북 모두 `mlflow_logging=True`, `run_final_test=True`가 기본입니다. 개인 계정은 기존 setup을 그대로 사용합니다. 모델을 비교하는 중에는 `run_final_test=False`로 바꿉니다.

ML 노트북은 `training_mode="yaml"`이면 일반 학습, `"optuna"`이면 선택한 알고리즘의 best 탐색만 실행합니다. 전체 셀 실행으로 두 방식이 중복 실행되지 않습니다. 일반 학습은 완료한 각 모델과 validation 지표를 기록하고, Optuna는 trial을 전송하지 않고 best 하나를 기록합니다. 최종 평가는 이번 실행의 `run_dirs` 또는 `best_run_dir`를 직접 사용하므로 YAML의 다른 과거 모델 경로를 평가하지 않습니다.

ML 일반 학습의 API는 `train_clinical(..., mlflow_logging=True)`, DL은 `train_raw_ecg(..., mlflow_logging=True)`입니다. API 자체는 오프라인 호환성을 위해 logging이 기본 False이고 노트북에서 활성화합니다. ML 일반/Optuna 공통 전송 재시도는 `Members.sang.model.tracking.log_best.log_ml_training_to_mlflow(run_dir)`를 사용합니다.

## 6. 처음부터 실행하는 순서

### 6.1 Jupyter 시작

2절에서 본인 계정을 등록하고, 준비된 프로젝트 환경에서 저장소 루트 기준으로 실행합니다. 아직 환경이 없다면 먼저 프로젝트의 `Global` 환경을 설치해야 합니다.

```bash
cd ~/sixletters-poc
Global/.venv/bin/python -m jupyterlab
```

Jupyter 커널은 `Global/.venv`의 Python을 선택합니다. 다른 Python 커널을 사용하면 계정 설정이나 패키지 설치가 같다고 가정할 수 없습니다. Jupyter 파일 목록에서 `Members/sang/model/`로 이동합니다.

### 6.2 ML: 일반 학습 또는 Optuna → MLflow → 최종 평가

1. [`ML/model_experiment_only_ML.ipynb`](../ML/model_experiment_only_ML.ipynb)를 엽니다.
2. 실행 설정에서 원하는 입력 버전과 학습 방식을 선택합니다.

   ```python
   model_version = 4        # 입력 묶음 1~4
   training_mode = "yaml"  # "yaml": 일반 학습 / "optuna": best 탐색
   model_name = "XGBoost"  # Optuna 알고리즘: XGBoost / CatBoost / Random Forest
   n_trials = 100          # Optuna일 때만 사용
   reuse_saved = False     # yaml: 새로 학습; True이면 동일 버전 저장 run 재사용
   mlflow_logging = True
   run_final_test = True   # 후보 비교·튜닝 중에는 False
   retry_mlflow = False
   ```

3. 일반 학습은 `ML/model_choice.yaml`의 `models`와 각 모델 `parameter.yaml`을 확인합니다. Optuna는 노트북의 objective 탐색 범위를 확인합니다.
4. 커널을 처음부터 실행합니다. 일반 학습과 Optuna 중 선택한 경로만 학습하고 자동 기록합니다. Optuna trial 모델은 전송하지 않습니다.
5. 마지막 평가 셀은 이번 실행의 일반 모델 목록 또는 Optuna best를 직접 평가합니다. 저장된 전처리·threshold를 적용하며, 학습과 test 평가는 별도 MLflow run입니다. 원본 `eval.yaml`을 수동으로 바꿀 필요가 없습니다.

일반 run은 `ML/models/<run_id>/`, Optuna best는 `ML/models/best/<algorithm>_vN/`, 최종 평가는 `evaluation/results/ml/<evaluation_run_id>/`에서 확인합니다. 실제 경로는 셀 출력에 표시됩니다.

### 6.3 DL: 학습 → MLflow → 최종 평가

1. [`DL/model_experiment_only_DL.ipynb`](../DL/model_experiment_only_DL.ipynb)를 엽니다.
2. `basic.yaml:data.raw_ecg`의 경로, split, source metadata, 품질 인덱스를 확인합니다. 현재 관측된 형식은 `(sample, 3)` float16이며, 전체 split 파일이 준비되면 같은 코드로 실행합니다. Sample rate·배열 축를 upstream에서 확인한 뒤에만 `source_metadata_confirmed: true`로 바꾸고, `quality_index_path`에 `pid,window_index,usable` CSV를 설정합니다. 정보가 없으면 학습이 중단됩니다.
3. `DL/model_choice.yaml`은 `cnn_model`, `sequence_model` 두 항목만 설정합니다. 모델 YAML은 flat 실험값만 사용하고, `DL/training.yaml`에서 epoch·학습률·segment cap·분류 threshold·MIL/MLP를 설정합니다. 유도 수는 YAML에서 고정하지 않고 실제 입력과 별도 lead/time mask로 처리합니다.
4. 실행 설정을 확인합니다.

   ```python
   mlflow_logging = True
   run_final_test = True   # 후보 비교·튜닝 중에는 False
   max_epochs = None
   limit_train_patients = None
   device_override = "auto"
   ```

5. 위에서 아래로 전체 셀을 실행합니다. 파일 감사 → 로더·모델 생성 → train/validation → best checkpoint와 MLflow 기록 → checkpoint 복원 → 최종 test 평가와 MLflow 기록 순서입니다.

짧은 smoke 실행은 `max_epochs=1`, `limit_train_patients=16`, `device_override="cpu"`, `run_final_test=False`로 바꿉니다. DL run은 `DL/models/<run_id>/`, 최종 평가는 `evaluation/results/dl/<evaluation_run_id>/`에 저장됩니다. 품질 CSV schema와 상세 설정은 [`DL/TRAINING.md`](../DL/TRAINING.md)에 있습니다.

### 6.4 학습 없이 저장 모델만 재평가

노트북의 마지막 평가 셀만 실행하려면 현재 커널에 해당 run 변수가 있어야 합니다. 커널을 재시작했거나 다른 저장 모델을 평가하려면 `evaluation/eval.yaml`의 기존 항목 중 다음을 바꿉니다. 아래 예시의 `<run_id>`를 실제 폴더명으로 교체합니다.

ML 예:

```yaml
evaluation:
  target: ml
  artifact_dir: ../ML/models/<run_id>
  split: test
  threshold_override: null
  mlflow_logging: true
```

Optuna ML이면 `artifact_dir: ../ML/models/best/xgboost_v4`처럼 해당 best 폴더를 지정합니다. DL 예:

```yaml
evaluation:
  target: dl
  model_path: ../DL/models/<run_id>/best_model.pt
  artifact_dir: null
  split: test
  threshold_override: null
  mlflow_logging: true
```

기존 YAML 전체를 위 예시로 대체하지 말고 해당 필드만 수정합니다. 저장소 루트에서 실행합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml --split test
```

평가는 저장 모델과 전처리를 로드하며 다시 학습하지 않습니다. DL의 `model_path`에는 `best_model.pt` 또는 run 폴더를 지정할 수 있고, metadata/config 파일도 같은 폴더에 보관합니다. 결과의 `predictions.csv`, `metrics.json`, `mlflow_status.json`을 확인합니다.

### 6.5 전송 상태와 재시도

학습·평가 셀에 출력되는 링크와 로컬 `mlflow_status.json`을 확인합니다. `FINISHED`는 전송 완료, `PENDING`은 로컬 결과가 저장됐지만 전송이 완료되지 않은 상태입니다. 전송만 다시 하려면 해당 run 경로를 지정해 다음 중 필요한 함수를 호출합니다.

```python
from pathlib import Path
from Members.sang.model.tracking.log_best import log_ml_training_to_mlflow
from Members.sang.model.tracking.log_dl import log_dl_training_to_mlflow
from Members.sang.model.tracking.log_evaluation import log_evaluation_to_mlflow

# 아래는 호출 예시입니다. <실제 ... 경로>를 교체하고 필요한 한 줄만 실행합니다.
# log_ml_training_to_mlflow(Path("<실제 ML run 또는 Optuna best 경로>"))
# log_dl_training_to_mlflow(Path("<실제 DL run 경로>"))
# log_evaluation_to_mlflow(Path("<실제 evaluation 결과 경로>"))
```

학습 run에는 모델과 집계 validation 지표를, 평가 run에는 집계 test/validation 지표를 기록합니다. 환자별 예측 CSV·원본 데이터·PID audit은 자동 전송하지 않습니다. 계정 연결·패키지 문제는 노트북 실행 전에 해결해야 하며, raw DL의 metadata/품질 인덱스 조건도 별도로 충족해야 합니다.
