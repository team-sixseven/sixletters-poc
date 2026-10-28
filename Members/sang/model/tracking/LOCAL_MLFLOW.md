# 이 PC의 MLflow 연동

팀원별 계정 등록, Wi-Fi/유선 접속 주소와 학습 전 연결 확인은 [TEAM_MLFLOW.md](TEAM_MLFLOW.md)를 참고하세요.

이 PC의 접속 주소는 `http://127.0.0.1:5000`, 실험 이름은 `sang-scd-ml`입니다. 기존 Docker Compose 서버를 사용합니다. 로그인 계정은 `sang`이고 비밀번호·접속 설정은 저장소 루트의 Git 제외 파일 `.poc.local.toml`에서 확인합니다. 서버 초기 관리용 접속 정보는 `.poc-cache/local-tracking-credentials.json`에 로컬로 보관합니다. 비밀번호를 코드나 노트북에 복사하지 않습니다.

팀원 PC에서 접속하려면 Windows LAN IP를 사용하는 포트 전달·방화벽 설정이 필요합니다. [WSL2 LAN 공유 절차](../../../../Global/docs/tracking-server.md#windows--wsl2에서-같은-lan으로-공유)의 `configure_mlflow_lan.ps1`을 Windows 관리자 PowerShell에서 실행합니다. 팀원은 `http://<Windows-LAN-IP>:5000`을 사용하고 본인의 계정으로 로그인합니다. 같은 LAN을 위한 설정이며, 다른 네트워크의 팀원은 VPN 등 별도 연결이 필요합니다. 각 계정의 초기 비밀번호는 Git 제외 파일 `.poc-cache/mlflow-team-accounts.json`에 보관합니다.

## 노트북 실행

`ML/model_experiment_only_ML.ipynb`의 실행 설정에서 `training_mode`(yaml/optuna), `model_version`, Optuna의 `model_name`·`n_trials`를 선택합니다. ML·DL 각 노트북은 아래 플래그를 기본으로 사용합니다.

```python
mlflow_logging = True
run_final_test = True  # 모델 선택·튜닝 중이면 False
```

일반 학습은 완료한 각 모델과 validation 결과를 기록합니다. Optuna는 탐색 완료 후 best 모델을 로컬에 저장하고 YAML 경로를 기록한 다음 MLflow로 전송합니다. 기본 Python API `ClinicalOptunaExperiment`는 오프라인 호출과의 호환성을 위해 `mlflow_logging=False`가 기본값이며, 노트북에서 명시적으로 활성화합니다. 로컬 실험만 하려면 준비 셀에서 `False`로 바꿉니다.

완료된 study의 best 하나가 MLflow run 한 개가 됩니다. 동일한 결과 재전송은 기존 run으로 재시도하고, 다른 study의 best는 새로운 run으로 기록합니다. trial별 모델을 서버에 올리지는 않습니다.

## 기록 내용

- Metrics: `validation.average_precision`, `validation.auroc`, sensitivity·specificity·F1·Brier 등 수치 지표와 confusion matrix의 TN·FP·FN·TP
- Parameters: 알고리즘, 입력 버전, 입력 열, 실제 모델 파라미터, threshold, training seed·조기 종료 설정, best trial 번호
- Artifacts: 최적 native 모델과 필요한 전처리·metadata·설정, `best_parameters.yaml`, `trials.csv`, `summary.json`

원본 임상 CSV, split CSV, ECG 파형과 환자별 validation 예측 CSV는 이 자동 전송 대상에 포함하지 않습니다. 로컬에 업로드용 모델을 중복 보관하지 않고 임시 복사본은 전송 후 정리합니다. 전송 재시도 정보는 기존 저장소 업로더의 `.poc-cache/uploads/`를 사용합니다. 모델 파일은 MLflow Artifacts의 `models/`에 보관하며 기존 `load_ml_run()`으로 다시 읽을 수 있습니다.

Optuna API 자체는 test를 평가하지 않습니다. ML 노트북이 학습 뒤 별도 공통 evaluation을 호출해, 선택한 best 모델·threshold로 test를 평가합니다. DL 노트북도 같은 방식이며 `run_final_test=True`와 `mlflow_logging=True`이면 평가 결과까지 자동 기록합니다. 평가 CLI도 `evaluation.mlflow_logging: true`이면 기록합니다. 전체 bundle 인수인계는 기존 `MLFLOW_HANDOFF.md`를 참고하세요.

## 저장 모델 평가 결과 기록

학습 run의 validation 점수를 덮어쓰지 않도록 평가마다 별도 run을 만듭니다. `stage=evaluation`, `split`, `model_name`, `model_version`, `source_run_id`로 구분하고, 학습 결과의 MLflow 기록이 있으면 `source_mlflow_run_id`와 `source_mlflow_url`로 연결합니다. MLflow에 전송한 학습 결과의 `summary.json`을 모델 폴더에 유지해야 이 연결을 찾을 수 있습니다. 평가 지표에는 `test.*` 또는 `validation.*` 접두사를 붙입니다.

업로드 파일은 집계 `metrics.json`, `resolved_config.yaml`, Fusion인 경우 `fusion_weights.json`입니다. 환자별 예측, 환자 목록·audit, 원본 데이터와 모델 파일은 평가 업로드에 포함하지 않습니다. 모델은 학습 run의 Artifacts에서 확인합니다. 이 방식은 평가마다 모델 파일을 중복 보관하지 않으며, 학습 모델을 MLflow에 먼저 전송하지 않았다면 평가 run만 생성됩니다.

평가 실행 예시는 `evaluation/README.md`를 참고하세요. 같은 평가 폴더 재전송은 같은 MLflow run을 재사용합니다. 새로운 평가 실행은 새로운 로컬 폴더와 run을 생성합니다. 서버가 꺼져 있어도 로컬 평가 파일은 유지되며 `mlflow_status.json`에 `PENDING`을 남깁니다. 전송만 다시 하려면 다음을 실행합니다.

```python
from Members.sang.model.tracking.log_evaluation import log_evaluation_to_mlflow

result = log_evaluation_to_mlflow(evaluation_dir)  # 기존 평가 결과 폴더 Path
print(result["url"])
```

오프라인 평가에는 `evaluation.mlflow_logging: false`를 사용합니다. 이 기능은 공통 `evaluation` 진입점의 ML·DL·Fusion 평가에 적용하며, `fusion.py`를 직접 호출하는 실행에는 자동 적용되지 않습니다. Test 결과로 모델·threshold·가중치를 재조정하지 않습니다.

## 전송 확인과 재시도

노트북에 출력되는 MLflow 링크와 로컬 `summary.json`의 `mlflow.status`를 확인합니다. 성공하면 `FINISHED`, 실패하면 `PENDING`과 오류 종류를 기록합니다. 서버 연결에 실패해도 학습한 로컬 best 모델은 유지합니다.

서버를 다시 켠 뒤 노트북의 **MLflow 전송 재시도 (필요할 때만)** 셀을 실행하거나 다음 함수를 호출합니다.

```python
from Members.sang.model.tracking.log_best import log_ml_training_to_mlflow

result = log_ml_training_to_mlflow(run_dir)  # 일반 학습 또는 Optuna best 경로
print(result["url"])
```

학습을 다시 할 필요는 없습니다. 같은 조합의 다음 탐색이 로컬 best를 교체하기 전에 미전송 결과를 먼저 전송하세요.

## 서버 시작·종료

다음 명령은 저장소 루트에서 실행합니다. Docker가 실행 중이어야 합니다.

```bash
./dev tracking status
./dev tracking up
./dev tracking logs --tail 30
./dev tracking down
```

서버 계정·실험 기록·모델은 Docker의 `tracking-data` 볼륨에 보관됩니다. 일반 종료·재시작 후에도 유지됩니다. PC 또는 Docker가 꺼져 있는 동안에는 서버에 접속할 수 없습니다. 서버 설정은 로컬 `.env.tracking`, 운영 절차는 `Global/docs/tracking-server.md`를 사용합니다.


## DL 노트북과 개인 계정

DL과 통합 fusion 노트북은 `mlflow_logging=True`를 기본으로 하며 각 PC의 `.poc.local.toml` 및 환경 변수에서 본인 계정을 읽습니다. `run_final_test=True`이면 고정한 학습 결과로 최종 평가까지 기록합니다. 모델을 비교 중이면 `False`로 설정합니다. DL 학습 전 계정 확인을 수행하고, 학습·평가 전송 실패는 로컬 결과를 보존한 채 `PENDING`으로 기록합니다.

```python
from Members.sang.model.tracking.log_dl import log_dl_training_to_mlflow

log_dl_training_to_mlflow(run_dir)  # 재학습 없이 전송 재시도
```

MLflow에는 checkpoint와 환자 정보 없는 network 설정, epoch 이력, 집계 validation/test 지표를 기록합니다. 환자별 예측·PID audit·raw quality index는 업로드하지 않습니다. 학습과 평가는 별도 run이며 평가 결과에 학습 run 링크를 기록합니다. 자세한 실행 순서는 [`../DL/TRAINING.md`](../DL/TRAINING.md)를 참고하세요.
