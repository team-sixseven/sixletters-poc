# MLflow 담당자 인수인계

## 준비 상태

Optuna 완료 직후 이 PC의 MLflow에 best 결과를 기록하는 경로는 [`LOCAL_MLFLOW.md`](LOCAL_MLFLOW.md)에 있습니다. 아래는 별도로 제공하는 bundle 생성·전달·명시적 업로드 절차입니다.

`export_runs.py`가 완료된 ML/DL 학습 run을 현재 importer가 읽는 contract v3 bundle로 변환합니다. exporter는 현재 환자 split을 다시 읽고 저장된 validation 예측과 모델 재추론 결과의 환자 ID·라벨·Average Precision을 대조합니다. ML 후보 여러 개 중 validation Average Precision이 가장 높은 한 모델에만 test 예측과 지표를 붙입니다. DL은 전달받은 단일 run을 test에서 평가합니다. threshold는 해당 run metadata의 학습 당시 값을 사용합니다.

bundle은 로컬에서 생성하고 검증할 뿐 자동 업로드하지 않습니다. MLflow 담당자는 개인정보 업로드 정책 및 승인된 tracking URI를 확인한 뒤 importer의 `--upload`만 별도로 실행합니다. bundle prediction CSV에는 환자 ID, 정답, 점수가 포함되므로 팀에서 허용한 위치에 보관하고 승인되지 않은 서버로 올리지 마세요. 원본 CSV, split CSV, NPY waveform은 bundle에 포함되지 않습니다.

세부 필드와 검증 규칙은 [`BUNDLE_CONTRACT.md`](BUNDLE_CONTRACT.md)를 참고하세요.

## 1. 학습 run 준비

명령은 저장소 루트(`/home/sang/sixletters-poc`)에서 실행합니다. 모델과 데이터 경로는 `Members/sang/model/basic.yaml` 및 실험별 YAML 설정을 따릅니다. 임상 및 ECG의 전체 학습 절차는 [`../TRAINING_GUIDE.md`](../TRAINING_GUIDE.md)를 참고하세요.

MLflow 모델 flavor 생성에는 MLflow full 패키지가 필요합니다. 저장소 `Global` 환경에 이미 필요한 extra/group가 설치되어 있으면 재설치하지 마세요. 환경을 처음 준비할 때만 프로젝트 규칙에 맞춰 동기화합니다.

```bash
uv sync --project Global --extra ml --extra gpu --group tracking-server
```

ML 실험 후보마다 학습 run 디렉터리를 확보합니다. run 디렉터리는 `Members/sang/model/ML/models/<run_id>/`이며 `run_status.json`이 `COMPLETE`여야 합니다. 학습은 `training.py`의 직접 CLI가 아니라 [`../TRAINING_GUIDE.md`](../TRAINING_GUIDE.md)에 안내된 노트북 흐름으로 실행합니다.

DL run 디렉터리는 `Members/sang/model/DL/models/<run_id>/`이며 `best_model.pt`, `metadata.json`, `resolved_config.yaml`, `validation_predictions.csv`, `metrics.json`, `run_status.json`을 포함해야 합니다. 실제 학습 실행법과 지원 조합은 Training Guide를 따르세요.

## 2. 로컬 bundle 생성 및 검증

`--run-dir`은 기존 모델 run 폴더 경로입니다. 상대 경로는 저장소 루트 기준입니다. `--output`은 새 디렉터리 또는 비어 있는 디렉터리여야 합니다. 이 명령은 현재 설정한 clinical/split/ECG 데이터를 읽어 재평가하므로 설정된 원본 데이터에 접근 가능한 환경에서 실행해야 합니다.

### Clinical ML

validation 비교에 포함할 모델별 run을 반복해서 전달합니다. 실제 존재하는 XGBoost, CatBoost, Random Forest 후보를 각각 학습한 경우 그 경로를 모두 넣습니다.

```bash
uv run --project Global --extra ml --extra gpu --group tracking-server python Members/sang/model/tracking/export_runs.py \
  --modality ml \
  --run-dir Members/sang/model/ML/models/<xgboost-run-id> \
  --run-dir Members/sang/model/ML/models/<catboost-run-id> \
  --run-dir Members/sang/model/ML/models/<random-forest-run-id> \
  --output Members/sang/model/outputs/handoff/ml-<bundle-name>
```

없는 후보는 생략합니다. 같은 모델의 여러 run을 한 번에 비교하는 방식은 지원하지 않으므로 후보별 최종 run 하나를 선택합니다. exporter가 validation AP로 선택하고 test를 평가합니다.

### ECG DL

DL bundle에는 run 하나를 전달합니다.

```bash
uv run --project Global --extra gpu --group tracking-server python Members/sang/model/tracking/export_runs.py \
  --modality dl \
  --run-dir Members/sang/model/DL/models/<dl-run-id> \
  --output Members/sang/model/outputs/handoff/dl-<bundle-name>
```

### bundle validator 재실행

Exporter가 생성 직후 importer의 offline validator를 실행합니다. 담당자는 전달 전후 동일 명령으로 다시 확인할 수 있습니다. `--upload`가 없으므로 MLflow server에는 접속하거나 쓰지 않습니다.

```bash
uv run --project Global python Members/sang/model/tracking/import_bundle_to_mlflow.py \
  --bundle Members/sang/model/outputs/handoff/<bundle-name>
```

정상 출력은 `"status": "VALIDATED_ONLY"`와 bundle SHA-256, 선택 모델, run 목록입니다. exporter 쪽 정상 출력은 `"status": "VALIDATED_LOCAL_BUNDLE"`입니다.

## 3. 승인 후 MLflow 업로드

업로드를 맡은 담당자는 팀이 승인한 URI 및 접근 정책을 확인하고, 예측 CSV의 환자 단위 정보 업로드가 허용되는지 확인한 다음 업로드를 실행합니다. 인증 토큰을 명령행 인자로 직접 넣지 말고 승인된 환경변수/credential 방식을 사용하세요.

```bash
export MLFLOW_TRACKING_URI="<approved-tracking-uri>"
uv run --project Global --group tracking-server python Members/sang/model/tracking/import_bundle_to_mlflow.py \
  --bundle Members/sang/model/outputs/handoff/<bundle-name> \
  --experiment sang-scd-model-comparison \
  --owner sang \
  --upload
```

Importer는 모델별 MLflow run을 만들고 validation/test metrics 및 prediction CSV, metadata, model flavor와 원본 checkpoint/model 파일을 기록합니다. bundle checksum을 태그로 사용해 같은 bundle의 완료 모델을 건너뜁니다. 업로드 후 생성된 run ID와 MLflow URL을 팀 기록에 연결하세요.

## 포함 범위와 제한

- 임상: XGBoost, CatBoost, Random Forest 모델 run을 지원합니다. 각 MLflow sklearn flavor는 설정된 clinical feature 컬럼을 입력으로 받고 두 열의 `predict_proba` 출력을 반환합니다.
- ECG: 현재 저장 가능한 CNN+LSTM 또는 CNN+Transformer run 하나를 지원합니다. MLflow PyTorch flavor 입력은 환자 단위 tensor `(batch, windows, leads, samples)`이고 출력은 환자 단위 logit입니다. 확률은 sigmoid를 적용합니다.
- Test 평가는 validation 선택 뒤 한 번 수행합니다. Test 결과를 이용한 후보 선택·threshold 조정은 하지 않습니다.
- ECG+clinical fusion 학습/가중치 결과는 현재 exporter가 내보내지 않습니다. 현재 fusion 모듈에 별도의 저장된 추론용 모델 artifact 계약이 없으므로 MLflow 담당자가 fusion serving 모델로 등록하면 안 됩니다.
- smoke 학습(`smoke_train_patient_limit` 설정 run)은 exporter가 거부합니다. 공식 handoff에는 전체 설정으로 완료한 run을 사용합니다.
- 실제 데이터 및 완료된 모델 artifacts가 없는 checkout에서는 bundle을 만들어 낼 수 없습니다. 이는 exporter 오류가 아니라 필요한 입력 산출물이 없는 상태입니다.

## 문제 해결 단서

- `Run is not COMPLETE`: 해당 run 학습이 끝나지 않았거나 실패했습니다.
- `Smoke-limited training run`: 전체 데이터 설정으로 다시 학습한 공식 run을 사용하세요.
- 환자 ID/label 불일치 또는 AP 불일치: 학습 이후 clinical/split/ECG 입력, 기본 설정 또는 모델 파일이 바뀌었는지 확인하세요. 현재 데이터 기준 평가가 학습 당시와 다르면 억지로 내보내지 마세요.
- `MLmodel` 누락 또는 checksum 오류: bundle 생성이 중단되었거나 전달 중 파일이 변했습니다. 새 빈 output 경로로 다시 export한 뒤 validator를 실행하세요.
- MLflow 연결 오류는 offline 검증에는 필요하지 않습니다. `--upload` 실행 시에만 URI 및 권한을 확인하세요.
