# MLflow 담당자 인수인계

## 사용 가능한 전달 경로

평가 스크립트는 모델별 결과를 계약 버전이 포함된 로컬 handoff bundle로 저장합니다. `import_bundle_to_mlflow.py`가 manifest와 예측 파일·모델 파일의 형식, ID/정답 지문, 파일 checksum을 확인한 다음 MLflow에 모델 후보별 run을 등록합니다. sklearn/PyTorch 모델은 flavor와 실행 환경이 포함된 MLflow Model 디렉터리도 함께 저장하므로 MLflow 모델 API로 불러올 수 있습니다. 기존 `sixletters.Experiment.save()` 데이터 계약에 결과를 끼워 맞추지 않습니다.

이 importer는 `Global` 프로젝트의 `mlflow-skinny==3.17.1`에 포함된 `MlflowClient`를 사용합니다. `--upload`가 없으면 bundle만 점검하고 네트워크에 접속하지 않습니다. 실제 업로드에는 명시적인 tracking URI가 필요합니다.

## 실행 절차

저장소 루트에서 평가기를 실행해 비어 있는 새 bundle 경로를 지정합니다.

```bash
uv run --project Global --extra ml python Members/sang/scripts/evaluate_preprocessed_clinical.py \
  --tracking-output Members/sang/model/outputs/handoff/clinical-<run-label>
```

```bash
uv run --project Global --extra gpu python Members/sang/scripts/smoke_p0002_holter.py
```

현재 Holter 범위는 실제 `P0002.npy` 하나의 입력 전처리와 모델 forward smoke뿐입니다. 학습·성능지표·Holter MLflow bundle은 생성하지 않습니다. `evaluate_ecg_architectures.py`는 split에 등록된 여러 환자의 NPY 파일을 학습·평가하므로 이 단일 환자 범위에서 실행하지 않습니다. 임상 bundle은 실행마다 고유한 `<run-label>`을 지정해 생성합니다. 다음은 업로드 없이 임상 bundle을 검사합니다.

```bash
uv run --project Global python Members/sang/model/tracking/import_bundle_to_mlflow.py \
  --bundle Members/sang/model/outputs/handoff/clinical-<run-label> \
  --experiment sang-scd-model-comparison
```

검토를 마치고 승인된 MLflow 서버로 전송할 때는 URI를 환경변수나 인자로 전달하고 `--upload`를 추가합니다. 인증 정보는 셸 기록이나 코드에 직접 넣지 말고, 팀이 승인한 환경 설정을 사용하세요.

```bash
export MLFLOW_TRACKING_URI="<team tracking URI>"
uv run --project Global python Members/sang/model/tracking/import_bundle_to_mlflow.py \
  --bundle Members/sang/model/outputs/handoff/clinical-<run-label> \
  --experiment sang-scd-model-comparison \
  --upload
```

여러 환자의 실제 Holter 데이터와 환자 단위 train/validation/test 구성이 확보된 뒤 cohort 평가를 실행하면, 해당 bundle도 같은 importer로 전달할 수 있습니다. 현재 `P0002.npy` 단일 환자 smoke 결과는 run이나 모델 성능으로 MLflow에 등록하지 않습니다. importer는 run 이름을 모델·dataset 조합으로 생성하고, tag에 dataset, 입력 그룹, 모델명, validation AP 기준 선택 여부와 bundle hash를 기록합니다. 파라미터에는 모델 설정을 평탄화해 저장하며, 지표에는 `validation_`, `validation_threshold_`, `training_`, `test_` 접두사를 붙입니다. AUROC·AP·Brier는 `roc_auc`, `average_precision`, `brier_score` 이름으로 저장합니다.

같은 bundle을 다시 가져오면 완료된 모델 run은 건너뛰고, 실패한 모델만 재시도합니다. 이미 진행 중인 같은 bundle import가 발견되면 중복 생성을 중단합니다. importer는 새 run의 ID와 이미 등록되어 건너뛴 모델 목록을 출력합니다.

## Bundle 내용

```text
handoff-bundle/
├── manifest.json
└── runs/<model-slug>/
    ├── run.json
    ├── validation_predictions.csv
    ├── test_predictions.csv        # validation AP 선택 모델만
    └── models/
        ├── validation_model.pkl                 # 임상 원본 estimator
        ├── validation_mlflow_model/             # 임상 MLflow sklearn flavor
        │   ├── MLmodel
        │   ├── conda.yaml
        │   ├── requirements.txt
        │   └── model.pkl
        ├── selected_test_mlflow_model/          # 선택 임상 모델만
        ├── model_state_dict.pt                  # Holter 원본 state dict
        └── mlflow_model/                        # Holter MLflow PyTorch flavor
```

각 MLflow run에는 해당 모델의 `run.json`, 예측 파일, 체크섬이 확인된 모델 artifact, manifest를 `handoff/` 경로에 저장합니다. 모델 flavor 디렉터리는 `handoff/models/<artifact-name>/`에 보존됩니다. 원시 임상 CSV, split 파일, NPY 파형은 importer가 업로드하지 않습니다. Test 지표와 test 예측은 validation AP로 선택된 모델에만 있어야 합니다. 이 문서의 Holter 모델 artifact 예시는 다수 환자 cohort 평가를 수행한 경우에만 생성됩니다.

### MLflow에 등록된 모델 불러오기

ML run URI는 다음처럼 사용할 수 있습니다. 기본 sklearn flavor의 `predict_proba()`는 `classes_` 순서대로 두 확률 열을 반환하며, 현재 이진 라벨에서는 두 번째 열이 `SCD=1` 확률입니다.

```python
import mlflow.sklearn

model = mlflow.sklearn.load_model(f"runs:/{run_id}/handoff/models/validation_mlflow_model")
scd_probability = model.predict_proba(features)[:, 1]
```

다수 환자로 학습한 Holter MLflow Model이 생기면 `mlflow.pytorch.load_model()`로 불러와 logits에 sigmoid를 적용합니다. 현재 P0002 smoke는 저장된 학습 모델이 아니므로 아래 코드를 실행할 Holter run은 없습니다.

```python
import mlflow.pytorch
import torch

model = mlflow.pytorch.load_model(f"runs:/{run_id}/handoff/models/mlflow_model")
with torch.no_grad():
    scd_probability = torch.sigmoid(model(ecg_tensor))
```

모델 디렉터리에는 환자 데이터가 포함되지 않습니다. Clinical signature는 feature 이름과 자료형만 담고 input example은 저장하지 않습니다. Holter flavor에는 모델 class 재구성에 필요한 `model/config.py`와 `model/DL/models.py`만 포함하며, dataset·split·outputs 디렉터리는 포함하지 않습니다. 모델 flavor는 Python pickle/cloudpickle 직렬화를 사용하므로 팀이 생성하고 체크섬을 검증한 신뢰 가능한 bundle만 로드하세요.

예측 CSV는 환자 ID·정답·점수를 포함합니다. 이 파일은 MLflow artifact로 업로드되므로 팀의 환자별 예측 데이터 업로드 정책이 허용된 서버에만 `--upload`를 실행하세요. 환자 ID와 정답은 MLflow params나 tags에는 기록하지 않습니다.

## 기존 공용 비교 흐름과의 경계

이 경로는 MLflow에 표준 run과 artifact를 직접 등록하는 인계 importer입니다. `sixletters.Experiment.save()` 및 `dev compare`가 사용하는 공통 prepared dataset 결과 계약으로 변환하지 않습니다. 임상 CSV는 현재 공용 prepared bundle과 동일성이 확인되지 않았고, Holter는 ECG 파일 보유자만 평가합니다. 따라서 이 결과는 해당 기존 비교 명령에 나타나지 않습니다. 기존 공용 비교에 포함하려면 공통 dataset/split 계약과 원시 ECG 입력 지원을 별도로 맞춰야 합니다.

## 담당자 확인 항목

1. 승인된 tracking URI와 experiment 권한을 설정합니다.
2. dry-run 출력을 확인하고 patient-level prediction artifact 업로드 정책을 확인합니다.
3. `--upload` 실행 후 생성된 run URL과 결과를 프로젝트 실험 기록에 연결합니다.
4. MLflow run에서 validation/test 지표 접두사, threshold, 모델 파일과 prediction artifact가 맞는지 확인합니다.
5. 테스트 환자 수가 작은 Holter 결과는 탐색 결과로 표시합니다.
