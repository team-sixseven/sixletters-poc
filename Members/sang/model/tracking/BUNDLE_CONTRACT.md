# Handoff bundle contract v3

이 문서는 `tracking/bundle.py`가 생성하고 `tracking/import_bundle_to_mlflow.py`가 검증하는 로컬 인계 bundle의 형식을 설명합니다. MLflow 담당자는 이 계약에 맞는 bundle을 받아 importer로 검증·등록할 수 있습니다. 계약 버전과 파일 검증을 변경할 때는 writer, validator, 이 문서를 함께 갱신하세요.

## 디렉터리 구조

```text
<bundle>/
├── manifest.json
└── runs/<portable-model-name>/
    ├── run.json
    ├── validation_predictions.csv
    ├── test_predictions.csv       # validation으로 선택된 모델만
    └── models/
        ├── mlflow_model/          # MLflow 모델 flavor 디렉터리
        ├── native_model.*         # 원본 checkpoint/model 파일
        ├── preprocessing.joblib   # 전처리가 별도 저장된 임상 모델
        └── metadata.json
```

각 run의 모델 artifact는 파일 또는 `MLmodel` 파일을 가진 MLflow 모델 디렉터리입니다. Contract v3 validator는 모든 artifact 파일의 상대 경로, 크기, SHA-256을 검사합니다. 심볼릭 링크는 허용하지 않습니다.

## manifest.json

writer가 생성하는 주요 필드:

- `contract`: `sixletters-model-handoff`
- `contract_version`: `3`
- `experiment`, `dataset`, `input_group`: 실험 및 입력 모달리티 식별값
- `split_summary`: train/validation/test 환자 수
- `selection.metric`: 현재 `validation_average_precision`
- `selection.model`: validation AP가 가장 높은 후보 이름
- `environment`: Python 및 사용 가능한 주요 라이브러리 버전
- `source_files`: 상대 경로와 SHA-256. 원본 데이터 파일은 등록하지 않음
- `runs`: 각 `run.json` 상대 경로와 모델 이름
- `contains_patient_level_predictions`: `true`
- `contains_raw_training_data`: `false`

## run.json

각 모델 run에는 다음을 기록합니다.

- `model`, `input_group`, `dataset`
- `selected_by_validation_average_precision`
- `parameters`: 모델 종류, 학습 설정, threshold 등 제한된 실행 설정
- `validation_metrics`, `validation_threshold_metrics`
- `test_metrics`: 선택 모델만 포함
- `prediction_files`: validation 및 선택 모델의 test CSV 위치
- `prediction_sha256`, `prediction_identity`: 각 prediction CSV checksum과 환자 ID/라벨 집합 지문
- `model_artifacts`: artifact 위치와 파일별 checksum
- `warnings`: 확인된 주의사항 목록

prediction CSV는 정확히 `patient_id,target,score` 세 열을 가집니다. ID는 비어 있거나 중복될 수 없고 target은 0/1, score는 유한한 0–1 확률이어야 합니다. 모든 모델의 validation prediction은 같은 환자와 라벨 집합이어야 합니다. Test prediction/metrics는 manifest에서 선택한 단일 모델에만 허용됩니다.

`patient_id` 및 `target`은 validation/test prediction artifact 안에만 있습니다. 이를 params, tags, model signature, input example에 넣지 않습니다. 원본 clinical CSV, split CSV, ECG waveform은 bundle에 포함하지 않습니다. Prediction artifact에 환자별 정보가 있다는 사실은 업로드 전에 별도로 검토해야 합니다.

## 현재 modality와 모델 flavor

| 입력 | `input_group` | 저장 flavor | 추론 계약 |
|---|---|---|---|
| Clinical ML | `clinical` | MLflow sklearn/pyfunc 모델 디렉터리 | configured clinical feature DataFrame → `predict_proba`, 클래스 순서 `[0, 1]` |
| ECG DL | `holter_raw_waveform` | MLflow PyTorch 모델 디렉터리 | 환자 tensor `(batch, windows, leads, samples)` → SCD logit; 확률은 sigmoid |

Exporter는 ML 후보 run을 validation Average Precision으로 선택하며 선택 후 test를 한 번 평가합니다. DL exporter는 지정된 단일 run의 validation 및 test 지표를 포함합니다. 현재 ECG+clinical fusion 모델은 exporter 지원 범위 밖입니다.

## 검증 및 업로드 경계

```bash
uv run --project Global python Members/sang/model/tracking/import_bundle_to_mlflow.py \
  --bundle Members/sang/model/outputs/handoff/<bundle-name>
```

기본 실행은 offline bundle 검증만 합니다. 실제 MLflow 등록은 승인된 URI를 사용해 `--upload`를 명시했을 때만 진행합니다. Importer는 signature나 모델 입력 데이터를 만들지 않고 bundle에 있는 MLflow flavor 디렉터리를 artifact로 등록합니다.
