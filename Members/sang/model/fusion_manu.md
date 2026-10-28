# ECG + clinical Fusion 실험 매뉴얼

이 문서는 저장된 clinical ML 모델과 ECG DL 모델을 함께 사용하는 순서입니다. 아래 명령은 **Linux/WSL 터미널의 저장소 루트**에서 실행합니다. `~/sixletters-poc`는 본인의 저장소 위치로 바꿉니다.

```text
ML·DL을 각각 학습하고 저장
→ fusion.yaml에서 두 모델 선택
→ validation에서 결합 가중치 비교 또는 학습
→ 모델·가중치·분류 임계값 확정
→ 같은 조건으로 최종 test 평가
→ 개인 계정 MLflow에 평가 결과 기록
```

현재 구현은 **확률 결합(late fusion)**입니다. 두 모델의 학습 파라미터를 함께 업데이트하는 구조가 아닙니다. Fusion 실행 중에는 저장된 기반 모델을 불러와 추론하며, `learn` 모드도 결합 가중치 두 개만 학습합니다.

```text
p_final = weight_ml × p_ml + weight_dl × p_dl
y_pred = 1 if p_final >= threshold else 0
```

`p_ml`, `p_dl`은 각각 SCD 양성 확률입니다. 이미 확률이므로 결합 결과에 sigmoid를 다시 적용하지 않습니다.

## 1. 실행 환경 준비

### 1-1. 저장소로 이동하고 Python 확인

```bash
cd ~/sixletters-poc
Global/.venv/bin/python --version
Global/.venv/bin/python -c "import torch, pandas, sklearn, yaml, xgboost, catboost; print('환경 확인 완료'); print('CUDA:', torch.cuda.is_available())"
```

프로젝트는 Python 3.12를 사용합니다. 환경이 없거나 패키지가 빠져 있다면 프로젝트에 고정된 uv 버전으로 설치합니다.

```bash
uvx --from uv==0.9.26 uv sync --project Global --locked --extra gpu --extra ml
```

Fusion에는 ML과 DL 패키지가 모두 필요합니다. 위 설치는 프로젝트의 Linux/WSL 환경 기준입니다. 이후에는 `Global/.venv/bin/python`으로 실행해 환경을 통일합니다.

### 1-2. 노트북 커널 준비

노트북을 사용할 경우 처음 한 번 커널을 등록합니다.

```bash
Global/.venv/bin/python -m ipykernel install --user \
  --name sixletters-poc-global \
  --display-name "Python (sixletters-poc Global)"
Global/.venv/bin/python -m jupyterlab
```

브라우저에서 **Python (sixletters-poc Global)** 커널을 선택합니다. 터미널에서 진행하려면 6단계를 사용합니다.

## 2. 개인 MLflow 계정 연결

### 2-1. 본인 계정 등록

아래 `euna`와 서버 주소를 본인의 owner·계정·실제 서버 주소로 바꿉니다. 비밀번호는 프롬프트에서 입력합니다.

```bash
Global/.venv/bin/python -m sixletters setup \
  --owner euna \
  --username euna \
  --uri http://192.168.10.88:5000 \
  --experiment sang-scd-fusion \
  --password
```

서버 PC 자체에서는 `http://127.0.0.1:5000`을 사용할 수 있습니다. 다른 PC에서는 접근 가능한 실제 서버 주소가 필요합니다. 설정은 Git 제외 파일 `.poc.local.toml`에 저장됩니다. 브라우저 로그인과 Python 계정 설정은 별도입니다.

### 2-2. 연결 확인

```bash
Global/.venv/bin/python -m Members.sang.model.tracking.check_connection
```

출력된 owner·username·experiment·주소가 맞는지 확인합니다. 이 명령은 인증과 experiment 조회를 확인하며 run을 생성하지 않습니다.

환경 변수 `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_EXPERIMENT_NAME`의 비어 있지 않은 값이 파일 설정보다 우선합니다. 계정이 다르게 보이면 실행 중인 터미널과 커널의 환경 변수를 확인합니다. 비밀번호를 노트북이나 YAML에 적지 않습니다.

Wi-Fi ↔ 유선 접속과 계정 설정의 자세한 절차는 [TEAM_MLFLOW.md](tracking/TEAM_MLFLOW.md)를 참고합니다. 오프라인에서는 노트북의 `mlflow_logging=False`, CLI 설정의 `evaluation.mlflow_logging: false`로 진행합니다.

## 3. 저장된 ML·DL 모델과 데이터 준비

### 3-1. 기반 모델을 각각 학습

아직 저장된 모델이 없다면 다음 문서 순서로 각각 학습합니다.

| 대상 | 실행 문서 | 학습 노트북 |
| --- | --- | --- |
| clinical ML | [ML_manu.md](ML/ML_manu.md) | [model_experiment_only_ML.ipynb](ML/model_experiment_only_ML.ipynb) |
| ECG DL | [DL_manu.md](DL/DL_manu.md) | [model_experiment_only_DL.ipynb](DL/model_experiment_only_DL.ipynb) |

학습 완료 시 출력되는 **저장 run 폴더 전체**를 확보합니다. 예를 들어 ML은 `ML/models/<run_id>/` 또는 `ML/models/best/xgboost_v4/`, DL은 `DL/models/<run_id>/`입니다. 파일 하나만 복사하지 않습니다.

### 3-2. 모델 폴더의 구성 확인

| 대상 | 주요 파일 |
| --- | --- |
| ML 공통 | `metadata.json`, `resolved_config.yaml`, 해당 모델 파일 |
| XGBoost | `best_model.json`, `preprocessing.joblib` |
| Random Forest | `best_model.joblib`, `preprocessing.joblib` |
| CatBoost | `best_model.cbm` |
| 현재 raw DL | `best_model.pt`, `metadata.json`, `resolved_config.yaml` 및 학습 때 저장된 감사 자료 |

학습 결과 폴더를 그대로 보관하면 metrics, validation 예측, 상태 파일 등도 함께 유지됩니다. Fusion에서 지정하는 경로는 `best_model.pt` 같은 **파일 경로가 아니라 그 파일이 있는 폴더**입니다. 각 모델의 입력 feature 버전과 구조는 저장된 metadata·설정으로 복원합니다.

Optuna의 `ML/models/best/...`는 후속 학습에서 교체될 수 있습니다. 재현할 실험에 사용한 모델 폴더는 별도의 고정 경로에 전체 복사해 두고 선택하거나, 최종 평가까지 해당 경로를 다시 학습으로 덮어쓰지 않습니다. `saved` 모드는 모델 파일을 복제해 보관하지 않습니다.

### 3-3. 데이터와 split 일치 확인

[basic.yaml](basic.yaml)의 경로에 실제 데이터가 있어야 합니다.

- ML: `ML/data/clinical_clean.csv`, `ML/data/final_split.csv`.
- 현재 raw DL: `DL/data/final_split_v2_augmented.csv`, `DL/data/wave/`의 환자별 NPY, 실제 전처리에서 만든 품질 인덱스.
- 기존 SQI-window DL 모델: `basic.yaml`의 `data.ecg` 경로에 해당 모델용 ECG와 sidecar가 필요합니다.

환자 ID가 양쪽에 있으면 split과 SCD 라벨이 같아야 합니다. 두 모델의 양성 클래스, target column, target semantics도 일치해야 합니다. Fusion은 환자 ID로 예측을 맞추며 **양쪽 예측이 모두 있는 환자만 평가**합니다. 한쪽만 있는 환자는 `excluded_patients.csv`와 감사 자료로 보고하고, 누락 확률을 0으로 채우지 않습니다.

**현재 기본 raw 설정은 `source_metadata_confirmed: false`, `quality_index_path: null`이므로 그대로는 raw 학습·평가를 완료할 수 없습니다.** 실제 원본의 sampling rate와 축 의미를 확인하고, 학습과 일치하는 품질 자료를 준비해야 합니다. 단순히 실행을 통과시키려고 확인 값을 바꾸거나 모든 구간을 정상으로 표시하지 않습니다. 준비 절차는 [DL_manu.md](DL/DL_manu.md)에 있습니다.

현재 raw DL은 checkpoint format v2를 사용합니다. 과거 고정 유도 모델은 원래 실행 코드나 재학습이 필요합니다. 평가 시 raw split·품질 인덱스·전처리 조건이 저장된 학습 조건과 다르면 중단될 수 있으므로 임의로 설정을 바꾸지 않습니다.

## 4. fusion.yaml 설정

[fusion.yaml](fusion.yaml)을 편집합니다. 아래 `실제_ML_run`, `실제_DL_run`을 3단계에서 확보한 폴더명으로 바꿉니다. 경로는 **fusion.yaml이 있는 `model/` 기준**입니다. 절대 경로도 가능합니다.

```yaml
basic_config: basic.yaml
ml:
  artifact_dir: ML/models/실제_ML_run
dl:
  artifact_dir: DL/models/실제_DL_run
fusion:
  mode: manual
  saved_run_dir: null
  weights:
    ml: 0.5
    dl: 0.5
  learning:
    split: validation
    objective: binary_cross_entropy
    parameterization: two_class_softmax
    optimizer: LBFGS
    max_iterations: 100
    tolerance_grad: 1.0e-9
classification:
  threshold: 0.5
missing_prediction_policy: intersection_and_report
results_directory: evaluation/results/fusion
```

`0.5/0.5`는 시작용 예시이며 최적값이라는 의미가 아닙니다. `choose_ml_run`, `choose_dl_run` 기본 placeholder는 실제 폴더로 반드시 바꿉니다.

### 4-1. 수동 가중치: manual

`mode: manual`에서 `weights`를 직접 지정합니다. 예를 들어 clinical에 0.7, ECG에 0.3을 줄 때는 다음과 같습니다. 기존 `fusion` 블록에서 이 항목들만 바꿉니다.

```yaml
fusion:
  mode: manual
  saved_run_dir: null
  weights:
    ml: 0.7
    dl: 0.3
```

가중치는 유한한 0 이상 값이며 합이 1이어야 합니다. 다른 `learning` 설정은 유지해도 manual 모드에서는 사용하지 않습니다. 가중치를 비교할 때는 validation 결과로 선택합니다.

### 4-2. validation으로 가중치 학습: learn

4단계 전체 설정에서 `fusion.mode`만 `learn`으로 바꾸고 `saved_run_dir: null`을 유지합니다. `learning` 블록은 위 값을 그대로 사용합니다.

학습은 **항상 validation의 공통 환자**에서 진행합니다. 목적함수는 binary cross entropy이며 softmax로 가중치를 0 이상·합 1로 제한합니다. 기존 `weights`는 learn 모드에서 사용하지 않습니다. 공통 validation 환자에 양성과 음성이 모두 있어야 합니다.

이 validation 점수는 가중치 학습에 사용한 데이터의 점수이므로 독립적인 최종 성능이 아닙니다. 코드도 `validation_metrics_independent_of_weight_fit: false`로 기록합니다. 수동 후보를 여러 번 비교한 validation 역시 최종 test를 대신하지 않습니다.

### 4-3. 저장된 결합 재사용: saved

validation 평가 후 생성된 **fusion 결과 폴더**를 `saved_run_dir`로 지정하고 `mode: saved`로 바꿉니다. 자세한 실행 순서는 6단계에 있습니다.

이 모드는 결과 폴더의 `fusion_weights.json`, `resolved_config.yaml`에서 **가중치와 ML·DL 모델 경로**를 읽습니다. 현재 `ml.artifact_dir`, `dl.artifact_dir`, `weights`를 바꿔도 saved 모드의 모델 선택은 바뀌지 않습니다. 저장된 모델 경로의 실제 파일이 계속 있어야 합니다. 다른 PC에서는 모델과 결과 폴더를 준비하고 저장된 경로의 유효성도 확인해야 합니다.

**분류 임계값은 saved 모드에서도 현재 `classification.threshold` 또는 평가 override를 사용합니다.** validation 때 확정한 임계값을 test에도 동일하게 설정합니다. 평가에 쓰는 `basic.yaml`과 데이터 조건도 유지합니다.

## 5. 노트북으로 실행하기

### 5-1. 실험 노트북 열기

[ECG+clinical_experiment.ipynb](ECG+clinical_experiment.ipynb)를 열고 1단계의 Global 커널을 선택합니다. YAML을 편집하고 저장한 뒤 노트북 공통 설정 셀을 다음처럼 맞춥니다.

```python
mlflow_logging = True
run_final_test = False
fusion_config = repo / "Members/sang/model/fusion.yaml"
```

기존 import와 `repo`를 찾는 셀은 그대로 둡니다. **현재 노트북 기본값은 `run_final_test=True`이므로 후보 비교 중에는 먼저 False로 바꿉니다.** 오프라인이면 `mlflow_logging=False`로 둡니다.

### 5-2. validation으로 비교

1. 커널을 재시작합니다.
2. 위에서 아래로 코드 셀을 순서대로 실행합니다.
3. 출력된 `Validation:` 결과 폴더와 metrics를 확인합니다.
4. `fusion.yaml`의 모델 또는 manual 가중치를 바꾸고 저장합니다.
5. 평가 셀을 다시 실행해 새 validation 결과를 비교합니다.

`learn` 모드라면 이 과정에서 validation 가중치도 학습됩니다. 결과 폴더의 `fusion_weights.json`으로 실제 가중치를 확인합니다. 결과는 실행마다 새 폴더로 저장됩니다.

### 5-3. 최종 test 평가

모델·가중치 선택 방식·분류 임계값을 확정한 뒤 `run_final_test=True`로 바꾸고 공통 설정 셀과 평가 셀을 순서대로 실행합니다.

노트북은 validation 평가 후 **해당 validation run을 saved 모드로 연결해 같은 가중치로 test를 평가**합니다. `Validation:`과 `Test:` 결과 폴더가 각각 출력되고, MLflow가 켜져 있으면 각각 평가 run으로 기록합니다. 기반 모델은 재학습하지 않습니다.

노트북은 임시 YAML을 사용하므로 원본 `fusion.yaml`, `evaluation/eval.yaml`의 target·split·mode를 자동으로 바꾸지 않습니다. 이미 선택한 validation run을 정확히 재사용하려면 4-3처럼 원본 `fusion.yaml`을 saved로 설정하거나, 아래 CLI에서 그 run을 지정합니다. Test 결과를 보고 가중치를 다시 고르지 않습니다.

## 6. 터미널로 실행하기

노트북과 같은 공통 평가 모듈을 사용합니다. `fusion.py`를 직접 실행하는 별도의 CLI는 없습니다.

### 6-1. 공통 평가 YAML 설정

[evaluation/eval.yaml](evaluation/eval.yaml)을 아래처럼 설정합니다. 이 파일의 경로는 **`evaluation/` 기준**입니다.

```yaml
basic_config: ../basic.yaml
evaluation:
  target: fusion
  fusion_config: ../fusion.yaml
  model_name: XGBoost
  model_version: null
  model_path: null
  artifact_dir: null
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

Fusion의 두 모델은 `fusion.yaml`에서 선택합니다. 위 `model_name`, `model_version`, `model_path`, `artifact_dir`는 fusion의 모델 선택에 사용하지 않습니다. `threshold_override: null`이면 `fusion.yaml`의 임계값을 사용합니다. 오프라인이면 `mlflow_logging: false`로 설정합니다.

공통 평가 진입점에서는 결과 위치를 `evaluation.results_directory/fusion/`으로 결정합니다. 위 예시이면 `model/evaluation/results/fusion/`입니다. `fusion.yaml`의 `results_directory`는 Python에서 `run_fusion()`을 직접 호출할 때 사용합니다. 직접 호출은 MLflow 업로드를 자동 실행하지 않으므로, 이 매뉴얼에서는 공통 평가 진입점을 사용합니다.

### 6-2. validation 실행

4단계에서 `manual` 또는 `learn`을 설정한 뒤 실행합니다. 결과를 찾기 쉽도록 run ID를 지정합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml \
  --split validation \
  --run-id fusion_candidate_01_validation
```

결과 폴더는 다음과 같습니다.

```text
Members/sang/model/evaluation/results/fusion/fusion_candidate_01_validation/
```

같은 run ID가 이미 있으면 덮어쓰지 않고 중단합니다. 다른 후보는 `fusion_candidate_02_validation`처럼 새 ID를 사용합니다. ID에는 영문·숫자·`_`·`-`·`.`을 사용할 수 있습니다. `--run-id`를 생략하면 자동 생성됩니다.

### 6-3. 결과와 학습된 가중치 확인

```bash
Global/.venv/bin/python - <<'PY'
import json
from pathlib import Path
import pandas as pd

run = Path("Members/sang/model/evaluation/results/fusion/fusion_candidate_01_validation")
for filename in ("fusion_weights.json", "metrics.json"):
    print(filename)
    print(json.dumps(json.loads((run / filename).read_text()), ensure_ascii=False, indent=2))
print(pd.read_csv(run / "validation_predictions.csv").head())
print(pd.read_csv(run / "excluded_patients.csv").head())
PY
```

후보별 validation 성능, 사용 환자 수, 제외 환자, 가중치를 함께 비교합니다. AUROC·AP가 `null`이면 단일 클래스 등 계산 불가 사유도 확인합니다.

### 6-4. 확정한 validation run으로 가중치 고정

`fusion.yaml`의 기존 `fusion` 블록에서 다음 두 항목만 수정합니다. 나머지는 유지합니다.

```yaml
fusion:
  mode: saved
  saved_run_dir: evaluation/results/fusion/fusion_candidate_01_validation
```

이 경로는 `fusion.yaml` 기준입니다. 실제로 선택한 후보가 02라면 폴더명도 02로 바꿉니다. `classification.threshold`는 validation 때 확정한 값을 그대로 유지하고 `evaluation.threshold_override`도 확인합니다.

### 6-5. 최종 test 실행

```bash
Global/.venv/bin/python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml \
  --split test \
  --run-id fusion_final_test_01
```

저장된 ML·DL 모델과 결합 가중치로 test를 추론합니다. 결과는 `Members/sang/model/evaluation/results/fusion/fusion_final_test_01/`에 생깁니다. `test_predictions.csv`, `metrics.json`을 확인합니다.

임계값 override가 꼭 필요하면 `--threshold-override 0.4`처럼 지정할 수 있습니다. 이 경우 validation과 test 양쪽에 같은 확정 값을 적용합니다. Test를 본 뒤 임계값을 바꾸는 용도로 사용하지 않습니다.

## 7. 결과 파일과 MLflow 확인

| 파일/폴더 | 확인할 내용 |
| --- | --- |
| `validation_predictions.csv` 또는 `test_predictions.csv` | patient ID, 정답, `p_ml`, `p_dl`, `p_final`, 가중치, 임계값, 예측 클래스, 기반 모델 run ID |
| `metrics.json` | 평가 split·환자 수·AUROC·AP·분류 지표·혼동행렬·실제 적용 임계값 |
| `fusion_weights.json` | 사용한 가중치, 모드, 가중치 학습 정보, ML·DL run ID |
| `resolved_config.yaml` | 실제 선택된 모델 경로와 metadata, 데이터 설정, 적용 임계값 |
| `excluded_patients.csv` | 해당 평가 split에서 한쪽 예측만 있어 fusion에서 제외된 환자 |
| `data_audit/` | 데이터 검증, 사용 가능 환자 목록, 앞 단계 제외 사유 |
| `mlflow_status.json` | MLflow 전송을 요청했을 때 성공 정보 또는 `PENDING` 상태 |

원자료나 모델 로딩 자체가 잘못되면 평가가 중단될 수 있습니다. `intersection_and_report`가 잘못된 파일을 무시하는 옵션은 아닙니다. 전체 제외 사유는 `excluded_patients.csv`와 `data_audit/`을 함께 확인합니다.

MLflow에서 설정한 experiment를 열고 `model_name=Fusion`, `stage=evaluation`인 run을 찾습니다. Validation/test 집계 지표와 가중치, `ml_source_run_id`, `dl_source_run_id`로 원본 모델을 확인할 수 있습니다. 기반 모델 학습 run은 각각의 학습 실행에서 별도로 기록합니다. Fusion 업로드는 환자별 예측 CSV와 원시 ECG를 전송하지 않습니다.

### 7-1. 업로드 실패 시 전송만 재시도

`mlflow_status.json`이 `PENDING`이면 연결을 복구하고 아래를 실행합니다. 예시는 test 결과이며 재전송할 실제 폴더로 바꿉니다.

```bash
Global/.venv/bin/python -m Members.sang.model.tracking.check_connection
Global/.venv/bin/python - <<'PY'
from pathlib import Path
from Members.sang.model.tracking.log_evaluation import log_evaluation_to_mlflow

run = Path("Members/sang/model/evaluation/results/fusion/fusion_final_test_01")
result = log_evaluation_to_mlflow(run)
print(result)
PY
```

평가나 가중치 학습을 반복하지 않고 저장된 결과를 전송합니다. 노트북에서도 같은 함수를 import한 뒤 `log_evaluation_to_mlflow(fusion_run)` 또는 `log_evaluation_to_mlflow(test_run)`으로 재시도할 수 있습니다. 노트북은 MLflow를 켜면 사전 연결 확인에서 중단될 수 있으므로, 서버 없이 먼저 평가하려면 logging을 끄고 실행합니다.

## 8. 실행이 막힐 때

| 증상 | 확인·조치 |
| --- | --- |
| `No module named Members` 또는 패키지 import 실패 | 저장소 루트에서 Global Python으로 실행하고 노트북 커널 확인 |
| `choose_ml_run`, `choose_dl_run` 관련 파일 없음 | `fusion.yaml` placeholder를 실제 저장 폴더로 수정 |
| metadata·전처리 파일·checkpoint 없음 | 모델 파일 하나가 아닌 학습 run 폴더 전체 준비 |
| raw provenance·품질 인덱스 검증 실패 | [DL_manu.md](DL/DL_manu.md)의 실제 데이터 준비와 학습 조건 확인 |
| label/split/target semantics 불일치 | 원본 split과 모델 metadata 확인; 평가를 위해 라벨이나 split을 임의 수정하지 않음 |
| 공통 환자 없음 | 두 모델의 데이터 경로·평가 split·ID와 감사 자료 확인 |
| learn 모드에서 양쪽 클래스 필요 오류 | 공통 validation 환자에 양성과 음성이 모두 있는지 확인; test로 가중치를 학습하지 않음 |
| 가중치 합 오류 | manual 가중치를 음수 없이 합 1로 설정 |
| saved에서 예전 모델을 읽음 | 선택은 saved run의 `resolved_config.yaml`에서 가져옴; 새 모델 조합은 manual/learn validation부터 실행 |
| 출력 폴더가 이미 존재 | 새 `--run-id` 사용 또는 옵션 생략 |
| MLflow 연결·인증 실패 | 서버 주소·계정·환경 변수·LAN 경로 확인 후 전송 재시도 |

현재 원시 데이터의 provenance·품질 자료 준비 여부와 개인 서버 연결 여부는 각 실행 환경에서 확인해야 합니다. 이 문서는 실행 절차이며 실제 성능 수치나 서버 업로드 성공을 보장하는 실험 결과가 아닙니다.
