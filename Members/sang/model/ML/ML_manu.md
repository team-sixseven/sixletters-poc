# Clinical ML 사용 매뉴얼 — 준비부터 학습·Optuna·평가·MLflow까지

이 문서는 `Members/sang/model/ML`의 현재 코드를 사용하는 순서입니다. 기존 모델별 파라미터와 Optuna 탐색 범위 설명은 맨 아래 부록에 유지했습니다.

```text
환경·개인 계정 준비 → 데이터·입력 버전 확인
→ 일반 YAML 학습 또는 Optuna 중 하나 선택
→ train 학습 / validation AP로 모델 선택 → MLflow 기록
→ 저장 모델로 최종 test 평가 → 평가 MLflow 기록
```

아래 터미널 명령은 **Linux/WSL의 저장소 루트**에서 실행합니다. Windows 사용자는 WSL Ubuntu 터미널을 사용합니다. `<run_id>`와 계정·저장소·서버 경로는 본인의 실제 값으로 바꿉니다. 예시에서 부분 YAML이라고 안내한 블록으로 기존 파일 전체를 덮어쓰지 않습니다.

## 1. 실행 환경 준비

### 1-1. 저장소 루트로 이동

```bash
cd ~/sixletters-poc
```

루트에 `Global/`, `Members/`가 있어야 합니다. 경로가 다르면 본인 저장소 위치로 바꿉니다.

### 1-2. 기존 Python 환경 확인

```bash
Global/.venv/bin/python --version
Global/.venv/bin/python -c "import pandas, sklearn, xgboost, catboost, optuna, torch; print('ML / common evaluation imports OK')"
```

프로젝트는 Python 3.12와 고정한 uv 버전을 사용합니다. 기존 환경이 정상이라면 다음 단계로 갑니다. 환경이 없거나 패키지가 빠졌다면:

```bash
uvx --from uv==0.9.26 uv sync --project Global --locked --extra ml --extra gpu
```

ML estimator의 현재 기본 설정은 CPU입니다. 다만 공통 평가 코드가 DL 로더도 import하므로, ML 학습부터 공통 평가까지 사용하는 Linux/WSL 환경은 위처럼 두 extra를 설치합니다. 실제 GPU 사용 여부와 PyTorch 설치 여부는 별개입니다. 이미 DL 실행 환경이 있다면 같은 환경을 사용하면 됩니다.

### 1-3. Jupyter 실행

커널 등록은 처음 한 번만 진행합니다.

```bash
Global/.venv/bin/python -m ipykernel install --user \
  --name sixletters-poc-global \
  --display-name "Python (sixletters-poc Global)"
Global/.venv/bin/python -m jupyterlab
```

브라우저에서 커널을 **Python (sixletters-poc Global)**로 선택하고 `Members/sang/model/ML/`로 이동합니다.

## 2. 개인 MLflow 계정 연결

### 2-1. 본인 계정 한 번 등록

다음은 `euna`의 예시입니다. owner·username은 본인 계정, uri는 실제 서버 주소로 바꿉니다. 비밀번호는 프롬프트에서 입력합니다.

```bash
Global/.venv/bin/python -m sixletters setup \
  --owner euna \
  --username euna \
  --uri http://192.168.10.88:5000 \
  --experiment sang-scd-ml \
  --password
```

접속 주소·팀 계정·네트워크 절차는 [TEAM_MLFLOW.md](../tracking/TEAM_MLFLOW.md)를 확인합니다. 서버 PC 자체에서는 주소가 `http://127.0.0.1:5000`일 수 있습니다.

계정 설정은 Git 제외 파일 `.poc.local.toml`에 보관합니다. 브라우저 로그인과 Python 업로드 계정은 별도입니다. 비밀번호를 코드·YAML·노트북에 적거나 다른 사람의 설정 파일을 복사하지 않습니다.

### 2-2. 연결·인증 확인

```bash
Global/.venv/bin/python -m Members.sang.model.tracking.check_connection
```

서버 주소, owner, username, experiment가 맞는지 확인합니다. 비밀번호를 출력하거나 run을 생성하지 않고 인증·실험 조회를 확인합니다.

`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MLFLOW_EXPERIMENT_NAME` 환경 변수가 설정돼 있으면 파일보다 우선합니다. 예상과 다른 계정이 나오면 현재 커널과 환경 변수를 확인합니다.

오프라인 실행은 이후 노트북에서 `mlflow_logging=False`로 설정합니다. MLflow를 켜면 학습 전 연결 검사 실패는 중단하고, 학습 완료 후 전송 실패는 로컬 모델을 보존합니다.

## 3. 데이터와 경로 확인

[basic.yaml](../basic.yaml)의 관련 항목입니다. 기존 파일의 아래 값만 확인·수정합니다.

```yaml
data:
  clinical:
    path: ML/data/clinical_clean.csv
    id_column: Patient ID
    label_column: SCD
  split:
    path: ML/data/final_split.csv
```

- 임상 CSV의 환자 ID와 split CSV의 `pid`를 문자열로 연결합니다.
- Split은 train/val/test를 유지하고 다른 split에서 환자를 옮기지 않습니다.
- SCD는 제공된 binary label 0/1이며 임의로 만들거나 바꾸지 않습니다.
- 선택한 입력 변수에 누락 열·결측값이 있으면 원본과 전처리 결과를 확인합니다. 이 ML pipeline은 feature를 imputation하지 않습니다.
- 전처리는 train에만 fit하고 validation/test에는 저장된 전처리를 적용합니다.
- ML 단독 학습·평가에는 학습된 DL 모델이 필요하지 않습니다. Raw DL의 품질 인덱스 준비 조건도 ML 학습 조건이 아닙니다.

공통 데이터 감사:

```bash
Global/.venv/bin/python Members/sang/scripts/audit_model_data.py
```

출력된 ML train/validation/test 환자 수와 label 분포, 제외 사유를 확인합니다. 임상 ML 감사와 학습은 ECG 파일을 읽지 않습니다. 감사 명령은 모델을 학습하거나 MLflow로 전송하지 않습니다.

상대 경로는 해당 YAML 위치 기준입니다. `basic.yaml`의 임상 경로는 `model/` 기준이고, 모델 `parameter.yaml`의 상대 경로는 각 알고리즘 디렉터리 기준입니다.

## 4. 알고리즘과 입력 변수 버전 선택

### 4-1. 알고리즘 선택

[model_choice.yaml](model_choice.yaml)의 `models`는 **일반 YAML 학습**에 사용할 모델 목록입니다. 하나만 쓰려면 목록을 해당 모델 하나로 바꿉니다.

```yaml
basic_config: ../basic.yaml
models:
  - XGBoost
  - CatBoost
  - Random Forest
model_configs:
  XGBoost: xgboost/parameter.yaml
  CatBoost: catboost/parameter.yaml
  Random Forest: random_forest/parameter.yaml
model_output_dir: models
```

Optuna는 노트북의 `model_name`으로 알고리즘 하나를 고릅니다. 이름과 대소문자를 그대로 사용하며 `Random Forest`에는 공백이 있습니다.

### 4-2. 입력 버전 선택

`model_version`은 모델 저장 횟수가 아니라 **임상 입력 변수 묶음 번호**입니다. 모든 알고리즘에서 같은 버전을 사용할 수 있습니다.

| 버전 | 변수 수 | 구성 |
|---|---:|---|
| 1 | 4 | 성별·나이·LVEF·NYHA |
| 2 | 8 | 버전 1 + eGFR·허혈성 원인·Albumin·Diabetes |
| 3 | 21 | 버전 2 + 제공된 심장 구조·기능·병력·혈압 등 |
| 4 | 23 | 버전 3 + 흡연량·음주량 |

정확한 열 이름과 순서는 [feature_versions.py](feature_versions.py)에 있습니다. 터미널에서도 확인할 수 있습니다.

```bash
Global/.venv/bin/python -c "from Members.sang.model.ML.feature_versions import get_feature_columns; print(get_feature_columns(4))"
```

노트북은 기본 버전 4를 사용합니다. 버전이 다른 저장 모델을 선택한 버전의 모델로 재사용하지 않습니다.

## 5. 모델별 YAML 수정

| 모델 | 수정할 파일 | 모델 파일 형식 |
|---|---|---|
| XGBoost | [xgboost/parameter.yaml](xgboost/parameter.yaml) | `best_model.json` |
| CatBoost | [catboost/parameter.yaml](catboost/parameter.yaml) | `best_model.cbm` |
| Random Forest | [random_forest/parameter.yaml](random_forest/parameter.yaml) | `best_model.joblib` |

기본 역할은 다음과 같습니다.

- `model`: estimator에 전달할 하이퍼파라미터
- `training`: XGBoost/CatBoost의 early stopping 등 trainer 설정
- `classification.threshold`: 확률을 0/1로 나누는 threshold
- `model_path`, `model_paths`: 기존 저장 모델을 선택하는 경로

현재 XGBoost의 학습 관련 부분:

```yaml
model:
  n_estimators: 500
  max_depth: 3
  learning_rate: 0.03
training:
  early_stopping_rounds: 30
classification:
  threshold: 0.5
```

현재 CatBoost의 학습 관련 부분:

```yaml
model:
  iterations: 500
  depth: 4
  learning_rate: 0.03
training:
  early_stopping_rounds: 30
classification:
  threshold: 0.5
```

현재 Random Forest의 학습 관련 부분:

```yaml
model:
  n_estimators: 500
  min_samples_leaf: 2
  class_weight: balanced
classification:
  threshold: 0.5
```

위는 발췌입니다. 기존 파일의 저장 경로 항목은 보존합니다. 상세 파라미터 의미와 Optuna 탐색 범위는 부록에 있습니다.

XGBoost·CatBoost는 validation AP로 best iteration을 선택합니다. Random Forest는 설정한 tree 수로 학습하고 validation AP를 기록하며 early stopping을 하지 않습니다. 세 알고리즘 간 최종 승자를 자동 선택하는 기능은 없습니다.

## 6. 노트북 실행 설정

[model_experiment_only_ML.ipynb](model_experiment_only_ML.ipynb)를 열고 처음부터 실행합니다. 처음 실행하거나 코드·환경이 바뀌었으면 커널을 재시작합니다.

실행 설정의 현재 기본값:

```python
model_version = 4
training_mode = "optuna"  # "yaml" 또는 "optuna"
model_name = "XGBoost"   # Optuna에서 사용할 알고리즘 하나
n_trials = 100           # Optuna에서만 사용
reuse_saved = False      # 일반 학습에서 기존 run을 재사용할지 선택
mlflow_logging = True
run_final_test = True    # 후보 비교·튜닝 중에는 False
retry_mlflow = False
```

- 전체 셀을 실행해도 `training_mode`로 고른 경로 하나만 학습합니다.
- `yaml`은 model_choice.yaml의 모델 목록을 사용합니다. 이때 model_name/n_trials는 사용하지 않습니다.
- `optuna`는 model_name 하나를 탐색합니다. 이때 일반 학습 셀은 건너뜁니다.
- 계정은 해당 PC의 기존 MLflow 설정을 읽습니다.
- 학습 후 자동 평가에는 이번 실행의 모델 경로를 직접 사용합니다. 원본 eval.yaml을 수동으로 바꾸지 않습니다.
- 설정을 비교하는 중에는 `run_final_test=False`로 둡니다. Test를 본 뒤 설정을 다시 고르지 않습니다.

## 7. 방법 A — YAML 고정값으로 일반 학습

### 7-1. 실행 설정 변경

```python
model_version = 4
training_mode = "yaml"
reuse_saved = False
mlflow_logging = True
run_final_test = False   # 초기 실행·후보 비교 시
retry_mlflow = False
```

### 7-2. 셀을 위에서 아래로 실행

1. 공통 import와 입력 변수·실행 설정을 실행합니다.
2. YAML 고정값 학습 셀이 model_choice.yaml의 각 모델을 학습합니다.
3. 완료한 각 run의 모델·전처리·하이퍼파라미터·집계 validation 지표를 개인 계정 MLflow에 기록합니다.
4. 결과 확인 셀에서 `run_dirs`, validation 확률, AP 및 전송 상태를 확인합니다.
5. Optuna 관련 실행 셀은 자동으로 건너뜁니다.
6. 최종 모델·입력 버전·threshold가 확정됐으면 run_final_test=True인 최종 실행 또는 11절의 저장 모델 평가를 진행합니다.

일반 결과는 `ML/models/<run_id>/`에 각 모델별로 저장됩니다. 목록에 여러 알고리즘이 있으면 run도 여러 개입니다.

### 7-3. 저장 모델 재사용과 새 학습의 차이

| 설정 | 동작 |
|---|---|
| 노트북 기본 `reuse_saved=False` | YAML 파라미터로 새 run 학습 |
| `reuse_saved=True`이고 유효한 저장 경로 있음 | 해당 run 재사용; 변경한 학습값을 새로 fit하지 않음 |
| `reuse_saved=True`이고 저장 모델 없음 | 새로 학습 후 parameter YAML에 저장 경로 등록 |
| 저장 파일은 있으나 구성 불완전/모델 종류 불일치 | 오류로 중단 |

현재 코드에서 `reuse_saved=False` 일반 학습은 parameter YAML의 model_path를 자동 갱신하지 않습니다. 노트북의 최종 평가는 `run_dirs`를 직접 사용하므로 새 run을 정확히 평가합니다. 이후 별도 평가에서는 실제 run을 artifact_dir로 지정하거나, 10절 방법으로 경로를 등록합니다.

Python API는 노트북과 다르게 기본 `reuse_saved=True`, `mlflow_logging=False`입니다. 직접 호출할 때 아래 예시처럼 원하는 값을 명시합니다.

## 8. 방법 B — Optuna로 best 모델 탐색

### 8-1. 실행 설정 변경

```python
model_version = 4
training_mode = "optuna"
model_name = "XGBoost"
n_trials = 100
mlflow_logging = True
run_final_test = False   # 탐색·후보 비교 중에는 False
retry_mlflow = False
```

빠른 동작 점검은 `n_trials`를 줄여 실행할 수 있지만, 짧은 탐색이 충분한 성능 탐색을 의미하지는 않습니다.

### 8-2. 준비 → objective → 탐색 → 저장 순서로 실행

1. 공통 설정 셀을 실행합니다.
2. Optuna 준비 셀에서 알고리즘·입력 버전·개인 계정 연결과 best 저장 위치를 확인합니다.
3. `objective(trial)` 셀에서 탐색 범위와 조건부 옵션을 확인합니다. 수정한다면 모델별 조건을 유지합니다.
4. 탐색 실행 셀을 실행합니다. 각 trial은 train으로 학습하고 validation AP를 반환합니다.
5. 탐색이 성공하면 best 하나를 저장하고 모델 YAML의 model_path/model_paths에 등록합니다.
6. 결과 셀에서 `best_run_dir`, best trial/파라미터/AP와 MLflow 링크를 확인합니다.
7. 최종 모델을 고정한 뒤 run_final_test=True의 최종 평가 셀 또는 11절의 저장 모델 평가를 실행합니다.

각 알고리즘×입력 버전은 별도 study입니다. 한 번 실행해서 3개 알고리즘×4개 버전을 모두 자동 탐색하지 않습니다.

### 8-3. 저장 정책과 다음 실행

- Best 저장: `ML/models/best/<algorithm>_vN/`, 예: `best/xgboost_v4/`
- Trial 모델은 임시 저장하고 최종 best만 보관합니다.
- 같은 알고리즘·버전의 다음 study가 성공하면 관리 중인 best 폴더를 새 best로 갱신합니다.
- 다른 알고리즘·버전은 유지합니다. 실패하면 이전 best를 보존합니다.
- Trial별 파라미터/AP/상태는 `trials.csv`에 보관합니다.
- 원본 parameter YAML의 기본 학습값은 유지하고, best 전체 설정은 `best_parameters.yaml`에 저장합니다.

이미 완료한 experiment를 재사용해 탐색 셀만 반복 실행하지 않습니다. 새로운 탐색은 준비 셀부터 새 experiment를 만듭니다. 보관할 기존 best가 있으면 새 study가 그 폴더를 교체한다는 점을 고려해 먼저 별도 보관합니다.

## 9. 터미널에서 실행하는 방법

### 9-1. 명시적 버전·MLflow가 필요한 일반 학습

현재 Python API로 실행하는 예입니다. 저장소 루트에서 명령 전체를 실행합니다. YAML은 5절에서 먼저 설정합니다.

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
    print(run)
PY
```

이 명령은 학습·validation·학습 결과 전송까지 실행합니다. Test는 11절에서 별도로 평가합니다. 오프라인이면 mlflow_logging=False로 바꿉니다. 다른 알고리즘은 model_names 값을 바꿉니다.

작은 train subset으로 학습 경로만 확인하려면 API에 `limit_train_patients=32`를 추가할 수 있습니다. Validation은 축소하지 않으며, 축소 학습을 전체 모델 성능으로 해석하지 않습니다.

### 9-2. 기존 호환 CLI

```bash
Global/.venv/bin/python Members/sang/scripts/evaluate_preprocessed_clinical.py --help
Global/.venv/bin/python Members/sang/scripts/evaluate_preprocessed_clinical.py \
  --models "Random Forest" \
  --run-id clinical_rf_001
```

이 스크립트는 이름과 달리 저장 모델 재사용/일반 학습 진입점입니다. 현재 옵션은 --models, --run-id, --limit-train-patients이며 **model_version·MLflow·강제 새 학습 옵션을 노출하지 않습니다.** API 기본값으로 실행하므로 위의 명시적 API 명령이나 노트북을 사용하는 편이 동작을 분명히 지정할 수 있습니다. 이 CLI를 실행한다고 자동으로 test나 MLflow까지 실행되지는 않습니다.

Optuna는 현재 노트북 또는 `ClinicalOptunaExperiment` Python API로 실행하며, 별도 Optuna CLI는 제공하지 않습니다.

## 10. 저장 결과와 경로 등록

### 10-1. 필요한 파일 유지

| 결과 | 주요 내용 |
|---|---|
| native 모델 | XGBoost JSON / CatBoost CBM / Random Forest joblib |
| `preprocessing.joblib` | XGBoost·Random Forest의 train-fitted 전처리 |
| `metadata.json` | 알고리즘·입력 버전·열·threshold·양성 클래스·모델 파일 |
| `resolved_config.yaml` | 입력 YAML과 실제 estimator/training 설정 |
| `metrics.json` | 집계 validation 지표 |
| `validation_predictions.csv` | 환자별 validation label·확률·threshold |
| `run_status.json` | 학습 완료 시 COMPLETE |
| 일반 run의 history·audit | `training_history.csv`, `data_audit/` |
| Optuna best 추가 파일 | `summary.json`, `trials.csv`, `best_parameters.yaml` |
| MLflow 활성 시 상태 파일 | `summary.json`의 mlflow와 `mlflow_status.json` |

원본 run 폴더를 함께 유지합니다. 저장 경로 검증은 model 종류·버전·입력 열·완료 상태·필수 companion 파일을 확인합니다. native 모델 하나만 옮겨 설정하면 실패할 수 있습니다.

### 10-2. model_path와 model_paths

모델 parameter YAML의 경로 예시입니다. 경로는 해당 YAML 기준입니다.

```yaml
model_path: ../models/best/xgboost_v4/best_model.json
model_paths:
  4: ../models/best/xgboost_v4/best_model.json
```

- model_path: 가장 최근 등록 경로
- model_paths: 입력 버전별 등록 경로
- Optuna save_best는 성공 후 두 경로를 갱신합니다.
- 일반 새 run을 등록하지 않아도 명시적 artifact_dir로 바로 평가할 수 있습니다.

### 10-3. 새 일반 run을 학습 없이 등록하려는 경우

터미널에서 Python을 엽니다.

```bash
Global/.venv/bin/python
```

다음 예시에서 `<run_id>`를 실제 일반 학습 폴더명으로 바꿉니다. Metadata의 모델 이름을 읽어 맞는 parameter 파일에 등록합니다.

```python
import json
from pathlib import Path
from Members.sang.model.config import load_yaml, resolve_config_path
from Members.sang.model.ML.artifacts import save_model_path

choice_path = Path("Members/sang/model/ML/model_choice.yaml").resolve()
run = Path("Members/sang/model/ML/models/<run_id>").resolve()
metadata = json.loads((run / "metadata.json").read_text())
choice = load_yaml(choice_path)
parameter_path = resolve_config_path(choice["model_configs"][metadata["model"]], choice_path)
print(save_model_path(parameter_path, run))
```

이 함수는 새로 학습하지 않고 해당 parameter YAML의 경로만 갱신합니다.

## 11. 저장 모델만 불러와 evaluation

### 11-1. 가장 명확한 방법: 평가할 run을 직접 지정

[evaluation/eval.yaml](../evaluation/eval.yaml)의 **다음 항목만 수정하고 metrics 등 나머지는 유지**합니다.

```yaml
evaluation:
  target: ml
  fusion_config: null
  model_path: null
  artifact_dir: ../ML/models/<run_id>
  split: test
  threshold_override: null
  mlflow_logging: true
```

Optuna best면 `artifact_dir: ../ML/models/best/xgboost_v4`처럼 지정합니다. 이 경로는 **eval.yaml이 있는 evaluation 디렉터리 기준**입니다. 명시적 artifact_dir가 있으면 실제 모델/입력 버전은 저장 metadata를 따릅니다. `evaluation.model_path`는 DL용 필드이므로 ML에서는 null로 둡니다.

### 11-2. parameter YAML의 등록 경로로 선택

```yaml
evaluation:
  target: ml
  fusion_config: null
  model_name: XGBoost
  model_version: 4
  model_path: null
  artifact_dir: null
  split: test
  threshold_override: null
  mlflow_logging: true
```

artifact_dir가 null이면 선택한 알고리즘 parameter YAML을 읽습니다. model_version=4이면 model_paths의 4번 경로를 검증하며, null이면 최신 model_path를 사용합니다. 평가에서는 파일이 없다고 모델을 학습하지 않고 오류로 알립니다.

### 11-3. 평가 실행

노트북에서 선택한 모델을 그대로 평가할 경우에는 학습 노트북의 마지막 평가 셀을 사용합니다. 현재 커널에 run_dirs 또는 best_run_dir가 있어야 합니다. 커널을 재시작했거나 다른 저장 모델을 평가할 때는 위 YAML을 설정한 뒤 다음 CLI 또는 [공통 평가 노트북](../evaluation/evaluation_experiment.ipynb)을 사용합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml \
  --split test
```

Validation 재평가는 `--split validation`으로 바꿉니다. 저장된 모델·전처리·threshold를 불러와 환자 확률을 계산하며 재학습하지 않습니다. 평가 중 현재 parameter YAML의 하이퍼파라미터로 모델을 새로 fit하지 않습니다.

### 11-4. 평가 결과 확인

기본 결과는 `evaluation/results/ml/<evaluation_run_id>/`입니다.

- `predictions.csv`: PID, split, 실제 label, 확률, threshold, 이진 예측, 모델/run/입력 버전
- `metrics.json`: 환자 수·클래스 수, AUROC, Average Precision, sensitivity/recall, specificity, precision, F1, confusion matrix 등
- `resolved_config.yaml`: 실제 artifact 경로와 모델·평가 설정
- `data_audit/`: 환자와 제외 사유
- `mlflow_status.json`: 전송을 켠 경우 상태와 링크

AP는 sklearn Average Precision이며 사다리꼴 PR-AUC와 구분합니다. 단일 클래스/분모 0 등의 계산 불가 지표는 null과 이유를 기록합니다. Test 결과로 모델·입력 버전·threshold를 반복 선택하지 않습니다.

Threshold만 재평가하려면 `threshold_override` 또는 CLI `--threshold-override`를 사용합니다. Validation에서 정하고 test에 고정 적용하며, override는 저장 모델 자체를 바꾸지 않습니다.

## 12. MLflow 결과 확인과 재시도

학습과 평가는 별도 MLflow run입니다. 평가 run은 학습 결과의 원래 run ID와 가능한 경우 MLflow 링크를 기록합니다. 원본 CSV·환자별 예측·PID audit은 자동 전송하지 않습니다.

- COMPLETE: 로컬 학습 완료
- FINISHED: MLflow 전송 완료
- PENDING: 로컬 결과는 있지만 전송 미완료
- Logging=False: 전송 상태 파일이 없을 수 있음

학습 전 계정 확인 실패는 중단합니다. 학습 후 전송이 실패하면 모델을 보존하며, 새로 학습하지 않고 전송 함수만 재실행합니다.

저장소 루트에서 Python을 열어 아래 예시의 run 경로를 바꿉니다.

```python
from pathlib import Path
from Members.sang.model.tracking.log_best import log_ml_training_to_mlflow

run = Path("Members/sang/model/ML/models/<run_id>")
result = log_ml_training_to_mlflow(run)
print(result["url"])
```

이 함수는 일반 학습과 Optuna best를 모두 지원합니다. Optuna라면 run을 실제 `models/best/<algorithm>_vN` 경로로 바꿉니다.

평가 전송만 재시도:

```python
from pathlib import Path
from Members.sang.model.tracking.log_evaluation import log_evaluation_to_mlflow

evaluation_dir = Path("Members/sang/model/evaluation/results/ml/<evaluation_run_id>")
result = log_evaluation_to_mlflow(evaluation_dir)
print(result["url"])
```

같은 결과는 공통 업로더의 기존 run을 재사용합니다. 노트북의 재시도 셀을 사용할 때는 현재 run 변수를 유지한 채 `retry_mlflow=True`로 바꾸고 해당 셀만 실행합니다. 설정 셀 전체를 다시 실행하면 run 변수들이 초기화되므로 재전송을 위해 학습을 반복하지 않습니다.

## 13. 오류 발생 시 확인할 내용

| 증상 | 확인할 내용·행동 |
|---|---|
| Members import 실패 | 저장소 루트·현재 커널·Global 환경 확인 |
| 패키지 import 실패 | 설치 환경과 ml/gpu extra 확인 |
| 필수 입력 열/결측값 오류 | clinical CSV와 선택한 model_version의 열 확인; label/feature 임의 생성 금지 |
| 중복 PID·split/label 불일치 | 원본 임상·split 매핑 수정 근거 확인; 환자를 다른 split으로 이동하지 않음 |
| 저장 모델은 있는데 companion 누락 | 원본 run 폴더를 함께 보관했는지 확인 |
| 모델/입력 버전 불일치 | 알고리즘, model_version, metadata feature_names 확인 |
| 파라미터를 바꿨는데 재학습 안 함 | reuse_saved=False인지 확인 |
| 새 일반 run 대신 과거 모델이 평가됨 | 실제 run을 artifact_dir로 지정하거나 올바른 경로 등록 |
| model_path 없음 | 학습/Optuna 저장 완료 확인; 평가 중 자동 학습은 안 함 |
| Unsupported model keys | YAML model 키가 해당 estimator에 존재하는지 확인 |
| CatBoost bootstrap 옵션 충돌 | Bayesian/Bernoulli 조건과 auto_class_weights 충돌 설명 확인 |
| Random Forest bootstrap/max_samples 충돌 | bootstrap=False일 때 max_samples=None 유지 |
| 닫힌 Optuna experiment 오류 | Optuna 준비 셀부터 새 experiment 생성 |
| MLflow 인증·서버 오류 | 본인 계정·주소·실험 권한·환경 변수 우선순위 확인 |
| MLflow PENDING | 연결 해결 후 전송만 재시도 |
| 같은 run 이름이 이미 존재 | 새 run-id 사용; 기존 결과를 덮어쓰지 않음 |

## 14. 팀 공유·검증과 다른 실행 경로

자동 테스트는 일반 학습에 필수는 아니지만 코드·환경을 바꾼 뒤 검증할 수 있습니다.

```bash
Global/.venv/bin/python -m pytest Members/sang/model/ML/tests/test_workflow.py -q
```

작은 합성 데이터의 native 모델 학습·재사용·저장 평가, Optuna best-only 기록, 모의 MLflow와 노트북 실행 경로를 확인합니다. 실제 임상 성능이나 실제 서버 전송 검증으로 해석하지 않습니다.

팀에는 코드·YAML·매뉴얼·테스트를 공유하고 개인 계정 설정·비밀번호·실제 데이터·생성 모델/결과·캐시는 Git에서 제외합니다. 상세 계정 절차는 [TEAM_MLFLOW.md](../tracking/TEAM_MLFLOW.md), ECG 단독 실험은 [DL_manu.md](../DL/DL_manu.md)를 참고하세요.

Clinical+ECG 결합은 별도 [fusion.yaml](../fusion.yaml)에 저장 ML·DL 경로와 가중치를 정하고 [통합 노트북](../ECG+clinical_experiment.ipynb) 또는 target=fusion 평가를 사용합니다. ML 단독 평가의 target은 ml입니다.

## 부록 — 모델 파라미터·저장 정책·Optuna 탐색 범위

이 문서는 현재 ML trainer가 읽는 모델별 `parameter.yaml` 키를 설명합니다. 값과 적용 방식은 각 YAML과 `ML/training.py`, `ML/models.py` 구현 기준입니다.

### 설정 파일 연결

실험에서 사용할 후보는 [`model_choice.yaml`](model_choice.yaml)의 `models`에 적고, `model_configs`가 각 후보의 parameter 파일을 지정합니다.

| 후보 | 설정 파일 |
|---|---|
| XGBoost | [`xgboost/parameter.yaml`](xgboost/parameter.yaml) |
| CatBoost | [`catboost/parameter.yaml`](catboost/parameter.yaml) |
| Random Forest | [`random_forest/parameter.yaml`](random_forest/parameter.yaml) |

각 파일에는 모델 선택에 직접 영향을 주는 값만 둡니다. 실행 편의와 재현성을 위한 기본값은 `ML/models.py`와 `ML/training.py`에서 관리하고, 실제 적용된 값은 각 run의 `resolved_config.yaml`에 기록합니다.

YAML에는 `model`, `classification`을 둡니다. XGBoost와 CatBoost는 validation 기준 best iteration 선택에 필요한 `training.early_stopping_rounds`도 설정합니다. Random Forest는 `training` 설정 없이 코드에서 early stopping을 끕니다.

- `model`: estimator에 전달되는 모델 자체의 파라미터
- `training`: 학습 중단·subset sampling 등 trainer 동작 설정
- `classification`: 확률을 0/1로 나눌 때 사용할 기본 threshold

현재 `model_choice.yaml`은 validation Average Precision을 높이는 방향으로 학습 후보를 평가합니다. XGBoost와 CatBoost는 validation AP로 best iteration을 저장하고, Random Forest는 설정된 tree 수를 사용해 validation AP를 기록합니다. 후보 run 중 어떤 모델을 최종 평가할지는 자동 결정되지 않습니다.

`model_choice.yaml`에는 사용할 모델·각 parameter 파일 경로·결과 경로만 둡니다. 노트북은 model_version의 입력 열 목록을 사용합니다. model_version을 생략한 기존 Python 호출만 CSV에서 ID·라벨을 제외한 전체 열을 사용합니다. 범주형 열 7개는 `config.py`의 기본 목록으로 관리합니다. validation Average Precision 최대화와 전처리 기본값은 코드에서 관리하고 run의 `resolved_config.yaml`에 기록합니다.

### 입력 변수 버전

노트북의 `model_version`은 알고리즘 이름과 별개인 **입력 변수 묶음 번호**입니다. `ML/feature_versions.py`의 제공 목록을 사용하며 모든 ML 알고리즘과 Optuna에 동일하게 적용합니다.

| 버전 | 변수 수 | 구성 |
|---|---:|---|
| 1 | 4 | Gender, Age, LVEF, NYHA |
| 2 | 8 | 버전 1 + eGFR, Ischemic etiology, Albumin, Diabetes |
| 3 | 21 | 버전 2 + 제공된 심장 구조·기능, 병력, 혈압, Pro-BNP, Cardiothoracic ratio |
| 4 | 23 | 버전 3 + cigarettes /year, alcohol consumption |

기본값 4는 전체 23개를 사용합니다. `Mitral valve insufficiency`의 끝 공백은 CSV 로더와 동일하게 정리했습니다. 존재하지 않는 열이나 1~4 이외 값은 오류로 표시합니다. 범주형 처리는 선택한 입력 열에 포함된 범주형 변수에만 적용합니다.

`train_clinical(..., model_version=2)`와 같이 지정할 수 있습니다. 버전을 생략한 기존 Python 호출은 이전처럼 ID·라벨을 제외한 CSV 열을 자동 선택합니다. `model_version`을 지정한 run 이름에는 `v1`~`v4`가 들어가고 `metadata.json`, `resolved_config.yaml`, validation 예측에 실제 버전과 입력 열을 기록합니다. 선택된 목록의 순서도 유지합니다.

### 저장 모델 경로

각 모델 `parameter.yaml`의 `model_path`에 저장 모델 파일을 지정합니다. 상대 경로는 해당 parameter YAML의 디렉터리 기준이며, 초기값 `null`은 저장 모델을 아직 지정하지 않았다는 뜻입니다.

```yaml
model_path: ../models/선택한_run_id/best_model.json
```

XGBoost는 `best_model.json`, CatBoost는 `best_model.cbm`, Random Forest는 `best_model.joblib`을 지정합니다. 이 저장소에서 생성한 run 디렉터리도 지정할 수 있습니다. 모델 옆의 `metadata.json`, `resolved_config.yaml`, `metrics.json`, `validation_predictions.csv`, `run_status.json`과 XGBoost·Random Forest의 `preprocessing.joblib`을 함께 유지합니다.

- 일반 `train_clinical()`은 경로가 존재하면 모델 종류와 완료 상태를 검사하고 해당 run을 재사용합니다. 학습 파라미터를 새로 적용하지 않습니다.
- 경로가 비었거나 파일이 없으면 YAML 파라미터로 학습하고 새 모델 경로를 YAML에 기록합니다.
- 파일은 있는데 run 파일이 누락되거나 모델 종류가 다르면 오류로 중단합니다. 임의로 다른 모델을 학습하지 않습니다.
- `reuse_saved=False`를 명시하면 기존 경로와 관계없이 새로 학습합니다. Optuna의 trial은 이 옵션을 사용합니다.

Optuna 탐색 실행 셀은 성공적으로 끝날 때 선택된 모델의 최신 경로인 `model_path`와 버전별 `model_paths`를 원본 YAML에 갱신합니다. 기본 하이퍼파라미터는 유지하며, 최적값 전체는 별도의 `best_parameters.yaml`에 저장합니다. 모델 로딩 코드는 `ML/artifacts.py`에 있고 기존 `inference.py`의 함수 경로도 계속 사용할 수 있습니다.

평가는 `evaluation/eval.yaml`에서 `target: ml`, `model_name`을 선택한 이름, `artifact_dir: null`로 설정하면 모델 YAML의 저장 경로를 읽습니다. `evaluation.model_version: 1`~`4`를 지정하면 `model_paths`에서 해당 버전을 고르고, `null`이면 최신 `model_path`를 사용합니다. 버전 또는 입력 열이 일치하지 않는 모델은 선택한 버전의 저장 모델로 인정하지 않습니다. 명시적 `artifact_dir`를 설정하면 그 run이 우선합니다. 평가에서 저장 모델이 없으면 오류를 표시하며, 평가 과정에서 학습하지 않습니다. Threshold는 저장 모델의 metadata 값을 사용하며 변경은 `threshold_override`로 지정합니다.

### XGBoost

현재 값은 [`xgboost/parameter.yaml`](xgboost/parameter.yaml)에 있습니다.

| 키 | 현재 값 | 역할 |
|---|---:|---|
| `model.n_estimators` | `500` | boosting 최대 단계 수입니다. 실제 저장 단계는 validation early stopping이 고른 best iteration일 수 있습니다. |
| `model.max_depth` | `3` | 각 tree의 최대 깊이입니다. 깊이를 키우면 더 복잡한 상호작용을 학습할 수 있습니다. |
| `model.learning_rate` | `0.03` | 각 boosting 단계의 기여 크기를 조절합니다. |
| `training.early_stopping_rounds` | `30` | validation AP가 개선되지 않는 boosting 단계가 이 횟수에 도달하면 멈춥니다. best iteration의 모델을 저장합니다. |
| `classification.threshold` | `0.5` | 양성 확률이 이 값 이상이면 양성으로 분류합니다. 모델 학습이나 AP 기준 best iteration 선택에는 영향을 주지 않습니다. |

`random_state=42`, `n_jobs=1`, train subset seed `42`, `verbosity=0`은 코드 기본값입니다. 임상 feature는 결측 없이 제공되어야 하며, 결측값을 채우는 처리는 하지 않습니다. XGBoost 입력은 train split에서 fit한 scaling·one-hot 전처리를 거치고 validation에는 train에서 배운 전처리만 적용합니다.

### CatBoost

현재 값은 [`catboost/parameter.yaml`](catboost/parameter.yaml)에 있습니다.

| 키 | 현재 값 | 역할 |
|---|---:|---|
| `model.iterations` | `500` | boosting 최대 반복 수입니다. `use_best_model=True`로 학습하고 validation AP로 best iteration을 저장합니다. |
| `model.depth` | `4` | tree의 최대 깊이입니다. |
| `model.learning_rate` | `0.03` | 각 boosting 단계의 기여 크기를 조절합니다. |
| `training.early_stopping_rounds` | `30` | validation AP가 개선되지 않는 반복 횟수 기준입니다. trainer가 validation AP를 평가 지표로 전달합니다. |
| `classification.threshold` | `0.5` | 양성 확률을 0/1 분류로 바꾸는 기본 경계입니다. 학습·best iteration 선택과는 별개입니다. |

`random_seed=42`, `verbose=false`, `thread_count=1`, `allow_writing_files=false`, train subset seed `42`는 코드 기본값입니다. CatBoost에는 categorical feature를 문자열로 변환해 전달합니다. XGBoost/Random Forest처럼 one-hot encoder를 적용하지 않습니다.

### Random Forest

현재 값은 [`random_forest/parameter.yaml`](random_forest/parameter.yaml)에 있습니다.

| 키 | 현재 값 | 역할 |
|---|---:|---|
| `model.n_estimators` | `500` | 학습할 decision tree 수입니다. |
| `model.min_samples_leaf` | `2` | 각 leaf에 필요한 최소 train 환자 수입니다. 너무 작은 leaf가 생기는 것을 제한합니다. |
| `model.class_weight` | `balanced` | train label의 클래스 빈도에 반비례하도록 클래스 가중치를 줍니다. |
| `classification.threshold` | `0.5` | 양성 확률을 분류 결과 0/1로 바꾸는 기본 경계입니다. |

`max_depth=null`, `max_features=sqrt`, `random_state=42`, train subset seed `42`는 코드 기본값입니다. Random Forest는 XGBoost와 같은 train-fitted scaling·one-hot 전처리를 씁니다. 결측 feature를 채우지 않으며, Validation AP는 기록하지만 best tree 수를 고르거나 조기 종료하지 않습니다.

### Threshold와 재평가

`classification.threshold`는 estimator가 확률을 만드는 방식을 바꾸지 않고, 확률을 이진 예측과 threshold 의존 지표로 바꿀 때 사용합니다. 저장 모델 평가에서 threshold만 바꾸려면 `evaluation/eval.yaml`의 `threshold_override` 또는 평가 CLI의 `--threshold-override`를 사용하면 재학습이 필요 없습니다. 이 override는 해당 평가 결과에만 적용됩니다.

학습 run의 validation 기록에 저장되는 threshold 기본값을 바꾸려면 해당 모델의 `parameter.yaml` 값을 조정하고 새 학습 run을 만들어야 합니다. Threshold나 후보 비교를 위해 test 결과를 반복해서 사용하지 마세요.

### Seed 구분

Estimator와 train subset의 seed 기본값은 코드에서 `42`로 고정합니다. 전체 train cohort를 쓰는 일반 학습에서는 subset seed가 샘플을 바꾸지 않습니다. 각 run의 `resolved_config.yaml`에서 YAML 입력값과 실제 적용된 estimator·training 기본값을 확인할 수 있습니다.

### Optuna 실험

[`model_experiment_only_ML.ipynb`](model_experiment_only_ML.ipynb)의 `Optuna 실험`에서 모델 하나와 실험 횟수를 정하고 탐색할 수 있습니다. 모델별 탐색 범위는 `objective(trial)` 셀의 `suggest_int`, `suggest_float`, `suggest_categorical` 호출에 있습니다.

현재 YAML은 기본값으로 읽고 `train_clinical(..., model_names=[model_name], model_parameters=params, reuse_saved=False, model_version=model_version)`로 해당 trial의 모델 파라미터만 덮어씁니다. trial 중에는 원본 YAML을 변경하지 않으며 적용값은 각 trial run의 `resolved_config.yaml`에 기록합니다. 탐색 실행 셀이 완료되면 선택된 run의 최신 경로와 해당 버전 경로를 YAML에 기록합니다. 다른 버전의 경로는 유지합니다. 이 인자는 모델 하나를 선택한 경우에만 지원합니다. Threshold·조기 종료 횟수는 YAML 값을 유지하고 validation AP로 최적 trial을 선택합니다.

최적 모델 경로는 노트북의 `best_run_dir`이며 `ML/models/best/<algorithm>_vN/`에 최적 모델 한 개와 `summary.json`, `trials.csv`, `best_parameters.yaml`을 보관합니다. 다른 trial의 모델 파일은 영구 저장하지 않습니다. Optuna API 자체는 test를 평가하지 않습니다. 노트북은 run_final_test=True일 때 마지막 셀에서 저장 best를 별도로 평가합니다.

### Optuna 저장 정책

`ML/optuna_experiment.py`의 `ClinicalOptunaExperiment`가 임시 trial 실행과 best 보관을 관리합니다. 같은 알고리즘·버전의 새 study가 성공하면 기존 관리 폴더를 새 study의 best로 갱신합니다. 단일 모델 파일과 최적 파라미터 외에 저장 모델 재로드에 필요한 전처리·metadata·설정·validation 예측·지표·완료 상태를 함께 유지합니다. 현재 study의 모든 trial은 파라미터·AP·상태만 `trials.csv`에 남깁니다.

탐색 중에는 이전에 저장된 모델을 유지합니다. 새 best를 별도 위치에 준비한 다음 교체하고, YAML 등록이 실패하면 이전 모델을 복구합니다. 서로 다른 알고리즘·버전 폴더는 건드리지 않습니다. 기존의 날짜별 run이나 검증용 폴더는 자동 정리 대상이 아닙니다. 현재 방식은 단일 study의 순차 trial 실행을 기준으로 합니다.

### Optuna 확장 탐색 범위

아래 값은 이 프로젝트에서 정한 초기 탐색 범위이며 라이브러리 기본값이나 최적 범위라는 의미는 아닙니다. 실제 범위는 노트북 `objective(trial)`에서 수정합니다. 현재 기본 `n_trials`는 조합별 100회입니다. TPE의 첫 10회는 무작위 탐색이며, 100회도 최적값 발견을 보장하지 않습니다. [Optuna TPE 공식 문서](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)

#### XGBoost

| 파라미터 | 탐색 범위 | 역할 |
|---|---|---|
| `n_estimators` | 100~1500, 100 간격 | boosting 최대 단계 수 |
| `max_depth` | 2~8 | tree 깊이 제한 |
| `learning_rate` | 0.005~0.2, 로그 범위 | 단계별 학습 기여도 |
| `min_child_weight` | 1~20 | 자식 노드의 최소 Hessian 합. 환자 수 자체가 아님 |
| `gamma` | 0~5, 0.5 간격 | 분할에 필요한 최소 손실 감소량 |
| `subsample` | 0.6~1.0, 0.1 간격 | 학습 행 샘플링 비율 |
| `colsample_bytree` | 0.6~1.0, 0.1 간격 | tree별 feature 샘플링 비율 |
| `reg_alpha` | 0, 0.001, 0.01, 0.1, 1, 10 | leaf 가중치 L1 규제 |
| `reg_lambda` | 0.1~30, 로그 범위 | leaf 가중치 L2 규제 |
| `scale_pos_weight` | 1, 2, 4, 8 | 양성 클래스 학습 가중치 |

각 파라미터의 의미와 허용값은 [XGBoost 공식 문서](https://xgboost.readthedocs.io/en/stable/parameter.html)를 기준으로 확인했습니다. Best iteration 선택과 조기 종료는 기존 validation AP를 사용합니다.

#### CatBoost

| 파라미터 | 탐색 범위 | 역할 |
|---|---|---|
| `iterations` | 100~1500, 100 간격 | boosting 최대 반복 수 |
| `depth` | 2~8 | tree 깊이 |
| `learning_rate` | 0.005~0.2, 로그 범위 | 단계별 학습 기여도 |
| `l2_leaf_reg` | 0.1~30, 로그 범위 | leaf 가중치 L2 규제 |
| `random_strength` | 0~5, 0.5 간격 | 분할 점수의 무작위성 |
| `rsm` | 0.6~1.0, 0.1 간격 | 분할 후보 feature 비율 |
| `border_count` | 16, 32, 64, 128 | 수치 feature 양자화 경계 수 |
| `auto_class_weights` | `None`, `Balanced`, `SqrtBalanced` 문자열 | 자동 클래스 가중치 |
| `bootstrap_type` | `Bayesian`, `Bernoulli` | 학습 표본 가중·샘플링 방식 |
| `bagging_temperature` | 0~5, 0.5 간격 | Bayesian bootstrap에서만 탐색 |
| `subsample` | 0.6~1.0, 0.1 간격 | Bernoulli bootstrap에서만 탐색 |

한 trial에서는 bootstrap 방식에 맞는 마지막 두 옵션 중 하나만 사용합니다. `auto_class_weights`와 별도의 `class_weights`·`scale_pos_weight`는 함께 적용하지 않습니다. Optuna trial이 bootstrap·자동 클래스 가중치를 지정하면 trainer가 상충하는 YAML 기본 옵션을 제거하고 trial 값을 우선합니다. [CatBoost 공식 문서](https://catboost.ai/docs/en/references/training-parameters/common)

#### Random Forest

| 파라미터 | 탐색 범위 | 역할 |
|---|---|---|
| `n_estimators` | 100~1000, 100 간격 | tree 개수 |
| `max_depth` | `None`, 2, 3, 4, 5, 6, 8, 10, 15 | 최대 깊이. `None`은 깊이 제한 없음 |
| `min_samples_split` | 2~20 | 노드를 나눌 최소 샘플 수 |
| `min_samples_leaf` | 1~12 | leaf 최소 샘플 수 |
| `max_features` | `sqrt`, `log2`, 0.5, `None` | 전처리 후 분할 후보 feature 수·비율 |
| `criterion` | `gini`, `entropy` | 분할 품질 기준 |
| `class_weight` | `None`, `balanced`, `balanced_subsample` | 클래스 가중치 |
| `ccp_alpha` | 0, 0.0001, 0.001, 0.005, 0.01 | 비용 복잡도 가지치기 강도 |
| `bootstrap` | `True`, `False` | 복원 추출 사용 여부 |
| `max_samples` | 0.6~1.0, 0.1 간격 | bootstrap이 켜졌을 때만 샘플 비율 탐색 |

bootstrap이 꺼지면 `max_samples=None`을 적용합니다. `max_samples`를 지정하고 bootstrap을 끄는 조합은 사용하지 않습니다. [scikit-learn 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

경우의 수를 늘린 효과는 같은 validation에서 비교합니다. 입력 버전·파라미터·threshold를 확정한 다음 test를 평가하며, test 결과로 범위를 반복 조정하지 않습니다. 최적 모델 한 개만 보관하는 저장 정책은 동일하게 적용합니다.
