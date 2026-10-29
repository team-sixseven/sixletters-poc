# 모델 학습·평가 가이드

이 문서는 현재 `model/` 코드를 실제로 실행하는 가장 짧은 경로입니다. 모든 명령은 저장소 루트에서 실행합니다.

```text
환경 준비 → (MLflow 사용 시 계정 연결) → ML 또는 DL 학습
→ validation 평가·선택 → 설정 고정 → test 평가 → 결과 확인·기록
```

세부 하이퍼파라미터와 화면별 설명은 [ML_manu.md](ML/ML_manu.md), [DL_manu.md](DL/DL_manu.md), [fusion_manu.md](fusion_manu.md)에 있습니다.

| 목적 | 사용할 파일 |
| --- | --- |
| 공통 데이터 경로 | [basic.yaml](basic.yaml) |
| Clinical ML 학습 | [ML/model_experiment_only_ML.ipynb](ML/model_experiment_only_ML.ipynb) |
| ECG DL 학습 | [DL/model_experiment_only_DL.ipynb](DL/model_experiment_only_DL.ipynb) |
| 저장 모델 평가 | [evaluation/eval.yaml](evaluation/eval.yaml) |
| ML + ECG 결합 | [fusion.yaml](fusion.yaml), [ECG+clinical_experiment.ipynb](ECG+clinical_experiment.ipynb) |

## 1. 공통 준비

### 환경

```bash
cd ~/sixletters-poc
uvx --from uv==0.9.26 uv sync --project Global --locked --extra ml --extra gpu
Global/.venv/bin/python -c "import torch, pandas, sklearn, xgboost, catboost; print('CUDA:', torch.cuda.is_available())"
```

노트북을 쓸 경우 처음 한 번 커널을 등록합니다.

```bash
Global/.venv/bin/python -m ipykernel install --user \
  --name sixletters-poc-global \
  --display-name "Python (sixletters-poc Global)"
Global/.venv/bin/python -m jupyterlab
```

모든 노트북에서 **Python (sixletters-poc Global)** 커널을 선택합니다.

### MLflow 계정 연결

`username`은 서버 인증 계정, `owner`는 결과 작성자 태그와 개인 결과 폴더, `experiment`는 MLflow에서 run을 모을 공간입니다. 계정과 experiment 이름은 별개입니다. 팀원이 각자 experiment를 쓰려면 **각자 자기 MLflow 계정으로 로그인해 experiment를 한 번 생성**한 다음, 로컬 설정에서 그 이름을 선택합니다. 예를 들어 `euna`는 `euna-scd-ml`, `minjeong`은 `minjeong-scd-ml`을 사용합니다.

1. MLflow 웹 화면에 본인 계정으로 로그인해 **Experiments → Create Experiment**에서 고유한 experiment를 만듭니다. 이름은 `<owner>-scd-ml`처럼 팀에서 겹치지 않게 정합니다.
2. 저장소 루트에서 자기 `owner`, 인증 `username`, 서버 주소, 방금 만든 experiment 이름으로 설정합니다.
3. 연결 확인 출력에 본인 `username`·`owner`·`experiment`가 표시되는지 확인한 뒤 학습 노트북을 실행합니다.

예를 들어 은아는 다음처럼 설정합니다. 다른 팀원은 세 값을 모두 본인 것으로 바꾸고, `--experiment`에는 본인이 만든 이름을 넣습니다.

```bash
Global/.venv/bin/python -m sixletters setup \
  --owner euna \
  --username euna \
  --uri http://192.168.10.88:5000 \
  --experiment euna-scd-ml \
  --password
Global/.venv/bin/python -m Members.sang.model.tracking.check_connection
```

현재 연결 검사는 선택한 experiment가 이미 있고 본인 계정으로 조회 가능한지 확인합니다. 따라서 처음에는 MLflow UI에서 직접 생성해야 합니다. **Create Experiment 메뉴가 없거나 생성이 거부되면 서버 관리자에게 experiment 생성 권한을 요청하거나 이름을 전달해 사전 생성을 부탁하세요.** 같은 PC를 다른 사람이 이어서 쓸 때는 setup을 다시 실행합니다. 설정 우선순위와 네트워크·계정 관리는 [TEAM_MLFLOW.md](tracking/TEAM_MLFLOW.md)를 참고합니다.

`MLFLOW_TRACKING_*` 환경 변수가 있으면 로컬 `.poc.local.toml`보다 우선합니다. 연결 확인 출력에서 username·owner·experiment가 의도한 값인지 먼저 확인합니다.

### 데이터 확인

데이터는 저장소 안의 `ML/data`나 `DL/data`가 아니라, 연결된 Google Drive의 `DATA` 폴더에서 읽습니다. [basic.yaml](basic.yaml)의 `drive_root`가 이 폴더를 가리키고, `drive://` 경로는 그 아래에서 해석됩니다. 현재 설정된 항목은 다음과 같습니다.

| 용도 | Drive 경로 (`DATA/` 기준) | 설정 키 |
| --- | --- | --- |
| Clinical ML 입력 | `clincal/clinical_clean.csv` | `data.clinical.path` |
| Clinical ML 환자 분할 | `final_split.csv` | `data.split.path` |
| Raw DL 환자 분할·4년 SCD 라벨 | `ECG_tem/final_split_augmented.csv` | `data.raw_ecg.split_path` |
| Raw DL 파형 | `ECG_tem/ECG_tem_clean/clean_signal/{pid}.npy` | `data.raw_ecg.wave_directory` |
| 30초 구간 품질 정보 | `ECG_tem/segment_rpeaks.csv` | `data.raw_ecg.quality_index_path` |
| 실제 측정 유도 정보 | `ECG_tem/ECG_tem_clean.csv` | `data.raw_ecg.lead_metadata_path` |

실행 PC에서 Drive가 다른 위치에 마운트되면 `basic.yaml`의 `drive_root`만 해당 PC의 `DATA` 폴더로 지정합니다. `source_metadata_confirmed: true`, 200 Hz, 30초 구간 및 Drive 품질·유도 metadata 설정은 현재 연결된 Drive의 README와 파일 구조를 기준으로 구성되어 있습니다. raw DL은 품질 인덱스의 제외 구간과 `has_z` 유도 정보를 확인하므로, 예전 `DL/data/...` 경로나 `pid,window_index,usable` 형식의 sidecar를 준비할 필요가 없습니다.

Raw DL의 분할과 파형 파일 존재 여부는 다음 명령으로 먼저 확인할 수 있습니다.

```bash
uvx --from uv==0.9.26 uv run --project Global --locked \
  python -m Members.sang.model.DL.dataloader.segmented_ecg
```

이 명령은 품질 인덱스와 유도 metadata까지 검증하지는 않습니다. 그 검증은 DL 노트북의 cohort 로딩 단계에서 수행됩니다. Clinical ML과 Raw DL은 서로 다른 Drive 분할 파일을 사용하므로 두 결과를 결합할 때는 환자 ID·분할·라벨이 일치하는지 별도로 확인합니다.

## 2. Clinical ML 학습

가장 간단한 방법은 [ML 노트북](ML/model_experiment_only_ML.ipynb)입니다. 후보 비교 중에는 다음처럼 설정하고 test는 끕니다.

```python
model_version = 4
training_mode = "optuna"  # 또는 "yaml"
model_name = "XGBoost"
mlflow_logging = True
run_final_test = False
```

모델·입력 버전·threshold를 validation으로 고른 뒤에만 `run_final_test=True`로 다시 실행합니다.

터미널에서 XGBoost v4를 새로 학습하고 MLflow에 기록하는 예시입니다.

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
    print(run.resolve())
PY
```

`reuse_saved=False`는 현재 YAML 값으로 새 모델을 학습합니다. 결과는 `ML/models/<run_id>/`에 저장됩니다. 이 API는 test를 자동 평가하지 않으므로 4절에서 저장 모델을 평가합니다.

## 3. ECG DL 학습

모델 선택은 [DL/model_choice.yaml](DL/model_choice.yaml)에서 합니다.

```yaml
cnn_model: ResNet
sequence_model: BiLSTM
```

학습 옵션은 [DL/training.yaml](DL/training.yaml), CNN·시간 모델 옵션은 각 하위 YAML에서 바꿉니다. 실제 유도 수와 기록 길이는 파일에서 읽으며 고정 `channels`나 고정 raw 길이를 설정하지 않습니다.

DL 데이터 준비가 끝난 후 학습합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.DL.raw_pipeline \
  --run-id ecg_resnet_bilstm_01 \
  --device auto \
  --mlflow-logging
```

실행 경로만 확인하는 작은 학습은 다음과 같습니다. 전체 성능 실험으로 해석하지 않습니다.

```bash
Global/.venv/bin/python -m Members.sang.model.DL.raw_pipeline \
  --run-id ecg_smoke_01 \
  --max-epochs 1 \
  --limit-train-patients 16 \
  --device cpu \
  --mlflow-logging
```

결과 run 폴더에는 `best_model.pt`, `metadata.json`, `resolved_config.yaml`, `metrics.json`이 저장됩니다. checkpoint 파일 하나만 옮기지 않고 run 폴더 전체를 보관합니다.

## 4. 저장 모델 evaluation

[evaluation/eval.yaml](evaluation/eval.yaml)에서 평가 대상을 설정합니다. 경로는 `evaluation/` 폴더 기준입니다.

| 대상 | `target` | 선택 방법 |
| --- | --- | --- |
| Clinical ML | `ml` | `artifact_dir: ../ML/models/<run_id>` |
| ECG DL | `dl` | `model_path: ../DL/models/<run_id>/best_model.pt` |
| Fusion | `fusion` | `fusion_config: ../fusion.yaml` |

ML 평가 예시:

```yaml
evaluation:
  target: ml
  artifact_dir: ../ML/models/실제_ML_run
  split: validation
  threshold_override: null
  mlflow_logging: true
```

DL 평가 예시:

```yaml
evaluation:
  target: dl
  model_path: ../DL/models/실제_DL_run/best_model.pt
  artifact_dir: null
  split: validation
  threshold_override: null
  mlflow_logging: true
```

Validation 평가:

```bash
Global/.venv/bin/python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml \
  --split validation
```

모델·threshold·fusion 가중치를 확정한 뒤 같은 저장 모델로 test를 한 번 평가합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.evaluation.evaluation \
  --config Members/sang/model/evaluation/eval.yaml \
  --split test \
  --run-id final_test_01
```

`--threshold-override 0.4`는 저장 모델의 확률을 바꾸지 않고 분류 threshold만 평가 시점에 바꿉니다. validation에서 선택한 값은 test에도 동일하게 유지합니다.

결과는 `evaluation/results/<ml|dl|fusion>/<run_id>/`에 저장됩니다. `metrics.json`, 예측 CSV, `resolved_config.yaml`, `data_audit/`를 확인합니다.

## 5. ML + ECG Fusion

Fusion은 저장된 ML과 DL의 환자별 확률을 결합합니다. 기반 모델을 다시 학습하지 않습니다.

```text
p_final = weight_ml × p_ml + weight_dl × p_dl
```

[fusion.yaml](fusion.yaml)에서 실제 run 폴더를 지정합니다.

```yaml
ml:
  artifact_dir: ML/models/실제_ML_run
dl:
  artifact_dir: DL/models/실제_DL_run
fusion:
  mode: learn  # manual, learn, saved
  saved_run_dir: null
classification:
  threshold: 0.5
```

- `manual`: `weights.ml`, `weights.dl`을 직접 지정하며 0 이상·합 1이어야 함
- `learn`: validation 공통 환자에서 결합 가중치만 학습
- `saved`: 이전 fusion validation run의 모델 경로·가중치를 재사용

[ECG+clinical_experiment.ipynb](ECG+clinical_experiment.ipynb)에서는 후보 비교 중 `run_final_test=False`로 validation만 실행합니다. 최종 실행에서 `run_final_test=True`로 바꾸면 validation 결과의 저장 가중치를 test에 재사용합니다.

CLI로 final test를 할 때는 먼저 `fusion.mode: saved`, `fusion.saved_run_dir: evaluation/results/fusion/<validation_run_id>`로 고정한 뒤 4절의 evaluation 명령을 `target: fusion`으로 실행합니다. 상세 절차는 [fusion_manu.md](fusion_manu.md)에 있습니다.

Fusion은 양쪽 예측이 있는 환자만 사용합니다. 동일 환자의 split·SCD 라벨·target semantics가 다르면 중단하며, 한쪽이 없는 환자는 `excluded_patients.csv`로 기록합니다.

## 6. MLflow 기록·오프라인 실행·재전송

연결되면 학습·평가 후 `mlflow_status.json`의 `FINISHED`와 URL을 확인합니다. `PENDING`이면 로컬 결과는 저장됐지만 전송이 완료되지 않은 상태입니다.

서버에 연결할 수 없으면 **ML/DL 노트북에서 `mlflow_logging=False`**로 바꾼 뒤 실행합니다. `True`일 때 학습 전 연결 검사가 실패하면 학습을 시작하지 않습니다. DL CLI는 `--mlflow-logging`을 생략하고, 평가 CLI는 `evaluation/eval.yaml`에서 `evaluation.mlflow_logging: false`로 설정합니다. ML·DL 노트북은 실행 설정의 `mlflow_logging` 값으로 학습과 연결된 평가 기록을 함께 제어합니다.

결과는 ML `ML/models/<run_id>/`, DL `DL/models/<run_id>/`, 평가 `evaluation/results/<target>/<run_id>/`에 남습니다. 학습은 `run_status.json`의 `COMPLETE`, 평가는 `metrics.json`과 예측 CSV가 있는지 확인하고 run 폴더를 보존합니다. `mlflow_logging=False`로 실행한 결과는 `mlflow_status.json`이 없을 수 있습니다.

서버 연결이 복구되면 개인 계정·experiment를 설정하고 연결 검사를 통과한 뒤, 재학습·재평가 없이 해당 run 폴더를 전송합니다.

```bash
Global/.venv/bin/python - <<'PY'
from pathlib import Path
from Members.sang.model.tracking.log_best import log_ml_training_to_mlflow
from Members.sang.model.tracking.log_dl import log_dl_training_to_mlflow
from Members.sang.model.tracking.log_evaluation import log_evaluation_to_mlflow

# 실행할 결과 종류만 남기고 경로를 실제 run 폴더로 바꿉니다.
print(log_ml_training_to_mlflow(Path("Members/sang/model/ML/models/<run_id>")))
# print(log_dl_training_to_mlflow(Path("Members/sang/model/DL/models/<run_id>")))
# print(log_evaluation_to_mlflow(Path("Members/sang/model/evaluation/results/ml/<run_id>")))
PY
```

## 7. 완료 기준과 현재 상태

| 단계 | 확인 파일 |
| --- | --- |
| ML 학습 | `ML/models/<run_id>/run_status.json`, `mlflow_status.json` |
| DL 학습 | `DL/models/<run_id>/best_model.pt`, `run_status.json`, `mlflow_status.json` |
| 평가 | `evaluation/results/<target>/<run_id>/metrics.json`, 예측 CSV |
| Fusion | `fusion_weights.json`, `excluded_patients.csv`, `resolved_config.yaml` |

현재 `basic.yaml`에 설정된 Clinical ML 입력·분할, Raw DL 분할·파형 폴더·품질 인덱스·유도 metadata 경로는 연결된 Drive에서 확인됩니다. Raw DL은 학습 전 1절의 감사 명령과 노트북 cohort 로딩을 통과해야 합니다. 경로가 존재한다는 사실만으로 전체 환자의 파형·품질 정보가 학습 조건을 만족한다고 단정하지 않으며, 감사 결과를 확인하기 전에는 전체 학습이나 fusion 검증 완료로 간주하지 않습니다.
