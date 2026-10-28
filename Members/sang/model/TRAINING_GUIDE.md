# 모델 학습·평가 가이드

이 문서는 현재 `model/` 코드를 실제로 실행하는 가장 짧은 경로입니다. 모든 명령은 저장소 루트에서 실행합니다.

```text
환경·MLflow 연결 → ML 또는 DL 학습 → validation 평가·선택
→ 설정 고정 → test 평가 → MLflow 확인
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

`username`은 서버 인증 계정, `owner`는 결과 작성자 태그와 개인 결과 폴더, `experiment`는 결과를 모을 공간입니다. 계정과 experiment 이름은 별개입니다.

팀 공용 experiment에 기록하는 예시입니다.

```bash
Global/.venv/bin/python -m sixletters setup \
  --owner euna \
  --username euna \
  --uri http://192.168.10.88:5000 \
  --experiment sang-scd-ml \
  --password
Global/.venv/bin/python -m Members.sang.model.tracking.check_connection
```

개인 공간을 쓰려면 `--experiment euna-scd-ml`처럼 본인 이름으로 바꿉니다. 서버에 그 experiment가 존재하고 본인 계정에 조회·기록 권한이 있어야 합니다. 자세한 네트워크·권한 설명은 [TEAM_MLFLOW.md](tracking/TEAM_MLFLOW.md)를 봅니다.

`MLFLOW_TRACKING_*` 환경 변수가 있으면 로컬 `.poc.local.toml`보다 우선합니다. 연결 확인 출력에서 username·owner·experiment가 의도한 값인지 먼저 확인합니다.

### 데이터 확인

ML은 `ML/data/clinical_clean.csv`, `ML/data/final_split.csv`를 사용합니다. 경로는 [basic.yaml](basic.yaml)에서 변경합니다.

Raw DL은 다음이 모두 있어야 합니다.

- `DL/data/final_split_v2_augmented.csv`
- `DL/data/wave/{pid}.npy`
- 확인된 sample rate·sample/lead 축 근거
- `pid,window_index,usable` 열의 실제 30초 품질 인덱스

현재 기본 설정은 `source_metadata_confirmed: false`, `quality_index_path: null`입니다. 따라서 raw DL 학습은 의도적으로 중단됩니다. 원본 정보와 품질 인덱스를 확인 없이 만들어 넣거나 모든 구간을 usable로 처리하지 않습니다.

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

## 6. MLflow 완료 확인·재전송

로컬 학습 성공은 `run_status.json`의 `COMPLETE`, 원격 업로드 성공은 `mlflow_status.json`의 `FINISHED`와 URL로 확인합니다. `PENDING`이면 모델·평가 결과는 로컬에 남았고 전송만 재시도하면 됩니다.

평가 결과 재전송 예시입니다. ML/DL/Fusion 경로로 바꿔 사용합니다.

```bash
Global/.venv/bin/python - <<'PY'
from pathlib import Path
from Members.sang.model.tracking.log_evaluation import log_evaluation_to_mlflow

run = Path("Members/sang/model/evaluation/results/ml/실제_평가_run")
print(log_evaluation_to_mlflow(run))
PY
```

학습 결과 재전송은 ML에 `log_ml_training_to_mlflow`, DL에 `log_dl_training_to_mlflow`를 사용합니다. 정확한 명령은 각 세부 매뉴얼에 있습니다. 재시도 전에는 항상 연결을 다시 확인합니다.

```bash
Global/.venv/bin/python -m Members.sang.model.tracking.check_connection
```

## 7. 완료 기준과 현재 제한

| 단계 | 확인 파일 |
| --- | --- |
| ML 학습 | `ML/models/<run_id>/run_status.json`, `mlflow_status.json` |
| DL 학습 | `DL/models/<run_id>/best_model.pt`, `run_status.json`, `mlflow_status.json` |
| 평가 | `evaluation/results/<target>/<run_id>/metrics.json`, 예측 CSV |
| Fusion | `fusion_weights.json`, `excluded_patients.csv`, `resolved_config.yaml` |

최근 ML·DL 회귀 검증은 92개 테스트를 통과했습니다. 실제 ML 학습·평가·MLflow 기록도 확인했습니다. 현재 raw DL 전체 실행과 fusion은 waveform 부족, 원본 metadata 미확인, 품질 인덱스 부재 때문에 진행할 수 없습니다. 데이터가 준비되면 3절부터 실행하면 됩니다.
