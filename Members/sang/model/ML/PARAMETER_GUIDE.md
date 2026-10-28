# ML 모델 파라미터 가이드

이 문서는 현재 ML trainer가 읽는 모델별 `parameter.yaml` 키를 설명합니다. 값과 적용 방식은 각 YAML과 `ML/training.py`, `ML/models.py` 구현 기준입니다.

## 설정 파일 연결

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

`model_choice.yaml`에는 사용할 모델·각 parameter 파일 경로·결과 경로만 둡니다. 임상 feature는 CSV에서 환자 ID와 정답 라벨을 제외한 모든 열을 CSV 순서대로 사용합니다. 범주형 열 7개는 `config.py`의 기본 목록으로 관리합니다. validation Average Precision 최대화와 전처리 기본값은 코드에서 관리하고 run의 `resolved_config.yaml`에 기록합니다.

## 입력 변수 버전

노트북의 `model_version`은 알고리즘 이름과 별개인 **입력 변수 묶음 번호**입니다. `ML/feature_versions.py`의 제공 목록을 사용하며 모든 ML 알고리즘과 Optuna에 동일하게 적용합니다.

| 버전 | 변수 수 | 구성 |
|---|---:|---|
| 1 | 4 | Gender, Age, LVEF, NYHA |
| 2 | 8 | 버전 1 + eGFR, Ischemic etiology, Albumin, Diabetes |
| 3 | 21 | 버전 2 + 제공된 심장 구조·기능, 병력, 혈압, Pro-BNP, Cardiothoracic ratio |
| 4 | 23 | 버전 3 + cigarettes /year, alcohol consumption |

기본값 4는 전체 23개를 사용합니다. `Mitral valve insufficiency`의 끝 공백은 CSV 로더와 동일하게 정리했습니다. 존재하지 않는 열이나 1~4 이외 값은 오류로 표시합니다. 범주형 처리는 선택한 입력 열에 포함된 범주형 변수에만 적용합니다.

`train_clinical(..., model_version=2)`와 같이 지정할 수 있습니다. 버전을 생략한 기존 Python 호출은 이전처럼 ID·라벨을 제외한 CSV 열을 자동 선택합니다. 각 run 이름에는 `v1`~`v4`가 들어가고 `metadata.json`, `resolved_config.yaml`, validation 예측에 실제 버전과 입력 열을 기록합니다. 선택된 목록의 순서도 유지합니다.

## 저장 모델 경로

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

## XGBoost

현재 값은 [`xgboost/parameter.yaml`](xgboost/parameter.yaml)에 있습니다.

| 키 | 현재 값 | 역할 |
|---|---:|---|
| `model.n_estimators` | `500` | boosting 최대 단계 수입니다. 실제 저장 단계는 validation early stopping이 고른 best iteration일 수 있습니다. |
| `model.max_depth` | `3` | 각 tree의 최대 깊이입니다. 깊이를 키우면 더 복잡한 상호작용을 학습할 수 있습니다. |
| `model.learning_rate` | `0.03` | 각 boosting 단계의 기여 크기를 조절합니다. |
| `training.early_stopping_rounds` | `30` | validation AP가 개선되지 않는 boosting 단계가 이 횟수에 도달하면 멈춥니다. best iteration의 모델을 저장합니다. |
| `classification.threshold` | `0.5` | 양성 확률이 이 값 이상이면 양성으로 분류합니다. 모델 학습이나 AP 기준 best iteration 선택에는 영향을 주지 않습니다. |

`random_state=42`, `n_jobs=1`, train subset seed `42`, `verbosity=0`은 코드 기본값입니다. 임상 feature는 결측 없이 제공되어야 하며, 결측값을 채우는 처리는 하지 않습니다. XGBoost 입력은 train split에서 fit한 scaling·one-hot 전처리를 거치고 validation에는 train에서 배운 전처리만 적용합니다.

## CatBoost

현재 값은 [`catboost/parameter.yaml`](catboost/parameter.yaml)에 있습니다.

| 키 | 현재 값 | 역할 |
|---|---:|---|
| `model.iterations` | `500` | boosting 최대 반복 수입니다. `use_best_model=True`로 학습하고 validation AP로 best iteration을 저장합니다. |
| `model.depth` | `4` | tree의 최대 깊이입니다. |
| `model.learning_rate` | `0.03` | 각 boosting 단계의 기여 크기를 조절합니다. |
| `training.early_stopping_rounds` | `30` | validation AP가 개선되지 않는 반복 횟수 기준입니다. trainer가 validation AP를 평가 지표로 전달합니다. |
| `classification.threshold` | `0.5` | 양성 확률을 0/1 분류로 바꾸는 기본 경계입니다. 학습·best iteration 선택과는 별개입니다. |

`random_seed=42`, `verbose=false`, `thread_count=1`, `allow_writing_files=false`, train subset seed `42`는 코드 기본값입니다. CatBoost에는 categorical feature를 문자열로 변환해 전달합니다. XGBoost/Random Forest처럼 one-hot encoder를 적용하지 않습니다.

## Random Forest

현재 값은 [`random_forest/parameter.yaml`](random_forest/parameter.yaml)에 있습니다.

| 키 | 현재 값 | 역할 |
|---|---:|---|
| `model.n_estimators` | `500` | 학습할 decision tree 수입니다. |
| `model.min_samples_leaf` | `2` | 각 leaf에 필요한 최소 train 환자 수입니다. 너무 작은 leaf가 생기는 것을 제한합니다. |
| `model.class_weight` | `balanced` | train label의 클래스 빈도에 반비례하도록 클래스 가중치를 줍니다. |
| `classification.threshold` | `0.5` | 양성 확률을 분류 결과 0/1로 바꾸는 기본 경계입니다. |

`max_depth=null`, `max_features=sqrt`, `random_state=42`, train subset seed `42`는 코드 기본값입니다. Random Forest는 XGBoost와 같은 train-fitted scaling·one-hot 전처리를 씁니다. 결측 feature를 채우지 않으며, Validation AP는 기록하지만 best tree 수를 고르거나 조기 종료하지 않습니다.

## Threshold와 재평가

`classification.threshold`는 estimator가 확률을 만드는 방식을 바꾸지 않고, 확률을 이진 예측과 threshold 의존 지표로 바꿀 때 사용합니다. 저장 모델 평가에서 threshold만 바꾸려면 `evaluation/eval.yaml`의 `threshold_override` 또는 평가 CLI의 `--threshold-override`를 사용하면 재학습이 필요 없습니다. 이 override는 해당 평가 결과에만 적용됩니다.

학습 run의 validation 기록에 저장되는 threshold 기본값을 바꾸려면 해당 모델의 `parameter.yaml` 값을 조정하고 새 학습 run을 만들어야 합니다. Threshold나 후보 비교를 위해 test 결과를 반복해서 사용하지 마세요.

## Seed 구분

Estimator와 train subset의 seed 기본값은 코드에서 `42`로 고정합니다. 전체 train cohort를 쓰는 일반 학습에서는 subset seed가 샘플을 바꾸지 않습니다. 각 run의 `resolved_config.yaml`에서 YAML 입력값과 실제 적용된 estimator·training 기본값을 확인할 수 있습니다.

## Optuna 실험

[`model_experiment_only_ML.ipynb`](model_experiment_only_ML.ipynb)의 `Optuna 실험 (선택)`에서 모델 하나와 실험 횟수를 정하고 탐색할 수 있습니다. 모델별 탐색 범위는 `objective(trial)` 셀의 `suggest_int`, `suggest_float`, `suggest_categorical` 호출에 있습니다.

현재 YAML은 기본값으로 읽고 `train_clinical(..., model_names=[model_name], model_parameters=params, reuse_saved=False, model_version=model_version)`로 해당 trial의 모델 파라미터만 덮어씁니다. trial 중에는 원본 YAML을 변경하지 않으며 적용값은 각 trial run의 `resolved_config.yaml`에 기록합니다. 탐색 실행 셀이 완료되면 선택된 run의 최신 경로와 해당 버전 경로를 YAML에 기록합니다. 다른 버전의 경로는 유지합니다. 이 인자는 모델 하나를 선택한 경우에만 지원합니다. Threshold·조기 종료 횟수는 YAML 값을 유지하고 validation AP로 최적 trial을 선택합니다.

최적 모델 경로는 노트북의 `best_run_dir`이며 `ML/models/best/<algorithm>_vN/`에 최적 모델 한 개와 `summary.json`, `trials.csv`, `best_parameters.yaml`을 보관합니다. 다른 trial의 모델 파일은 영구 저장하지 않습니다. Test 평가는 최적값 선택 후 별도로 실행합니다.

## Optuna 저장 정책

`ML/optuna_experiment.py`의 `ClinicalOptunaExperiment`가 임시 trial 실행과 best 보관을 관리합니다. 같은 알고리즘·버전의 새 study가 성공하면 기존 관리 폴더를 새 study의 best로 갱신합니다. 단일 모델 파일과 최적 파라미터 외에 저장 모델 재로드에 필요한 전처리·metadata·설정·validation 예측·지표·완료 상태를 함께 유지합니다. 현재 study의 모든 trial은 파라미터·AP·상태만 `trials.csv`에 남깁니다.

탐색 중에는 이전에 저장된 모델을 유지합니다. 새 best를 별도 위치에 준비한 다음 교체하고, YAML 등록이 실패하면 이전 모델을 복구합니다. 서로 다른 알고리즘·버전 폴더는 건드리지 않습니다. 기존의 날짜별 run이나 검증용 폴더는 자동 정리 대상이 아닙니다. 현재 방식은 단일 study의 순차 trial 실행을 기준으로 합니다.

## Optuna 확장 탐색 범위

아래 값은 이 프로젝트에서 정한 초기 탐색 범위이며 라이브러리 기본값이나 최적 범위라는 의미는 아닙니다. 실제 범위는 노트북 `objective(trial)`에서 수정합니다. 현재 기본 `n_trials`는 조합별 100회입니다. TPE의 첫 10회는 무작위 탐색이며, 100회도 최적값 발견을 보장하지 않습니다. [Optuna TPE 공식 문서](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)

### XGBoost

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

### CatBoost

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

### Random Forest

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
