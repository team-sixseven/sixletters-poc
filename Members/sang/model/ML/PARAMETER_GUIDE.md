# ML 모델 파라미터 가이드

이 문서는 현재 ML trainer가 읽는 모델별 `parameter.yaml` 키를 설명합니다. 값과 적용 방식은 각 YAML과 `ML/training.py`, `ML/models.py` 구현 기준입니다.

## 설정 파일 연결

실험에서 사용할 후보는 [`model_choice.yaml`](model_choice.yaml)의 `models`에 적고, `model_configs`가 각 후보의 parameter 파일을 지정합니다.

| 후보 | 설정 파일 |
|---|---|
| XGBoost | [`xgboost/parameter.yaml`](xgboost/parameter.yaml) |
| CatBoost | [`catboost/parameter.yaml`](catboost/parameter.yaml) |
| Random Forest | [`random_forest/parameter.yaml`](random_forest/parameter.yaml) |

각 파일은 세 구역으로 나뉩니다.

- `model`: estimator에 전달되는 모델 자체의 파라미터
- `training`: 학습 중단·subset sampling 등 trainer 동작 설정
- `classification`: 확률을 0/1로 나눌 때 사용할 기본 threshold

현재 `model_choice.yaml`은 validation Average Precision을 높이는 방향으로 학습 후보를 평가합니다. XGBoost와 CatBoost는 validation AP로 best iteration을 저장하고, Random Forest는 설정된 tree 수를 사용해 validation AP를 기록합니다. 후보 run 중 어떤 모델을 최종 평가할지는 자동 결정되지 않습니다.

## XGBoost

현재 값은 [`xgboost/parameter.yaml`](xgboost/parameter.yaml)에 있습니다.

| 키 | 현재 값 | 역할 |
|---|---:|---|
| `model.n_estimators` | `500` | boosting 최대 단계 수입니다. 실제 저장 단계는 validation early stopping이 고른 best iteration일 수 있습니다. |
| `model.max_depth` | `3` | 각 tree의 최대 깊이입니다. 깊이를 키우면 더 복잡한 상호작용을 학습할 수 있습니다. |
| `model.learning_rate` | `0.03` | 각 boosting 단계의 기여 크기를 조절합니다. |
| `model.tree_method` | `hist` | histogram 기반 tree split 방법을 사용합니다. |
| `model.random_state` | `42` | XGBoost estimator의 난수 seed입니다. |
| `model.n_jobs` | `1` | estimator가 사용할 CPU 병렬 작업 수입니다. |
| `model.verbosity` | `0` | XGBoost 로그 출력 수준입니다. `0`은 조용한 실행입니다. |
| `training.early_stopping_rounds` | `30` | validation AP가 개선되지 않는 boosting 단계가 이 횟수에 도달하면 멈춥니다. best iteration의 모델을 저장합니다. |
| `training.seed` | `42` | 기능 확인용 train 환자 subset을 뽑을 때 쓰는 seed입니다. 전체 train cohort에서는 estimator seed인 `model.random_state`가 난수 설정을 담당합니다. |
| `classification.threshold` | `0.5` | 양성 확률이 이 값 이상이면 양성으로 분류합니다. 모델 학습이나 AP 기준 best iteration 선택에는 영향을 주지 않습니다. |

XGBoost 입력은 train split에서 fit한 imputation·scaling·one-hot 전처리를 거칩니다. validation 데이터에는 train에서 배운 전처리만 적용합니다.

## CatBoost

현재 값은 [`catboost/parameter.yaml`](catboost/parameter.yaml)에 있습니다.

| 키 | 현재 값 | 역할 |
|---|---:|---|
| `model.iterations` | `500` | boosting 최대 반복 수입니다. `use_best_model=True`로 학습하고 validation AP로 best iteration을 저장합니다. |
| `model.depth` | `4` | tree의 최대 깊이입니다. |
| `model.learning_rate` | `0.03` | 각 boosting 단계의 기여 크기를 조절합니다. |
| `model.random_seed` | `42` | CatBoost estimator의 난수 seed입니다. |
| `model.verbose` | `false` | CatBoost 학습 로그를 출력할지 정합니다. |
| `model.thread_count` | `1` | CatBoost가 사용할 CPU thread 수입니다. |
| `model.allow_writing_files` | `false` | CatBoost가 학습 중 별도 정보 파일을 기록하지 않도록 합니다. |
| `training.early_stopping_rounds` | `30` | validation AP가 개선되지 않는 반복 횟수 기준입니다. trainer가 validation AP를 평가 지표로 전달합니다. |
| `training.seed` | `42` | 기능 확인용 train 환자 subset을 뽑을 때 쓰는 seed입니다. 전체 cohort 학습에서 estimator 난수 seed는 `model.random_seed`가 담당합니다. |
| `classification.threshold` | `0.5` | 양성 확률을 0/1 분류로 바꾸는 기본 경계입니다. 학습·best iteration 선택과는 별개입니다. |

CatBoost에는 categorical feature를 문자열로 변환해 전달합니다. XGBoost/Random Forest처럼 one-hot encoder를 적용하지 않습니다.

## Random Forest

현재 값은 [`random_forest/parameter.yaml`](random_forest/parameter.yaml)에 있습니다.

| 키 | 현재 값 | 역할 |
|---|---:|---|
| `model.n_estimators` | `500` | 학습할 decision tree 수입니다. |
| `model.max_depth` | `null` | 최대 tree 깊이 제한이 없습니다. 다른 분할 조건과 `min_samples_leaf`에 따라 tree가 자랍니다. |
| `model.min_samples_leaf` | `2` | 각 leaf에 필요한 최소 train 환자 수입니다. 너무 작은 leaf가 생기는 것을 제한합니다. |
| `model.max_features` | `sqrt` | 각 split에서 후보로 살펴볼 feature 수를 전체 feature 수의 제곱근으로 정합니다. 적용 대상은 전처리 후 feature입니다. |
| `model.class_weight` | `balanced` | train label의 클래스 빈도에 반비례하도록 클래스 가중치를 줍니다. |
| `model.random_state` | `42` | estimator의 난수 seed입니다. |
| `model.n_jobs` | `1` | estimator가 사용할 CPU 병렬 작업 수입니다. |
| `training.early_stopping` | `false` | Random Forest는 validation 기반 early stopping을 사용하지 않습니다. 설정한 tree 수로 학습합니다. |
| `training.seed` | `42` | 기능 확인용 train 환자 subset을 뽑을 때 쓰는 seed입니다. 전체 train cohort에서는 estimator seed인 `model.random_state`가 난수 설정을 담당합니다. |
| `classification.threshold` | `0.5` | 양성 확률을 분류 결과 0/1로 바꾸는 기본 경계입니다. |

Random Forest는 XGBoost와 같은 train-fitted imputation·scaling·one-hot 전처리를 씁니다. Validation AP는 기록하지만 best tree 수를 고르거나 조기 종료하지 않습니다.

## Threshold와 재평가

`classification.threshold`는 estimator가 확률을 만드는 방식을 바꾸지 않고, 확률을 이진 예측과 threshold 의존 지표로 바꿀 때 사용합니다. 저장 모델 평가에서 threshold만 바꾸려면 `evaluation/eval.yaml`의 `threshold_override` 또는 평가 CLI의 `--threshold-override`를 사용하면 재학습이 필요 없습니다. 이 override는 해당 평가 결과에만 적용됩니다.

학습 run의 validation 기록에 저장되는 threshold 기본값을 바꾸려면 해당 모델의 `parameter.yaml` 값을 조정하고 새 학습 run을 만들어야 합니다. Threshold나 후보 비교를 위해 test 결과를 반복해서 사용하지 마세요.

## Seed 구분

각 모델의 `model.random_state` 또는 `model.random_seed`가 estimator 난수를 설정합니다. `training.seed`는 코드상 `limit_train_patients`로 train subset을 선택할 때 쓰입니다. 따라서 전체 train cohort로 실행할 때 `training.seed`만 바꿔도 estimator의 난수 동작은 바뀌지 않습니다.
