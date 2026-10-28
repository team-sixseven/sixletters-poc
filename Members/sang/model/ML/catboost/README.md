# CatBoost

CatBoost estimator 설정은 [`parameter.yaml`](parameter.yaml)에 있고, 학습 시 범주형 변수 목록은 공통 [`config.py`](../../config.py)의 기본 `categorical_features` 목록을 사용합니다. 후보 선택과 실행은 [`model_choice.yaml`](../model_choice.yaml) 및 공통 trainer가 담당합니다.
