# Random Forest

Estimator 설정과 classification threshold는 [`parameter.yaml`](parameter.yaml)에 있습니다. 수치형 결측치 대체, 범주형 결측치 대체 및 one-hot encoding은 train split에만 fit합니다. Random Forest는 boosting 방식이 아니므로 early stopping 없이 설정한 tree 수로 학습하고 validation Average Precision을 기록합니다. 모델 선택과 실행은 공통 ML 설정 및 trainer를 사용합니다.
