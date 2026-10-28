# Random Forest

Estimator 설정과 classification threshold는 [`parameter.yaml`](parameter.yaml)에 있습니다. 임상 feature에 결측값이 있으면 오류로 중단하며 대체하지 않습니다. 표준화와 범주형 one-hot encoding은 train split에만 fit합니다. Random Forest는 boosting 방식이 아니므로 early stopping 없이 설정한 tree 수로 학습하고 validation Average Precision을 기록합니다. 모델 선택과 실행은 공통 ML 설정 및 trainer를 사용합니다.
