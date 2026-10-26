# 임상 평가

`clinical.py`는 임상 모델 평가에서 공통적으로 쓰는 AUROC, Average Precision, Brier와 호출자가 지정한 threshold의 Sensitivity/Specificity를 제공합니다. 학습 runner와 prediction 재평가 CLI는 validation에서 정한 threshold를 사용합니다.
