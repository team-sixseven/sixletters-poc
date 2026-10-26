# CNN 계열

실제 구현은 `../models.py`의 `ResNetECG`, `InceptionTimeECG`, `TCNECG`, `RawECGMIL`입니다. 각 raw-window encoder는 `(batch,3,2000)`을 받고 평가기는 환자별 8개 구간 `(batch,8,3,2000)`을 전달합니다. 앞의 세 모델은 구간 logit 평균, Raw ECG MIL은 구간 embedding attention으로 환자 단위 출력을 만듭니다. Architecture 설정은 [`model_set.yaml`](model_set.yaml)에서 읽습니다.
