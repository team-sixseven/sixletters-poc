# LSTM 계열

구현은 `../models.py`의 `AttentionLSTMFCN`과 `BiLSTMECG`입니다. 각 10초 NPY window를 CNN으로 임베딩한 뒤, 기록 전체에서 균등 추출된 8개 window를 시간 순서대로 LSTM에 입력합니다. 10초는 원본 샘플링 주파수 200 Hz 가정에 따른 값입니다. 설정은 [`model_set.yaml`](model_set.yaml)에서 읽습니다.
