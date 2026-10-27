# Holter ECG window 입력

- 현재 smoke 입력: `Members/sang/datas/Holter/windows/P0002.npy` 한 파일만 사용
- cohort 학습 입력: 사용이 합의된 경우 `Members/sang/datas/Holter/windows/Pxxxx.npy`와 임상 split을 함께 사용
- 배열 형식: 환자당 `(N_windows, 3 leads, 2000 samples)`, 현재 확인된 dtype은 `float16`
- Loader: [`holter_windows.py`](holter_windows.py)
- 라벨과 split membership: `Members/sang/datas/clincal/splits/{train,val,test}.csv`

[`smoke_p0002_holter.py`](../../../scripts/smoke_p0002_holter.py)는 지정된 `P0002.npy` 하나만 읽어 8개 window를 고르고 입력 전처리와 후보 모델의 forward shape만 확인합니다. 이 smoke는 모델을 학습하지 않고 성능 지표나 MLflow 평가 bundle을 만들지 않습니다.

cohort runner는 split에 등록된 환자 중 대응하는 모든 NPY를 읽습니다. 원시 모델 입력 shape은 `(patient, 8, 3, 2000)`이며, 각 선택 window·lead별 z-score를 적용합니다. 상수 lead는 해당 window에서 0으로 만들고, 세 lead가 모두 상수인 window는 오류로 처리합니다. 200 Hz 기준으로 window 하나는 10초이며, NPY 자체에는 sampling-rate metadata가 없어 이 값은 원본 데이터 사양을 따른 가정입니다. Segment Attention MIL과 MIL-TCN에는 동일 window에서 계산한 16개 통계량을 사용합니다. 전체 기록 중 80초만 모델 입력에 쓰는 탐색용 표본이므로 긴 시간의 리듬 변화를 충분히 학습하지 못할 수 있습니다.
