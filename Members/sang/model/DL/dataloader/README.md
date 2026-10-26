# Holter ECG window 입력

- 입력 파일: `Members/sang/datas/Holter/windows/Pxxxx.npy`
- 배열 형식: 환자당 `(N_windows, 3 leads, 2000 samples)`, 현재 확인된 dtype은 `float16`
- Loader: [`holter_windows.py`](holter_windows.py)
- 라벨과 split membership: `Members/sang/datas/clincal/splits/{train,val,test}.csv`

각 환자의 NPY는 메모리 매핑으로 열고, 전체 저장 sequence에서 시간 순서대로 균등한 간격의 8개 window를 선택합니다. 원시 모델 입력 shape은 `(patient, 8, 3, 2000)`이며, 각 선택 window·lead별 z-score를 적용합니다. 상수 lead는 해당 window에서 0으로 만들고, 세 lead가 모두 상수인 window는 오류로 처리합니다. 이를 split별 coverage에 기록합니다. 200 Hz 기준으로 window 하나는 10초이며, NPY 자체에는 sampling-rate metadata가 없어 이 값은 원본 데이터 사양을 따른 가정입니다. Segment Attention MIL과 MIL-TCN에는 동일 window에서 계산한 16개 통계량을 사용합니다. 전체 기록 중 80초만 모델 입력에 쓰는 탐색용 표본이므로 긴 시간의 리듬 변화를 충분히 학습하지 못할 수 있습니다. 파일이 없는 환자는 split별 coverage에 기록되고, ECG가 매칭된 환자만 해당 Holter 실험에 포함됩니다.
