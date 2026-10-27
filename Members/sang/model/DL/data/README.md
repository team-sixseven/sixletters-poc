# ECG 파형 데이터 경로

파형 파일을 복사하지 않습니다. 현재 입력 smoke는 `../../datas/Holter/windows/P0002.npy` 한 파일만 읽습니다. Cohort 평가 runner는 `../../datas/Holter/windows/Pxxxx.npy`와 `../../datas/clincal/splits/{train,val,test}.csv`를 사용하므로, 단일 파일 smoke 용도로 실행하지 않습니다.

환자별 NPY는 `(N_windows,3,2000)` 형식입니다. loader는 기록 전체에 걸쳐 균등하게 8개 window를 선택합니다. 선택된 각 window는 10초(200 Hz 기준)이며, 모델 입력은 `(N,8,3,2000)`입니다. 전처리 가정은 `../dataloader/README.md`를 참고하세요.
