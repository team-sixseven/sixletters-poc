# 임상 데이터 입력

- Feature 원본: `Members/sang/datas/clincal/clinical_clean.csv`
- Split membership: `Members/sang/datas/clincal/splits/{train,val,test}.csv`
- Loader: [`clinical.py`](clinical.py)

Clean CSV에서 `Patient ID`, `SCD` 이외의 임상 feature를 읽습니다. split CSV는 환자 ID와 라벨 검증 및 train/validation/test membership 지정에만 사용합니다. ID 중복·누락, split 중복, clean CSV와 라벨 불일치, split 누락 환자가 있으면 학습 전에 실패합니다.
