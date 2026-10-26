# 임상 데이터 경로

데이터를 복사하지 않습니다. 임상 feature 원본은 `../../datas/clincal/clinical_clean.csv`, patient split은 `../../datas/clincal/splits/{train,val,test}.csv`입니다.

Loader는 clean CSV에서 feature를 읽고, split CSV에서 ID membership과 label 일치 여부만 확인합니다. 공통 분할과 실험 제한은 `../../docs/preprocessed_data_model_comparison.md`를 참고하세요.
