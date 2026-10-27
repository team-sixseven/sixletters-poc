# 남상기 모델 실험

임상 ML, Holter DL, 저장 모델 확률 결합과 공통 평가 코드는 [`model/README.md`](model/README.md)에 있습니다. 입력 원본 CSV·NPY는 수정하거나 복사하지 않습니다.

저장소 루트에서 실행:

```bash
uv run --project Global --extra ml python Members/sang/scripts/audit_model_data.py
uv run --project Global --extra ml python Members/sang/scripts/evaluate_preprocessed_clinical.py
uv run --project Global --extra gpu python Members/sang/scripts/evaluate_ecg_architectures.py
uv run --project Global python -m Members.sang.model.evaluation.evaluation
```

공통 설정은 `model/basic.yaml`이 관리합니다. 임상은 `datas/clincal/clinical_clean.csv`와 `split_team_v1.csv`를 환자 ID로 join합니다. Holter는 `datas/Holter/windows/{pid}.npy`와 `datas/Holter/index/windows.csv.gz`의 row/SQI 정보를 사용합니다. 현재 확인된 환자 수, modality별 사용 환자, 라벨 분포, 제외 사유는 audit 결과와 [`model/README.md`](model/README.md)에 기록합니다.

P1030은 NPY 입력·전처리·CNN+LSTM/Transformer forward 확인에 사용합니다. 임상 라벨이 없으므로 supervised 학습·성능 계산에는 사용하지 않습니다. 과거 더미 실험과 이전 split 비교는 `docs/`의 이력이며, 현재 `split_team_v1.csv` 기반 성능 결과로 간주하지 않습니다.
