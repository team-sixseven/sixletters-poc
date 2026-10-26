# SCD 모델 실험

이 폴더에는 임상 데이터와 Holter ECG 데이터를 각각 입력으로 사용하는 모델 비교 코드가 있습니다. 입력 데이터 원본은 `datas/`에 두며 평가 코드는 원본 파일을 수정하지 않습니다.

## 입력 데이터와 실행

- 임상: `datas/clincal/clinical_clean.csv` 및 `datas/clincal/splits/{train,val,test}.csv`
- Holter: `datas/Holter/windows/Pxxxx.npy` 및 임상 split 파일의 환자 ID·SCD 라벨
- 임상 비교: 저장소 루트에서 `python Members/sang/scripts/evaluate_preprocessed_clinical.py`
- Holter 비교: 저장소 루트에서 `python Members/sang/scripts/evaluate_ecg_architectures.py`

Global 프로젝트 환경에서는 ML extra에 XGBoost·CatBoost가, GPU extra에 CUDA PyTorch가 포함됩니다. 평가 결과를 MLflow로 넘길 때는 [bundle importer 실행 안내](model/tracking/MLFLOW_HANDOFF.md)를 따릅니다.

```sh
uv run --project Global --extra ml python Members/sang/scripts/evaluate_preprocessed_clinical.py
uv run --project Global --extra gpu python Members/sang/scripts/evaluate_ecg_architectures.py
```

각 실행은 모델을 학습·평가하고 `Members/sang/model/outputs/evaluation/` 아래에 `report.md`와 `results.json`을 생성합니다. 학습 설정은 각 모델 디렉터리의 YAML에서 읽습니다. 실행 명령에서 옵션을 지정하면 YAML 기본값을 덮어쓸 수 있습니다.

## 입출력 개요

- 임상 CSV: `Patient ID`, `SCD`를 제외한 23개 feature를 읽어 모델에 입력합니다. 모델 입력은 `(N, 23)`이고 예측 결과는 환자별 확률 `(N,)`입니다.
- Holter NPY: 환자별 `(N_windows, 3, 2000)` 배열에서 8개 구간을 균등하게 선택합니다. 원시 파형 모델은 `(batch, 8, 3, 2000)`, 구간 통계 모델은 `(batch, 8, 16)`을 입력받고 `(batch, 1)` logit을 출력합니다.
- NPY에 sampling-rate 정보가 없어 200 Hz는 원본 데이터 사양에 따른 가정입니다. Holter 결과는 현재 파일이 있는 환자 subset의 탐색 결과로 해석합니다.

상세 구조와 설정은 [모델 디렉터리 안내](model/README.md), 데이터 입력 규칙은 각 `model/ML/dataloader/README.md`와 `model/DL/dataloader/README.md`를 참고하세요. MLflow 담당자용 로컬 번들 계약은 [인수인계 문서](model/tracking/MLFLOW_HANDOFF.md)에 있습니다.

## 기존 탐색 기록

더미 입력 확인, P0001 파형 확인, 50명·100명 ECG 파일럿 등의 이전 기록은 `docs/` 아래에 보존되어 있습니다. 이 결과는 현재의 정식 임상·Holter 비교 실행과 별도이며, 표본 규모와 split 구성을 확인한 뒤 탐색 결과로 해석해야 합니다.
