# SCD 모델 실험 구조

이 디렉터리는 실험 진입점과 모델 설정을 데이터 종류별로 정리합니다. 기존 학습·평가 구현과 데이터는 복제하지 않습니다.

```text
model/
├── DL/                                  # ECG 딥러닝 모델과 학습 설정
│   ├── CNN/                             # CNN 계열
│   ├── LSTM/                            # LSTM 계열
│   ├── Segment_MIL/                     # ECG 구간 집계 모델
│   ├── Transformer/                     # Transformer 계열
│   ├── data/                            # 데이터 경로·형식 안내
│   ├── dataloader/                      # ECG 로딩·전처리
│   ├── model_choice.yaml                # 평가 모델과 입력 설정
│   ├── model_experiment_only_DL.ipynb   # DL 실험 노트북
│   ├── models.py                        # DL 네트워크 정의
│   └── training.yaml                    # 공통 학습 설정
├── ECG+clinical_experiment.ipynb        # ECG·임상 통합 실험 노트북
├── ML/                                  # 임상 데이터 머신러닝 모델과 설정
│   ├── __pycache__/                     # Python 임시 캐시
│   ├── catboost/                        # CatBoost 모델
│   ├── data/                            # 데이터 경로·형식 안내
│   ├── dataloader/                      # 임상 CSV·split 로딩
│   ├── model_choice.yaml                # 평가 모델·feature 설정
│   ├── model_experiment_only_ML.ipynb   # ML 실험 노트북
│   ├── models.py                        # 모델 생성·공통 전처리
│   └── xgboost/                         # XGBoost 모델
├── README.md                            # model 디렉터리 안내
├── __pycache__/                         # Python 임시 캐시
├── config.py                            # YAML 설정·경로 처리
├── evaluation/                          # 예측 평가와 보고서 생성
│   ├── DL/                              # ECG 평가 지표
│   ├── ML/                              # 임상 평가 지표
│   ├── README.md                        # 평가 모듈 안내
│   ├── __init__.py                      # Python 패키지 표시
│   ├── __pycache__/                     # Python 임시 캐시
│   ├── core.py                          # 공통 평가 로직
│   ├── evaluate_predictions.py          # 저장된 예측 재평가
│   ├── evaluation_choice.yaml           # 평가 설정
│   ├── evaluation_experiment.ipynb      # 평가 노트북
│   └── reporting.py                     # 결과 보고서 생성
├── outputs/                             # 로컬 실행 결과
│   ├── evaluation/                      # 평가 보고서·결과 JSON
│   └── handoff/                         # MLflow 전달 bundle
└── tracking/                            # 결과 bundle 생성·MLflow 등록
    ├── MLFLOW_HANDOFF.md                # MLflow 담당자 안내
    ├── __init__.py                      # Python 패키지 표시
    ├── __pycache__/                     # Python 임시 캐시
    ├── bundle.py                        # 결과 bundle 생성
    ├── handoff.yaml                     # bundle 계약·파일 정책
    ├── import_bundle_to_mlflow.py       # bundle 검증·MLflow 업로드
    └── model_flavors.py                 # sklearn·PyTorch MLflow Model 저장
```

## 실행 기준

- 임상 모델의 실제 구현은 `../scripts/evaluate_preprocessed_clinical.py`이며 XGBoost와 CatBoost를 같은 환자 분할에서 비교합니다.
- 임상 feature 입력은 `../datas/clincal/clinical_clean.csv`, split membership은 `../datas/clincal/splits/{train,val,test}.csv`입니다. 모델 팩토리는 `ML/models.py`, loader는 `ML/dataloader/clinical.py`, 비교 runner는 `../scripts/evaluate_preprocessed_clinical.py`입니다.
- Holter 모델의 실제 구현은 `../scripts/evaluate_ecg_architectures.py`, 모델 정의는 `DL/models.py`입니다. 파형 입력은 `../datas/Holter/windows/Pxxxx.npy` 환자별 파일입니다.
- 현재 Holter 확인 범위는 `../datas/Holter/windows/P0002.npy` 한 파일의 forward smoke이며, 학습·성능지표·MLflow bundle은 만들지 않습니다. [`smoke_p0002_holter.py`](../scripts/smoke_p0002_holter.py)를 사용합니다.
- NPY windows loader는 `DL/dataloader/holter_windows.py`입니다. `(N_windows,3,2000)`에서 환자별 8개 window를 기록 전체에 균등하게 골라 `(N,8,3,2000)`으로 구성합니다.
- `ML/models.py`와 `DL/models.py`가 공통 전처리와 네트워크 구현을 보유합니다. 평가 스크립트는 `model_choice.yaml`, 모델별 `parameter.yaml`·`model_set.yaml`·`transformer.yaml`을 읽어 후보 목록, 데이터 경로, 학습값과 architecture 설정을 적용합니다. 모델별 하위 Python 파일은 공통 구현을 노출하는 entry point입니다.
- 두 평가 스크립트는 `evaluation/ML/clinical.py`, `evaluation/DL/holter.py`의 metric 함수를 사용합니다. 저장된 prediction만 다시 평가하는 방법은 [evaluation README](evaluation/README.md)에 있습니다.
- `ML/data`와 `DL/data`는 데이터 복사본이 아닙니다. 원본은 각각 `../datas/clincal/clinical_clean.csv`와 `../datas/Holter/windows/`입니다.
- 실험 노트북은 공통 평가 스크립트를 호출하는 실행 진입점입니다. 실제 결과는 스크립트가 `report.md`와 `results.json`으로 저장합니다. 명령은 저장소 루트에서 실행합니다.
- MLflow 담당자는 로컬 handoff bundle을 검사하고 MLflow run으로 등록하는 [importer 사용법](tracking/MLFLOW_HANDOFF.md)을 참고하세요.
- 평가 스크립트의 `--tracking-output`은 실행할 때마다 모델별 번들을 새 폴더에 저장합니다. 로컬 추적 준비물은 `.gitignore`에서 제외되며, 자동 MLflow 업로드는 하지 않습니다.
