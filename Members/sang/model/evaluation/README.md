# Evaluation 코드

이 디렉터리에는 임상·Holter 모델별 학습 및 평가 결과를 검토하는 공통 지표 코드와, 저장된 prediction CSV를 다시 계산하는 CLI가 있습니다. 실제 모델별 학습은 `Members/sang/scripts/`의 두 비교 실행기가 담당합니다. 각 비교 실행기는 후보 모델을 각각 학습하고, 실행 때마다 `report.md`와 `results.json`을 반드시 남깁니다.

```text
evaluation/
├── evaluation_experiment.ipynb
├── evaluation_choice.yaml
├── evaluate_predictions.py
├── reporting.py
├── core.py
├── ML/
│   ├── clinical.py
│   └── evaluation_only_ML.ipynb
└── DL/
    ├── holter.py
    └── evaluation_only_DL.ipynb
```

## 공통 입력 형식

CSV에는 환자당 한 행, 다음 세 열이 있어야 합니다.

```csv
patient_id,target,score
P0001,0,0.13
```

`score`는 양성 클래스 확률이며 `[0,1]` 범위여야 합니다. Validation/Test 환자는 서로 겹치면 안 됩니다. ROC-AUC 계산을 위해 각 평가 split에는 양성과 음성이 모두 있어야 합니다.

## CLI

### 실제 데이터에서 모델별 학습·비교

저장소 루트에서 실행합니다. 기본 저장 위치는 `Members/sang/model/outputs/evaluation/`이며, 매 실행마다 timestamp가 붙은 새 폴더를 만듭니다. 폴더가 이미 있으면 덮어쓰지 않습니다. `report.md`는 실행 시작 시 먼저 생성되고, 성공적으로 끝나면 비교표와 최종 결과로 갱신됩니다. `results.json`에는 전체 지표와 실행 설정이 기록됩니다.

```bash
python Members/sang/scripts/evaluate_preprocessed_clinical.py
uv run --project Global --extra gpu python Members/sang/scripts/smoke_p0002_holter.py
```

출력 위치를 바꾸려면 `--report-root /path/to/evaluation-runs`를 지정합니다. 임상은 각 후보 모델을 train split에서 각각 학습하고 validation Average Precision으로 후보를 선택한 뒤, 선택된 모델만 train+validation으로 다시 학습해 test를 한 번 평가합니다. 현재 Holter는 `P0002.npy` 한 파일의 forward smoke만 수행하며 학습·지표·평가 보고서를 만들지 않습니다. `evaluate_ecg_architectures.py`는 split에 등록된 모든 환자의 NPY를 cohort로 학습·평가하므로 현재 단일 환자 범위에서는 실행하지 않습니다. 임상과 Holter는 환자 집합과 입력 조건이 달라 같은 순위표로 합치지 않습니다.

실행 도중 오류나 중단이 나면 시작 시 작성된 `report.md`가 `RUNNING` 상태로 남아 해당 폴더가 미완료 실행임을 표시합니다. 오류가 발생한 실행은 지표가 완성된 결과로 취급하지 않습니다.

### 저장된 예측 재평가

저장소 루트에서 handoff bundle 안의 모델별 예측 파일을 지정합니다.

```bash
python Members/sang/model/evaluation/evaluate_predictions.py \
  --validation Members/sang/model/outputs/handoff/clinical-552-RUN/runs/CatBoost/validation_predictions.csv \
  --test Members/sang/model/outputs/handoff/clinical-552-RUN/runs/CatBoost/test_predictions.csv \
  --input-group clinical
```

모델 이름과 bundle 실행 label은 실제 manifest 및 폴더명에 맞춰 바꿉니다. 기본 `--threshold-mode auto`는 validation Youden J threshold를 선택합니다. 별도 운영 threshold가 정해졌다면 `--threshold-mode fixed --threshold 0.35`처럼 validation/test 평가 전에 명시합니다. Test 점수로 threshold를 조정하지 않습니다.

이 CLI도 기본으로 `outputs/evaluation/prediction-review-<timestamp>/report.md`와 `results.json`을 생성합니다. `--report-root`로 상위 경로를 바꿀 수 있습니다. `--output result.json`을 추가하면 별도 경로에도 집계 지표를 저장합니다. 기존 파일은 덮어쓰지 않습니다. 환자 ID는 출력 보고서나 JSON에 넣지 않습니다.

Holter의 모델 폴더명과 입력 계열은 실제 `run.json`을 보고 선택합니다. 예를 들어 bundle의 선택 모델이 `MIL-TCN`이면 `runs/MIL-TCN/validation_predictions.csv`와 `test_predictions.csv`를 넘기고 `--input-group holter_segment_statistics`를 지정합니다. 원시 파형 모델은 `holter_raw_waveform`을 사용합니다.

## 지표와 조건

- AUROC: `auroc`
- AUPRC: Average Precision을 `auprc_average_precision`으로 기록
- Calibration 요약: Brier score (`brier`); calibration curve/slope/intercept는 계산하지 않음
- Sensitivity/Specificity: 반드시 사용한 threshold와 함께 기록
- 임상·Holter: 각 모델의 validation에서 Youden J 최대화; 동률이면 sensitivity가 높은 threshold, 이어 낮은 threshold를 선택

이 코드는 분류 지표 집계기입니다. 생존분석, censoring, 위험 horizon, 확률 calibration 보정은 수행하지 않습니다.
