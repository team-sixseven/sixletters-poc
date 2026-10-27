# 전처리 임상 데이터 모델 비교 및 추천

실행일: 2026-10-23
데이터 경로: `Members/sang/datas/clincal/`
실행 코드: [`evaluate_preprocessed_clinical.py`](../scripts/evaluate_preprocessed_clinical.py)

이 임상 모델 평가는 2026-10-23 시점의 실행이다. 이후 환자 ID raw ECG 한 쌍을 확보해 단일 환자 smoke test를 수행했으나, 아래 임상 성능 결과나 ECG 코호트 성능으로 합치지 않았다. [Transformer 후보 실제 데이터 비교](transformer_ecg_model_comparison.md)를 참고한다.

## 요약

전처리된 환자 단위 train/validation/test split으로 실제 비교가 가능한 정형 모델 4개(Random Forest, XGBoost, CatBoost, MLP)를 학습했다. Validation AUPRC(Average Precision)를 우선 기준으로 보면 CatBoost가 가장 높았고, 선택된 CatBoost를 train+validation으로 다시 학습해 held-out test에서 한 번 평가했다.

**현재 추천:** 다음 반복 실험의 우선 후보는 **CatBoost**, 비교 challenger는 **Random Forest**다. CatBoost가 validation AP와 선택 후 test AP에서 앞섰다. Random Forest는 validation AUROC와 Brier가 근소하게 더 좋았다. validation/test 각각 SCD 양성이 10명뿐인 단일 분할이라 확정적인 모델 우열 또는 임상 효용으로 해석하면 안 된다.

## 1. 데이터 검증

| 확인 항목 | 결과 |
|---|---|
| Clean 파일 | `clinical_clean.csv`: 552행 |
| split 파일 합계 | train 386 + val 83 + test 83 = 552행 |
| split 합집합 | clean 파일 환자 ID 집합과 일치 |
| 중복 / split 간 겹침 | 중복 ID 없음; train/val/test 간 ID 겹침 없음 |
| SCD 양성 | 전체 66명; train 46, val 10, test 10 |
| 입력 열 수 | 총 25열 중 `Patient ID`, `SCD` 제외 23개 feature |
| 결측 | 세 split의 feature/target에서 결측 0개 |
| ECG 자료 | `ECG_num/` 및 `ECG_tem/` 현재 파일 0개 |

### 열 수 불일치

요청 문맥에는 임상 feature 24개라고 적혀 있지만 실제 split CSV는 25열이며 이 중 2열이 patient ID와 target이므로 **예측 입력은 23개**다. 분석 코드에서는 CSV에 실제 들어 있는 23개를 사용했고, 임의로 열을 추가하지 않았다.

`SCD` 열의 0/1 라벨을 target으로 사용했다. 이 폴더에는 event horizon, censoring, non-SCD death 처리 기준을 설명하는 protocol/codebook이 없어, 결과는 제공받은 이진 라벨을 그대로 학습한 점수다. 라벨의 임상적 정의와 팀이 계획한 789명 코호트가 동일한지는 별도 확인이 필요하다.

## 2. 실행 방법 및 공정 비교 조건

- 모든 모델은 같은 train 386명으로 학습하고 validation 83명에서 비교했다. test는 validation AP가 가장 높은 모델 선택 후 그 모델만 한 번 평가했다.
- `Patient ID`는 입력에서 제외했다. feature 전처리기는 train에만 fit했다.
- 범주형으로 처리한 7개 변수: `NYHA class`, `Right ventricle contractility (altered=1)`, `Mitral valve insufficiency `, `Diabetes (yes=1)`, `Prior Myocardial Infarction (yes=1)`, `Gender (male=1)`, `Ischemic etiology`. 데이터 dictionary가 없어 열 이름과 관측 코드에 기반해 분류한 실험 가정이다.
- Random Forest, XGBoost, MLP는 범주형 one-hot, 연속형 median imputation 및 scaling을 사용했다. 이번 파일에는 결측이 없어 대치 동작은 데이터상 필요하지 않았지만, 전처리 pipeline에는 포함했다.
- CatBoost는 위 범주 열을 native categorical로 지정했다.
- 모델별 설정은 고정 seed 42의 작은 baseline 설정이며 하이퍼파라미터 탐색·교차검증은 하지 않았다.
- AUPRC는 Average Precision으로 계산했다. 학습 시간은 CPU 단회 `fit` 시간이다.

## 3. Validation 비교 결과

Validation: N=83, SCD=10. 정렬은 AP 내림차순이다.

| 모델 | AUROC ↑ | AUPRC / AP ↑ | Brier ↓ | Train 시간 | 비고 |
|---|---:|---:|---:|---:|---|
| **CatBoost** | 0.733 | **0.390** | 0.095 | 0.177초 | AP 1위; 범주형 native 처리 |
| Random Forest | **0.782** | 0.369 | **0.094** | 0.431초 | AUROC 1위, Brier 최저 |
| MLP | 0.708 | 0.269 | 0.127 | 0.545초 | 현재 고정 구조에서 비교 |
| XGBoost | 0.695 | 0.240 | 0.107 | 0.046초 | 고정 기본 설정 |

## 4. 단일 held-out test 결과

Validation에서 AP가 가장 높은 CatBoost를 선택했다. CatBoost를 train+validation 469명으로 다시 학습한 뒤 test split을 한 번 평가했다.

| 선택 모델 | Test N / SCD | AUROC ↑ | AUPRC / AP ↑ | Brier ↓ | Test 사용 방식 |
|---|---:|---:|---:|---:|---|
| CatBoost | 83 / 10 | 0.781 | 0.335 | 0.094 | validation에서 선택 후 최종 점수 1회 산출 |

test 지표는 모델 선택이나 파라미터 조정에 사용하지 않았다. 양성 10건만으로 계산한 점수이므로 불확실성이 크며, 더 넓은 성능 주장을 뒷받침하지 않는다.

## 5. 판단 및 다음 단계

1. **CatBoost를 다음 개발 반복의 주 후보로 둔다.** Validation AP가 0.390으로 가장 높았고, 선택 후 test AP는 0.335였다.
2. **Random Forest를 병렬 challenger로 유지한다.** Validation AUROC 0.782로 가장 높고 Brier 0.094로 CatBoost와 비슷하거나 약간 낮았다.
3. 아직 최종 배포 모델을 정하지 않는다. 각 holdout의 양성 사례가 10명이고 신뢰구간이나 반복 분할 결과가 없으며, 임상 endpoint 및 코호트 789명 기준도 확인되지 않았다.
4. 우선 데이터 담당자에게 실제 feature 수(23 vs 계획 24), SCD 예측 기간, censoring/타 사망 처리, 789명 선정 기준, 범주형 코드북을 확인한다.
5. 입력 정의가 확정된 뒤 train 안에서 stratified cross-validation/반복 seed로 CatBoost와 Random Forest를 비교한다. 합의 후에만 test를 최종 재평가한다.
6. 임상 비교 당시에는 환자 ID 단일 smoke만 수행했으나, 이후 ECG 50명 파일럿에서 7개 ECG 모델을 추가 비교했다. 표본이 작고 test SCD 양성이 1명이라 성능 결론은 보류한다. [실험 케이스와 ECG 파일럿 결과](experiment_cases_50.md)를 참고한다.

## 6. Notion 「실험 케이스」 입력용 기록

아래 행은 스크린샷의 database 속성 순서에 맞췄다. 각 행을 C1 임상 케이스의 모델별 복사본으로 입력하면 된다.

| 케이스 | 입력 데이터 | 중요도 | 모델 구분 | 모델 | 담당 | 상태 | ROC-AUC | PR-AUC (AP) | Sensitivity* | Specificity* | Brier | MLflow 링크 | 메모 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| C1 임상 – Random Forest | 임상 | 해당 없음 | ML | Random Forest | 남상기 | 완료 | 0.782 | 0.369 | 0.000 | 1.000 | 0.094 | — | 23 features; validation N=83/SCD=10; train 386/SCD=46; fit 0.421초 |
| C1 임상 – XGBoost | 임상 | 해당 없음 | ML | XGBoost | 남상기 | 완료 | 0.695 | 0.240 | 0.100 | 0.973 | 0.107 | — | 23 features; validation N=83/SCD=10; train 386/SCD=46; fit 0.080초 |
| C1 임상 – CatBoost | 임상 | 해당 없음 | ML | CatBoost | 남상기 | 완료 | 0.733 | 0.390 | 0.100 | 1.000 | 0.095 | — | 23 features; validation N=83/SCD=10; train 386/SCD=46; validation AP 기준 선택. Test N=83/SCD=10: AUC 0.781, AP 0.335, sensitivity 0.000, specificity 0.986, Brier 0.094 |
| C1 임상 – MLP | 임상 | 해당 없음 | DL | MLP | 남상기 | 완료 | 0.708 | 0.269 | 0.300 | 0.918 | 0.127 | — | 23 features; validation N=83/SCD=10; train 386/SCD=46; fit 0.519초 |

\* Sensitivity/Specificity는 validation 확률 임계값 0.5 기준이다. PR-AUC 칸에는 Average Precision을 기록했다. 일부 모델의 threshold 0.5 sensitivity가 매우 낮으므로, 이는 threshold 조정이나 calibration을 반영한 운영 성능이 아니다.

ECG 시계열/파형의 성능 실험은 다수 환자 입력이 없어 미실행이다. 이후 확보한 환자 ID 실파일은 실제 파형 입력 smoke에만 사용했으며 성능 결과와 분리해 [별도 기록](transformer_ecg_model_comparison.md)했다. 앞선 합성 더미 결과도 성능 결과와 분리해 관리한다.

## 7. 실제 성능 비교 범위

| 모델 | 실제 임상 split으로 평가 | 상태 |
|---|---|---|
| Random Forest | 가능 | Validation 결과 기록 |
| XGBoost | 가능 | Validation 결과 기록 |
| CatBoost | 가능 | Validation + 선택 후 test 결과 기록 |
| MLP | 가능 | Validation 결과 기록 |
| 1D-CNN / ECG Transformer | 성능평가 불가 | 환자 ID 단일 train 환자로 실제 파형 smoke PASS; cohort metrics 미산출 |
| Segment Attention MIL / MIL–TCN / Raw ECG MIL | 성능평가 불가 | 환자 ID 단일 실제 파형 smoke PASS; 임시 구간 특징 및 cohort metrics 미산출 |

## 8. 결정 기록

| 날짜 | 결정 | 이유 |
|---|---|---|
| 2026-10-23 | 임상 데이터 모델의 우선 후보로 CatBoost를 선정(잠정). Random Forest는 비교 후보로 유지. | CatBoost가 validation AP 0.390으로 최고였고, validation 기준으로 선택한 뒤 test에서도 AUROC 0.781 / AP 0.335 / Brier 0.094를 기록했다. Random Forest는 validation AUROC 0.782 및 Brier 0.094로 근접하거나 일부 지표에서 앞섰다. 각 split SCD 양성이 10명뿐이고 라벨 horizon·789명 기준이 미확정이라 최종 확정은 보류한다. |

## 재현

저장소 루트에서:

```sh
PYTHONPATH=/tmp/scd-smoke-extra:/home/sang/.cache/uv/archive-v0/TIDVvkcYCJEzCu0b \
  /home/sang/dnn_tf_ex/.venv/bin/python Members/sang/scripts/evaluate_preprocessed_clinical.py
```

환경은 Python 3.12.3, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.0, XGBoost 3.4.1, CatBoost 1.2.10이었다. Docker 공통환경이 아닌 기존 호스트 venv와 격리 패키지 경로 조합이다.
