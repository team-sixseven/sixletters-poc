# 실험 케이스 및 50명 ECG 파일럿 비교

작성일: 2026-10-25
담당: 남상기
실험 상태: ECG 50명 파일럿 실행 완료 (성능 결론은 보류)

## 목적

모델 후보 조사 결과를 바탕으로 데이터 조합별 실험을 구분하고, 실행 결과를 Notion 「실험 케이스」 표와 같은 항목으로 기록한다. 이 문서의 임상 표 결과는 기존 전처리 임상 데이터 실험이고, ECG 표 결과는 추가로 확보한 환자 50명의 ECG 원시 파형 파일럿이다.

## 일정 및 요구사항

- 2026-10-26: 모델 선정 회의 및 담당 확정
- 2026-10-26 ~ 2026-11-01: Sprint 2 모델 학습·평가
- 평가 지표: ROC-AUC, PR-AUC (Average Precision), Sensitivity, Specificity, Brier
- 입력·전처리·환자 분할을 통일한다. 임계값은 validation에서만 정하고 test는 선택 후 1회 평가한다.
- 최종 데모는 환자별 위험 점수, 주요 기여 변수, 모델 성능을 제시한다.

> 이 문서는 50명 파일럿의 역사적 기록이다. 이후 100명 ECG 비교를 수행했으며 최신 ECG 결과와 후보 판단은 [Holter ECG 100명 모델 비교](ecg_100_model_comparison.md)를 기준으로 한다.

## 1. 후보 모델 및 데이터 적용

| 케이스 | 입력 데이터 | 후보 모델 | 목적 | 상태 |
|---|---|---|---|---|
| C1 임상 | 임상 정형 변수 (실제 CSV 입력 23개) | Logistic Regression, Random Forest, XGBoost, CatBoost, MLP | 임상 baseline 및 정형 모델 비교 | 완료; 기존 결과 참조 |
| C2 ECG 통계 특징 | 30초 ECG 구간에서 계산한 검증된 특징 | XGBoost | ECG 정량 특징 단독 예측력 | 특징 정의·코호트 구성 전 |
| C3 임상 + ECG 특징 | 임상 변수 + 검증된 ECG 특징 | XGBoost | ECG 특징의 추가 가치 | 특징 정의·코호트 구성 전 |
| C4 Segment Attention MIL | 환자별 ECG 구간 특징 `(B,S,F)` | Attention MIL | 구간별 중요도 집계 | 현재 ECG 50명 파일럿의 16개 통계 특징으로 탐색 비교 |
| C5 MIL–TCN | 시간순 ECG 구간 특징 `(B,S,F)` | TCN + Attention MIL | 구간의 시간 순서 효과 비교 | 현재 ECG 50명 파일럿의 16개 통계 특징으로 탐색 비교 |
| C6 Raw ECG | ECG 구간 `(B,S,3,2500)` | 1D-CNN, Raw ECG MIL | 원시 파형 학습과 환자 단위 집계 | 현재 ECG 50명 파일럿 |
| C7 CNN + 순서 모델 | ECG 구간 `(B,S,3,2500)` | CNN–LSTM, CNN–TST | 순서 모델 간 탐색 비교 | 현재 ECG 50명 파일럿 |
| C8 Patch Transformer | ECG 구간 `(B,S,3,2500)` | Patch ECG Transformer | 파형 patch attention 비교 | 현재 ECG 50명 파일럿 |

Transformer 문헌 후보는 CNN–TST를 첫 비교 prototype, ConvTran을 후속 독립 baseline으로 두고 Medformer와 FormerTime은 후속 후보로 기록했다. 연구 후보와 로컬 prototype 구현은 구분한다. 조사 근거 및 환자 ID smoke 결과는 [Transformer 후보 조사 문서](transformer_ecg_model_comparison.md)를 참고한다.

| 모델 | 장점 | 제한 및 확인 사항 |
|---|---|---|
| Logistic Regression | 빠른 baseline, 계수 해석 가능 | 비선형 관계를 직접 학습하지 않음 |
| Random Forest | 안정적인 비선형 앙상블, 변수 중요도 비교 가능 | 외삽·확률 보정 확인 필요 |
| XGBoost / CatBoost | 비선형 상호작용 학습; 트리 기반 설명 도구 적용 가능 | 소표본 과적합·튜닝 영향, 임계값 및 calibration 확인 필요 |
| MLP | 정형 변수의 비선형 조합 학습 | 표본이 작아 과적합; 설명성 보완 필요 |
| 1D-CNN | 짧은 ECG 파형의 국소 패턴 학습 | 긴 기록은 구간화·환자 집계 필요 |
| CNN–LSTM | CNN 구간 특징과 순서 정보를 결합 | 긴 sequence 학습 비용 및 환자 수 부족 |
| Patch ECG Transformer / CNN–TST | patch 또는 segment token 간 관계 학습 | 작은 데이터에서 과적합·seed 민감; prototype은 논문 공식 구현 아님 |
| Segment Attention MIL / MIL-TCN | 환자 내 구간 가중치·시간 변화 집계 | 구간 특징과 시간 정렬을 먼저 정의해야 함 |
| Raw ECG MIL | 원시 구간 CNN 특징을 환자 단위로 attention 집계 | 아티팩트·채널·구간 선택에 민감; 전체 Holter 검증 필요 |

## 2. 공통 실험 조건

- **50명 제한:** 완성 ECG 파일 pair는 50명이다. 전체 다운로드가 시작된 뒤 중단 전에 완성된 44명분을 포함하고, train에서 SCD 양성 3명·음성 3명을 추가했다. 나머지 전체 코호트 파일은 계속 다운로드하지 않는다. 중단 전 다운로드로 분석 표본 밖의 미완성 `.dat.part` 16개(총 688,050,176 bytes)와 header-only 16개가 남아 있으며 평가 스크립트는 이를 읽지 않는다. 중단 전 44명은 환자 ID 순으로 먼저 내려받은 편의 표본이므로 전체 환자를 대표하는 무작위 표본이 아니다.
- **분할:** 기존 환자 단위 `train.csv`, `val.csv`, `test.csv` 소속을 유지한다. 현재 선정된 50명은 train 36명, validation 7명, test 7명이다.
- **입력 구간:** 각 환자 기록의 앞 100초를 8개 연속 12.5초 segment로 사용한다. 3 lead, 200 Hz, segment당 2,500 sample이며 raw 모델 입력은 `(B,8,3,2500)`이다.
- **파형 변환:** WFDB format-16 헤더의 gain과 ADC zero로 physical signal을 계산한다. Raw waveform 모델은 segment별·lead별 z-score 입력을 받는다.
- **통계 특징 모델:** 각 segment에서 lead별 평균·표준편차·RMS·범위(12개), XYZ 벡터 크기 평균·표준편차·RMS·범위(4개)를 계산한다. HRV/임상 검증 특징이 아니며 탐색 비교용이다.
- **학습:** seed 42, GPU 우선, BCEWithLogitsLoss의 train 양성 비율 보정, Adam, validation AP 기준 checkpoint/모델 선택을 적용한다. test는 선택된 모델 하나만 평가한다.
- **임계값:** validation에서 Youden J 기준으로 모델별 임계값을 정한다. 소표본이므로 임계값은 매우 불안정할 수 있다.
- **해석:** 각 split의 환자 수와 양성 수를 항상 함께 기록한다. 이 50명 파일럿은 실행·파이프라인 및 초기 모델 동작 비교이지, 일반화 성능이나 임상 유효성의 확정 근거가 아니다.
- **표본 한계:** 현재 분할은 train 36명/SCD 7명, validation 7명/SCD 4명, test 7명/SCD 1명이다. validation/test가 매우 작고 SCD 비율도 다르므로 AUROC·AP·threshold 지표는 크게 흔들린다. 성능 순위 또는 최종 모델 선정에 사용하지 않는다.

## 3. 실험 케이스 결과 기록

아래 속성은 Notion 표와 동일한 순서다: 케이스, 입력 데이터, 중요도, 모델 구분, 모델, 담당, 상태, ROC-AUC, PR-AUC, Sensitivity, Specificity, Brier, MLflow 링크, 메모.

### C1 임상 모델 비교

기존 동일 환자 분할의 validation 결과다 (N=83, SCD=10). 임계값은 0.5로 계산했다. CatBoost를 validation AP로 선택한 뒤 별도 test 1회 결과는 메모에 구분해 기재한다.

| 케이스 | 입력 데이터 | 중요도 | 모델 구분 | 모델 | 담당 | 상태 | ROC-AUC | PR-AUC (AP) | Sensitivity | Specificity | Brier | MLflow 링크 | 메모 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| C1 임상 – Logistic Regression | 임상 | 해당 없음 | 전통 ML | Logistic Regression | 남상기 | 완료 | 0.771 | 0.260 | 0.100 | 0.973 | 0.111 | — | Validation N=83/SCD=10; 23 features; CPU |
| C1 임상 – Random Forest | 임상 | 해당 없음 | 전통 ML | Random Forest | 남상기 | 완료 | 0.782 | 0.369 | 0.000 | 1.000 | 0.094 | — | Validation N=83/SCD=10; 23 features; CPU |
| C1 임상 – XGBoost | 임상 | 해당 없음 | 전통 ML | XGBoost | 남상기 | 완료 | 0.695 | 0.240 | 0.100 | 0.973 | 0.107 | — | Validation N=83/SCD=10; 23 features; CPU |
| C1 임상 – CatBoost | 임상 | 해당 없음 | 전통 ML | CatBoost | 남상기 | 완료 | 0.733 | 0.390 | 0.100 | 1.000 | 0.095 | — | Validation N=83/SCD=10; AP 최고. Test N=83/SCD=10: AUROC 0.781, AP 0.335, Brier 0.094 |
| C1 임상 – MLP | 임상 | 해당 없음 | 딥러닝 | MLP | 남상기 | 완료 | 0.708 | 0.269 | 0.300 | 0.918 | 0.127 | — | Validation N=83/SCD=10; 23 features; CPU |

### C4–C8 ECG 50명 파일럿

각 행의 지표는 validation 7명(SCD 4명) 결과다. threshold는 validation Youden J 기준이다. Validation AP 최고 모델이 동률인 경우 현재 코드의 고정 후보 순서상 1D-CNN을 선택해 test를 1회 실행했다. Test는 7명 중 SCD 양성이 1명이다.

| 케이스 | 입력 데이터 | 중요도 | 모델 구분 | 모델 | 담당 | 상태 | ROC-AUC | PR-AUC (AP) | Sensitivity | Specificity | Brier | MLflow 링크 | 메모 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| C6 ECG 원시 파형 – 1D-CNN | Holter ECG 50명 | 높음 | 딥러닝 | 1D-CNN | 남상기 | 완료 (파일럿) | 1.000 | 1.000 | 1.000 | 1.000 | 0.248 | — | Val N=7/SCD=4; 12 epochs; 1.95초, peak 50.8 MiB. Test N=7/SCD=1: AUC/AP 1.000, sensitivity 1.000, specificity 0.500, Brier 0.247 |
| C7 ECG 원시 파형 – CNN-LSTM | Holter ECG 50명 | 높음 | 딥러닝 | CNN-LSTM | 남상기 | 완료 (파일럿) | 0.750 | 0.854 | 0.500 | 1.000 | 0.249 | — | Val N=7/SCD=4; 12 epochs; 0.79초, peak 51.0 MiB |
| C8 ECG 원시 파형 – Patch ECG Transformer | Holter ECG 50명 | 중간 | 딥러닝 | Patch ECG Transformer | 남상기 | 완료 (파일럿) | 0.667 | 0.830 | 0.500 | 1.000 | 0.240 | — | Val N=7/SCD=4; 12 epochs; 1.10초, peak 32.9 MiB |
| C7 ECG 원시 파형 – CNN-TST | Holter ECG 50명 | 높음 | 딥러닝 | CNN-TST | 남상기 | 완료 (파일럿) | 0.750 | 0.893 | 0.750 | 1.000 | 0.250 | — | Val N=7/SCD=4; 12 epochs; 0.76초, peak 51.4 MiB; 로컬 prototype |
| C4 ECG 구간 통계 – Attention MIL | Holter ECG 50명 | 중간 | 딥러닝 | Segment Attention MIL | 남상기 | 완료 (파일럿) | 1.000 | 1.000 | 1.000 | 1.000 | 0.232 | — | Val N=7/SCD=4; 12 epochs; 0.27초, peak 20.7 MiB; 탐색용 16개 통계 특징, HRV 아님 |
| C5 ECG 구간 통계 – MIL-TCN | Holter ECG 50명 | 중간 | 딥러닝 | MIL-TCN | 남상기 | 완료 (파일럿) | 0.917 | 0.950 | 0.750 | 1.000 | 0.244 | — | Val N=7/SCD=4; 12 epochs; 0.35초, peak 20.8 MiB; 탐색용 16개 통계 특징, HRV 아님 |
| C6 ECG 원시 파형 – Raw ECG MIL | Holter ECG 50명 | 높음 | 딥러닝 | Raw ECG MIL | 남상기 | 완료 (파일럿) | 1.000 | 1.000 | 1.000 | 1.000 | 0.250 | — | Val N=7/SCD=4; 12 epochs; 0.31초, peak 51.3 MiB; 구간별 CNN + attention pooling |

**Test 단회 결과:** validation AP 최고 동률에서 고정 후보 순서로 선택된 1D-CNN은 Test N=7/SCD=1에서 AUROC 1.000, AP 1.000, Brier 0.247을 기록했다. Validation에서 선택한 threshold `0.49695` 적용 시 sensitivity 1.000, specificity 0.500이었다. 양성 1건으로 계산한 지표이므로 모델 우월성 증거가 아니다.

**해석:** validation AP가 1.0인 모델이 세 개이고 split이 매우 작아 작은 점수 변화가 순위를 바꿀 수 있다. Brier는 모든 모델에서 약 0.23–0.25로 확률 보정이 좋다고 보기 어렵다. 이 파일럿 결과만으로 ECG 후보 중 최종 모델을 고르지 않는다.

GPU: NVIDIA GeForce RTX 4060 Laptop GPU, Python 3.12.3, PyTorch 2.14.0+cu130, CUDA 13.0. 모델별 시간은 12 epoch 학습 wall time이고 peak 메모리는 CUDA allocated 기준이다. MLflow 링크는 연결된 run이 없어 `—`로 두고, 원 JSON 결과를 재현 자료로 둔다.

## 4. 결정 기록

| 날짜 | 결정 | 이유 |
|---|---|---|
| 2026-10-23 | 임상 데이터의 다음 개발 반복 우선 후보로 CatBoost 선정(잠정); Random Forest challenger 유지 | Validation AP 0.390으로 최고. Random Forest는 AUROC 0.782 및 Brier 0.094로 일부 지표에서 앞섰다. Validation/test 양성이 각각 10명뿐이며 최종 선정은 보류한다. |
| 2026-10-25 | ECG 원시 파형은 전체 다운로드 대신 50명으로 파일럿 범위 제한 | 실제 GPU 모델 비교와 데이터 입출력 검증에 필요한 최소 비교를 우선한다. 소표본 결과를 임상 성능 주장에 사용하지 않는다. |
| 2026-10-25 | ECG Transformer 첫 prototype 후보로 CNN–TST를 비교하고 Patch ECG Transformer를 병행 | CNN frontend를 공유하는 CNN–LSTM과 비교 구조를 만들 수 있다. 연구 논문 공식 구현 재현과는 구분한다. |
| 2026-10-25 | 임상과 Holter ECG 모델을 별도로 평가·선정하기로 결정 | 입력 데이터와 평가 대상이 다르다. 임상 CatBoost 결과와 ECG 50명 파일럿 결과는 직접적인 모델 순위 비교에 사용할 수 없다. |
| 2026-10-25 | Holter ECG를 100명까지 확장해 7개 모델 비교; MIL-TCN을 validation 우선 후보로 기록하되 최종 선정은 보류 | Validation AP/AUROC는 각각 0.917/0.955였지만 선택 후 test AUROC/AP는 0.409/0.208이었다. Test 13명/SCD 2명이며 기존 50명 파일럿 test 환자도 포함되어 독립적인 최종 근거가 아니다. 원시 파형만 보면 1D-CNN이 validation AP 최고(0.827)였다. |

## 5. 최종 선정

- **임상 데이터 선정: CatBoost를 현재 우선 후보로 선정하고 Random Forest를 challenger로 둔다.** CatBoost는 validation AP 최고(0.390), 선택 후 held-out test에서 AUROC 0.781 / AP 0.335 / Brier 0.094였다. Random Forest는 validation AUROC 0.782와 Brier 0.094로 근소하게 앞섰고 AP 0.369였다. 이 선정은 임상 입력 모델끼리의 현재 개발 우선순위다.
- **Holter ECG 우선 실험 후보: MIL-TCN(구간 통계 특징).** 100명 비교에서 validation AP 0.917/AUROC 0.955로 가장 높았다. 그러나 test AUROC 0.409/AP 0.208, sensitivity 0.000이어서 최종 모델로 선정하지 않는다. 원시 파형 계열의 우선 실험 후보는 1D-CNN(validation AP 0.827)이다. 상세 결과와 제한은 [100명 보고서](ecg_100_model_comparison.md)에 기록했다.
- **두 입력 데이터 계열 간 직접 비교 금지:** 임상 모델 결과와 Holter 모델 결과는 사용 환자 수, 입력, split 규모가 달라 서로 같은 조건의 성능 비교가 아니다. 향후 임상+ECG 융합 모델을 평가할 때는 두 입력이 모두 있는 동일 환자 집단과 공통 split에서 Clinical-only / ECG-only / Fusion을 비교한다.
- **요구 지표에 대한 상태:** AUROC, AP, Sensitivity, Specificity 및 Brier를 기록했다. 다만 현재 sensitivity/specificity는 임계값 선택에 민감하고, Brier 하나만으로 calibration을 충분히 평가했다고 볼 수 없다. 임상 위험도 사용 전 validation에서 임계값을 정하고 calibration curve 및 필요 시 보정법을 별도로 검증한다. 현재 CatBoost test sensitivity는 기록된 0.5 threshold에서 0.000이므로 임상 알림 모델로 바로 사용하지 않는다.
- 따라서 회의·발표에서는 **“임상: CatBoost 우선 후보, Random Forest challenger / Holter ECG 통계 특징: MIL-TCN 탐색 후보 / Holter 원시 파형: 1D-CNN 탐색 후보 / 최종 모델은 추가 검증 후 확정”**으로 데이터 유형별 구분해 표현한다. 각 결과에 환자 수, SCD 양성 수, split, threshold 및 파일럿 한계를 함께 제시한다.

## 재현 파일

- 평가 스크립트: [`evaluate_ecg_architectures.py`](../scripts/evaluate_ecg_architectures.py)
- 모델 prototype: [`models.py`](../model/DL/models.py)
- ECG 50명 원시 결과/환자 ID manifest: [`ecg_50_model_comparison.json`](ecg_50_model_comparison.json)
- 임상 성능 비교: [`preprocessed_data_model_comparison.md`](preprocessed_data_model_comparison.md)
- Transformer 조사 및 환자 ID smoke: [`transformer_ecg_model_comparison.md`](transformer_ecg_model_comparison.md)
