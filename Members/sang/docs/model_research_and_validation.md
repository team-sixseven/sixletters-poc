# SCD 위험 예측 모델 조사 및 입출력 검증

**담당:** 남상기
**기간:** 2026.10.23 ~ 10.25
**상태:** 진행 중

> 최신 사용자 제공 전처리 split(`Members/sang/datas/clincal/splits`)을 사용한 모델 비교와 추천은 [전처리 데이터 모델 비교 보고서](preprocessed_data_model_comparison.md)를 기준으로 한다. 이 문서에 포함된 961명 공통 prepared bundle 결과는 별도 데이터 기준의 이전 비교 결과다.
> 환자 ID 실제 파형 smoke test 및 시계열 Transformer 후보 조사는 [Transformer 후보 조사 및 실제 데이터 비교](transformer_ecg_model_comparison.md)를 참고한다. 임상 split 결과와 ECG 단일 환자 smoke test를 성능 비교로 합치지 않는다.

## 1. 목적 및 데이터 현황

**목표:** 전통 머신러닝과 딥러닝 모델의 입력 적합성 및 실행 가능성을 확인하고, 실제 데이터에서 가능한 초기 비교를 수행해 10/25 회의에서 실험 방향을 정한다.

**기업 요구사항**

- ML ↔ DL 모델 성능 및 장단점 비교
- NVIDIA GPU 우선 활용
- Furiosa NPU 추론 추가 검토

| 항목 | 확인 내용 |
|---|---|
| 데이터셋 | PhysioNet MUSIC v1.0.1 |
| 원본 환자 | 992명, SCD 94건 (원본 통계) |
| 현재 사용자 전처리 임상 데이터 | `clinical_clean.csv` 552명; train 386 / val 83 / test 83 |
| 현재 라벨 | 제공된 `SCD` 기준 전체 66명; train 46 / val 10 / test 10 |
| 현재 임상 입력 | CSV는 총 25열이며 `Patient ID`, `SCD` 제외 23개 feature |
| 현재 ECG 자료 | MUSIC Holter 100명 완전한 `.dat/.hea` pair; 3채널 200 Hz split 기준 Train 72/Val 15/Test 13. 100명 파일럿 모델 결과는 [별도 비교 보고서](ecg_100_model_comparison.md); 최종 성능 확정 불가 |
| 별도 공통 prepared bundle (이전 비교) | 961명 (SCD 94명), train 671 / validation 144 / test 146; 64 clinical features |
| 예측 목표 | 현재 split은 이진 `SCD` 제공; event horizon·censoring 정의는 별도 확인 필요 |
| 예측 목표 | 관찰된 추적 기간 중 SCD 여부; 고정 4년 위험 또는 time-to-event가 아님 |

※ 789명 선정 기준과 해당 코호트의 SCD 수는 확정되지 않았다. 현재 실제 평가 결과는 임의로 789명을 구성하지 않고 저장소 공통 prepared dataset의 환자 단위 split을 사용했다. Endpoint는 실험 프로토콜 기준이며 최종 연구 정의를 팀에서 확인해야 한다.

## 2. 후보 모델 조사

| 구분 | 모델 | Input → Output | 장점 | 제한 / 확인 사항 | 우선순위 |
|---|---|---|---|---|---|
| ML | XGBoost | `(N,F)` → `(N,)` 확률 | 비선형 관계, SHAP 활용 가능 | 과적합·튜닝 점검 | 1차 |
| ML | CatBoost | `(N,F)` → `(N,)` 확률 | 범주형 처리 지원 | 범주형 코드 정의 확인 | 1차 |
| ML | Random Forest | `(N,F)` → `(N,)` 확률 | 안정적인 앙상블 비교 | 실제 데이터 결과 미측정 | 선택 |
| DL | MLP | `(N,F)` → `(N,1)` logit | 정형 데이터 DL 비교 | 소규모 데이터 과적합 가능; 더미 fit에서 수렴 경고 | 2차 |
| DL | 1D-CNN | `(B,C,T)` 또는 `(B,S,C,T)` → `(B,1)` logit | 파형 특징 자동 추출; MIL 집계 가능 | 실제 파형 전처리·환자 집계 필요 | 1차 후보 |
| DL | ECG Transformer | `(B,C,T)` → `(B,1)` logit | 시계열 장거리 관계 학습 | 실제 입력 길이·메모리 평가 필요 | 2차 |
| DL | Segment Attention MIL | `(B,S,F)` → `(B,1)` logit | 환자 내 segment 중요도 집계 | 실제 구간 특징 필요 | 추가 실험 |
| DL | MIL–TCN | `(B,S,F)` → `(B,1)` logit | 시간순 segment 관계 학습 후 집계 | 실제 구간 순서·특징 필요 | 추가 실험 |
| DL | Raw ECG MIL | `(B,S,C,T)` → `(B,1)` logit | 각 파형 구간을 CNN 인코딩 후 환자 단위 집계 | 실제 파형·채널·샘플링 정보 필요 | 추가 실험 |

`N`: 환자 수, `F`: 정형 특징 수, `B`: batch, `S`: 환자당 구간 수, `C`: ECG 채널 수, `T`: 구간당 시계열 길이. 정형 출력은 양성 클래스 확률, 딥러닝 모델 출력은 학습용 logit이다.

## 3. 데이터와 모델 입출력 연결

| 데이터 | 현재 확인된 형태 | 적용 실험 | 필요한 전처리 |
|---|---|---|---|
| 임상 | prepared schema 64개 변수 | LR, XGBoost, CatBoost, MLP | train-only 결측 대치, 범주형 인코딩; 24개 변수 확정 시 schema 재정의 |
| ECG 환자별 정량지표 | 공통 schema 5개 | ECG baseline, clinical+ECG fusion | train-only 결측 대치·범주형 처리 |
| Holter 구간별 특징 | 현재 자료에서 확인 불가 | Segment Attention MIL, MIL–TCN | 환자/기록 ID, segment 시간순서, 구간별 특징 및 라벨 연결 |
| Holter 원시 파형 | 환자 ID 단일 WFDB pair 확인 | 1D-CNN, Transformer, Raw ECG MIL | 환자 ID smoke test 완료; 코호트 비교를 위해 추가 환자 파형, 전처리·구간 규칙 필요 |

**데이터 연동 원칙**

- `patient_id`를 기준으로 임상·ECG·라벨 연결
- 같은 환자의 모든 ECG 구간은 한 split에만 배치
- 구간별 예측은 환자 단위로 집계
- 결측치 대치·변환은 train에서만 fit
- test split은 모델 선택에 사용하지 않고 최종 평가까지 보류

## 4. 더미 데이터 실행 결과

**실행일:** 2026.10.23, MIL 계열 추가 검증 2026.10.23
**목적:** 합성 입력에서 모델 API, shape, Forward, Loss, Backward가 동작하는지 확인. 예측 성능 평가는 아님.

| 모델 | 더미 Input | 결과 | 실행 장치 / 실행 시간 |
|---|---|---|---|
| Random Forest | `(256,24)` | PASS | CPU / 0.167초 |
| XGBoost | `(256,24)` | PASS | CPU / 1.016초 |
| CatBoost | `(256,24)` | PASS | CPU / 0.108초; 범주형 입력 미시험 |
| MLP | `(256,24)` | PASS* | CPU / 0.027초 |
| 1D-CNN | `(8,3,2500)` | PASS | CPU 0.148초, CUDA 1.103초 |
| ECG Transformer | `(8,3,2500)` | PASS | CPU 0.129초, CUDA 0.348초 |
| Segment Attention MIL | `(8,12,16)` | PASS | CPU 0.060초, CUDA 0.507초; CUDA peak 16.30 MiB |
| MIL–TCN | `(8,12,16)` | PASS | CPU 0.130초, CUDA 0.377초; CUDA peak 16.37 MiB |
| Raw ECG MIL (1D-CNN + MIL) | `(2,4,3,2500)` | PASS | CPU 0.019초, CUDA 0.152초; CUDA peak 20.62 MiB |

**확인 결과**

- 정형 모델 더미 테스트에서 `fit()` 및 확률 출력 `(256,)` 확인
- CNN/Transformer와 MIL 계열에서 Forward, BCE loss, Backward 및 유한 출력·loss·gradient 확인
- MLP는 `max_iter=30`에서 수렴 경고가 발생해 입출력 검증만 PASS
- 위 MIL 입력의 구간/특징 수와 파형 shape은 smoke test용 가정이며 실제 MUSIC 구조를 나타내지 않음
- 단회 실행 시간은 모델 성능 벤치마크가 아님

### 테스트 환경

| 항목 | 환경 |
|---|---|
| OS | WSL2 Ubuntu 24.04 |
| Python | 3.12.3 |
| PyTorch | 2.14.0+cu126 |
| CUDA runtime | 12.6 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB |
| scikit-learn | 1.9.0 |
| XGBoost / CatBoost | 3.4.1 / 1.2.10 |
| Random Seed | 42 |
| 기존 더미 실행 | `Members/sang/scripts/smoke_model_api.py` |
| MIL 더미 실행 | `Members/sang/scripts/smoke_mil_models.py` |

CUDA 검증은 GPU 장치 접근이 허용된 실행 환경에서 수행했다. 일반 sandbox 실행은 GPU를 노출하지 않을 수 있다. 패키지는 여러 기존/격리 환경에서 사용했으므로 저장소 공통 Docker 환경의 완전 재현은 별도 검증이 필요하다.

## 5. 실제 데이터 Validation 비교

5.1은 최신 사용자 전처리 데이터의 validation 83명(SCD 10명) 결과다. 5.2는 다른 `Global/data/prepared` bundle로 이전에 계산한 기록이며 현재 모델 추천에는 사용하지 않는다. AUPRC는 Average Precision(AP)이다.

### 5.1 Clinical 모델 비교

| 모델 | 입력 | AUROC ↑ | AUPRC/AP ↑ | Brier ↓ | 학습 시간 |
|---|---|---:|---:|---:|---:|
| XGBoost | Clinical 23 features | 0.695 | 0.240 | 0.107 | 0.046초 |
| CatBoost | Clinical 23 features | 0.733 | **0.390** | 0.095 | 0.177초 |
| Random Forest | Clinical 23 features | **0.782** | 0.369 | **0.094** | 0.431초 |
| MLP | Clinical 23 features | 0.708 | 0.269 | 0.127 | 0.545초 |

최신 사용자 전처리 split 결과이며, 세부 조건과 test 평가·모델 추천은 [전처리 데이터 모델 비교 보고서](preprocessed_data_model_comparison.md)를 참고한다. 앞선 64개 feature 결과는 다른 prepared bundle을 쓴 이전 실행이다.

### 5.2 이전 별도 실험 기록: 공통 prepared bundle

이 아래 비교는 validation 144명(SCD 14명), clinical 64개 변수 및 ECG feature 5개를 사용한 별도 bundle 결과다. 아래 결과는 원시 ECG 파형 비교가 아니며, 이후 환자 ID 단일 파형 smoke와 직접 합치지 않는다.

| 순서 | 실험 | 입력 | N / SCD | AUROC ↑ | AUPRC/AP ↑ | Brier ↓ | 학습 시간 |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Clinical Baseline | 임상 64개 | 144 / 14 | 0.713 | 0.194 | 0.087 | 0.134초 |
| 2 | ECG Features Baseline | ECG 정량지표 5개 | 144 / 14 | 0.505 | 0.116 | 0.093 | 0.025초 |
| 3 | Clinical + ECG Fusion | 임상 64개 + ECG 5개 | 144 / 14 | 0.718 | 0.186 | 0.088 | 0.081초 |

Clinical와 Fusion AUROC는 각각 0.713, 0.718이나 Fusion AP는 0.186으로 Clinical 0.194보다 낮다. validation SCD 사례가 14명인 단일 split 결과이므로 결합 효과나 모델 우위를 확정할 수 없다. 두 Clinical XGBoost 기록의 학습 시간 차이는 단회 측정의 실행 변동이며 정확도 차이를 뜻하지 않는다.

### 5.3 추천 실험 4–6: 실제 성능 평가 상태

| 순서 | 실험 | 입력 | 실제 데이터 평가 |
|---:|---|---|---|
| 4 | Segment MIL | Holter segment별 통계 특징 | 100명 탐색 비교 완료; 자세한 수치와 제한은 [보고서](ecg_100_model_comparison.md) |
| 5 | MIL–TCN | 시간순 Holter segment 특징 | 100명 탐색 비교 완료; validation 우선 후보, test 일반화 미확인 |
| 6 | Raw ECG Model | Holter 원시 파형 | 100명 탐색 비교 완료; raw 계열 validation 우선 후보 1D-CNN |

환자별 ECG 정량지표 CSV를 구간 특징이나 원시 파형으로 간주하지 않았다. 환자 ID 파형 smoke 결과와 Transformer 후보별 비교 범위는 [별도 보고서](transformer_ecg_model_comparison.md)를 참고한다. smoke PASS는 실제 임상 성능 검증을 뜻하지 않는다.

## 6. 모델 선정 기준 및 추천

| 평가 항목 | 판단 기준 |
|---|---|
| 성능 | AUROC, AUPRC, Brier, calibration; 환자 단위 공통 split |
| 데이터 적합성 | 입력 변수 정의, 파형·구간 구조, 결측 처리 |
| 안정성 | 과적합, 클래스 불균형, 재현성 및 불확실성 |
| 설명 가능성 | 회귀계수, SHAP, 구간별 기여도 |
| 개발 가능성 | 일정, GPU 자원, 실제 입력 확보 |
| 추가 검토 | Furiosa NPU 추론 호환성 |

**현재 추천:** 정형 데이터는 XGBoost와 CatBoost를 비교한다. ECG 파형 비교는 1D-CNN, CNN–LSTM, CNN–TST 순서를 권한다. ConvTran은 추가 시계열 baseline, Medformer/FormerTime은 후속 후보로 둔다. 환자 ID smoke 실행은 완료했으나 단일 환자로 임상 성능이나 최종 후보를 고를 수 없다. Transformer 후보 근거와 실데이터 실행 범위는 [별도 비교 보고서](transformer_ecg_model_comparison.md)에 기록했다. Random Forest는 더미 실행은 통과했으나 현재 실제 데이터 비교에는 포함되지 않았다.

## 7. 미확정 사항 및 다음 작업

### 요청된 진행 순서와 현재 상태

| 단계 | 작업 | 상태 / 결과 |
|---|---|---|
| STEP 1 | 실제 데이터·라벨·환자 ID 검증 | 현재 split 552명과 clean 파일의 ID·라벨 일치 및 split 간 중복 없음 확인. 789명 기준 및 label horizon 미확정 |
| STEP 2 | 정형 데이터 Baseline 재현 및 비교 | 사용자 전처리 train/val에서 LR, RF, XGBoost, CatBoost, MLP 실행. 현재 split XGBoost AUROC 0.695; 예전 64-feature bundle의 0.713과 데이터가 다름 |
| STEP 3 | Clinical / ECG / Fusion 입력 비교 | Clinical 결과 완료. ECG 100명 파일럿은 [별도 보고서](ecg_100_model_comparison.md)에 기록했으며 test가 13명/SCD 2명이라 탐색 결과로 해석. 임상+ECG fusion은 두 입력이 모두 있는 동일 환자 집단에서 별도 실행 필요 |

이번 최신 데이터 실행은 **clinical 23개 변수만** 사용했다. 앞선 64개 clinical / 5개 ECG-feature 비교 결과는 공통 prepared bundle에서 별도로 수행된 기록이며, 현재 사용자 제공 데이터의 성능 비교로 간주하지 않는다.

STEP 1에서 확인한 endpoint는 공통 프로토콜상 관찰 추적 기간 중 SCD 발생 여부다. 이는 팀이 최종 확정한 라벨 정의를 뜻하지 않는다. 789명 선정 규칙, 4년 고정 예측 여부, 검열 및 비SCD 사망 처리 기준은 데이터 담당자/팀 결정이 남아 있다.

재현 명령:

```sh
PYTHONPATH=Global/src:/tmp/scd-smoke-extra:/home/sang/.cache/uv/archive-v0/TIDVvkcYCJEzCu0b \
  /home/sang/dnn_tf_ex/.venv/bin/python Members/sang/scripts/evaluate_actual_clinical.py \
  --data-dir Global/data/prepared/music-scd-basic-v1
```

### 회의에서 확정할 사항

- 분석 대상 789명 선정 기준 및 최종 양성 수
- SCD 예측 기간과 중도 검열·다른 원인 사망 처리
- 임상 변수 24개 목록 및 ECG 관련 32개 컬럼의 실제 사용 변수
- Holter 파일 제공 위치, 입력 채널·구간 길이·환자별 집계 방식
- 최종 학습 모델과 반복/교차검증 및 calibration 평가 기준
- Furiosa NPU 대상 장치와 SDK/runtime 접근 가능성

### 진행 현황

- 후보 모델 조사 및 일반 입력/출력 형태 정리 완료
- 정형 모델 5종과 파형 CNN/Transformer 더미 실행 확인
- Segment Attention MIL, MIL–TCN, Raw ECG MIL 구조 추가 및 CPU/CUDA 더미 Forward·Backward 확인
- 공통 prepared split에서 Clinical 모델 및 ECG/Fusion XGBoost 실제 validation 결과 기록
- 실제 Holter 원시 파형·segment 데이터 연동 및 평가 미완료
- 동등한 반복 평가, calibration, 최종 모델 선정 미완료
- Furiosa NPU 호환성 미검증

### Sprint 2 (10/26~11/1)

1. **데이터 연동:** 전처리 CSV·파형·라벨 연결 및 확정된 환자 단위 cohort/split 적용
2. **정형 모델:** 확정 clinical 변수와 ECG 특징에서 LR / XGBoost / CatBoost 비교
3. **시계열 모델:** 실제 Holter 입력 구조를 확인한 뒤 1D-CNN/MIL 소규모 학습 검증
4. **평가:** AUROC, AUPRC, Brier, calibration 및 주요 기여 변수 확인
5. **결과 정리:** 동일 평가 기준의 비교표, 재현 설정 및 선정 근거 작성

※ Transformer 본학습과 Furiosa NPU 검증은 실제 데이터, 장치, SDK와 일정을 확인한 후 진행한다.

## 8. 결론

현재 사용자 전처리 데이터 비교에서는 CatBoost가 validation AP 0.390으로 가장 높았고, validation에서 선택해 test를 한 번 평가했을 때 AUROC 0.781/AP 0.335/Brier 0.094를 기록했다. Random Forest는 validation AUROC와 Brier에서 가장 좋았다. 양성 표본이 각각 10건뿐이라 CatBoost를 잠정 후보로 두고 Random Forest를 challenger로 유지한다. 환자 ID 한 건은 실제 ECG smoke에 사용했지만 ECG 기반 임상 성능은 평가하지 않았다.

CNN, Transformer, Segment MIL, MIL–TCN, Raw ECG MIL은 더미 입력 및 환자 ID 단일 파형으로 CPU/CUDA 실행 구조를 확인했다. 파형 기반 SCD 성능은 아직 측정하지 않았다. 다음 단계는 코호트·endpoint·특징 목록을 확정하고 다수 환자의 Holter 입력을 연동하는 것이다.

**참고 문서:** [실제 모델 결과](actual_model_results.md), [추천 실험 6개](recommended_experiments.md), [더미 실행 기록](model_smoke_results.md)

**주요 참고 자료:** PhysioNet MUSIC, scikit-learn, XGBoost, CatBoost, torch_ecg, UTU ResNet/Transformer, FuriosaAI 공식 문서
