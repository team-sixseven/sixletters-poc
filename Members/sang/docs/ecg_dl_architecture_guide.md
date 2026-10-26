# ECG 딥러닝 모델 구조와 비교 가이드

작성일: 2026-10-25

대상: MUSIC Holter ECG 파일럿에서 실행한 딥러닝 모델 7종

## 1. 먼저 구분할 용어

- **1D-CNN:** ECG 원시 파형의 짧은 시간 패턴을 합성곱 필터로 추출한다.
- **CNN-LSTM:** 각 ECG 구간을 CNN으로 요약한 뒤, LSTM이 구간 순서를 처리한다.
- **Transformer:** 입력을 token으로 바꾸고 self-attention으로 token 사이 관계를 학습한다. 이번 비교에는 파형을 작은 patch token으로 만든 `Patch ECG Transformer`와, CNN 구간 embedding을 Transformer에 넣는 `CNN-TST` 두 local prototype이 있다.
- **MIL (Multiple Instance Learning):** 한 환자의 여러 구간을 하나의 bag으로 보고, 구간별 표현을 환자별 예측 하나로 모은다.
- **TCN:** 시간축 합성곱으로 순서 정보를 학습한다. 이번 `MIL-TCN`의 입력은 raw ECG가 아니라 구간별 통계 특징이다.

이번 실험의 모든 모델은 로컬에서 만든 소형 prototype이며, 사전학습 가중치나 논문 저자의 공식 구현을 그대로 사용하지 않았다. Patch token 구조는 Vision Transformer의 patch-to-token 아이디어, 시계열 Transformer는 TST 계열, 구간 attention pooling은 attention-based MIL의 개념과 연결되지만 구현·학습 설정은 이 프로젝트 코드 기준이다. [Vision Transformer 원 논문](https://arxiv.org/abs/2010.11929), [다변량 시계열 Transformer 논문](https://doi.org/10.1145/3447548.3467401), [Attention-based MIL 논문](https://proceedings.mlr.press/v80/ilse18a.html)

## 2. 이번 실험의 입력과 출력

각 환자에게서 기록 앞 100초를 8개 연속 구간으로 나눴다. 구간 하나는 12.5초, 3개 lead, 200 Hz이므로 raw 입력은 환자 batch 기준 `(B, 8, 3, 2500)`이다. 모델은 최종적으로 환자별 SCD logit `(B, 1)`을 출력하고, 추론 점수는 sigmoid를 적용한 확률이다.

통계 특징 모델은 각 12.5초 구간에서 16개 숫자를 계산해 `(B, 8, 16)`으로 입력했다.

- X/Y/Z lead 각각 평균, 표준편차, RMS, 진폭 범위: 12개
- XYZ 벡터 크기의 평균, 표준편차, RMS, 진폭 범위: 4개

이 16개는 파일럿을 위한 단순 통계량이다. 임상적으로 검증된 HRV 특징이나 심전도 판독 feature로 해석하면 안 된다.

## 3. 모델별 구조와 입력 차이

| 모델 | 실제 입력 | 주요 계산 흐름 | 환자 단위 집계 | 이 모델이 배우는 것 |
|---|---|---|---|---|
| **1D-CNN** | raw ECG `(B,8,3,2500)` | 구간마다 Conv1d → ReLU → MaxPool → Conv1d → Global Average Pool | 구간별 logit 8개를 평균 | 짧은 ECG 파형에서 lead 조합과 국소 모양 패턴 |
| **CNN-LSTM** | raw ECG `(B,8,3,2500)` | 각 구간을 CNN embedding 32차원으로 변환 → 순서대로 LSTM | 마지막 LSTM hidden state로 환자 예측 | 구간 내용과 구간 간 순서·변화 |
| **Patch ECG Transformer** | raw ECG `(B,8,3,2500)` | 각 구간을 길이 50 sample의 patch로 자름 → 50 token × 32차원 → 2-layer Transformer encoder | 구간별 logit을 평균 | 한 12.5초 구간 안에서 떨어진 파형 patch끼리의 관계 |
| **CNN-TST** | raw ECG `(B,8,3,2500)` | 구간별 CNN embedding 32차원 → 환자당 8개 token → 위치 인코딩 → 2-layer Transformer encoder | 8 token 평균 pooling | 100초 내 구간 간 장거리 관계와 순서 |
| **Segment Attention MIL** | 통계 특징 `(B,8,16)` | 각 구간을 32차원으로 인코딩 → attention weight 계산 | 중요 구간의 가중 평균 | 환자 예측에 기여하는 통계 특징 구간 |
| **MIL-TCN** | 통계 특징 `(B,8,16)` | 구간 특징을 dilated Conv1d (dilation 1, 2, 4)로 처리 → attention | 시간 특징의 가중 평균 | 통계 특징이 시간에 따라 바뀌는 패턴 |
| **Raw ECG MIL** | raw ECG `(B,8,3,2500)` | 각 구간을 CNN embedding 32차원으로 변환 → attention | 중요 파형 구간의 가중 평균 | 환자 내 어떤 raw ECG 구간이 점수에 기여하는지 |

### Transformer 둘의 차이

- **Patch ECG Transformer:** Transformer token이 원시 파형의 patch다. 구간 길이 2,500 sample을 patch size 50으로 나눠 구간당 50 token을 만든다. 각 구간의 logit을 평균하므로 8개 구간 사이 시간 순서는 이 모델에서 직접 attention하지 않는다.
- **CNN-TST:** 먼저 CNN이 각 12.5초 구간을 하나의 embedding으로 줄인다. 환자당 8개 embedding이 Transformer token이므로 8개 구간 사이의 순서를 학습한다. 원시 파형 내 patch 간 attention은 하지 않는다.

## 4. 모델별 장단점

| 모델 | 장점 | 주의할 점 | 이 프로젝트 파일럿에서의 validation 결과 |
|---|---|---|---|
| 1D-CNN | 구조가 단순하고 연산이 가벼움; raw waveform baseline으로 설명하기 쉬움 | 장기 구간 관계를 직접 모델링하지 않음; 구간 평균이 유익한 segment 차이를 약화할 수 있음 | raw 계열 중 최고: AUROC 0.795, AP 0.827 |
| CNN-LSTM | 구간 특징과 시간 순서를 함께 반영; Transformer보다 구조 해석이 직관적 | 구간 순서 길이에 따라 순차 계산; 소표본에서는 LSTM 효과 검증이 어려움 | AUROC 0.682, AP 0.594 |
| Patch ECG Transformer | 한 구간 내부의 긴 범위 patch 관계를 attention으로 학습; patch 길이로 token 수 조절 가능 | patch 크기·위치 인코딩 영향; 작은 표본에서 과적합·seed 민감성 우려 | AUROC 0.682, AP 0.671; Brier는 일곱 모델 중 최저 0.208 |
| CNN-TST | CNN으로 raw signal 차원을 줄인 뒤 환자 내 segment 관계를 학습; CNN-LSTM과 시간 집계 방식 비교 가능 | 8개 token만 쓰는 얕은 prototype이며 정식 저자 구현이나 검증된 TST 결과가 아님 | AUROC 0.523, AP 0.546 |
| Segment Attention MIL | 구간별 가중치를 확인할 수 있어 환자 단위 집계 설명에 유리; 매우 가벼움 | attention weight는 인과 설명이 아님; 통계 특징 정의에 성능이 좌우됨 | 통계 계열 2위: AUROC 0.909, AP 0.875 |
| MIL-TCN | dilation으로 여러 시간 간격의 구간 변화를 보고 attention으로 집계 | 현재 입력은 HRV/raw ECG가 아니라 임시 통계 특징; validation 결과가 test에서 재현되지 않음 | 전체 validation 1위: AUROC 0.955, AP 0.917 |
| Raw ECG MIL | raw 파형 구간별 중요도 학습; 환자당 variable segment 수로 확장하기 쉬움 | attention과 CNN이 작은 데이터에서 불안정할 수 있음; 이번 구성은 100초만 사용 | AUROC 0.409, AP 0.273 |

## 5. 100명 실험 결과 요약

| 항목 | 결과 |
|---|---|
| 데이터 split | Train 72명/SCD 13명, Validation 15명/SCD 4명, Test 13명/SCD 2명 |
| 학습 환경 | RTX 4060 Laptop GPU, PyTorch 2.14.0+cu126, CUDA 12.6 |
| 학습 조건 | seed 42, 최대 12 epochs, batch size 8, validation AP로 checkpoint 선택 |
| Validation 우선 후보 | MIL-TCN: AP 0.917, AUROC 0.955; 16개 구간 통계 특징을 사용 |
| Raw waveform 우선 후보 | 1D-CNN: AP 0.827, AUROC 0.795 |
| Transformer 비교 | Patch ECG Transformer AP 0.671; CNN-TST AP 0.546 |
| 선택 모델 test | MIL-TCN, N=13/SCD=2: AUROC 0.409, AP 0.208, sensitivity 0.000, specificity 0.818, Brier 0.229 |

Test 결과가 낮아 validation 최고 모델을 최종 모델로 확정하지 않았다. Test는 양성 2명뿐이고 이전 50명 파일럿에서 평가한 test 환자 7명을 다시 포함하므로 완전히 독립적인 test로 볼 수 없다. 전체 결과·threshold·자원 측정은 [100명 모델 비교 보고서](ecg_100_model_comparison.md), 환자 ID는 [다운로드 manifest](ecg_100_download_manifest.json)를 참고한다.

## 6. 현재 추천

1. **임상 입력 모델과 Holter 모델은 별도로 선정한다.** 임상 CatBoost 결과를 Holter CNN/Transformer 결과와 같은 순위표로 직접 비교하지 않는다.
2. **Holter 구간 통계 특징 연구 후보:** MIL-TCN을 다음 반복 실험 후보로 둔다. 다만 test 결과가 좋지 않아 확정 모델로 쓰지 않는다.
3. **원시 ECG 연구 후보:** 1D-CNN을 baseline으로 유지하고 CNN-TST 또는 Patch ECG Transformer를 challenger로 비교한다. 이번 100명 결과에서는 Transformer의 우위가 관찰되지 않았다.
4. 새 cohort 또는 미사용 holdout을 확보한 뒤 여러 환자 단위 split/seed에서 같은 입력 집합으로 반복 평가한다. Sensitivity/Specificity 임계값은 validation에서 정하고 Brier뿐 아니라 calibration curve도 확인한다.

## 7. 참고 문헌 및 구현

- [Vision Transformer: An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) — patch를 token으로 변환하는 설계 개념 참고.
- [A Transformer-based Framework for Multivariate Time Series Representation Learning (TST)](https://doi.org/10.1145/3447548.3467401) — 시계열 Transformer 표현 학습 논문. 이 프로젝트의 CNN-TST는 저자 구현 재현이 아니라 축약 prototype.
- [Attention-based Deep Multiple Instance Learning](https://proceedings.mlr.press/v80/ilse18a.html) — instance attention과 bag-level pooling 개념 참고.
- 구현 코드: [`models.py`](../model/DL/models.py), 평가 코드: [`evaluate_ecg_architectures.py`](../scripts/evaluate_ecg_architectures.py).
