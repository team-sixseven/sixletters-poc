# Holter ECG 100명 모델 비교 결과

작성일: 2026-10-25

실험 상태: 100명 파일럿 완료; 최종 모델 선정은 보류

## 1. 결론

- **Holter 파일럿 validation 최고 후보: MIL-TCN.** 동일 validation 환자에서 AUROC 0.955, AP 0.917, Brier 0.217을 기록했다. 다만 이 모델은 원시 파형 대신 ECG 구간별 통계 특징 16개를 입력한다.
- **원시 파형 모델 중 우선 후보: 1D-CNN.** 원시 파형 계열에서 validation AUROC 0.795, AP 0.827로 가장 높았다. Transformer 후보(Patch ECG Transformer: AP 0.671, CNN-TST: AP 0.546)보다 이번 validation 결과가 높았다.
- **최종 Holter 모델은 아직 선정하지 않는다.** Validation에서 선택한 MIL-TCN의 test 성능은 AUROC 0.409, AP 0.208, sensitivity 0.000, specificity 0.818, Brier 0.229였다. Test가 13명/SCD 2명이고 과거 50명 파일럿의 test 환자 7명을 다시 포함하므로 일반화 우위의 증거로 보기 어렵다.
- 임상 정형 데이터 모델과 Holter 모델은 입력 및 환자 구성이 다르므로 별도 결과로 평가한다. 임상 모델 비교는 [임상 데이터 모델 비교](preprocessed_data_model_comparison.md)를 참고한다.

## 2. 데이터 및 실행 조건

| 항목 | 조건 |
|---|---|
| 원자료 | [MUSIC v1.0.1 Holter ECG](https://physionet.org/content/music-sudden-cardiac-death/1.0.1/); 개별 `.dat/.hea` 파일은 공개 PhysioNet AWS Open Data 버킷에서 받음 |
| 완전한 파형 쌍 | 100명; 추가 50명은 기존 50명과 환자 ID 중복 없음 |
| 환자 단위 split | Train 72명/SCD 13명, Validation 15명/SCD 4명, Test 13명/SCD 2명 |
| 입력 파형 | 환자 기록 앞 100초; 8개 연속 segment × 12.5초, 3 lead, 200 Hz, segment당 2,500 sample |
| Raw 입력 shape | `(B, 8, 3, 2500)`; 모델별 segment/환자 집계는 구현에 따름 |
| 통계 특징 입력 | `(B, 8, 16)`; 각 구간의 lead별 평균·표준편차·RMS·범위 12개와 XYZ 벡터 크기 통계 4개. HRV 특징은 아님 |
| 학습 설정 | Seed 42, Adam, learning rate `1e-3`, batch size 8, 최대 12 epochs, train SCD 비율 기반 BCE 가중치 |
| 모델 선택·임계값 | Validation AP 기준 checkpoint 및 후보 선택; Sensitivity/Specificity 임계값은 validation Youden J 기준 |
| Test 정책 | validation 최고 AP인 MIL-TCN만 한 번 평가; test 값으로 모델이나 threshold를 다시 선택하지 않음 |
| 실행 환경 | Python 3.12.3, PyTorch 2.14.0+cu126, CUDA 12.6, NVIDIA GeForce RTX 4060 Laptop GPU |
| 다운로드/파일 검증 | 100개 pair와 3 lead·200 Hz 확인. 기존 미완성 파일은 이어받음. 환자 ID는 2 lead, 환자 ID는 원격 header 404로 제외하고 대체 환자를 선택 |

정확한 환자 ID, 라벨, split, 파일 크기는 [다운로드 manifest](ecg_100_download_manifest.json)에 기록했다. 데이터는 비무작위 편의 표본이며 분석 코호트 전체를 대표하지 않는다.

## 3. Validation 결과

Validation: `N=15`, `SCD=4`. AP는 Average Precision이며 Sensitivity/Specificity는 각 모델의 validation Youden J 임계값 기준이다. 학습 시간은 12 epoch 이하의 단회 학습 시간이며 반복 평균이 아니다.

| 모델 | 입력 계열 | AUROC ↑ | AP ↑ | Sensitivity ↑ | Specificity ↑ | Brier ↓ | 학습 시간 (초) | CUDA peak (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **MIL-TCN** | 구간 통계 특징 | **0.955** | **0.917** | **1.000** | 0.818 | 0.217 | 0.57 | 20.83 |
| Segment Attention MIL | 구간 통계 특징 | 0.909 | 0.875 | 0.750 | **1.000** | 0.226 | 0.46 | 20.68 |
| 1D-CNN | 원시 파형 | 0.795 | 0.827 | 0.750 | **1.000** | 0.254 | 1.90 | 51.24 |
| Patch ECG Transformer | 원시 파형 | 0.682 | 0.671 | 0.500 | **1.000** | **0.208** | 1.68 | 32.86 |
| CNN-LSTM | 원시 파형 | 0.682 | 0.594 | 0.500 | 0.909 | 0.252 | 0.71 | 51.42 |
| CNN-TST | 원시 파형 | 0.523 | 0.546 | 0.500 | 0.909 | 0.234 | 1.28 | 51.89 |
| Raw ECG MIL | 원시 파형 | 0.409 | 0.273 | 1.000 | 0.182 | 0.244 | 0.57 | 51.78 |

모델 입력이 통계 특징과 원시 파형으로 다르므로, 이 표는 Holter 기반 파이프라인의 탐색 비교다. 같은 입력 표현에서 아키텍처만 바꾼 통제 실험은 아니다.

## 4. Validation에서 선택된 모델의 Test 결과

Validation AP 최고인 MIL-TCN을 선택했다. Threshold `0.52217`은 validation에서 정한 값이다.

| 모델 | Test N / SCD | AUROC ↑ | AP ↑ | Sensitivity | Specificity | Brier ↓ |
|---|---:|---:|---:|---:|---:|---:|
| MIL-TCN | 13 / 2 | 0.409 | 0.208 | 0.000 | 0.818 | 0.229 |

Validation과 test 결과가 크게 달라 작은 표본에서의 순위 불안정을 보여준다. 이 결과를 근거로 MIL-TCN을 최종 또는 임상 적용 모델로 확정하지 않는다. Brier는 확률 오차의 요약치이며, calibration curve·기울기·절편은 별도로 평가하지 않았다.

## 5. 모델 선정 기록

| 데이터 입력 | 현재 우선 후보 | 비교 후보 | 선정 상태와 근거 |
|---|---|---|---|
| 임상 정형 데이터 | CatBoost | Random Forest | 임상 모델끼리 별도로 비교. 자세한 수치와 한계는 [임상 결과](preprocessed_data_model_comparison.md) 참고 |
| Holter 구간 통계 특징 | MIL-TCN | Segment Attention MIL | Validation AP/AUROC가 가장 높아 다음 반복 우선 실험 후보. Test 성능이 낮아 최종 모델은 아님 |
| Holter 원시 파형 | 1D-CNN | Patch ECG Transformer, CNN-TST, CNN-LSTM | Raw 입력 모델 중 validation AP/AUROC가 가장 높음. 독립 코호트 검증 전 파일럿 후보 |

**발표용 표현:** “현재 임상 모델 후보는 CatBoost, Holter 통계 특징 모델의 탐색 후보는 MIL-TCN, 원시 파형 모델의 탐색 후보는 1D-CNN이다. 데이터 계열별로 후보를 나눠 평가하며, ECG test 결과와 표본 규모가 충분하지 않아 전체 최종 모델은 추가 검증 후 정한다.”

## 6. 한계와 다음 검증

1. 전체 MUSIC 임상 split 552명 중 ECG가 있는 100명을 사용했으며, 현재 파일 선택은 무작위 코호트 표본으로 보장되지 않는다.
2. Train/validation/test의 SCD 양성은 각각 13/4/2명이다. 특히 test 2건으로는 지표 신뢰구간과 임상 민감도를 안정적으로 추정할 수 없다.
3. 50명 파일럿에서 test 환자 7명을 이미 확인했기 때문에 이번 100명 test는 완전히 새로운 독립 holdout이 아니다. 후속 결과는 탐색적 근거로 취급하고, 최종 보고에는 새로운 사전 고정 test 또는 외부 코호트가 필요하다.
4. 24시간 파형 전체가 아니라 기록 앞 100초만 평가했다. segment 위치·길이·아티팩트 처리의 임상 적합성도 확정되지 않았다.
5. 고정 seed 한 번, 제한된 구조와 최대 12 epoch만 비교했다. 교차검증, 여러 seed, hyperparameter search, 신뢰구간, calibration curve는 수행하지 않았다.
6. 다음 비교는 환자 단위 stratified 반복 검증으로 MIL-TCN과 Segment Attention MIL(통계 특징 계열), 1D-CNN과 CNN-TST(원시 파형 계열)를 별도 비교한다. 동일 ECG 환자에서 Clinical-only / ECG-only / Fusion 결과를 보고 싶으면 세 입력 조합 모두 같은 환자 집단과 split을 사용한다.

## 7. 재현

저장소 루트 기준:

```sh
PYTHONPATH=Members/sang /home/sang/hf_pytorch/.venv/bin/python \
  Members/sang/scripts/evaluate_ecg_architectures.py \
  --data-dir Members/sang/datas/ECG_tem/ECG_tem_clean \
  --splits-dir Members/sang/datas/clincal/splits \
  --epochs 12 --batch-size 8 --seed 42
```

실행 결과 원본 JSON: [`ecg_100_model_comparison.json`](ecg_100_model_comparison.json)

평가 코드: [`evaluate_ecg_architectures.py`](../scripts/evaluate_ecg_architectures.py)
