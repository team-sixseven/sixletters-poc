# 추천 실험 6개: 실행 결과와 진행 상태

이 문서의 실제 ECG feature 결과는 공통 prepared bundle의 별도 비교다. 최신 사용자 제공 clinical split(552명) 기준 모델 결과와 추천은 [전처리 데이터 모델 비교](preprocessed_data_model_comparison.md)를 참고한다. 현재 원시 ECG는 환자 ID pair 한 개만 있다. Transformer 후보 조사와 환자 ID 실행 비교는 [Transformer ECG 모델 비교](transformer_ecg_model_comparison.md)를 참고한다.

작성일: 2026-10-23

## 실험 요약

| 순서 | 실험 | 입력 데이터 | 모델 | 목적 | 상태 |
|---:|---|---|---|---|---|
| 1 | Clinical Baseline | 임상 변수 64개 | XGBoost | 임상 기반 기준 성능 | 실행 완료 |
| 2 | ECG Features Baseline | 환자별 ECG 정량지표 5개 | XGBoost | ECG 지표만의 예측력 확인 | 실행 완료 |
| 3 | Clinical + ECG Fusion | 임상 64개 + ECG 5개 | XGBoost | 두 입력 그룹의 결합 효과 확인 | 실행 완료 |
| 4 | Segment MIL | Holter 구간별 특징 | Attention MIL | 중요한 구간의 집계 | 환자 ID 임시 16개 통계 특징 smoke PASS; 성능 미평가 |
| 5 | MIL–TCN | 시간순 Holter 구간별 특징 | TCN + Attention MIL | 시간적 변화 학습 효과 | 환자 ID 임시 16개 통계 특징 smoke PASS; 성능 미평가 |
| 6 | Raw ECG Model | Holter 원시 파형 | 1D-CNN + MIL | 수작업 특징 없이 파형 학습 | 환자 ID 실제 파형 smoke PASS; 성능 미평가 |

## 1–3번 실제 데이터 결과

| 실험 | 입력 데이터 | N | SCD 수 | AUROC | AUPRC (AP) | Brier | 학습 시간 | 비고 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Clinical Baseline | Clinical (64 features) | 144 | 14 | 0.713 | 0.194 | 0.087 | 0.134 s | XGBoost, CPU |
| ECG Features Baseline | ECG features (5) | 144 | 14 | 0.505 | 0.116 | 0.093 | 0.025 s | XGBoost, CPU |
| Clinical + ECG Fusion | Clinical + ECG (69) | 144 | 14 | 0.718 | 0.186 | 0.088 | 0.081 s | XGBoost, CPU |
| Segment MIL | 합성 구간 특징 | — | — | — | — | — | — | 실제 구간 입력 미제공; 더미 Forward/Backward PASS |
| MIL–TCN | 합성 시간순 구간 특징 | — | — | — | — | — | — | 실제 구간 입력 미제공; 더미 Forward/Backward PASS |
| Raw ECG Model | 합성 ECG 파형 조각 | — | — | — | — | — | — | 실제 파형 미제공; 더미 Forward/Backward PASS |

### 결과 해석 범위

- 평가 데이터는 공통 train/validation/test 환자 분할 중 validation 144명(SCD 14명)이다. train은 671명(SCD 65명)이며 test 146명은 사용하지 않았다.
- 세 실험은 동일한 XGBoost 설정(seed 42, 100 estimators, depth 3, learning rate 0.05), train-only 결측치 대치 및 같은 환자 분할을 사용한다. 범주형 변수는 train 데이터에서 one-hot 변환 규칙을 학습했다.
- 임상 입력은 준비된 MUSIC 스키마의 64개다. 제안 초안의 “임상 24개 변수”에 해당하는 확정 변수 목록은 저장소 자료에서 찾지 못해 임의 선정하지 않았다.
- ECG 입력 5개는 `ecg_rhythm`, `ecg_q_waves`, `ecg_qrs_duration_ms`, `ecg_qt_interval_ms`, `ecg_average_rr_ms`다. 이는 환자별 기존 정량 특징이며 구간별 HRV 또는 원시 파형이 아니다.
- Clinical baseline과 Fusion의 AUROC는 각각 0.713, 0.718이다. 반면 Fusion의 AP(0.186)는 Clinical(0.194)보다 낮다. 작은 validation 양성 수와 단일 split 결과이므로 결합 효과가 개선됐다고 단정할 수 없다.
- 이 평가의 endpoint는 관찰 추적 기간 중 SCD 여부다. 고정 4년 위험도나 생존분석 결과가 아니다. split/endpoint 세부는 [실제 모델 결과 기록](actual_model_results.md)을 참고한다.

## 4–6번 실행에 필요한 입력

이 문서의 초기 작성 시점에는 `환자 ID.dat/.hea` 한 쌍만 있어 입력 smoke만 가능했다. 이후 Holter 파형 100명으로 MIL/TCN/CNN/Transformer를 비교했다. 최신 파일럿 성능은 [Holter ECG 100명 모델 비교](ecg_100_model_comparison.md)를 참고한다.

후속 성능 평가에서는 더 많은 ECG 환자, 환자 ID/기록 ID, SCD label 연결 규칙, 확정된 구간 시간·전처리·특징 정의가 필요하다. 현재 100명 파일럿도 표본이 제한되고 이전 파일럿 test 환자를 포함하므로 validation/test 성능은 탐색 결과로만 본다. 다음 실험은 새 patient-level split 또는 독립 코호트를 사용해야 한다.

### 합성 데이터 모델 smoke test

4–6번 입력을 흉내 낸 합성 Tensor로 입력/출력 shape, Forward, BCE loss, Backward gradient, NaN/Inf를 검사했다. 이 결과는 모델 실행 구조 확인일 뿐 실제 ECG에 대한 성능 결과가 아니다.

| 모델 | 더미 입력 | 출력 | CPU Forward+Loss+Backward | CUDA Forward+Loss+Backward | CUDA peak allocated |
|---|---|---|---:|---:|---:|
| Segment MIL | `(8, 12, 16)` | `(8, 1)` | 0.060 s | 0.507 s | 16.30 MiB |
| MIL–TCN | `(8, 12, 16)` | `(8, 1)` | 0.130 s | 0.377 s | 16.37 MiB |
| Raw ECG Model | `(2, 4, 3, 2500)` | `(2, 1)` | 0.019 s | 0.152 s | 20.62 MiB |

- CPU와 CUDA에서 세 모델 모두 Forward, BCE loss, Backward 및 finite output/loss/gradient 검사를 통과했다.
- CUDA 환경은 PyTorch 2.14.0+cu126, CUDA runtime 12.6, NVIDIA GeForce RTX 4060 Laptop GPU였다. CUDA 시간은 단일 실행이며 첫 실행 비용을 포함한다. 워밍업/반복 평균이 아니다.
- CUDA peak allocated는 PyTorch가 측정한 peak tensor allocation이며 전체 GPU VRAM 점유량이 아니다.
- 구간 특징은 더미에서 12개 segment × 16개 feature로 가정했다. 원시 파형은 4개 segment × 3 lead × 2500 samples로 구성했다. 실제 데이터의 segment 길이, 채널 및 특징 수를 뜻하지 않는다.
- 재현 명령: `PYTHONPATH=/tmp/scd-smoke-cuda:Members/sang python Members/sang/scripts/smoke_mil_models.py` (CUDA 접근이 허용된 실행 환경에서). CUDA 접근 불가 시 같은 명령이 CPU로 자동 대체된다.

## 재현 명령

저장소 루트에서 공통 prepared bundle과 프로젝트 Python 환경을 사용한다.

```sh
PYTHONPATH=Global/src:/tmp/scd-smoke-extra:/home/sang/.cache/uv/archive-v0/TIDVvkcYCJEzCu0b \
  /home/sang/dnn_tf_ex/.venv/bin/python Members/sang/scripts/evaluate_feature_sets.py \
  --data-dir Global/data/prepared/music-scd-basic-v1
```

학습 시간은 단일 CPU 실행에서 측정한 fit 시간이며, 반복 평균이나 성능 벤치마크가 아니다. validation 평가용 결과이며 held-out test 성능을 나타내지 않는다.
