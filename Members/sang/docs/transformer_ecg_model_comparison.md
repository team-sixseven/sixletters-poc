# ECG 시계열 Transformer 후보 조사 및 실파형 비교

작성일: 2026-10-25 (Asia/Seoul)

## 요약 및 선정안

현재 프로젝트의 첫 Transformer 비교 대상으로 **CNN–TST Encoder**를 추천한다. `CNN–LSTM`과 동일 구조의 1D-CNN 구간 인코더를 쓰고, 뒤의 시간 집계부만 LSTM과 TST Encoder로 달리하면 비교하기 쉽다. 별도 시계열 분류 모델로 **ConvTran**을 다음 후보로 둔다. Medformer와 FormerTime은 다중 채널·다중 시간 규모 또는 장시간 처리 가설이 필요할 때 후속 실험으로 검토한다.

이 선정은 논문에서 보고한 다른 데이터셋의 성능 순위가 아니다. SCD 위험도, MUSIC Holter 코호트에서 후보를 실행하고 비교한 결과를 근거로 한 **실험 우선순위**다.

후속 100명 비교에서는 원시 파형 기준 Patch ECG Transformer validation AP 0.671, CNN–TST AP 0.546을 기록했고, 1D-CNN의 AP 0.827보다 낮았다. 100명 결과와 제한은 [Holter ECG 100명 모델 비교](ecg_100_model_comparison.md)를 기준으로 한다. 이 문서의 환자 ID smoke와 이전 50명 파일럿 기록은 당시 실행 내용을 보존한다.

| 우선순위 | 후보 | 입력과 주요 아이디어 | 프로젝트 적용 판단 |
|---|---|---|---|
| 1 | CNN–TST Encoder | CNN이 파형을 시간 구간 표현으로 바꾸고 TST Encoder가 구간 순서를 학습 | 로컬 소형 prototype을 환자 ID에서 실행함. CNN–LSTM과 CNN 및 구간을 공유해 집계부만 비교 |
| 2 | ConvTran | 합성곱 기반 시계열 임베딩, 시간 절대 위치 인코딩(tAPE), 상대 위치 인코딩(eRPE) | CNN을 포함한 독립 시계열 분류 후보. 기존 CNN에 encoder만 단순 부착하지 말고 별도 모델로 비교 |
| 3 | Medformer | 의료 시계열용 다중 granularity patch와 채널 간 관계 학습 | 3채널 XYZ 및 시간 규모 가설을 검증할 후속 후보. 구현·튜닝 부담이 커 첫 비교에서는 보류 |
| 4 | FormerTime | 계층적 다중 시간 규모 표현으로 시계열 분류 | 전체 Holter를 긴 구간 계층으로 처리해야 할 때 후보. 첫 CNN–LSTM 비교의 필수 구성은 아님 |

### 논문 근거와 적용 범위

- **TST**: Transformer encoder를 이용한 다변량 시계열 표현 학습 프레임워크이며, 분류·회귀 및 자기지도 사전학습을 다룬다. CNN 출력 token을 입력하는 비교 모델로 활용할 수 있다. [KDD 2021 논문](https://doi.org/10.1145/3447548.3467401)
- **ConvTran**: 시계열 분류를 대상으로 convolution 기반 입력 표현과 tAPE/eRPE 위치 인코딩을 결합했다. 논문 평가는 다변량 시계열 벤치마크에 대한 것이며 SCD 위험 예측 검증은 아니다. [공식 논문](https://link.springer.com/article/10.1007/s10618-023-00948-2), [저자 구현](https://github.com/Navidfoumani/ConvTran)
- **Medformer**: 의료 시계열 분류를 위해 cross-channel patching, 다중 granularity embedding, intra/inter-granularity attention을 제안했고 ECG 포함 의료 데이터셋을 평가했다. 이는 MUSIC의 장기 SCD endpoint에 대한 성능 증거가 아니다. [NeurIPS 2024 논문 및 구현 링크](https://papers.neurips.cc/paper_files/paper/2024/hash/3fe2a777282299ecb4f9e7ebb531f0ab-Abstract-Conference.html)
- **FormerTime**: 계층적 다중 규모 표현으로 시계열 분류를 다루는 모델이다. 전체 Holter를 긴 환자 단위 시퀀스로 다룰 때 참고할 수 있으나, 현재 확보한 한 환자만으로는 이 구조의 성능 이점을 시험할 수 없다. [논문](https://arxiv.org/abs/2302.09818)

## 환자 ID 실제 데이터 현황

| 항목 | 확인 결과 |
|---|---|
| ECG 파일 | `환자 ID.dat` 98,763,600 bytes, `환자 ID.hea` 105 bytes |
| 헤더 입력 | 3채널 X/Y/Z, 200 Hz, 채널당 16,460,600 samples (약 22.86시간) |
| 임상 ID 연결 | `환자 ID`이 임상 `train.csv`에 한 건 존재, `SCD=0` |
| 실험에 사용한 파형 | 기록 앞부분의 연속 8개 12.5초 구간, 총 100초 |
| 데이터 분할 | 환자 ID은 train 환자. validation/test 평가 표본으로 사용하지 않음 |
| 추가 코호트 규모 점검 | clinical split 552명 중 공개 목록에서 525명 pair 확인; 전체 파일은 받지 않고 50명 pair만 로컬 확보 |

코호트 규모는 [PhysioNet MUSIC Holter 파일 목록](https://physionet.org/content/music-sudden-cardiac-death/1.0.1/Holter_ECG/)과 공개 객체 목록을 대조했다. 후속 파일럿에서 전체 파형을 받지 않고 50명 pair만 확보했다. split coverage와 실제 비교 수치는 [50명 ECG 실험 케이스 보고서](experiment_cases_50.md) 및 [원시 결과 JSON](ecg_50_model_comparison.json)에 기록했다.

## 실제 파형 실행 결과

다음 결과는 모델 체크포인트를 학습한 성능 비교가 아니라, 실제 환자 ID 파형과 매칭된 환자 라벨을 사용해 환자 단위 출력, BCE, Backward가 가능한지 확인한 GPU smoke test다. 모든 모델은 초기화 가중치이며, 시간은 워밍업/반복 평균 없는 단회 측정이다.

| 로컬 모델 구현 | 입력 shape | 환자 출력 shape | 초기화 모델 sigmoid 출력 | Forward + BCE + Backward | CUDA peak allocated | 결과 |
|---|---|---|---:|---:|---:|---|
| 1D-CNN | `(8,3,2500)` | `(1,1)` | 0.493 | 0.412초 | 20.55 MiB | PASS |
| Patch ECG Transformer | `(8,3,2500)` | `(1,1)` | 0.411 | 0.116초 | 18.35 MiB | PASS |
| Segment Attention MIL | `(1,8,16)` | `(1,1)` | 0.480 | 0.044초 | 16.67 MiB | PASS |
| MIL–TCN | `(1,8,16)` | `(1,1)` | 0.539 | 0.019초 | 16.76 MiB | PASS |
| Raw ECG MIL | `(1,8,3,2500)` | `(1,1)` | 0.530 | 0.002초 | 20.80 MiB | PASS |
| CNN–LSTM | `(1,8,3,2500)` | `(1,1)` | 0.527 | 0.066초 | 27.66 MiB | PASS |
| CNN–TST | `(1,8,3,2500)` | `(1,1)` | 0.405 | 0.008초 | 20.97 MiB | PASS |

`1D-CNN`과 Patch ECG Transformer는 구간별 logit 8개를 평균해 환자 점수 하나로 만들었다. CNN–LSTM과 CNN–TST는 동일한 작은 CNN segment encoder 후 시간 집계 방식을 달리한 로컬 prototype이며 출력은 환자 단위다. MIL 계열은 모델 내부에서 구간을 환자 출력 하나로 집계했다. Segment Attention MIL과 MIL–TCN의 16개 구간 특징은 테스트용 통계량(리드별 평균·표준편차·RMS·범위 12개, 3리드 벡터 크기 통계 4개)이며 HRV 특징 또는 확정된 임상 전처리가 아니다.

Patch ECG Transformer와 CNN–TST는 저장소의 작은 실험 prototype이다. CNN–TST는 CNN 구간 인코더 뒤에 sinusoidal 위치 정보를 포함한 Transformer encoder를 붙인 구성으로, 논문 저자의 TST 코드를 그대로 가져온 것이 아니며 CNN 가중치도 CNN–LSTM과 공유하지 않는다. **정식 TST, ConvTran, Medformer, FormerTime 구현의 결과와 혼동하지 않는다.** 후보별 현황은 다음과 같다.

| 연구 후보 | 문헌 조사 | 저장소 구현을 이용한 실제 환자 ID 비교 | 성능 평가 |
|---|---|---|---|
| TST / CNN–TST | 완료 | CNN–TST 소형 prototype 실행; 원 저자 TST 구현은 미연동 | 미실행 |
| ConvTran | 완료 | 미실행 (공식 구조 미연동) | 미실행 |
| Medformer | 완료 | 미실행 (공식 구조 미연동) | 미실행 |
| FormerTime | 완료 | 미실행 (계층 구조 미연동) | 미실행 |

## 해석과 한계

- 한 명의 SCD 음성 환자와 초기화된 모델로는 AUROC, AUPRC, Brier, calibration 또는 일반화 성능을 계산할 수 없다. 표시한 sigmoid 출력과 loss를 위험 예측값이나 후보 간 성능 순위로 해석하지 않는다.
- 환자 ID은 train split 사례이며 위 표의 smoke test만을 위한 단일 사례다. 이후 확보한 50명 실험은 별도 탐색 비교다. train 36명, validation 7명, test 7명이며 test SCD 양성이 1명뿐이라 일반화 성능을 판단할 수 없다.
- 전체 22.86시간 원파형을 그대로 self-attention에 넣는 대신, 구간별 CNN 임베딩을 만들고 시간순 token을 Transformer에 전달하는 계층 구조가 필요하다. 매우 긴 환자 시퀀스는 chunking/pooling 등으로 token 수를 줄여야 한다.
- SCD endpoint의 예측 시점과 horizon, 중도 검열 및 다른 사망 원인 처리 규칙이 확정되어야 모델 성능 지표를 의미 있게 비교할 수 있다.

### 다음 비교 단계

1. 환자 단위 split과 SCD label/추적 기간 정의를 확정한다.
2. 필요한 학습·검증 환자의 ECG를 확보하고 리드·샘플링·결측/아티팩트 처리 및 segment 길이를 고정한다.
3. 공통 CNN frontend를 둔 CNN–LSTM과 CNN–TST를 먼저 비교한다.
4. ConvTran을 독립 baseline으로 추가하고, 채널/다중시간 규모가 핵심 가설이면 Medformer를 후속 평가한다.
5. threshold 및 튜닝은 train/validation에서만 정하고, held-out test는 모델 선택 후 한 번 평가한다.

실제 환자 ID smoke test 재현 스크립트: [`smoke_p0001_ecg_models.py`](../scripts/smoke_p0001_ecg_models.py). 실행 환경은 Python 3.12.3, PyTorch 2.14.0+cu126, CUDA 12.6, NVIDIA GeForce RTX 4060 Laptop GPU다.
