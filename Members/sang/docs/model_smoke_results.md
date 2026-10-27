# 4. 더미 데이터 실행 기록

실행일: 2026-10-23 (Asia/Seoul)

## 4.1 테스트 목적

실제 SCD 예측 성능을 평가하지 않고, 합성 더미 입력에 대해 모델의 학습·추론 API와 입력/출력 형태가 정상 동작하는지 확인했다.

**검증 범위**

- 입력 및 출력 shape
- ML `fit` / `predict_proba` API
- 딥러닝 Forward Pass
- `BCEWithLogitsLoss` 계산과 Backward Pass
- 출력, loss, gradient의 NaN/Inf 여부
- CPU 및 GPU 실행 여부
- 실행 시간과 GPU peak allocated memory

## 4.2 현재 테스트 결과

| 모델 | 더미 입력 형태 | 실행 결과 | 확인 내용 |
|---|---|---|---|
| Random Forest | `(256, 24)` | PASS | CPU, 확률 `(256,)`, 범위 `0.024–0.784`; fit+predict 0.167초 |
| XGBoost | `(256, 24)` | PASS | CPU, 확률 `(256,)`, 범위 `0.024–0.812`; fit+predict 1.016초 |
| CatBoost | `(256, 24)` | PASS | CPU, 확률 `(256,)`, 범위 `0.016–0.847`; fit+predict 0.108초. 범주형 입력은 미시험 |
| MLP | `(256, 24)` | PASS* | CPU, 확률 `(256,)`, 범위 `0.025–0.828`; fit+predict 0.027초 |
| 1D-CNN | `(8, 3, 2500)` | PASS | CPU/CUDA forward, BCE, backward; 출력 `(8, 1)`; CUDA 1.103초, CPU 0.148초 |
| ECG Transformer | `(8, 3, 2500)` | PASS | CPU/CUDA forward, BCE, backward; 출력 `(8, 1)`; CUDA 0.348초, CPU 0.129초 |
| Segment Attention MIL | `(8, 12, 16)` | PASS | CPU/CUDA forward, BCE, backward; 출력 `(8, 1)`; CUDA 0.507초, CPU 0.060초; CUDA peak 16.30 MiB |
| MIL–TCN | `(8, 12, 16)` | PASS | CPU/CUDA forward, BCE, backward; 출력 `(8, 1)`; CUDA 0.377초, CPU 0.130초; CUDA peak 16.37 MiB |
| Raw ECG MIL (1D-CNN + MIL) | `(2, 4, 3, 2500)` | PASS | CPU/CUDA forward, BCE, backward; 출력 `(2, 1)`; CUDA 0.152초, CPU 0.019초; CUDA peak 20.62 MiB |

모든 ML 확률 출력은 `(256,)`, 유한값이며 `[0,1]` 안에 있었다. CNN과 Transformer는 CPU와 CUDA 모두 `(8,1)` logits를 출력했고 loss 및 gradient가 유한함을 확인했다.

추가된 MIL 계열 모델도 CPU와 CUDA 모두 출력 shape, BCE loss, finite output/loss/gradient 및 Backward를 통과했다. 구간 특징 더미 shape `(8,12,16)`은 batch 8, 환자당 12개 구간, 구간당 16개 특징이라는 smoke test용 가정이다. Raw ECG MIL 입력 `(2,4,3,2500)`은 batch 2, 환자당 4개 구간, 3개 리드, 구간당 2500 sample이라는 가정이다. 실제 MUSIC 데이터의 입력 형식을 의미하지 않는다.

`PASS*`인 MLP는 `max_iter=30`에서 `ConvergenceWarning`이 발생했다. 더미 입력 학습 호출과 입출력 검증은 성공했으나 수렴을 뜻하지 않으므로, 실제 모델 학습 전에는 반복 횟수·수렴 조건을 별도로 결정해야 한다.

**기록 기준**

- PASS: 해당 더미 실행 검증 성공
- PASS*: API 및 입출력 검증 성공, 수렴 경고 확인 필요
- FAIL: 실행 또는 검증 실패
- SKIP/미실행: 테스트하지 않음

※ 실행 시간은 한 번의 fit+predict 또는 forward+loss+backward 측정이며 워밍업이나 반복 평균이 아니다. 성능 벤치마크로 비교하면 안 된다. 확률 평균과 loss도 모델 성능 비교 지표가 아니다.

※ CatBoost 범주형 입력, 실제 임상 결측값 처리, XGBoost/CatBoost GPU 옵션은 시험하지 않았다.

## 4.3 테스트 환경 기록

| 항목 | 현재 상태 |
|---|---|
| OS | WSL2 Ubuntu 24.04, kernel `6.18.33.2-microsoft-standard-WSL2`, x86_64 |
| Python 버전 | 3.12.3 |
| PyTorch 버전 | 2.14.0+cu126, CUDA runtime 12.6 |
| scikit-learn 버전 | 1.9.0 |
| XGBoost 버전 | 3.4.1 |
| CatBoost 버전 | 1.2.10 |
| 실행 장치 | 전통 ML/MLP는 CPU; CNN/Transformer는 CPU 및 CUDA |
| NVIDIA GPU 테스트 | PASS, NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB) |
| NVIDIA Driver | Windows driver 610.55, KMD 610.78, CUDA UMD 13.3 |
| Random Seed | NumPy/sklearn/XGBoost/CatBoost/PyTorch `42` |
| 실행 코드 경로 | [`scripts/smoke_model_api.py`](../scripts/smoke_model_api.py) |
| 실행 시간 | LR 0.022s; RF 0.167s; XGBoost 1.016s; CatBoost 0.108s; MLP 0.027s; CNN CUDA 1.103s / CPU 0.148s; Transformer CUDA 0.348s / CPU 0.129s |
| GPU 메모리 | CNN 20.5 MiB, Transformer 18.3 MiB peak allocated; 전통 ML은 미측정/해당 없음 |

새 MIL 계열 smoke test는 [`scripts/smoke_mil_models.py`](../scripts/smoke_mil_models.py)로 실행했고, 모델 정의는 [`model/DL/models.py`](../model/DL/models.py)에 추가했다. 실행 명령은 `PYTHONPATH=/tmp/scd-smoke-cuda:Members/sang python Members/sang/scripts/smoke_mil_models.py`다. CUDA 접근 가능한 권한 실행 환경에서 NVIDIA GeForce RTX 4060 Laptop GPU를 확인했다.

### 재현 조건 및 한계

임상용 합성 데이터는 `make_classification(n_samples=256, n_features=24, weights=[0.8, 0.2], random_state=42)`로 생성했다. ECG 입력은 `(8,3,2500)` float tensor이다. NumPy와 PyTorch seed는 42로 설정했고 모델별 seed도 고정했다. GPU는 Windows에서 정상 인식되지만 이 세션의 일반 샌드박스는 WSL GPU 장치 접근을 제한하므로, CUDA 검증은 승인된 권한 실행 경로에서 수행했다.

측정에는 scikit-learn을 `dnn_tf_ex`의 기존 가상환경에서, CatBoost와 CUDA PyTorch를 `/tmp` 격리 설치에서 사용했다. XGBoost는 로컬 uv cache를 사용했다. 따라서 이 기록은 **실제 모델 API 실행 결과**지만 저장소 Docker 공통환경이나 `Members/sang`의 고정 lock을 사용한 재현성 검증은 아니다. 해당 환경 기준 재현은 Docker socket 권한이 해결된 뒤 수행해야 한다.

CUDA 테스트에서는 `torch.cuda.is_available() == True`였다. GPU 메모리는 PyTorch의 `max_memory_allocated()` 기준이며, 드라이버가 표시하는 전체 VRAM 사용량과는 다른 값이다.

## 4.4 환자 ID 실제 ECG 파형 기반 CNN/Transformer smoke test

실행일: 2026-10-25 (Asia/Seoul)

더미 신호 대신 MUSIC `환자 ID`의 `.dat`/`.hea` 쌍을 사용해 1D-CNN과 ECG Transformer의 실제 파형 입력 경로를 확인했다. 이 검증은 **한 환자의 파일 앞부분에서 연속 8개 12.5초 구간(총 100초)**만 사용했다. 헤더에는 3채널 X/Y/Z, 200 Hz, 채널당 16,460,600 samples가 기록되어 있고, `.dat` 크기 98,763,600 bytes는 `samples × 3 × 2 bytes`와 일치했다. 전체 파형을 읽어 WFDB format 16의 invalid sentinel(-32768) 개수를 확인했으며 0개였다.

| 모델 | 실제 파형 입력 | 환자 출력 | sigmoid 출력 | 실행 결과 | GPU Forward + Loss + Backward | PyTorch peak allocated | 비고 |
|---|---|---|---:|---|---:|---:|---|
| 1D-CNN | `(8, 3, 2500)` | `(1, 1)` | `0.493` | PASS | 0.412초 | 20.55 MiB | 8개 segment logits 평균 집계, CUDA |
| ECG Transformer | `(8, 3, 2500)` | `(1, 1)` | `0.411` | PASS | 0.116초 | 18.35 MiB | 8개 segment logits 평균 집계, CUDA |

입력은 WFDB format 16의 little-endian int16/sample-major 채널 데이터를 읽고, 헤더의 gain 100 및 ADC zero 0을 적용한 다음 각 12.5초 구간·채널별 z-score 표준화를 했다. format, gain, offset, 신호 스케일 변환은 [WFDB 헤더 규격](https://wfdb.io/spec/header-files.html)과 [신호 파일 규격](https://wfdb.io/spec/signal-files.html)을 참고했다. 헤더에 물리 단위는 기재되어 있지 않다. 이 표준화는 **smoke test 전용 임시 처리**이며 최종 학습 전처리로 확정한 것이 아니다.

파형 파일에는 라벨이 없지만, 임상 `train.csv`에서 `환자 ID`의 `SCD=0`을 ID로 연결했다. 환자 라벨 하나와 비교하기 위해 8개 segment logits를 평균해 `(1,1)` 환자 점수를 만들었다. 환자 ID은 train split에 포함되어 있고 모델은 학습된 체크포인트 없이 seed 42로 초기화했으므로, 출력 확률과 loss는 일반화 성능이나 위험도로 해석할 수 없다. AUROC/AUPRC 및 모델 선택 결과는 산출하지 않았다.

| 검증 항목 | 결과 | 근거 |
|---|---|---|
| 실제 파일/헤더 크기 정합 | PASS | 헤더 기대 크기와 `.dat`가 모두 98,763,600 bytes |
| 실제 ECG segment 로딩 | PASS | 첫 8개 segment의 입력 shape `(8,3,2500)` |
| 환자 ID/라벨 연결 | PASS | `환자 ID`, train split, SCD `0` |
| 1D-CNN Forward, BCE, Backward | PASS | segment 출력 `(8,1)`을 환자 출력 `(1,1)`로 평균 집계; finite logits/loss/gradient |
| ECG Transformer Forward, BCE, Backward | PASS | segment 출력 `(8,1)`을 환자 출력 `(1,1)`로 평균 집계; finite logits/loss/gradient |
| CUDA 실행 | PASS | NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.14.0+cu126, CUDA runtime 12.6 |
| 임상 성능 평가 | 미실행 | 단일 train 환자, 모델 미학습, 성능지표 산출 불가 |

재현 명령 (저장소 루트, 해당 smoke-test 패키지 경로가 준비된 환경):

```sh
PYTHONPATH=/tmp/scd-smoke-cuda:/tmp/scd-smoke-extra:/home/sang/.cache/uv/archive-v0/TIDVvkcYCJEzCu0b:Members/sang \
  python Members/sang/scripts/smoke_p0001_ecg_models.py
```

실행 코드는 [`smoke_p0001_ecg_models.py`](../scripts/smoke_p0001_ecg_models.py)다. 시간은 워밍업·반복 평균이 아닌 단일 forward/loss/backward 측정이며, GPU 메모리는 해당 모델 실행 중 PyTorch peak allocated 값이다. 설치 위치 `/tmp` 및 사용자 캐시의 패키지 경로를 사용한 실행이므로, 저장소 Docker 환경에서의 재현성 검증은 별도다.

## 4.5 환자 ID 구간 기반 MIL 모델 smoke test

같은 환자 ID에서 앞의 8개 연속 구간(각 12.5초)을 사용해 Segment Attention MIL, MIL–TCN, Raw ECG MIL도 추가 확인했다. MIL은 여러 구간을 한 환자 입력으로 묶으므로 feature 모델은 `(환자 1명, 구간 8개, 특징 16개)`, Raw ECG MIL은 `(환자 1명, 구간 8개, 리드 3개, 샘플 2500개)`를 입력한다.

| 모델 | 환자 ID 입력 | 출력 | sigmoid 출력 | 실행 결과 | GPU Forward + Loss + Backward | PyTorch peak allocated | 비고 |
|---|---|---|---:|---|---:|---:|---|
| Segment Attention MIL | `(1, 8, 16)` | `(1, 1)` | `0.480` | PASS | 0.044초 | 16.67 MiB | 유한 logits/loss/gradient |
| MIL–TCN | `(1, 8, 16)` | `(1, 1)` | `0.539` | PASS | 0.019초 | 16.76 MiB | 구간 순서 유지, 유한 logits/loss/gradient |
| Raw ECG MIL | `(1, 8, 3, 2500)` | `(1, 1)` | `0.530` | PASS | 0.002초 | 20.80 MiB | 파형 구간별 CNN 인코딩 후 attention pooling |

Segment Attention MIL과 MIL–TCN용 구간 특징 16개는 **이번 입력 연결 테스트를 위해 임시로 구성**했다: 리드별 평균·표준편차·RMS·범위 12개, 3리드 벡터 크기의 평균·표준편차·RMS·범위 4개. 이 특징은 HRV나 임상적으로 검증된 ECG 특징 목록이 아니며, 실제 학습에 사용할 확정 전처리가 아니다. Raw ECG MIL은 앞서 사용한 구간별 z-score 파형을 입력으로 썼다.

이번 세 모델도 체크포인트 없이 초기화 가중치로 실행했다. 환자 ID에 매칭된 실제 SCD 라벨 0을 BCE/Backward 경로 확인에 사용했다. 다만 환자 ID은 train split의 환자 한 명일 뿐이고 모델이 학습되지 않았으므로, 보고된 loss와 출력은 성능 또는 위험도로 해석할 수 없다. 따라서 이 테스트는 **실제 ECG와 라벨을 연결한 환자 단위 입력·모델 실행 확인**까지만 PASS이며 임상 성능 평가는 미실행이다.

세 모델 모두 재현 스크립트 [`smoke_p0001_ecg_models.py`](../scripts/smoke_p0001_ecg_models.py)의 4.4와 동일한 명령으로 실행된다. 단일 환자 파일 하나로는 학습/검증 분할이나 AUROC/AUPRC를 계산할 수 없다. 성능 비교 단계에는 다수 환자의 ECG 구간, 환자별 SCD 라벨, 사전에 합의된 구간 길이·특징 추출, 환자 단위 train/validation/test 분할이 필요하다.

## 4.6 CNN–LSTM과 CNN–TST 비교 프로토타입

첨부된 설계의 직접 비교를 위해 같은 구간 CNN 인코더 뒤에 LSTM 또는 time-series Transformer encoder를 둔 소형 patient-level prototype을 추가했다. 환자 ID의 같은 8개 실제 파형 구간과 실제 `SCD=0` train label을 사용했다.

| 모델 | 입력 | 환자 출력 | sigmoid 출력 | 결과 | CUDA Forward + BCE + Backward | Peak allocated |
|---|---|---|---:|---|---:|---:|
| CNN–LSTM | `(1,8,3,2500)` | `(1,1)` | 0.527 | PASS | 0.066초 | 27.66 MiB |
| CNN–TST | `(1,8,3,2500)` | `(1,1)` | 0.405 | PASS | 0.008초 | 20.97 MiB |

이는 논문 저자의 TST 원본 구현이나 성능 재현이 아니라, 동일 구조의 CNN 구간 인코더를 각각 두고 집계부를 달리한 작은 실험 prototype이다. 두 모델은 CNN 가중치를 공유하지 않는다. 환자 ID 한 건과 초기화 모델만으로 두 구조의 정확도나 우열을 판단할 수 없다. 모델 정의는 [`model/DL/models.py`](../model/DL/models.py)에 있다.
