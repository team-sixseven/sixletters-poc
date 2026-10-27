# 8.5 실험 결과 기록

이 문서는 `Global/data/prepared/music-scd-basic-v1`의 961명/64 clinical feature 이전 비교다. 현재 `Members/sang/datas/clincal/splits`의 552명 전처리 데이터 모델 비교는 [전처리 데이터 모델 비교](preprocessed_data_model_comparison.md)를 기준으로 한다.

실행일: 2026-10-23

아래 임상 수치는 이 문서 작성 당시의 실제 결과다. 이후 확보한 Holter 100명 7개 모델 비교 결과는 [Holter ECG 100명 모델 비교](ecg_100_model_comparison.md)에 별도로 기록했다. 데이터 계열과 평가 규모가 달라 임상과 Holter 수치를 직접 순위 비교하지 않는다.

다음 결과는 실제 MUSIC 데이터의 **validation split**으로 계산했다. 모델 학습에는 공통 train split을 썼으며 held-out test split은 확인하거나 점수화하지 않았다.

| 모델 | 입력 데이터 | N | SCD 수 | AUROC | AUPRC | Brier | 학습 시간 | 비고 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| XGBoost | Clinical (64 features) | 144 | 14 | 0.713 | 0.194 | 0.087 | 0.109 s | ML; 기존 AUROC 0.713 재현 |
| CatBoost | Clinical (64 features) | 144 | 14 | 0.660 | 0.154 | 0.087 | 0.319 s | ML; 범주형 입력 사용 |
| MLP | Clinical (64 features) | 144 | 14 | 0.651 | 0.149 | 0.104 | 0.419 s | DL |
| 1D-CNN | Holter raw ECG | — | — | — | — | — | — | Holter 100명 비교는 별도 [보고서](ecg_100_model_comparison.md); 이 임상 validation 표와 직접 비교하지 않음 |
| ECG Transformer / CNN-TST 등 | Holter raw ECG | — | — | — | — | — | — | Holter 100명 비교는 별도 [보고서](ecg_100_model_comparison.md); 이 임상 validation 표와 직접 비교하지 않음 |

## 평가 기준과 해석

- 기본 분할은 환자 단위 공통 split(seed 42)다: train 671명 / SCD 65명, validation 144명 / SCD 14명, test 146명 / SCD 15명. 이 표의 `N`과 `SCD 수`는 validation 기준이다.
- `Members/sang/datas/`의 clinical/ECG/labels CSV는 각각 992개 `patient_id`를 가지며, 세 CSV의 ID 집합은 서로 같고 저장소 MUSIC 원본의 ID 집합과도 일치했다. 평가는 손실 추적/이식 제외 기준과 고정 split을 제공하는 공통 prepared bundle에서 수행했다. 이 기준에서 31명을 제외하고 961명을 분석 코호트로 구성한다.
- 입력은 공통 `music-1.0.1-scd-basic-v1`의 임상 feature group 64개다. 앞선 더미 smoke test의 `(256,24)`는 API 확인용이었고 실제 평가에서는 저장소의 버전 관리된 64개 clinical schema를 사용했다. 결측값 대치는 train에서만 fit했다. 범주형 변수는 XGBoost/MLP에서 one-hot 변환했고 CatBoost는 범주형 변수로 지정해 처리했다.
- AUPRC는 저장소 비교 관례에 따라 Average Precision(AP)으로 계산했다. Brier는 validation 확률과 정답으로 계산했다.
- 학습 시간은 CPU에서 한 번 측정한 `fit` 시간이며, 파이프라인 모델은 train 데이터 전처리 학습을 포함한다. CatBoost 시간에는 train 범주형 문자열 변환이 포함된다. 반복 평균이 아니므로 성능 벤치마크가 아니다.
- 현재 validation 양성은 14명뿐이므로 한두 명의 점수 순위 변화도 지표에 큰 영향을 줄 수 있다. 신뢰구간도 계산하지 않았다. 이 결과만으로 모델 우열이나 임상 효용을 결론내리지 않는다.
- 대상은 공통 experimental protocol의 **관찰된 추적 기간 중 SCD 발생 여부**다. 고정된 4년 위험이나 생존분석 결과가 아니다. 연구 원자료의 추적 기간 중앙값은 44개월이다. [PhysioNet MUSIC v1.0.1](https://physionet.org/content/music-sudden-cardiac-death/1.0.1/)

## Holter 모델을 실행하지 않은 이유

2026-10-23 평가 당시 `Members/sang/datas/`에는 clinical, ECG 정량 특징, labels, dictionary CSV만 있고 `.hea`/`.dat` 파형 파일은 없었다. 이후 확보한 환자 ID 실험은 단일 train 환자의 입력/실행 smoke이며, 이 표의 AUROC/AUPRC 등 성능값을 만들지 않았다. 세부 실행 결과는 [Transformer ECG 모델 비교 보고서](transformer_ecg_model_comparison.md)에 기록했다.

## 실행 환경 및 재현

- Python 3.12.3, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.0, XGBoost 3.4.1, CatBoost 1.2.10
- CPU 학습, seed 42. CNN/Transformer는 이 실제 데이터 표에 포함되지 않는다.
- 학습 코드: [`evaluate_actual_clinical.py`](../scripts/evaluate_actual_clinical.py)

저장소 루트에서 패키지가 준비된 프로젝트 Python 환경으로 실행한다.

```sh
PYTHONPATH=Global/src python Members/sang/scripts/evaluate_actual_clinical.py \
  --data-dir Global/data/prepared/music-scd-basic-v1
```

이 실행은 실제 환자별 입력을 읽지만 stdout에는 split 집계와 모델 지표만 출력한다. 환자별 ID나 예측값은 파일로 저장하지 않는다. 이번 실행은 호스트의 기존 scikit-learn 환경과 격리 설치한 XGBoost/CatBoost 조합으로 수행했고, 저장소 Docker 공통환경의 완전한 재현은 별도 확인이 필요하다.
