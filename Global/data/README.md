# 공통 데이터 준비

## 1. MUSIC 다운로드부터 환자 분할까지

Jupyter 또는 컨테이너 터미널에서 실행한다. Ubuntu 터미널에서는 `dev` 대신 `./dev`를 사용한다.

```sh
dev music
```

MUSIC **1.0.1**의 `subject-info.csv`, `subject-info_codes.csv`, `subject-info_definitions.csv`를 공식 저장소에서 내려받는다. 원본 3개의 SHA-256(파일 내용이 같은지 확인하는 값)을 확인한 뒤 정답·컬럼·환자 분할을 정리한다. 원시 파형은 내려받지 않는다.

예측 대상인 SCD는 심장돌연사를 뜻한다.

| 분석 | 명령 | 준비 데이터 경로 | 환자 / SCD | 입력 변수 |
|---|---|---|---:|---|
| 기본 | `dev music` | `Global/data/prepared/music-scd-basic-v1` | 961 / 94 | 임상 64개 + ECG 5개 |
| HRV 추가 | `dev music --cohort hrv` | `Global/data/prepared/music-scd-hrv-v1` | 571 / 51 | 임상 64개 + ECG 5개 + HRV 4개 |

HRV는 심장 박동 간격의 변동성이다. HRV 분석은 **Holter 기록이 동율동(심장의 정상 박동 생성 위치에서 시작하는 리듬)이며 HRV 4개 값이 모두 있는 환자**로 제한한다. 기본 분석과 대상 환자가 다르므로 두 데이터의 결과를 같은 조건의 성능 비교로 취급하지 않는다.

인터넷 없이 다시 준비하려면 `dev music --offline`을 사용한다. HRV 분석은 `dev music --cohort hrv --offline`이다. 원본 3개가 이미 내려받아져 있고 확인 값이 일치해야 한다.

같은 명령을 다시 실행하면 원본과 산출물을 검증한 뒤 재사용한다. 산출물의 누락·변경·추가가 있으면 덮어쓰지 않고 중단한다. 준비된 데이터를 수동으로 수정하지 않는다.

## 2. 공통 분석 기준

아래는 **버전을 고정한 기초 분석 규칙이며, 기업과 합의한 최종 임상 분석 기준은 아니다.** 목표·대상자·입력·분할을 변경할 때는 공통 코드와 데이터 버전을 PR에서 검토하고, 비교할 실험을 새 버전으로 다시 실행한다.

| 항목 | `v1` 규칙 |
|---|---|
| 예측 정답 | 관찰기간 중 `Cause of death=3`이면 `target=1`, 나머지는 `0` |
| 공통 제외 | `Exit of the study=1`(추적탈락), `2`(심장이식) 제외 |
| 예측기간 | 환자별 관찰기간 전체의 SCD 여부. **4년 이내 SCD 확률을 예측하는 데이터가 아님** |
| 입력 금지 | 사망 원인·연구 종료 상태·추적기간·`days_4years` 등 사후 정보 |
| 컬럼 정리 | 원본명을 고정된 영문 이름으로 매핑. 구분자·인코딩·소수점 쉼표를 공통 처리 |
| 결측·값 확인 | 빈 값은 결측으로 보존. `Age`에 `>89`로 기록된 값은 정확한 나이를 알 수 없으므로 결측으로 처리하고 내역 기록 |
| 환자 분할 | 같은 정답끼리 고정된 방식으로 정렬해 학습 70%·검증 15%·테스트 나머지로 배정 |

환자마다 추적기간이 다르며 다른 원인의 사망도 `target=0`에 포함된다. 이진 분류 결과를 특정 기간의 임상 위험도로 해석하지 않는다.

분할은 정답별로 `42:{target}:{patient_id}`의 SHA-256 값으로 정렬한 뒤 학습·검증 인원을 각각 내림하여 계산한다. 원본 행 순서와 실행 장비에 관계없이 같은 환자가 같은 집합에 들어간다.

| 데이터 | 학습 인원 / SCD | 검증 인원 / SCD | 테스트 인원 / SCD |
|---|---:|---:|---:|
| 기본 | 671 / 65 | 144 / 14 | 146 / 15 |
| HRV 추가 | 399 / 35 | 85 / 7 | 87 / 9 |

### 입력 변수

| 그룹 | 포함 내용 |
|---|---|
| `clinical` | 나이·성별·병력·증상·신체계측·심초음파·혈액검사·약물 등 임상 변수 64개 |
| `ecg` 기본 | `ecg_rhythm`, `ecg_q_waves`, `ecg_qrs_duration_ms`, `ecg_qt_interval_ms`, `ecg_average_rr_ms` |
| `ecg` HRV 추가 | 기본 ECG + `hrv_sdnn_ms`, `hrv_sdann_ms`, `hrv_rmssd_ms`, `hrv_pnn50_percent` |
| `combined` | 같은 환자의 `clinical`과 `ecg` 입력을 결합 |

의미·값 처리 확인이 필요한 PR 간격, 보정 QT, 일부 Holter 요약값, 흡연량(`cigarettes /year`), 정규화 troponin(심근 손상 검사값) 등은 입력에서 보류한다. 전체 컬럼의 원본명·변환명·그룹·보류 사유는 생성된 `column_manifest.json`에서 확인한다.

크레아티닌은 `creatinine`에 원본 수치를 유지한다. 원문 단위가 불명확하므로 단위를 확인하기 전에는 임의로 환산하지 않는다.

**결측치 대치·표준화·범주형 인코딩·변수 선택은 공동 준비에서 학습하지 않는다.** 각자 개인 학습 코드에서 학습 환자만 사용해 계산한다. 검증·테스트 환자를 그 계산이나 모델 학습에 사용하지 않는다.

## 3. 생성 파일과 개인 학습

원본 CSV 3개는 `Global/data/raw/music-1.0.1/`에 보관한다. 선택한 `Global/data/prepared/music-scd-basic-v1/` 또는 `Global/data/prepared/music-scd-hrv-v1/` 폴더에는 다음 파일을 생성한다.

| 파일 | 내용 |
|---|---|
| `dataset.json` | 개인 실험이 읽는 입력·정답·환자 분할과 식별 값 |
| `protocol.json` | 정답·대상자·컬럼 처리·분할 규칙 |
| `report.json` | 포함·제외 인원, 결측·변환 내역 등 집계 |
| `split_manifest.csv` | 환자별 학습·검증·테스트 배정 |
| `source_manifest.json` | 공식 원본 위치와 SHA-256 |
| `column_manifest.json` | 원본 컬럼별 사용·변환·보류 내역 |

개인 데이터 경로는 `dev init`에서 등록한다. 기존 설정이 있으면 사용할 분석의 MUSIC 경로인지 확인한다. 실험 생성·학습은 [첫 실험 절차](../../README.md#2-music-준비와-사전-실습)를 따른다. 한 번만 HRV 데이터로 실행하려면 해당 데이터를 먼저 준비한 뒤 학습 명령에 `--data Global/data/prepared/music-scd-hrv-v1`을 붙인다.

각자 원본을 다운로드해 같은 버전의 준비 기능을 실행하면 된다. 데이터 원본·환자 ID·분할 목록·준비 파일은 Git에 올리지 않는다. 환자별 예시는 이슈·PR·노트북 출력에도 넣지 않는다. 실행 방법을 설명할 예시가 필요하면 `dev demo`로 합성 데이터를 만든다.

공식 데이터는 [MUSIC 1.0.1](https://physionet.org/content/music-sudden-cardiac-death/1.0.1/)의 **Open Data Commons Open Database License 1.0**에 따라 사용하고, 보고서·제출 자료에는 공식 페이지의 데이터 및 논문 인용을 포함한다.

## 4. 다른 데이터의 공통 형식

MUSIC 이외의 데이터를 추가할 때는 아래 형식의 UTF-8·쉼표 구분 CSV와 변수 설명 JSON을 준비한다. 일반 `prepare` 명령은 이미 정한 정답·환자 분할을 검사하는 기능이다.

| CSV 열 | 값 |
|---|---|
| `patient_id` | 비어 있지 않은 고유 문자열. 앞자리 `0` 보존 |
| `target` | 숫자 `0` 또는 `1` |
| `split` | `train`, `validation`, `test` 중 하나 |
| 입력 변수 | 설명 JSON에 선언한 숫자형·범주형 열. 빈 셀은 결측치 |

같은 환자는 한 집합에만 넣는다. 학습·검증 집합에는 정답 `0`과 `1`이 모두 있어야 한다. 한쪽 정답이 없으면 준비·불러오기를 중단하고 해당 집합의 정답별 인원을 표시한다. 여러 파형 구간을 사용하더라도 공통 평가에는 환자별 예측을 제출한다.

`schema.json` 예시의 변수 이름은 형식 설명용이다.

```json
{
  "dataset_id": "cohort-v1",
  "target_definition": "정답 정의와 예측 기준",
  "feature_groups": {
    "clinical": ["clinical_numeric", "clinical_category"],
    "ecg": ["ecg_feature"]
  },
  "categorical_features": ["clinical_category"]
}
```

임상·ECG 변수를 중복 없이 선언하고 범주형 열만 `categorical_features`에 적는다. 숫자형의 누락 코드는 빈 셀로 바꾼다. 범주형 문자열 `NA`는 값으로 유지되므로 원본에서 누락을 뜻한다면 미리 빈 셀로 변환한다. 선언하지 않은 추가 열은 저장하지 않는다.

```sh
dev prepare --csv Global/data/processed/cohort.csv --schema Global/data/processed/schema.json --output Global/data/prepared/cohort-v1
```

`prepare`는 형식·환자 중복·분할·입력 열·자료형을 검증해 `dataset.json`에 저장하며 기존 출력은 덮어쓰지 않는다. 정답의 임상적 타당성이나 사후 정보 사용 여부는 별도로 검토한다.
