# 개발 환경 및 입출력 점검 기록

기준일: 2026-10-23 (Asia/Seoul)

## 환경 확인 결과

| 항목 | 확인 결과 |
|---|---|
| OS | WSL2 Linux, kernel `6.18.33.2-microsoft-standard-WSL2` |
| CPU 아키텍처 | x86_64 |
| 호스트 Python | 3.12.3 (`/home/sang/.local/bin/python`) |
| uv | 0.12.3 (`/home/sang/.local/bin/uv`); 저장소 공통환경은 컨테이너 내 Python 3.12.12 / uv 0.9.26 사용 |
| GPU / Driver | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB; Windows driver 610.55 / KMD 610.78 |
| CUDA toolkit (`nvcc`) | 미확인: 실행 파일 확인 불가 |
| Docker | CLI 29.8.1 / Compose 5.5.1; Docker daemon 소켓 권한 거부 |
| PyTorch | 2.14.0+cu126, CUDA runtime 12.6; 권한 실행 경로에서 CUDA 사용 가능 |
| XGBoost / CatBoost / WFDB | XGBoost 3.4.1, CatBoost 1.2.10 사용; WFDB 미확인 |
| 프로젝트 환경 파일 | `Global/pyproject.toml`, `Global/uv.lock` 존재. sang 개인 `.venv`는 생성하지 않음 |

일반 샌드박스에서는 `/dev/dxg`가 보이지 않고 `nvidia-smi`가 GPU access blocked 오류를 냈다. 권한 실행 경로에서 확인하니 NVIDIA GPU와 Windows driver가 정상이고 `nvidia-smi`가 동작했다. 같은 권한 경로에서 PyTorch CUDA 연산도 성공했다. 따라서 GPU 자체나 Windows driver가 고장 난 것이 아니라 이 작업의 일반 샌드박스가 WSL GPU 장치 접근을 제한한 상태다. 프로젝트 공통환경 대신 `/tmp`의 CUDA 런타임과 기존 로컬 환경을 조합해 검증했다. 호스트 uv는 프로젝트 pinned uv와 달라 공통환경 설치에는 사용하지 않았다.

## 모델 smoke test

| 모델 | CPU smoke test | GPU |
|---|---|---|
| Logistic Regression, Random Forest, XGBoost, CatBoost, MLP | PASS (CPU, 격리/기존 로컬 환경 조합) | 미대상: 해당 구현은 CPU로 실행 |
| 1D-CNN, ECG Transformer | PASS (CPU 및 CUDA forward/loss/backward) | PASS: RTX 4060 |

입출력 형태, 시간 측정, 관측 버전과 환경 제약은 [모델 smoke test 상세 결과](model_smoke_results.md)를 참고한다. 저장소 Docker 공통환경을 직접 사용한 재현은 daemon 권한 때문에 아직 수행하지 못했다. 실행 코드는 `scripts/smoke_model_api.py`에 있다.

## 데이터 구조 검사

실제 CSV는 `datas/`에 존재한다. 표준 라이브러리로 첫 행의 헤더만 읽어 확인한 결과 clinical 67열, ECG 33열, labels 5열이며 모두 `patient_id` 컬럼이 있다. 행 수, 결측, 중복 ID, 숫자/범주형 분류는 pandas 실행환경이 없어 **미확인**이다. `.hea` 파일은 0개이며 `.dat` 원시 파형은 확인되지 않았다. 확정된 24개 임상 변수 목록, `999` 코드 의미, ECG 검사 행과 환자 행의 관계, 두 테이블의 join 가능 환자 수는 코드북/전처리 담당자와 확인할 사항이다. 전체 데이터는 출력하거나 수정하지 않았다. 실행은 `python scripts/inspect_data.py --data datas`이다.

## Furiosa NPU 다음 확인

- 장비 모델과 접근 환경
- SDK 및 runtime 호환 버전
- 1D-CNN 연산자 지원
- 사용자 정의 Transformer 연산자 지원
- 모델 변환, 컴파일, 실행 가능 여부

SDK 설치나 NPU 배포는 수행하지 않았다. PyTorch 실행 여부만으로 NPU 호환성을 판단할 수 없다.
