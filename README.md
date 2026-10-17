# 식스세븐 · 식스레터스 PoC

공통 데이터·평가 기준을 사용하며 개인별 학습 코드를 관리한다.

## 환경 준비

Python 3.12와 uv 0.9.26을 사용한다. 저장소 루트에서 실행한다.

```sh
uv sync --project Global --locked --extra ecg
uv run --project Global poc --help
```

## 파일 관리

공통 코드는 `Global/`, 개인 코드는 `Members/<영문 소문자 ID>/`에 둔다. 원본 데이터·환자 ID·모델·비밀키·노트북 출력은 Git에 올리지 않는다.
## 공통 데이터

```sh
uv run --project Global poc music
# 실행 점검용 합성 데이터
uv run --project Global poc demo
```

준비 파일의 정답·대상·변수 기준은 기업과 확정한 뒤 사용한다.

## 개인 실험

```sh
uv run --project Global poc init --owner alex --data Global/data/prepared/demo-v1
uv run --project Global poc new exp001
uv run --project Global poc exec exp001 train.py
```

`Members/alex/experiments/exp001/`의 코드·설정을 수정한다. 추가 패키지가 없으면 공통 환경을 사용한다. `uv run --project Global poc env add exp001 xgboost`로 필요한 패키지만 개인 환경에 추가한다. `uv run --project Global poc env isolate exp001`은 해당 실험만 분리한다. `Experiment.save()`는 평가와 결과 파일을 `Global/artifacts/`에 저장한다.

## 결과 비교

```sh
uv run --project Global poc compare --list
uv run --project Global poc compare
```

같은 데이터·분할·평가 기준의 실행을 선택한다. 목록에서 환자·SCD 수, 점수, 신뢰구간을 함께 본다. 기준 실행을 지정하면 점수 차이의 신뢰구간도 비교한다.

## 결과 공유와 모델 보관

서버 연결 변수 `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`를 실행 환경에 설정한다. 기본 실험 이름은 `sixseven-poc`이다.

```sh
uv run --project Global poc sync
uv run --project Global poc compare --remote
uv run --project Global poc archive
uv run --project Global poc restore restored001 --remote
uv run --project Global poc new exp002 --from exp001
```

저장은 서버 응답을 기다리지 않는다. `sync`는 점수·설정·예측 결과를 전송하고 실패 원인을 표시한다. 선택한 실행의 모델·재현 파일은 `archive`로 추가 업로드한다.

## 최종 평가

```sh
uv run --project Global poc final freeze selected
uv run --project Global poc final status selected
```

후보와 임계값을 고정한 후 상태 안내에 따라 저장 모델로 예측하고 `uv run --project Global poc final report selected`를 실행한다. 시험 결과를 보고 후보를 다시 선택하지 않는다.

## 노트북과 사전 실습

```sh
uv run --project Global poc practice
uv run --project Global poc notebook exp001
uv run --project Global jupyter lab
```

개인 실습 노트북에서 MUSIC CSV·코드북·기본 모델·불균형·평가 지표를 확인한다. 개인 실험 노트북은 연결된 SixSeven 커널을 선택한다.
