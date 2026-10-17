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
