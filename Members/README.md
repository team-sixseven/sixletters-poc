# 개인 작업 공간

`./dev init`에 등록한 영문 소문자 ID로 실험과 노트북을 한곳에 모은다. 아래 `yourname`은 팀원과 중복되지 않게 정한 본인 ID로 바꾼다. 다른 사람의 폴더는 합의 후 수정한다. 추가 패키지가 없는 새 실험은 공통 환경을 사용하고, 아래 환경 파일·`.venv/`는 개인 환경이 필요할 때 만든다.

```text
Members/yourname/
├── requirements.in         여러 실험이 공유할 추가 패키지 목록
├── requirements.lock       고정한 설치 버전
├── environment.json        기준 환경 정보
├── .venv/                  이름별 설치 환경: Git 제외
├── experiments/
│   ├── exp001/
│   │   ├── fit_rf.py          학습 시작 파일: 이름 자유
│   │   ├── preprocessing.py   보조 코드
│   │   ├── config.toml        공통 실험 설정
│   │   └── configs/           추가 학습 설정
│   └── exp002/
└── notebooks/
    ├── 01_music_practice.ipynb
    └── 분석명.ipynb
```

[README의 사전 실습](../README.md#2-music-준비와-사전-실습)을 따라 `./dev practice`로 개인 노트북을 만든다. 셀을 실행해 MUSIC 원본·코드북을 확인하고 모델·평가지표를 비교한 뒤 해석을 작성한다. 기존 노트북은 덮어쓰지 않는다. 실습 집계표·그래프는 `Global/artifacts/yourname/practice/clinical/`에 저장된다.

`./dev new exp001`로 개인 실험을 만든다. 전처리·모델·학습 방법은 각자 작성하고, 팀 비교에 사용할 결과는 [공통 평가·저장 기능](../Global/docs/parallel-work.md#4-학습하고-결과-확인하기)에 전달한다. 공통 모듈로 옮길 의무는 없다. 예제 파일 `train.py`는 이름을 바꿀 수 있고 보조 코드를 여러 파일로 나눌 수 있다.

```sh
./dev env add exp001 xgboost
./dev exec exp001 fit_rf.py
```

위 실행 명령의 파일명은 실제 학습 파일명으로 바꾼다. 새 실험들은 이름별 환경을 공유하므로 한 번 추가한 패키지를 함께 사용할 수 있다. `./dev env status exp001`로 현재 환경을 확인하고, `./dev env isolate exp001`로 필요한 실험만 분리한다. 사용 중인 패키지 파일이 있으면 그 실험 폴더에 복사한다. 패키지 기록이 없으면 분리 후에도 공통 환경을 쓰며, 추가 패키지가 필요할 때 `.venv/`를 만든다. [환경 공유·분리 방법](../Global/docs/development.md#개인-실험-패키지)

생성·변경된 패키지 목록·고정 버전·환경 정보와 환경 선택 파일(`environment.toml`)은 코드와 함께 Git에 올린다. 공통 환경만 쓰는 실험에는 개인 환경 파일을 따로 만들지 않으며, `.venv/`는 Git에 올리지 않는다. 추가 설정의 보관 범위는 [실험 가이드](../Global/docs/parallel-work.md#4-학습하고-결과-확인하기)를 따른다.

개인 분석은 `Members/yourname/notebooks/`에 작성한다. 공용 [환경 점검 노트북](../Global/notebooks/00_environment_check.ipynb)은 개인 실험용으로 수정하지 않는다.

개인 PC의 절대 경로 대신 저장소 기준 경로를 사용한다. 실행 결과는 기본 `Global/artifacts/`에 저장한다. 커밋 검사에서 노트북 출력과 실행 번호가 제거되므로 필요한 결과는 먼저 저장한다. 환자 데이터·토큰은 코드나 Markdown 셀에 직접 적지 않는다.
