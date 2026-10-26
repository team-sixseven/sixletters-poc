# 식스세븐 · 식스레터스 PoC

개인 코드와 노트북을 실행할 공통 Docker 환경, 추가 패키지 관리, MLflow 결과 등록을 제공합니다. 데이터 전처리·모델 학습·평가 방법과 코드는 각자 작성합니다.

## 1. 시작

Windows는 WSL2의 Ubuntu 터미널에서, macOS는 터미널에서 실행합니다. Docker Desktop을 먼저 켜세요.

```bash
git clone -b develop https://github.com/team-sixseven/sixletters-poc.git
cd sixletters-poc
./dev up -d --wait
./dev setup --owner alice
./dev url
```

`alice`를 본인의 영문 소문자 ID로 바꾸세요. `url`에 표시되는 주소로 JupyterLab을 엽니다. NVIDIA GPU가 있는 Windows/WSL2에서는 시작 명령에 `--gpu`를 추가합니다.

```bash
./dev up --gpu -d --wait
./dev check --gpu
```

## 2. 데이터와 개인 작업

`setup`이 아래 빈 폴더를 준비합니다. 기존 파일은 덮어쓰거나 옮기지 않습니다. 기존 사용자도 `./dev setup --owner alice`를 다시 실행하면 됩니다.

```text
Global/data/DATA/                  # Drive의 DATA 폴더 내용 그대로
├── clinical/                     # clinical_clean.csv + split/{train,val,test}.csv
├── ECG_num/                      # ECG_clean.csv + split/{train,val,test}.csv
└── ECG_tem/                      # ECG_tem_clean.csv + ECG_tem_clean/ + split/
Members/alice/
├── notebooks/                    # 분석 노트북
├── experiments/                  # 개인 학습 코드·설정
├── results/                      # MLflow 등록 결과
└── models/                       # 학습한 모델 파일
```

Drive에서 받은 `DATA`를 **`Global/data/` 아래**에 넣으세요. `DATA/DATA`로 중첩하지 않습니다. ZIP으로 받았다면 먼저 압축을 풉니다. 데이터·결과·모델 파일은 Git에서 제외됩니다. [데이터 파일과 경로 안내](Global/data/README.md)

데이터는 각자 컴퓨터에 보관하며 Docker와 Jupyter에서도 같은 저장소 파일을 읽습니다. 다운로드·전처리·분할을 자동 실행하지 않습니다.

`Members/alice/` 아래 원하는 폴더에 코드와 노트북을 작성하세요. 파일명·모델·설정 형식은 자유입니다.

```bash
mkdir -p Members/alice/experiments/exp001
# exp001에 직접 작성한 train.py 실행
./dev exec Members/alice/experiments/exp001 train.py
# 빈 노트북 생성 + 개인 환경 커널 연결
./dev notebook Members/alice/notebooks
```

### 필요한 패키지 추가하기

pandas처럼 기본으로 제공되는 패키지는 바로 쓰면 됩니다. XGBoost가 추가로 필요하면 아래 명령을 실행하세요. `alice`는 본인 ID로 바꿉니다.

```bash
./dev env add Members/alice xgboost
```

내 컴퓨터의 개인 환경에 설치되며, 같은 이름 폴더 아래의 여러 실험에서 사용할 수 있습니다. 다른 팀원의 설치 환경은 바뀌지 않습니다. 노트북을 열어둔 경우 커널을 종료한 뒤 설치하고 다시 시작하세요.

설치하면 **다른 사람도 같은 패키지를 준비할 수 있도록 설치 목록이 함께 기록됩니다.** 직접 작성할 필요는 없습니다. 코드를 공유할 때 변경된 설치 목록도 함께 커밋하세요. 실제 설치 폴더·데이터·모델·비밀번호는 올리지 않습니다.

[설치 목록 공유·다른 팀원 코드 실행 방법](CONTRIBUTING.md#패키지-설치와-공유) · [특정 실험만 다른 패키지를 쓰는 방법](Global/docs/development.md#개인-패키지)

**Jupyter의 새 터미널에서는 `./dev` 대신 `dev`를 사용합니다.** 작업 폴더 안으로 이동했다면 경로도 현재 위치에 맞춰 입력하세요. 명령은 저장소 루트에서 실행하는 것이 가장 간단합니다.

## 3. 결과 등록

서버 운영자가 제공한 **컨테이너에서 접속 가능한 주소와 개인 계정**을 한 번 설정합니다. 비밀번호는 화면에 표시하지 않고 입력합니다.

```bash
./dev setup --uri http://서버주소:5000 --username 내계정 --password
```

접속 정보는 Git에서 제외한 `.poc.local.toml`에 저장됩니다. `.env`의 `MLFLOW_*` 설정도 사용할 수 있으며 비어 있지 않은 환경 변수가 우선합니다.

각자 계산한 결과를 아래 형식으로 저장하세요.

```text
Members/alice/results/exp001/
├── metrics.json       필수: 지표 이름과 숫자
├── params.json        선택: 모델명·설정·데이터 버전·평가 조건
└── artifacts/         선택: 그래프·보고서 등 공유할 파일
```

`metrics.json` 예시 — 아래 숫자는 형식 설명용입니다.

```json
{"roc_auc": 0.75, "average_precision": 0.31, "sensitivity": 0.80, "specificity": 0.65}
```

`params.json` 예시:

```json
{"model": "RandomForestClassifier", "n_estimators": 200, "split": "validation", "data_version": "v1"}
```

```bash
./dev upload Members/alice/results/exp001
# 선택한 모델만 같은 실행에 추가 보관
./dev upload Members/alice/results/exp001 --model Members/alice/models/model.pkl
```

완료 후 표시되는 MLflow 링크에서 **Metrics(지표), Parameters(설정), Artifacts(파일)**를 확인합니다. 기본 실험명은 `sixseven-poc`, 실행명은 결과 폴더명입니다. 평가나 팀 비교를 별도로 실행하지 않습니다.

- 같은 명령 재실행은 같은 실행에 재시도합니다. 성공한 파일은 다시 보내지 않습니다.
- 등록을 시작한 결과는 수정하지 마세요. 지표·설정·그래프·모델이 달라졌으면 새 결과 폴더를 만드세요. 모델은 처음 등록할 때 생략했다가 나중에 한 번 추가할 수 있습니다.
- 자동 전송은 없습니다. `upload`가 끝날 때까지 기다리며, 실패하면 같은 명령으로 재시도합니다. `.poc-cache/uploads/`는 재시도 기록이므로 유지하세요.
- `artifacts/`와 `--model`에 지정한 파일만 추가 전송합니다. 원본 데이터·인증 정보는 넣지 마세요.
- 모델은 **파일 보관**입니다. 모델 등록소 등록·자동 복원 실행·서비스 배포는 수행하지 않습니다.
- AP(`average_precision`)와 곡선 적분 방식의 PR-AUC는 다른 계산입니다. 실제 계산 방식에 맞게 이름을 사용하고 평가 대상·분할·임계값은 설정에 남기세요. 지표는 재계산하거나 평가 조건을 강제하지 않습니다.

## 4. 종료·협업

```bash
./dev down             # 개인 파일은 유지
./dev status
./dev hooks            # 최초 한 번: 커밋 전 검사 연결
./dev verify           # 코드·파일 정책·환경/등록 테스트
```

코드와 문서는 이슈 → 개인 브랜치 → PR → 검토 → `develop` 순서로 공유합니다. 데이터·모델·결과·계정 정보는 Git에 올리지 않습니다.

- [협업 규칙](CONTRIBUTING.md)
- [환경·패키지·문제 해결](Global/docs/development.md)
- [MLflow 서버 운영](Global/docs/tracking-server.md)
