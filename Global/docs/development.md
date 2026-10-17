# 개발 환경

| 구성 | 역할 |
|---|---|
| WSL2 | Windows에서 Linux를 실행 |
| Ubuntu | 저장소를 보관하고 Git·Docker 명령을 실행하는 작업 공간 |
| Docker 컨테이너 | 정해진 Python·공통 패키지로 Jupyter와 개인 학습 코드를 실행 |

Python과 학습 패키지는 컨테이너 안에 설치한다. 공통 버전은 **Python 3.12.12 · uv 0.9.26**이며 패키지는 `Global/uv.lock`에 고정한다. Docker Desktop을 사용하면 Ubuntu에 Docker Engine을 따로 설치하지 않는다.

| 실행 환경 | 포함 내용 | 사용 시점 |
|---|---|---|
| CPU | pandas·scikit-learn·Jupyter·WFDB·NeuroKit2·개발 도구 | 데이터 분석·기본 모델·파형 처리 |
| GPU | CPU 환경의 패키지 + PyTorch 2.13.0 / CUDA 12.6 런타임 | NVIDIA GPU로 신경망 학습 |

## 1. WSL2와 Docker 확인

먼저 [팀 저장소](https://github.com/team-sixseven/sixletters-poc)에 접근할 수 있는지 확인하고 [GitHub 클론 인증](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github)을 준비한다. 조직 초대가 있다면 수락한다.

Windows PowerShell에서 설치된 배포판과 WSL 버전을 확인한다.

```powershell
wsl --list --verbose
wsl --update
```

Ubuntu의 `VERSION`이 `2`여야 한다. Ubuntu가 없으면 [Microsoft 설치 안내](https://learn.microsoft.com/windows/wsl/install)에 따라 설치한다.

이미 WSL 안의 Docker Engine을 사용하고 있다면 기존 구성을 유지하고 아래 명령으로 확인한다. **Docker Desktop과 WSL 내부 Docker Engine을 중복 설치하지 않는다.** [Docker 안내](https://docs.docker.com/desktop/features/wsl/)

Docker가 없는 경우 [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)를 설치한다.

1. **Settings → General → Use WSL 2 based engine**을 켠다.
2. **Settings → Resources → WSL Integration**에서 사용할 Ubuntu를 켠다.
3. Linux containers 모드로 실행한다. WSL Integration 메뉴가 없으면 컨테이너 모드를 확인한다.

Ubuntu 터미널에서 확인한다.

```sh
docker info
docker compose version
git --version
```

Docker Compose는 **2.40.3 이상**을 사용한다. 버전이 낮으면 Docker Desktop을 업데이트한다. Docker Engine을 직접 설치한 환경에서는 Compose 플러그인을 업데이트한다.

Docker 서버 정보와 지원 버전을 확인하면 [README의 시작 절차](../../README.md#1-저장소와-실행-환경-준비)를 따른다. Docker Desktop은 작업하는 동안 실행 상태로 둔다.

저장소는 `~/projects/sixletters-poc`처럼 WSL의 Linux 파일 시스템에 둔다. Windows 드라이브의 `/mnt/c/` 아래보다 Linux 컨테이너의 파일 접근에 적합하다. [Docker WSL 권장 사항](https://docs.docker.com/desktop/features/wsl/best-practices/)

macOS에서는 Docker Desktop을 설치하고 macOS 터미널에서 같은 CPU 시작 명령을 사용한다. WSL2는 설치하지 않는다. 이 저장소의 NVIDIA GPU 환경은 NVIDIA GPU가 있는 Linux·WSL2 장비에서 사용한다.

## 2. NVIDIA GPU 사용

Windows에 WSL2를 지원하는 NVIDIA 드라이버가 필요하다. Windows 드라이버를 통해 WSL에서 GPU를 사용하므로 WSL 안에 Linux용 NVIDIA 디스플레이 드라이버를 별도로 설치하지 않는다. [NVIDIA WSL 안내](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

Ubuntu 터미널에서 장비를 확인한 뒤 GPU 환경을 시작한다.

```sh
nvidia-smi
./dev up --gpu -d --wait
```

시작할 때 컨테이너의 GPU 인식과 실제 GPU 연산을 점검하고, 통과하면 JupyterLab을 실행한다. 실패하면 해당 단계의 오류를 해결한 뒤 다시 시작한다.

| 실패한 단계 | 확인할 내용 |
|---|---|
| WSL의 `nvidia-smi` | Windows NVIDIA 드라이버와 WSL 업데이트 |
| 컨테이너 GPU 연결 | Docker Desktop의 WSL2 backend 또는 기존 Docker Engine의 NVIDIA Container Toolkit 설정 |
| Python GPU 연산 | 이미지 재빌드, 오류에 표시된 드라이버·CUDA 호환 조건 |

Docker Engine을 WSL에 직접 설치한 구성은 [NVIDIA Container Toolkit 안내](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)를 따른다. Docker Desktop 경로에 이 설치를 추가할 필요는 없다.

CPU·GPU 환경은 같은 `localhost:8888`을 사용하므로 하나만 실행한다. 전환할 때는 진행 중인 학습이 끝난 뒤 다음 명령을 실행한다.

```sh
./dev down
./dev up --gpu -d --wait
```

CPU로 돌아갈 때는 마지막 명령에서 `--gpu`를 빼면 된다. `./dev down`은 CPU·GPU 환경을 모두 정리한다. Docker 명령을 직접 사용할 때는 `docker compose --profile gpu down`을 쓴다. 별도로 재점검하려면 `./dev check --gpu`를 사용한다.

GPU 환경을 켜도 scikit-learn의 로지스틱 회귀·랜덤 포레스트가 자동으로 GPU를 사용하지는 않는다. 개인 신경망 코드에서 모델과 입력을 GPU에 배치하는 등 해당 라이브러리의 실행 방법을 적용한다.

## 3. Jupyter와 IDE

`./dev up -d --wait`로 시작한 뒤 `./dev url`로 개인 토큰을 확인하고 `http://localhost:8888`에 접속한다. 시작 후 같은 터미널에서 다음 명령을 실행할 수 있다. GPU 시작 명령도 접속 방법은 같다. 포트를 바꾸려면 시작 전에 `export POC_NOTEBOOK_PORT=8889`처럼 지정한다.

JupyterLab에서 **Launcher → Terminal**을 열면 별도 가상환경 활성화나 경로 설정 없이 프로젝트의 Python과 `poc` 명령을 사용할 수 있다. `poc --help`로 명령 목록을 확인한다. MUSIC 준비·이름 등록·실험 생성·학습·비교는 [README의 개인 실험 절차](../../README.md#2-music-준비와-사전-실습)를 따른다. Ubuntu 터미널에서 데이터를 준비할 때는 `./dev music`을 사용한다.

`Global/notebooks/00_environment_check.ipynb`로 노트북 실행도 확인할 수 있다. 개인 분석은 `Members/이름/notebooks/`에 작성한다.

IDE는 자유롭게 선택하되 **프로젝트 컨테이너의 Python**을 사용한다. Docker 또는 Docker Compose 인터프리터를 지원하는 IDE라면 다음 값을 기준으로 설정한다.

| 항목 | 설정 |
|---|---|
| 코드 | WSL에 복제한 저장소 |
| Compose 서비스 | 기본 `cpu`, NVIDIA GPU 학습은 `gpu` |
| Python | 공통 환경은 `/opt/venv/bin/python`, 개인 패키지는 아래 개인 환경의 Python |
| 실행 스크립트 | 본인의 `Members/이름/experiments/실험명/학습파일.py` |
| 작업 폴더 | 컨테이너의 `/workspace` |
| 데이터 경로 | `poc init`에 등록한 경로 또는 `--data` 인자 |

IDE에서 실행할 때는 `poc env sync exp001` 후 `poc env status exp001`에 표시된 Python을 인터프리터로 선택한다. 추가 패키지가 없는 새 실험은 `/opt/venv/bin/python`을 사용한다. 개인 환경은 `/workspace/Members/이름/.venv/bin/python`, 분리한 실험은 `/workspace/Members/이름/experiments/exp001/.venv/bin/python`이다. 노트북은 아래의 개인 커널을 선택한다. 커널은 노트북 셀을 실행하는 Python 환경이다.

실행 설정을 저장하면 같은 실험은 실행·디버그 버튼으로 시작할 수 있다. 공유할 학습 결과는 `poc exec 실험명 파일명.py`로 실행하면 파일명·인자도 함께 기록된다. IDE에서 직접 실행한 결과에는 이 기록이 없으므로 실행 방법을 실험 설명에 남긴다. Ubuntu 터미널에서 `./dev shell` 또는 `./dev shell --gpu`를 실행해 켜져 있는 환경에 들어가도 된다. 컨테이너 터미널을 `exit`로 닫아도 Jupyter 서버는 계속 실행된다.

### 개인 노트북 실행 환경

실험을 만든 뒤 Ubuntu 터미널의 저장소 폴더에서 실행한다.

```sh
./dev env add exp001 xgboost
./dev notebook exp001
```

`Members/이름/experiments/exp001/analysis.ipynb`를 JupyterLab에서 연다. 개인 패키지 준비와 커널 연결을 함께 처리하므로 바로 `import xgboost`를 실행할 수 있다. 기존 파일이면 셀은 보존하고 커널 정보만 갱신한다. 다른 파일명은 `--file eda.ipynb`로 지정한다. 이미 열려 있으면 저장 후 다시 연다. 기존 다른 노트북에 연결할 때는 `./dev env kernel exp001` 등록 후 **Kernel → Change Kernel → `SixSeven · 이름 / exp001`**을 선택한다. 커널은 실험별로 선택하며 공통·이름별·분리 환경 중 해당 실험의 환경을 사용한다. 추가 패키지가 없으면 `env add` 없이 `./dev notebook exp001`만 실행한다.

- 커널 등록은 해당 컴퓨터에만 적용되며 Docker를 다시 시작해도 유지된다. 다른 팀원의 실험은 코드와 패키지 고정 파일을 받은 뒤 자신의 환경에서 다시 등록한다.
- 패키지 변경·환경 갱신 전에는 그 환경을 공유하는 모든 학습과 커널을 종료한다. 커널은 **Kernel → Shut Down Kernel**로 종료하며, 탭만 닫으면 계속 실행 중일 수 있다. 작업 후 다시 커널을 선택한다.
- 같은 실험을 `./dev exec`로 실행하기 전에도 해당 커널을 종료한다. 서로 다른 실험은 같은 환경으로 동시에 실행할 수 있다.
- 같은 실험은 커널 하나를 사용한다. 여러 노트북을 함께 열면 이미 실행 중인 커널에 연결한다. 별도 Python 상태가 필요하면 실험 폴더를 나눈다.
- `env isolate`·`env share`는 이미 등록된 해당 실험의 커널 정보를 갱신한다. 전환 후 `./dev notebook exp001`로 필요한 패키지를 준비하고 파일을 다시 연다.
- CPU/GPU 전환이나 공통 환경 갱신 후에는 `./dev env kernel exp001`로 다시 등록한다. 기존 개인 환경의 기준이 달라졌다는 안내가 나오면 `--relock`을 붙인다. 생성·변경된 패키지 고정 파일만 확인 후 커밋한다.
- 노트북에서 실험 폴더의 보조 Python 파일을 불러올 수 있고, 상대 경로의 기준은 저장소 루트다. 노트북 셀 실행 자체는 MLflow에 자동 등록되지 않고 셀 코드도 실행 결과의 코드 사본에 포함되지 않는다. 제출 후보는 학습 코드를 실험 폴더의 `.py`에 반영하고 커널을 종료한 뒤 `./dev exec`로 실행해 저장한다.

## 4. 파일 보관과 환경 갱신

저장소 폴더가 컨테이너의 `/workspace`에 연결되어 코드·데이터·저장된 결과가 노트북에 남는다. `./dev down` 후에도 유지되며, 연결되지 않은 임시 경로에 결과를 두지 않는다. 공용 MLflow 서버는 별도로 실행되어 이 명령으로 종료되지 않는다.

| 바뀐 내용 | 필요한 작업 |
|---|---|
| 개인 코드·설정 | 다음 실행에 바로 반영. 이미지 재빌드 불필요 |
| 개인 패키지 | 같은 환경을 쓰는 모든 학습·커널 종료 후 `./dev env` 명령 사용. 이미지 재빌드 불필요 |
| `.env` 접속 정보, Docker·공통 패키지 파일, 작업 환경 시작 설정 | 학습·저장 완료 후 환경 종료·재시작 |
| CPU ↔ GPU | 기존 환경 종료 후 원하는 환경으로 시작 |

학습 중에는 해당 실험의 코드·설정·공통 데이터와 학습·평가 관련 공통 코드를 바꾸지 않는다. 공통 명령 화면·전송 등 보조 코드의 변경 허용 범위는 [실험 가이드](parallel-work.md#4-학습하고-결과-확인하기)를 따른다. 다른 학습 방법을 작성하려면 별도 실험 폴더를 사용한다. 환경을 종료하면 실행 중인 학습도 중단된다.

진행 중인 학습·저장이 끝난 뒤 환경을 갱신한다.

```sh
./dev down
./dev up -d --wait
```

GPU 환경은 마지막 명령에 `--gpu`를 붙인다. `./dev build`는 이미지만 만들며 실행 중인 환경을 갱신하지 않는다. 다시 빌드해도 의존성 파일이 같으면 기존 패키지 설치 단계를 재사용한다. 첫 빌드에는 네트워크·저장 공간이 필요하며 GPU 이미지가 더 크다.

## 5. 패키지 추가·변경

### 개인 실험 패키지

추가 패키지가 없는 새 실험은 **설치된 공통 환경을 바로 사용**한다. `env add`로 패키지를 추가할 때 이름별 개인 환경을 만들며, 이후 여러 실험이 그 환경을 공유한다. **공통 패키지는 기존 설치를 재사용**하고, 개인 환경에는 추가 패키지와 그 실행에 필요한 패키지 중 공통에 없는 것만 설치한다. 새 실험마다 같은 패키지를 다시 설치하지 않는다. Jupyter·컨테이너 터미널에서는 아래 명령을 사용하고, Ubuntu 터미널에서는 `poc`을 `./dev`로 바꾼다. 이름은 `poc init`의 등록값을 사용하며, 다른 작성자는 `--owner "이름"`으로 지정한다.

```sh
poc env status exp001
poc env add exp001 xgboost
poc exec exp001 fit_rf.py
```

`env status`는 환경 범위·준비 여부·사용 중인 학습/커널·다음 명령을 보여준다. 전체 환경·데이터·전송 상태는 `./dev status`로 확인한다. `exp001`이 이름별 환경을 쓰면 `env add`로 추가한 XGBoost를 `exp002` 등 같은 환경의 실험에서도 사용할 수 있다. `poc env remove exp001 xgboost`로 제거할 때도 같은 범위에 적용된다. 다른 팀원의 환경과 Docker 공통 환경은 바뀌지 않는다.

`fit_rf.py`는 실제 학습 파일명으로 바꾼다. 추가 패키지는 PyPI 이름과 필요한 버전 조건으로 지정한다. Git 주소·로컬 경로·인증 정보가 포함된 설치 주소는 지원하지 않는다. `poc exec`는 해당 실험의 공통 또는 개인 환경으로 파일을 실행하고 뒤의 인자를 그대로 전달한다. 직접 작성한 학습 코드를 사용하며 공통 학습 함수를 호출할 필요는 없다.

```sh
poc exec exp001 --owner "본인이름" fit_rf.py --data Global/data/prepared/music-scd-basic-v1
```

개인 환경 파일의 기본 위치는 `Members/이름/`이며, 아래에서 분리한 실험은 해당 실험 폴더에 저장한다. 공통 환경만 쓰는 새 실험에는 잠금 파일·환경 정보·`.venv/`를 만들지 않는다.

| 환경 파일 | 역할 | Git 공유 |
|---|---|---|
| `requirements.in` | 추가 패키지와 원하는 버전 범위 | 포함 |
| `requirements.lock` | 재현에 필요한 전체 패키지의 고정 버전(공통·개인 포함) | 포함 |
| `environment.json` | 고정 버전을 만든 Python·운영체제·공통 패키지 정보 | 포함 |
| `.venv/` | 개인 실행 환경과 공통에 없는 추가 패키지 | 제외 |

실험 폴더의 `environment.toml`은 공유 또는 분리 환경 선택을 기록하며, 생성되면 함께 커밋한다. `env isolate`·`env share`로 변경한다.

**특정 실험만 분리하려면** 다음을 실행한다. 사용 중인 패키지 기록이 있으면 복사하며, 이후 패키지 변경은 `exp002`에만 적용한다. 패키지 기록이 없는 실험은 분리 후에도 추가 패키지가 필요해질 때까지 공통 환경을 사용한다.

```sh
poc env isolate exp002
poc env add exp002 lightgbm
```

다시 이름별 환경을 사용하려면 `poc env share exp002`를 실행한다. 이름별 환경에 필요한 패키지·버전이 충족되는지 먼저 검사하며, 이름별 패키지를 자동으로 추가하거나 바꾸지 않는다. 전환해도 기존 실험의 패키지 파일을 삭제하지 않으며 `environment.toml`에 선택을 기록한다. 기존 실험 안에 패키지 고정 파일·환경 정보 또는 비어 있지 않은 패키지 목록이 있으면 분리 상태를 유지하므로 기존 실험의 환경은 자동 전환되지 않는다.

분리 여부와 관계없이 **공통 환경에 설치된 모든 패키지는 현재 공통 버전을 유지**한다. 추가 패키지가 이 버전과 충돌하면 설치를 중단한다. 다른 Python·공통 패키지 버전이 필요하면 아래의 공통 패키지 절차를 따른다.

Git에서 받은 실험의 고정 환경을 설치하거나 `.venv/`를 다시 만들 때는 다음 명령을 사용한다. `poc exec`도 처음 실행할 때 필요한 환경을 준비한다. 같은 패키지 목록·고정 버전·실제 설치가 일치하면 `env add`·`env sync`를 반복해도 재설치하지 않는다.

```sh
poc env sync exp001
```

공통 패키지까지 설치된 기존 개인 환경도 계속 사용할 수 있다. `poc env sync exp001 --repair`로 다시 만들거나 개인 패키지를 변경하면 공통 패키지를 재사용하는 방식으로 전환한다. 설치 방식과 관계없이 전체 패키지 버전을 계속 기록한다.

개인 환경의 `requirements.in`을 직접 수정했거나 공통 패키지(CPU/CUDA용 PyTorch 포함)·Python·실행 환경의 운영체제나 CPU 아키텍처가 바뀌면 이전 잠금 파일을 그대로 적용하지 않는다. 변경을 확인하고 해당 환경에 맞는 고정 버전을 다시 만든다. 공통 환경만 쓰는 실험은 개인 잠금 갱신이 필요 없다.

```sh
poc env sync exp001 --relock
```

실행이 멈추면 표시된 **원인과 다음 명령**을 따른다. `env status`에도 같은 안내가 나온다. 버전 기록을 갱신할 때는 `--relock`, 설치된 환경만 복구할 때는 `--repair`를 안내한다. 호스트 터미널에서는 `./dev`, Jupyter·컨테이너 안에서는 `poc`으로 표시되며 실험명·이름·GPU 선택도 포함한다. 해당 환경을 쓰는 학습이나 커널은 먼저 종료한다.

`env add`·`env remove`는 목록과 고정 버전을 함께 갱신하므로 별도 `--relock`이 필요 없다. 생성·변경된 환경 파일을 확인해 코드와 함께 커밋한다. 목록이 비어도 기존 고정 환경은 자동 삭제하지 않는다. 패키지는 `.venv/`나 공통 환경에 직접 설치하지 않고 이 명령으로 관리한다.

학습 결과에는 **실행 당시 Python·패키지 버전**을 기록하고, 개인 환경 파일이 있으면 사본도 보관한다. [복원](tracking.md#보관한-실험-복원)한 실험은 이름별 추가 패키지를 상속하지 않도록 분리한다. 개인 패키지 파일이 있으면 `poc env sync`로 설치하고, 없으면 현재 공통 환경을 사용하므로 원본의 환경 기록과 일치하는지 확인한다. 공통 코드 사본이 현재 `Global/` 코드를 자동 교체하지는 않는다.

같은 실험의 학습·커널은 하나만 실행한다. `exp001`, `exp002`처럼 서로 다른 실험은 같은 환경으로 동시에 실행할 수 있다. 패키지 추가·삭제·갱신은 그 환경을 쓰는 모든 학습과 커널을 종료한 뒤 진행한다.

NVIDIA GPU 실험은 GPU 작업 환경에서 같은 명령을 사용한다. 개인 환경도 해당 작업 환경의 Python·공통 패키지를 기준으로 만들며, 패키지를 설치하는 것만으로 GPU가 활성화되지는 않는다. 실제 장비에서 필요한 연산을 확인한다. 시스템 도구·드라이버가 필요한 패키지는 Python 패키지 설치와 별도로 공통 환경 변경을 검토한다.

### 공통 패키지

팀과 변경을 조율하고 `Global/pyproject.toml`(목록)·`Global/uv.lock`(고정 버전)·`Global/requirements.txt`(제출용 목록)를 한 PR에 반영한다. `Global/requirements.txt`는 직접 편집하지 않는다.

학습을 마친 뒤 **Ubuntu 터미널**에서 실행한다. `패키지이름`은 실제 이름으로 바꾸고, 개발 도구는 `add`에 `--dev`를 붙인다.

```sh
./dev deps add 패키지이름
./dev deps export
```

`deps`는 컨테이너의 uv로 파일을 갱신한다. GPU 환경만 켜져 있으면 해당 환경을 사용한다. 환경이 꺼져 있고 GPU 이미지만 준비했다면 `./dev deps --gpu add 패키지이름`, `./dev deps --gpu export`를 사용한다.

위의 [환경 갱신](#4-파일-보관과-환경-갱신)을 마치고 세 파일을 함께 검사·커밋한다. 실행 중인 컨테이너에만 설치한 패키지는 재현되지 않으므로 공통 목록과 고정 버전에 반영한다. GPU·NPU 패키지도 같은 검토를 거친다.

병합 충돌이 생기면 `Global/pyproject.toml`에서 필요한 패키지를 먼저 확정하고 `Global/uv.lock`은 가져온 `develop` 기준으로 다시 만든다.

```sh
git restore --source=origin/develop -- Global/uv.lock
./dev deps lock
./dev deps export
git add Global/pyproject.toml Global/uv.lock Global/requirements.txt
```

환경을 갱신한 뒤 [충돌 해결 절차](../../CONTRIBUTING.md#develop의-변경을-가져오거나-충돌을-해결할-때)에 따라 검사·병합 커밋·push한다.

## 기존 작업 파일 이동

저장소를 갱신한 뒤 `./dev up -d --wait`로 시작하면 기존 로컬 작업 파일을 새 위치로 이동한다. 같은 이름의 파일이 양쪽에 있으면 덮어쓰지 않고 중단하므로 표시된 경로를 확인한다.

| 기존 위치 | 새 위치 |
|---|---|
| `data/` | `Global/data/` |
| `artifacts/` | `Global/artifacts/` |
| `experiments/이름/` | `Members/이름/experiments/` |
| `notebooks/이름/` | `Members/이름/notebooks/` |

Docker 밖에서 Python·uv를 사용하는 경우 저장소 최상위에서 한 번 실행한다.

```sh
uv run --project Global --locked python Global/scripts/migrate_layout.py
```

이동한 데이터의 `.poc.local.toml` 경로도 갱신한다. 개인 코드와 저장된 실행 파일의 내용은 바꾸지 않는다. 개인 코드나 IDE 실행 설정에 경로를 직접 적었다면 새 위치를 확인한다. 직접 지정한 별도 데이터·결과 경로는 유지한다.

## 자주 막히는 지점

| 증상 | 조치 |
|---|---|
| 클론이 `Repository not found`·인증 오류로 실패 | GitHub 계정의 저장소 권한·조직 초대 수락·Git 인증 확인 |
| `./dev`를 찾지 못하거나 Ubuntu에서 `poc: command not found` | `cd ~/projects/sixletters-poc` 후 `./dev`로 실행. 컨테이너 터미널에서는 `poc` 사용 |
| Docker 서버에 연결할 수 없음 | Docker 실행·WSL Integration을 확인하고 `docker info` 실행 |
| 8888 포트가 이미 사용 중 | 같은 저장소의 CPU·GPU를 중복 실행하지 않았는지 확인. 종료 후 하나만 시작. 다른 앱이 쓰면 `export POC_NOTEBOOK_PORT=8889` 후 시작 |
| Jupyter 토큰을 찾지 못함 | `./dev url`로 개인 접속 토큰 확인 |
| 준비 상태·다음 작업이 불명확함 | `./dev status` 또는 `./dev env status 실험명` 확인 |

학습·저장을 마친 뒤 환경을 종료한다. 전송·비교 오류는 [팀 결과 수집](tracking.md#전송-실패와-재시도)을 확인한다.
