# 개발 환경

## 기본 환경

Docker Desktop을 실행하세요. Windows는 WSL2와 Docker의 Ubuntu 연동을 켜고 Ubuntu 안에 저장소를 복제합니다. GPU는 NVIDIA 드라이버와 WSL2 GPU 연결이 필요하며 macOS에서는 CPU 환경을 사용합니다.

Python 3.12.12·uv 0.9.26과 잠금 파일의 공통 패키지를 사용합니다. NumPy·pandas·scikit-learn·matplotlib·JupyterLab·WFDB·NeuroKit2·MLflow 클라이언트가 포함됩니다. GPU 이미지에는 PyTorch CUDA 패키지가 추가됩니다.

```bash
./dev up -d --wait
./dev url
./dev check
./dev shell
```

일반 공통 코드 변경은 컨테이너에 바로 반영됩니다. Python·공통 패키지 설정·잠금 파일을 바꿨으면 `./dev up -d --wait`로 이미지를 갱신하세요. GPU 환경은 `--gpu`를 추가합니다.

## 개인 패키지

```bash
./dev env add Members/alice xgboost
./dev env status Members/alice
./dev env remove Members/alice xgboost
./dev env sync Members/alice
```

`Members/alice/` 아래 작업은 기본적으로 이 환경을 공유합니다. 공통 패키지는 재사용하고 추가 패키지만 개인 환경에 설치합니다. 공통 패키지와 충돌하는 버전은 허용하지 않습니다.

특정 작업만 별도 패키지가 필요하면:

```bash
./dev env isolate Members/alice/work
./dev env add Members/alice/work shap
./dev env share Members/alice/work
```

기존 작업 폴더에 개인 잠금 파일이나 추가 패키지가 있으면 그 폴더의 환경을 유지합니다. `share`는 공통 복귀가 가능한지 확인하며 개인 요구사항을 임의로 버리지 않습니다. 실행 중인 코드나 커널이 사용하는 환경은 변경할 수 없습니다.

환경 갱신 안내가 나오면 출력된 명령을 실행합니다. 공통 패키지나 실행 장비가 달라지면 `./dev env sync <폴더> --relock`이 필요할 수 있습니다. 설치 손상은 `--repair`로 복구합니다.

```bash
./dev exec Members/alice/work custom_file.py --epochs 10
./dev notebook Members/alice/work --filename analysis.ipynb
```

노트북은 선택한 개인 환경 커널에 연결됩니다. 기존 노트북의 내용은 유지합니다. 추가 패키지 설치 후 실행 중인 커널은 재시작하세요. Jupyter 터미널에서는 `dev`를 사용합니다.

## 공통 패키지 변경

```bash
./dev deps add 패키지명
./dev deps export
./dev up -d --wait
```

`Global/pyproject.toml`·`uv.lock`·`requirements.txt`를 함께 검토합니다. 임의의 `pip install`은 재시작 후 재현을 보장하지 않습니다.

## 연결 문제

- Jupyter: `./dev url`로 현재 주소 확인. 토큰을 공유 문서에 붙이지 않습니다.
- MLflow: 주소는 브라우저뿐 아니라 **컨테이너 안에서** 연결되어야 합니다. 맥 호스트의 SSH 터널은 `host.docker.internal:<포트>` 사용 여부를 확인하세요.
- 등록 실패: 서버·주소·계정·실험 접근 권한 확인 후 같은 `upload` 재실행. `./dev setup --username 계정 --password`로 갱신합니다.
- 인증 파일은 로컬 전용입니다. `.poc.local.toml`을 공유하거나 업로드하지 않습니다. 비어 있지 않은 `MLFLOW_*` 환경 변수가 로컬 설정보다 우선합니다.

## 검증

`./dev verify`는 파일 정책과 남아 있는 환경·결과 등록 테스트를 실행합니다. 실제 Docker 통합 검사는 Docker 호스트에서 `RUN_DOCKER_TESTS=1 uv run --project Global --extra ecg pytest -c Global/pyproject.toml Global/tests/test_workspace_integration.py Global/tests/test_tracking_integration.py -q`로 실행합니다. 이 검사는 별도 프로젝트·볼륨을 생성하고 정리합니다.
