# 공용 MLflow 서버 운영

팀원 연결·실험 조회·모델 보관은 [팀 결과 수집](tracking.md)을 따른다.

## 서버 준비

**계속 켜 둘 서버 한 대에서만** 설정한다. CPU 서버로 충분하며, 학습용 CPU·GPU 환경과 별도의 Docker Compose 구성을 사용한다.

서버에도 저장소를 받은 뒤 Ubuntu 터미널에서 실행한다.

```sh
cp .env.tracking.example .env.tracking
openssl rand -hex 32
```

무작위로 생성된 값을 서버 비밀키로 사용하고 `.env.tracking`을 작성한다.

| 항목 | 설정 |
|---|---|
| `MLFLOW_AUTH_ADMIN_USERNAME` | 서버 관리자 계정. 영문·숫자·밑줄·점·하이픈 사용 |
| `MLFLOW_AUTH_ADMIN_PASSWORD` | 최초 관리자 비밀번호. 12자 이상이며 값 전체를 작은따옴표로 감싸서 입력 |
| `MLFLOW_FLASK_SERVER_SECRET_KEY` | 위 명령으로 생성한 비밀키. 재시작해도 유지 |
| `POC_MLFLOW_BIND_HOST` | 서버가 포트를 열 팀 네트워크/VPN 주소 |
| `POC_MLFLOW_PORT` | 기본값 `5000` |
| `MLFLOW_SERVER_ALLOWED_HOSTS` | 접속에 사용할 실제 `호스트:포트`를 쉼표로 구분 |

예를 들어 서버 주소가 `100.80.10.20`이면 바인딩 주소를 해당 IP로 설정하고, 허용 호스트는 `localhost:*,127.0.0.1:*,100.80.10.20:5000`으로 설정한다. 기본 바인딩 `127.0.0.1`은 서버 컴퓨터 내부 확인용이므로 다른 팀원이 접속할 수 없다. 접속할 모든 장비에서 실제 서버 주소에 도달할 수 있어야 한다.

```sh
./dev tracking up
./dev tracking status
```

첫 명령이 이미지 빌드·서버 시작·상태 확인을 수행한다. 기본 HTTP 구성은 접근이 제한된 팀 VPN에서 사용하고, 외부 네트워크에 제공할 때는 HTTPS 구성을 추가한다.

서버 운영자는 팀원별 계정을 만든다. 아래 `member01`은 실제 영문 계정명으로 바꾼다.

```sh
./dev tracking user member01
```

현재 관리자 비밀번호, 새 팀원 비밀번호와 확인값을 차례로 입력한다. 생성되는 계정은 팀 실험을 조회·기록할 수 있고 관리자 권한은 없다. 모든 팀원이 같은 `MLFLOW_EXPERIMENT_NAME`을 사용한다. 서버 운영자도 개인 학습에는 팀원 계정을 사용한다.

최초 서버 시작이 성공하면 `.env.tracking`의 `MLFLOW_AUTH_ADMIN_PASSWORD` 값을 지우고 다음 명령을 다시 실행해 컨테이너에 반영한다. 생성된 관리자 계정은 유지된다. 관리자 이름과 서버 비밀키는 유지하며, 비밀번호 변경은 이 환경 변수 대신 MLflow 계정 관리 기능을 사용한다.

```sh
./dev tracking up
```

| 서버 명령 | 동작 |
|---|---|
| `./dev tracking up` | 빌드·시작·상태 확인 |
| `./dev tracking status` | 실행 상태 확인 |
| `./dev tracking logs` | 서버 로그 확인 |
| `./dev tracking down` | 서버 종료. 보관 데이터 유지 |

서버의 계정·실험 기록·모델·예측 파일은 Docker의 `tracking-data` 볼륨에 남는다. `down` 후 다시 시작해도 유지되며, **볼륨 삭제 옵션을 사용하지 않는다.** 서버를 옮기거나 백업할 때는 서버를 멈춘 상태에서 해당 볼륨 전체를 복사한다. `.env.tracking`과 서버 비밀키도 별도로 보관한다. 로컬 결과 폴더를 지우기 전에는 필요한 모델·재현용 파일의 추가 전송과 백업까지 확인한다. 점수·설정·예측만 전송된 상태에서는 서버에 모델이 없다.
