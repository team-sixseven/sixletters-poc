# 공용 MLflow 서버 운영

팀원 연결·실험 조회·모델 보관은 [README](../../README.md)을 따른다.

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

### Windows + WSL2에서 같은 LAN으로 공유

WSL2 NAT 환경에서는 Docker 포트 공개 외에 Windows 포트 전달이 필요하다. Windows에 실제 할당된 LAN IPv4 주소를 확인하고 `.env.tracking`에 `POC_MLFLOW_BIND_HOST=0.0.0.0`을 설정한다. `MLFLOW_SERVER_ALLOWED_HOSTS`에는 `localhost:*`, `127.0.0.1:*`, WSL의 `IPv4:5000`, 공유할 Windows의 `IPv4:5000`을 명시한다. 기존 계정·DB 볼륨은 유지한 채 서버를 다시 실행한다.

```bash
./dev tracking up
```

그다음 **Windows PowerShell을 관리자 권한으로 실행**하고 다음 명령을 사용한다. 아래 `192.168.10.88`은 LAN 주소 예시이며 현재 PC 주소로 바꾼다. 저장소 경로와 WSL 배포판 이름도 실제 환경에 맞춘다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "\\wsl.localhost\Ubuntu-24.04\home\sang\sixletters-poc\Global\scripts\configure_mlflow_lan.ps1" -ListenAddress 192.168.10.88 -Distribution Ubuntu-24.04
```

스크립트는 Windows의 지정 IP, TCP/5000만 WSL의 TCP/5000으로 전달하고, 해당 포트의 Windows 방화벽 수신을 `LocalSubnet`으로 제한한다. 다른 서비스의 포트 규칙은 바꾸지 않는다. 관리자 권한, Windows IP 소유 여부, WSL health를 먼저 검사하고, 기존에 다른 용도로 사용 중인 포트 전달은 덮어쓰지 않는다. `ExecutionPolicy Bypass`는 이번 PowerShell 프로세스에만 적용한다.

기본 수신 범위는 `LocalSubnet`이며, 라우팅이 허용된 다른 팀 대역까지 필요하면 스크립트의 `-RemoteAddress`에 신뢰하는 서브넷을 명시할 수 있다. 같은 이름으로 생성한 기존 규칙도 이 범위로 갱신한다. 예: `-RemoteAddress "192.168.10.0/24","192.168.50.0/24"`. 이 옵션은 Windows 방화벽만 설정하며 공유기/사내 VLAN 간 통신은 별도로 허용되어 있어야 한다. 연결 해제된 어댑터에 남은 IP는 원격 접속용 주소로 사용할 수 없으며 스크립트도 연결 상태를 검사한다.

팀원은 같은 LAN에서 `http://<Windows-LAN-IP>:5000`에 본인 계정으로 로그인한다. 본인 checkout의 tracking URI도 이 주소로 설정한다. 관리자 비밀번호와 다른 팀원의 비밀번호를 공유하지 않는다. 원격 팀원이 다른 네트워크에 있다면 이 설정만으로 접속할 수 없으며 팀 VPN 또는 별도 HTTPS 구성이 필요하다. 공유기 인터넷 포트포워딩은 이 절차에서 설정하지 않는다.

WSL 또는 Windows 재시작 후 주소가 바뀌면 `.env.tracking` 허용 호스트를 갱신하고 이 스크립트를 다시 실행한다. Windows LAN 주소가 바뀌면 이전 주소의 규칙을 먼저 아래 `-Remove`로 제거한다. 되돌릴 때는 같은 관리자 PowerShell에서 사용한 주소와 함께 `-Remove`를 지정하고, `.env.tracking`의 바인딩을 `127.0.0.1`, 허용 호스트를 `localhost:*,127.0.0.1:*`로 복구한 뒤 서버를 재시작한다. 볼륨은 삭제하지 않는다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "\\wsl.localhost\Ubuntu-24.04\home\sang\sixletters-poc\Global\scripts\configure_mlflow_lan.ps1" -ListenAddress 192.168.10.88 -Remove
```

Windows 자체 health 확인 외에 실제 팀원 PC에서 로그인과 실험 조회까지 확인해야 LAN 접속 검증이 완료된다. 근거: [Microsoft WSL 네트워크 문서](https://learn.microsoft.com/windows/wsl/networking).

### 팀원 계정 관리

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

서버의 계정·실험 기록·모델·예측 파일은 Docker의 `tracking-data` 볼륨에 남는다. `down` 후 다시 시작해도 유지되며, **볼륨 삭제 옵션을 사용하지 않는다.** 서버를 옮기거나 백업할 때는 서버를 멈춘 상태에서 해당 볼륨 전체를 복사한다. `.env.tracking`과 서버 비밀키도 별도로 보관한다. 로컬 결과 폴더를 지우기 전에는 필요한 필요한 모델·설정 파일의 전송과 백업까지 확인한다. `--model`로 등록하지 않았다면 서버에 모델이 없다.
