# Google Drive ECG 데이터를 WSL GPU에서 학습하기

이 문서는 Windows에 연결된 Google Drive 데이터를 WSL의 PyTorch에서 읽어 간단한 GPU 학습을 실행하는 방법을 설명합니다. 2026-10-29에 이 PC에서 실행한 예시입니다.

## 전체 흐름

```text
Google Drive → Windows G: 드라이브 → Windows Python으로 필요한 신호 구간 읽기
             → 표준 출력으로 WSL에 전달 → WSL PyTorch → NVIDIA GPU 학습
```

Windows의 `G:`는 Google Drive 데스크톱 앱이 제공한 드라이브입니다. WSL에 `G:`를 마운트한 것은 아닙니다. 이 예제는 Windows Python을 WSL에서 실행해 신호 구간을 메모리로 전달합니다. 전체 원본을 `D:`에 복사하거나 학습 결과를 저장하지 않습니다. Google Drive는 읽은 파일의 일부나 전체를 로컬 캐시에 저장할 수 있습니다.

## 1. Google Drive 연결

이미 파일 탐색기에 `G:` 드라이브가 있고 대상 폴더가 열리면 이 단계는 완료된 상태입니다.

1. [Google Drive 데스크톱 앱 안내](https://support.google.com/drive/answer/10838124?co=GENIE.Platform%3DDesktop&hl=ko)를 참고해 앱을 설치하고 공유 폴더에 접근 권한이 있는 Google 계정으로 로그인합니다.
2. 공유받은 폴더가 탐색기에 보이지 않으면 웹 Google Drive에서 폴더를 찾아 우클릭한 뒤 **정리 → 바로가기 추가**로 내 드라이브에 추가합니다. [Google 안내](https://support.google.com/drive/answer/9700156?hl=ko)
3. 데스크톱 앱의 설정에서 파일 스트리밍을 사용하는지 확인합니다. 스트리밍 파일은 필요한 데이터를 읽을 때 전송하며 캐시가 생길 수 있습니다. [스트리밍과 미러링 안내](https://support.google.com/drive/answer/13401938?hl=ko)
4. 파일 탐색기에서 대상 폴더를 열고 실제 경로를 확인합니다. 드라이브 문자와 바로가기 경로는 PC마다 다를 수 있습니다.

이 PC의 대상 폴더는 다음과 같습니다.

```text
G:\.shortcut-targets-by-id\11rNObXs4cxNsiNEl_9jtgRXZz45CA_bW\식스세븐 PoC 프로젝트\DATA\ECG_tem\ECG_tem_clean\clean_signal
```

**Windows PowerShell**에서 접근을 확인합니다.

```powershell
$folder = 'G:\.shortcut-targets-by-id\11rNObXs4cxNsiNEl_9jtgRXZz45CA_bW\식스세븐 PoC 프로젝트\DATA\ECG_tem\ECG_tem_clean\clean_signal'
Test-Path -LiteralPath $folder
Get-ChildItem -LiteralPath $folder -Filter '*.npy' | Measure-Object
```

`True`와 파일 수가 출력되어야 합니다. 기존 확인에서는 1,335개였습니다. `False`라면 드라이브 앱 로그인과 공유 권한, 드라이브 문자 및 폴더 경로를 확인하세요.

## 2. Windows Python과 NumPy 준비

다음 명령은 **Windows PowerShell**에서 실행합니다. PowerShell 프롬프트에 Python 코드의 `import`를 직접 입력하지 않습니다.

```powershell
python --version
New-Item -ItemType Directory -Force C:\Temp | Out-Null
python -m pip install --target C:\Temp\holter_numpy numpy
```

이 명령은 NumPy를 `C:\Temp\holter_numpy`에 설치합니다. 기존 Windows Python의 기본 패키지 환경에 설치하지 않습니다. `python`을 찾을 수 없다면 Windows에 Python과 pip를 준비한 뒤 다시 실행하세요.

Windows Python의 실제 경로도 확인합니다.

```powershell
(Get-Command python).Source
```

이 PC에서는 `C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe`였습니다. 아래 WSL 스크립트의 `/mnt/c/Users/Admin/.../python.exe`는 이 경로를 가리킵니다. 다른 PC에서는 Windows 경로를 확인하고 해당 값을 WSL 경로로 바꾸세요. 예를 들어 `C:\Users\name\...`은 `/mnt/c/Users/name/...`입니다.

## 3. WSL의 프로젝트 환경과 GPU 확인

이후 명령은 **WSL Ubuntu 터미널**에서 실행합니다. 예시는 이 PC의 프로젝트 경로를 사용합니다.

```bash
cd /home/sang/sixletters-poc
nvidia-smi
Global/.venv/bin/python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA 사용 가능:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음")'
```

이 PC에서는 `torch 2.13.0+cu126`, `CUDA 사용 가능: True`, `NVIDIA GeForce RTX 4060 Laptop GPU`를 확인했습니다. 프로젝트 환경이 없다면 프로젝트의 [README](../../../README.md)와 `Global/pyproject.toml`을 먼저 확인하고 프로젝트 지침에 따라 GPU 환경을 준비하세요. CUDA가 `False`면 Windows NVIDIA 드라이버와 WSL GPU 설정부터 확인해야 합니다.

## 4. Windows에서 읽을 스크립트 만들기

아래 명령도 **WSL 터미널**에서 실행합니다. WSL의 `/mnt/c/Temp`는 Windows의 `C:\Temp`입니다. 다른 PC라면 `source`의 경로를 1단계에서 확인한 실제 `.npy` 경로로 바꾸세요.

```bash
cat > /mnt/c/Temp/holter_stream_smoke.py <<'PY'
import sys
sys.path.insert(0, r'C:\Temp\holter_numpy')
import numpy as np

source = r'G:\.shortcut-targets-by-id\11rNObXs4cxNsiNEl_9jtgRXZz45CA_bW\식스세븐 PoC 프로젝트\DATA\ECG_tem\ECG_tem_clean\clean_signal\P0218-1.npy'
a = np.load(source, mmap_mode='r', allow_pickle=False)
if a.ndim != 2 or a.shape[1] != 3 or a.shape[0] < 8 * 6000:
    raise ValueError(f'Unexpected input shape: {a.shape}')
windows = np.array(a[:8 * 6000], dtype=np.float32).reshape(8, 6000, 3)
np.save(sys.stdout.buffer, windows, allow_pickle=False)
sys.stdout.buffer.flush()
PY
```

Windows Python은 `np.load(..., mmap_mode='r')`로 파일을 열고 앞부분에서 6,000개 샘플씩 8개 구간을 가져옵니다. 200Hz 기준 각 구간은 30초입니다. Windows 표준 출력을 통해 NumPy 배열을 WSL 프로세스에 전달하므로 데이터 출력에 다른 로그를 섞지 않습니다.

## 5. WSL 학습 스크립트 만들고 실행하기

**WSL 터미널**에서 다음을 실행합니다. 다른 PC라면 `subprocess.run`에 지정된 Windows Python 실행 파일 경로를 2단계에서 확인한 경로에 맞게 바꾸세요.

```bash
cat > /tmp/holter_train_smoke.py <<'PY'
import io
import subprocess
import time
import numpy as np
import torch
from torch import nn

start = time.perf_counter()
proc = subprocess.run(
    ['/mnt/c/Users/Admin/AppData/Local/Programs/Python/Python312/python.exe', r'C:\Temp\holter_stream_smoke.py'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=True,
)
a = np.load(io.BytesIO(proc.stdout), allow_pickle=False)
if a.shape != (8, 6000, 3) or not np.isfinite(a).all():
    raise ValueError('Invalid input data')
mean = a.mean(axis=1, keepdims=True)
std = a.std(axis=1, keepdims=True)
a = (a - mean) / np.maximum(std, 1e-6)
x = torch.from_numpy(a.transpose(0, 2, 1).copy()).to('cuda')
torch.manual_seed(42)
model = nn.Sequential(nn.Conv1d(3, 16, 7, padding=3), nn.ReLU(), nn.Conv1d(16, 3, 7, padding=3)).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []
for step in range(10):
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(model(x), x)
    if not torch.isfinite(loss):
        raise ValueError('Non-finite loss')
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
torch.cuda.synchronize()
print('torch:', torch.__version__)
print('gpu:', torch.cuda.get_device_name(0))
print('windows:', tuple(a.shape), '(8 windows x 6000 samples x 3 leads)')
print('steps:', len(losses))
print('losses:', ', '.join(f'{v:.6f}' for v in losses))
print('peak_gpu_mib:', round(torch.cuda.max_memory_allocated() / 2**20, 2))
print('elapsed_seconds_including_data_read:', round(time.perf_counter() - start, 2))
if not losses[-1] < losses[0]:
    raise RuntimeError('Loss did not decrease')
PY

cd /home/sang/sixletters-poc
Global/.venv/bin/python /tmp/holter_train_smoke.py
```

스크립트는 각 구간·채널을 정규화한 뒤 작은 1D CNN이 입력 신호를 복원하도록 10단계 학습합니다. 정답 레이블을 이용한 환자 분류가 아닙니다. `/tmp`는 임시 경로이므로 재부팅 후 스크립트가 없어질 수 있습니다. 위 명령으로 다시 만들 수 있습니다.

## 6. 실행 결과와 해석

이 PC에서 실제로 확인한 결과입니다.

```text
torch: 2.13.0+cu126
gpu: NVIDIA GeForce RTX 4060 Laptop GPU
windows: (8, 6000, 3) (8 windows x 6000 samples x 3 leads)
steps: 10
losses: 0.900029, 0.855742, 0.812237, 0.769435, 0.727181, 0.685338, 0.643640, 0.602029, 0.560447, 0.519025
peak_gpu_mib: 11.55
elapsed_seconds_including_data_read: 3.94
```

손실은 신호 복원 오차이며, 10번의 학습에서 감소했습니다. 시간과 손실은 환경에 따라 달라집니다. 이 결과는 데이터 연결, GPU 사용, 순전파·역전파·가중치 갱신을 확인한 사례이며 모델 성능이나 의료적 유효성 평가는 아닙니다. 원본 ECG 한 파일의 앞부분만 사용했고 모델 파일은 저장하지 않았습니다.

## 문제가 생기면

- **PowerShell에서 `import` 오류:** Python 코드를 PowerShell에 직접 붙여 넣지 말고, 이 문서처럼 스크립트를 실행하거나 먼저 `python`을 입력해 `>>>` 프롬프트로 들어갑니다.
- **`No module named numpy`:** 2단계의 NumPy 설치 명령을 실행하고 스크립트의 `C:\Temp\holter_numpy`가 일치하는지 확인합니다.
- **파일 경로 오류:** Windows PowerShell에서 `Test-Path`로 실제 경로와 공유 권한을 확인하고 Windows 스크립트의 `source`를 수정합니다.
- **Windows Python 실행 파일을 찾을 수 없음:** `(Get-Command python).Source`로 경로를 확인하고 WSL 스크립트의 실행 파일 경로를 수정합니다.
- **CUDA 사용 불가:** WSL에서 `nvidia-smi`와 3단계 PyTorch 확인 명령을 실행하고 프로젝트 GPU 환경과 NVIDIA 드라이버를 확인합니다.
- **처음 읽을 때 지연 또는 시간 초과:** Google Drive 앱 로그인과 네트워크를 확인합니다. 첫 읽기는 Drive 캐시 상태와 네트워크 속도에 따라 오래 걸릴 수 있습니다. 필요하면 WSL 스크립트의 `timeout=180`을 조정합니다.

이전에 중지한 `D:\Holter` 다운로드의 일부 파일은 이 절차에 필요하지 않지만 별도로 남아 있을 수 있습니다.
