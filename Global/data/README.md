# 학습 데이터

Drive의 `DATA` 폴더를 압축 해제하여 `Global/data/` 아래에 넣습니다. 아래 경로의 파일은 전부 Git에서 제외되며 MLflow에 자동 전송되지 않습니다.

```text
Global/data/DATA/
├── clinical/
│   ├── clinical_clean.csv
│   └── split/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── ECG_num/
│   ├── ECG_clean.csv
│   └── split/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
└── ECG_tem/
    ├── ECG_tem_clean.csv
    ├── ECG_tem_clean/       # 실제 파형 파일
    └── split/
        ├── train.csv
        ├── val.csv
        └── test.csv
```

`setup`은 빈 폴더만 만듭니다. CSV 다운로드·생성·전처리·분할은 하지 않습니다. 실제 파일이 없는 폴더는 아직 데이터가 준비되지 않은 상태입니다. 필요한 종류만 내려받아도 됩니다.

- `clinical`: 임상 정보와 수치.
- `ECG_num`: 심전도 수치형 특징.
- `ECG_tem_clean.csv`: 환자 ID와 파형 파일을 연결하는 목록. 파형은 `ECG_tem_clean/`에 보관합니다.
- `split`: 팀에서 준비한 학습·검증·시험 데이터입니다. 다른 데이터 종류도 같은 환자 분할 기준을 사용하세요. 저장소가 분할을 다시 생성하거나 일치 여부를 검사하지는 않습니다.

## 코드에서 읽기

작업 폴더나 노트북 위치와 무관하게 저장소 기준 경로를 찾을 수 있습니다.

```python
from sixletters.local_settings import project_root
import pandas as pd

DATA = project_root() / "Global" / "data" / "DATA"
train = pd.read_csv(DATA / "clinical" / "split" / "train.csv")
```

CSV의 구분자·인코딩·환자 ID·정답 컬럼은 전달받은 실제 파일에 맞춰 지정하세요. 파형 목록에 적힌 파일 경로도 절대경로인지, 어느 폴더 기준의 상대경로인지 확인해야 합니다. 이 폴더 배치는 데이터 형식을 변환하지 않습니다.

데이터 버전을 바꿀 때 기존 파일을 무작정 섞지 말고 팀이 공유한 같은 버전을 사용하세요. 사용한 버전과 분할은 결과의 `params.json`에 기록합니다. 원본 데이터는 `results/.../artifacts/`에 넣지 않습니다.
