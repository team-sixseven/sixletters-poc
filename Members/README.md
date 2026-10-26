# 개인 작업 공간

`./dev setup --owner alice`는 아래 폴더를 만듭니다. ID는 본인의 영문 소문자 ID로 바꾸세요. 재실행해도 기존 파일은 유지됩니다.

```text
Members/alice/
├── notebooks/       # 분석 노트북
├── experiments/     # 실험별 코드·설정, 예: exp001/train.py
├── results/         # 결과별 metrics.json, params.json, artifacts/
└── models/          # 학습한 모델 파일
```

개인 추가 패키지는 이름 폴더에서 공유합니다. 코드 파일명과 실험 하위 폴더는 자유롭게 정할 수 있습니다. `models/`는 저장된 모델 전용이며 모델 구현 코드는 `experiments/`에 둡니다.

데이터는 개인별로 중복 복사하지 않고 각자 클론한 저장소의 `Global/data/DATA/`에서 읽습니다. 결과·모델·데이터는 커밋하지 않습니다. 결과 등록과 실행 명령은 루트 README를 확인하세요.
