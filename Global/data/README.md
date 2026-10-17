# 공통 데이터

`uv run --project Global poc music`으로 고정 버전의 MUSIC CSV 3개를 검증하고 준비한다. 원시 파형은 다운로드하지 않는다. `poc demo`는 합성 데이터만 생성한다.

정답·포함 대상·변수 정책은 생성된 `protocol.json`, `column_manifest.json`과 검토한다. 같은 환자가 학습·검증·시험 집합에 섞이지 않도록 고정 분할한다. 원본과 환자별 준비 파일은 Git에 올리지 않는다.
