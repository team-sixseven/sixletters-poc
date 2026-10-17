# 공유 결과와 제출 자료

환자 정보가 없는 집계표·그래프·실험 설명을 저장한다. 실행 원본은 `Global/artifacts/`에 유지하고 공유할 결과만 복사한다. 모델·환자별 예측·실행 당시 코드 사본은 Git에 올리지 않는다.

| 위치 예시 | 내용 |
|---|---|
| `Global/reports/이름/실험명.md` | 개인 실험의 목적·방법·결과 |
| `Global/reports/comparisons/비교명/` | 검토한 비교표·그래프 |
| `Global/reports/final/` | 최종 보고서·제출용 표·그래프 |

## 파일 형식과 크기

| 대상 | 파일당 최대 크기 |
|---|---:|
| `Global/reports/` 안의 PDF·PNG·JPG/JPEG·SVG·WebP·DOCX·PPTX | 20MiB |
| 그 밖의 파일과 경로(CSV·JSON·Markdown·노트북 포함) | 1MiB |

추적 파일 검사는 `uv run --project Global python Global/scripts/check_repository.py`로 실행한다. 20MiB를 넘는 제출 자료는 이미지 해상도·압축을 조정하거나, 검토한 파일을 **이 비공개 저장소의 [GitHub Release 첨부 파일](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)**로 올리고 `Global/reports/final/`의 Markdown에 릴리스 링크·파일명·버전을 기록한다. Release의 태그는 해당 제출 코드 커밋을 가리키도록 한다. 링크 수신자에게 저장소 접근 권한이 있어야 한다.

크기 예외는 보고서 형식에만 적용한다. 데이터·모델·환자별 예측·비밀정보는 보고서나 Release에도 넣지 않는다. 환자 식별 열이 있는 CSV·TSV와 MUSIC 환자 ID 값이 있는 CSV·TSV·TXT·JSON·JSONL·NDJSON은 보고서 폴더에서도 차단한다. 알려진 열 이름·ID 형식 이외의 환자 정보나 문서·이미지 내용까지 자동 판별하지는 않으므로 공유 전에 내용을 검토한다. 확장자를 바꾸거나 파일을 쪼개 제한을 우회하지 않는다.

## 실험 기록

개인 실험 기록에는 다음 내용을 적는다.

- 바꾼 방법과 비교 대상, 검증 결과와 한계
- 설정 경로·실행 ID·코드 커밋·실제 학습 파일의 `poc exec` 명령
- 데이터 버전·환자 분할 해시·입력 그룹·사용 장비
- 실제 모델·학습 설정·모델 읽기 방법·추가 패키지와 파일
- MLflow에 기록했다면 실행 링크. 모델 공유·재현을 요청한다면 [모델 보관 완료](../docs/tracking.md#선택한-실행의-모델-보관) 여부

`poc compare`로 생성한 표를 검토하고 복사한다. 여러 사람이 비교표를 손으로 고치지 않고 실행 결과에서 다시 생성한다. 다른 환자·분할의 결과를 같은 조건의 비교로 표시하지 않는다.

제출 자료에는 같은 환자 분할에서 임상 전용·ECG 전용·결합 입력을 비교한 결과, 선택한 개선 방법, 검증을 통한 선택 과정, 최종 평가와 한계를 포함한다. 데이터 출처에는 MUSIC 데이터셋과 사용한 버전을 기록하고 원 제공처의 인용 지침을 따른다.
