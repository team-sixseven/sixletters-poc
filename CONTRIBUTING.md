# 식스세븐 협업 규칙

첫 설치·학습은 [README](README.md)를 따른다. 아래 명령은 Docker를 켜 둔 상태에서 **Ubuntu·macOS 터미널의 저장소 폴더**에서 실행한다. Jupyter의 **Launcher → Terminal**에서는 데이터 준비·학습·비교 명령의 `./dev`를 `dev`로 바꿔 실행한다.

```text
이슈 → 작업 브랜치 → PR·리뷰·검사 → develop(팀 통합) → 릴리스 PR → main(제출)
```

`main`·`develop`에 직접 커밋·push하거나 강제 push·삭제하지 않는다.

## 1. 작업 시작

GitHub **Issues → New issue**에 목표와 수정할 파일을 적는다. 같은 목표의 반복 실험은 하나의 이슈·브랜치에서 진행한다. 아래 번호·작업명을 실제 값으로 바꾼다.

```sh
git switch develop
git pull --ff-only origin develop
git switch -c experiment/12-clinical-experiments
```

브랜치는 `종류/이슈번호-작업명`, 커밋·PR 제목은 `종류: 변경 내용`으로 쓴다. 종류는 `experiment`(실험), `feat`(기능), `fix`(오류), `perf`(성능), `docs`(문서), `chore`(환경)다.

- 개인 ID는 **공백 없는 영문 소문자**로 정하고 팀원 간 중복 없이 유지한다. 문서의 `yourname`은 `./dev init`에 등록한 본인 ID로 바꾼다.
- `Members/yourname/`에서 자유롭게 실험한다. 다른 사람의 폴더는 합의 후 수정한다.
- `Global/`·Docker·공통 패키지는 먼저 조율하고 PR을 순서대로 반영한다. 공용 노트북·보고서의 같은 파일을 동시에 수정하지 않는다.
- **학습 중인 코드·설정·데이터와 학습·평가 관련 공통 코드는 변경하지 않는다.** 영향 범위는 [공통 변경 기준](Global/docs/parallel-work.md#4-학습하고-결과-확인하기)을 따른다.
- 개인 실험에서 준비 데이터·환자 분할을 고치지 않는다. 공동 기준을 바꾸면 데이터 버전·준비 명령을 함께 갱신하고 비교할 실험을 다시 실행한다.

## 2. 변경 확인과 커밋

실제 작업한 경로만 추가한다. 개인 노트북도 공유한다면 해당 파일을 함께 추가한다.

```sh
git status
git diff
git add "Members/yourname/experiments/exp001/"
git diff --cached
./dev verify
```

- 검사는 노트북 출력·실행 번호를 지운다. 필요한 결과는 먼저 `Global/artifacts/`에 보관한다.
- 검사가 파일을 수정하면 확인 후 다시 `git add`하고 검사한다. 실패한 검사는 우회하지 않는다.
- `verify`는 작업 브랜치와 로컬 변경을 함께 검사한다. 개인 실험만 바뀌면 필요한 계약 검사를, 공통 코드·패키지가 바뀌면 전체 회귀 검사를 실행한다. `--full`은 전체 검사, `--staged`는 커밋할 파일만 확인할 때 사용한다. 학습 품질·데이터 누수는 실제 실행 결과와 리뷰로 확인한다.
- 개인 환경을 사용했다면 생성·변경된 `requirements.in`·`requirements.lock`·`environment.json`을 코드·설정과 함께 커밋한다. 기본 위치는 `Members/yourname/`이며, 분리한 실험은 해당 실험 폴더다. 공통 환경만 사용하면 개인 환경 파일을 만들지 않는다. 환경 선택 파일 `environment.toml`이 생성되면 함께 커밋한다. `.venv/`·커널 등록 정보·캐시·모델 출력은 제외한다.
- **데이터·환자 ID·분할 파일·모델·환자별 예측·개인 연결 설정·비밀키는 커밋하지 않는다.** 환자 식별 열이 있는 CSV·TSV와 MUSIC 환자 ID 값이 있는 텍스트 자료는 위치와 관계없이 차단한다. 환자 정보가 없는 집계 자료만 [보고서 파일·크기 기준](Global/reports/README.md)에 따라 공유한다.

검사 통과 후 커밋한다.

```sh
git commit -m "experiment: 임상 데이터 기본 모델 비교"
```

이어서 [최신 develop 반영·검사·push](#develop의-변경을-가져오거나-충돌을-해결할-때)를 마친다.

## 3. PR 만들기와 병합

1. GitHub **Pull requests → New pull request**에서 **base: `develop` / compare: 내 작업 브랜치**를 선택한다.
2. PR 양식에 `Closes #12`처럼 이슈를 연결한다. 실험은 데이터 버전·입력 그룹·실행 명령·실행 ID·집계 결과를, 사전 실습은 노트북 경로·확인 내용을 적는다. MLflow를 사용했다면 실행 링크도 남긴다.
3. **Reviewers**에서 동료에게 검토를 요청한다. 작업 중에는 제목에 `[WIP]`를 붙인다.
4. 병합 직전에도 최신 `develop`을 반영하고 아래 조건을 확인한 뒤 **Create a merge commit**으로 병합한다.

| 병합 조건 | 확인할 내용 |
|---|---|
| 동료 검토 | 작성자 외 팀원 1명이 최신 변경 승인 |
| 자동 검사 | 최신 `develop`을 반영한 커밋의 `validate` 성공 |
| 검토 의견 | 수정 요청·검토 의견 처리 완료 |
| 작업 상태 | `[WIP]` 제거·병합 충돌 없음 |

Free 비공개 저장소에서는 조건이 강제되지 않으므로 직접 확인한다. 검사 미실행은 성공이 아니다. [CI 실행 기준](Global/docs/github-setup.md#자동-검사)

**실험 리뷰에서 확인할 것:**

- 전처리·변수 선택·학습에 학습 환자만 사용했는가? 검증·테스트 정답이나 사후 정보가 섞이지 않았는가?
- 검증 환자 ID와 예측 확률이 정확히 연결되고, `Experiment.save()`에 실제 모델·시드·학습 설정을 기록했는가?
- `./dev exec`로 기록한 실행 파일·인자와 실제 학습 방법이 일치하는가? 모델 읽기 방법·장비 조건·개인 패키지 고정 파일·추가 설정을 포함했는가?
- 제출 후보는 저장한 모델을 재학습 없이 불러와 검증 예측을 재현할 수 있는가? 테스트 전에 후보·임계값·비교 기준 모델을 확정했는가?
- [비교 조건](Global/docs/parallel-work.md#5-두-가지-비교를-구분하기)을 지켰고 기준·대상 실행 ID와 비교 명령을 남겼는가? 점수 차이의 신뢰구간도 확인하고, 바뀔 수 있는 목록 번호만 적지 않는다.
- MLflow 링크를 공유했다면 `./dev sync`로 실패·서버 삭제 여부를 확인했는가? 모델 공유·재현을 요청했다면 [모델 보관 완료](Global/docs/tracking.md#선택한-실행의-모델-보관)도 확인한다.

공통 검사는 개인 학습 코드의 데이터 누수까지 자동 검증하지 않는다. 리뷰 수정은 같은 브랜치에서 검사·커밋·push하고, 새 커밋의 검사와 승인을 다시 확인한다.

## 4. 병합 후 다음 작업

GitHub에서 병합을 확인한 뒤 작업 브랜치만 **Delete branch**로 삭제한다. 아래 브랜치명도 실제 값으로 바꾼다.

```sh
git switch develop
git pull --ff-only origin develop
git branch -d experiment/12-clinical-experiments
git fetch --prune origin
```

환경 파일이 바뀌었으면 [환경 갱신](Global/docs/development.md#4-파일-보관과-환경-갱신)을 진행한다. 개인 실험 폴더는 유지하고 다음 작업은 새 이슈·브랜치에서 시작한다.

## develop의 변경을 가져오거나 충돌을 해결할 때

**진행 중인 학습을 마치고 변경을 커밋한 뒤, 작업 브랜치에서** 실행한다.

```sh
git fetch origin
git merge origin/develop
```

충돌이 나면 `git status`의 파일을 관련 팀원과 확인하고 두 변경을 반영해 충돌 표시를 지운다. 상대방 변경을 통째로 덮어쓰지 않는다. 수정한 파일을 `git add`한다.

환경 변경은 [갱신 절차](Global/docs/development.md#4-파일-보관과-환경-갱신)를 적용한 뒤 아래 검사를 실행한다. 충돌 없이 병합된 경우도 검사한다.

```sh
./dev verify
```

검사가 수정한 파일은 확인 후 다시 추가·검사한다. 충돌을 해결했다면 `git commit`으로 병합을 마친다. 이후 `git push`한다. 최초 push는 `git push -u origin HEAD`를 사용한다.

## 패키지를 추가하거나 변경할 때

추가 패키지가 없는 새 실험은 공통 환경을 사용한다. 패키지를 추가하면 `Members/yourname/`의 개인 환경을 만들어 여러 실험이 공유한다. 공통 패키지는 재사용하고, 공통에 없는 패키지만 개인 환경에 설치한다. 전체 패키지 버전 기록은 함께 커밋한다. 특정 실험만 별도로 관리하려면 `./dev env isolate 실험명`을 사용한다. 패키지 추가·삭제·환경 갱신 전에는 **해당 환경을 공유하는 모든 학습과 노트북 커널**을 종료한다. 다른 팀원의 환경은 바뀌지 않는다. [환경 확인·분리·변경 절차](Global/docs/development.md#개인-실험-패키지)

개인 노트북은 `./dev notebook 실험명`으로 만들면 패키지 준비·커널 연결을 함께 처리한다. 같은 실험을 `./dev exec`로 실행하기 전에는 해당 커널을 종료한다. 서로 다른 실험은 같은 환경으로 동시에 실행할 수 있다. 노트북 탭만 닫아서는 커널이 종료되지 않는다. 사용 중인 실험·커널은 `./dev env status 실험명`으로 확인한다. [커널 선택·종료 방법](Global/docs/development.md#개인-노트북-실행-환경)

공통 패키지 변경은 먼저 조율하고 `Global/pyproject.toml`·`Global/uv.lock`·`Global/requirements.txt`를 한 PR에 반영한다. [공통 패키지 변경 절차](Global/docs/development.md#공통-패키지)

## 제출 버전을 main에 반영할 때

통합된 코드·같은 환경에서 검증 결과를 확인하고 [최종 평가](Global/docs/final-evaluation.md)로 후보·임계값을 고정한 뒤 테스트한다. 테스트 결과로 후보를 다시 고르거나 설정을 개선하지 않는다. [필수 비교·제출 자료](Global/docs/parallel-work.md#6-공유하고-제출-후보-선택하기)를 확인한다. **base가 `main`, compare가 `develop`인 릴리스 PR**에 같은 병합 조건을 적용하고 merge commit으로 병합한다. `develop`은 삭제하지 않는다.
