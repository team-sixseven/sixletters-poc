"""Personal environments and MLflow uploads."""

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

from filelock import Timeout

from .local_settings import project_root, read_settings, validate_owner, write_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="식스세븐 개발 환경과 결과 등록")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="개인 ID와 MLflow 접속 설정")
    setup.add_argument("--owner")
    setup.add_argument("--uri")
    setup.add_argument("--username")
    setup.add_argument("--experiment")
    setup.add_argument("--password", action="store_true", help="화면에 표시하지 않고 비밀번호 입력")
    upload = commands.add_parser("upload", help="완성한 결과 폴더 등록·재시도")
    upload.add_argument("directory", type=Path)
    upload.add_argument("--model", type=Path)
    commands.add_parser("status", help="개인 ID와 환경 설정 상태")
    execute = commands.add_parser("exec", help="개인 패키지 환경에서 Python 파일 실행")
    execute.add_argument("directory", type=Path)
    execute.add_argument("script", type=Path)
    execute.add_argument("args", nargs=argparse.REMAINDER)
    notebook = commands.add_parser("notebook", help="개인 노트북 생성·커널 연결")
    notebook.add_argument("directory", type=Path)
    notebook.add_argument("--filename", default="analysis.ipynb")
    notebook.add_argument("--relock", action="store_true")
    env = commands.add_parser("env", help="개인 추가 패키지 관리")
    env.add_argument(
        "action", choices=["add", "remove", "sync", "status", "kernel", "isolate", "share"]
    )
    env.add_argument("directory", type=Path)
    env.add_argument("packages", nargs="*")
    env.add_argument("--relock", action="store_true")
    env.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    try:
        root = project_root()
        if args.command == "setup":
            settings = read_settings(root)
            owner = args.owner or settings.get("owner") or input("개인 ID (영문 소문자): ").strip()
            validate_owner(owner)
            settings = {
                k: v
                for k, v in settings.items()
                if k in {"owner", "uri", "username", "password", "experiment"}
            }
            settings["owner"] = owner
            for key in ("uri", "username", "experiment"):
                value = getattr(args, key)
                if value is not None:
                    settings[key] = value
            if args.password:
                settings["password"] = getpass.getpass("MLflow 비밀번호: ")
            folders = [
                root / "Members" / owner / name
                for name in ("notebooks", "experiments", "results", "models")
            ] + [
                root / "Global/data/DATA" / name
                for name in (
                    "clinical/split",
                    "ECG_num/split",
                    "ECG_tem/split",
                    "ECG_tem/ECG_tem_clean",
                )
            ]
            for folder in folders:
                for path in (folder, *folder.parents):
                    if path == root:
                        break
                    if path.is_symlink() or (path.exists() and not path.is_dir()):
                        raise ValueError(f"일반 폴더가 필요합니다: {path.relative_to(root)}")
            for folder in folders:
                folder.mkdir(parents=True, exist_ok=True)
            write_settings(root, settings)
            print(f"개인 설정 저장: Members/{owner} (접속 정보는 Git에 포함하지 않습니다)")
        elif args.command == "upload":
            from .upload import upload

            result = upload(root, args.directory, args.model)
            print(
                f"등록 완료 · 지표 {result['metrics']} · 설정 {result['params']} · 파일 {result['files']}"
            )
            print(result["url"])
        elif args.command == "status":
            settings = read_settings(root)
            print(f"개인 ID: {settings.get('owner', '미설정')}")
            print(
                "MLflow 주소: "
                + (
                    "설정됨"
                    if os.environ.get("MLFLOW_TRACKING_URI") or settings.get("uri")
                    else "미설정 (dev setup --uri 서버주소)"
                )
            )
            print("결과 전송: dev upload Members/<ID>/results/<결과명>")
        else:
            from . import personal_env as pe

            if args.command == "exec":
                return pe.execute_script(args.directory, args.script, args.args)
            if args.command == "notebook":
                from .notebook_workspace import prepare_notebook

                print(
                    prepare_notebook(args.directory, filename=args.filename, relock=args.relock)[
                        "path"
                    ]
                )
            else:
                if args.packages and args.action not in {"add", "remove"}:
                    raise ValueError("패키지명은 add/remove에만 사용하세요.")
                if args.action in {"add", "remove"}:
                    if not args.packages:
                        raise ValueError("패키지명을 입력하세요.")
                    result = getattr(pe, f"env_{args.action}")(args.directory, args.packages)
                elif args.action == "sync":
                    result = pe.env_sync(args.directory, relock=args.relock, repair=args.repair)
                elif args.action == "kernel":
                    result = pe.env_kernel(args.directory, relock=args.relock)
                else:
                    result = getattr(pe, f"env_{args.action}")(args.directory)
                print(
                    json.dumps(result, ensure_ascii=False, default=str)
                    if isinstance(result, dict)
                    else result
                )
        return 0
    except Timeout:
        print("같은 폴더의 작업이 진행 중입니다. 완료 후 다시 실행하세요.", file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError) as error:
        print(f"실행 실패: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
