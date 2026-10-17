"""Lock and run independent experiment environments without changing the shared runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock, Timeout
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from .local_settings import project_root

UV_VERSION = "0.9.26"
INPUT_FILE = "requirements.in"
LOCK_FILE = "requirements.lock"
MANIFEST_FILE = "environment.json"
STATE_FILE = ".poc-env-state.json"
ENV_DIR = ".venv"
SHARED_PTH = "_sixletters_shared.pth"
SCOPE_FILE = "environment.toml"
LOCK_NAMES = (INPUT_FILE, LOCK_FILE, MANIFEST_FILE)
MAX_REQUIREMENTS_BYTES = 64 * 1024
MAX_LOCK_BYTES = 1024 * 1024
GATE_TIMEOUT_SECONDS = 600
LOCK_REFRESH_REASON = (
    "개인 패키지 설정·잠금 파일 또는 공통 패키지·Python·실행 환경이 저장된 기록과 다릅니다. "
    "현재 환경에 맞게 패키지 버전을 다시 고정해야 합니다."
)
INSTALL_REPAIR_REASON = (
    "패키지 버전 기록은 유효하지만 설치된 환경이 없거나 기록과 다릅니다. "
    "기록된 버전으로 다시 설치해야 합니다."
)


class EnvironmentRefreshError(ValueError):
    """A recoverable package-record/install problem, distinct from unrelated errors."""

    def __init__(self, reason: str, option: str, *, experiment: Path | None = None):
        self.reason = reason
        self.option = option
        self.experiment = experiment
        message = reason
        if experiment is not None:
            message += "\n다음: " + _env_command(experiment, "sync", option)
        super().__init__(message)


@contextmanager
def _environment_guidance(experiment: Path):
    try:
        yield
    except EnvironmentRefreshError as error:
        active = _active_readers(environment_directory(experiment))
        if active:
            raise _busy(active) from error
        if error.experiment is not None:
            raise
        raise EnvironmentRefreshError(error.reason, error.option, experiment=experiment) from error


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _directory(path: Path) -> Path:
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Environment directories must not contain symbolic links")
    if not path.is_dir():
        raise ValueError("Select an existing experiment directory")
    return path.resolve()


def _member_directory(path: Path) -> Path | None:
    if path.parent.name == "experiments" and path.parent.parent.parent.name == "Members":
        return path.parent.parent
    return None


def environment_directory(experiment_dir: Path) -> Path:
    """Resolve the active scope without creating or changing an environment."""
    path = _directory(experiment_dir)
    selector = path / SCOPE_FILE
    if selector.is_symlink():
        raise ValueError("environment.toml must not be a symbolic link")
    owner = _member_directory(path)
    scope = None
    if selector.exists():
        if not selector.is_file() or selector.stat().st_size > MAX_REQUIREMENTS_BYTES:
            raise ValueError("environment.toml must be a text file smaller than 64 KiB")
        try:
            value = tomllib.loads(selector.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(
                'environment.toml must contain scope = "shared" or "isolated"'
            ) from error
        if (
            set(value) != {"scope"}
            or not isinstance(value["scope"], str)
            or value["scope"] not in {"shared", "isolated"}
        ):
            raise ValueError('environment.toml must contain only scope = "shared" or "isolated"')
        scope = value["scope"]
    if scope == "shared" and owner is None:
        raise ValueError("Shared environments require Members/<owner>/experiments/<name>")
    if scope == "isolated" or owner is None:
        return path
    if scope is None:
        # Preserve previously prepared experiment environments, including hand-edited inputs.
        if any(
            (path / name).exists() or (path / name).is_symlink()
            for name in (LOCK_FILE, MANIFEST_FILE)
        ):
            return path
        source = path / INPUT_FILE
        if source.is_symlink():
            raise ValueError("requirements.in must not be a symbolic link")
        if source.exists() and _requirements(_input_content(path)):
            return path
    return _directory(owner)


def environment_resources(experiment_dir: Path) -> dict[str, bytes]:
    """Snapshot only active package records; virtual environments are never archived."""
    path = environment_directory(experiment_dir)
    resources = {}
    for name in LOCK_NAMES:
        source = path / name
        if source.is_symlink():
            raise ValueError(f"Environment {name} must not be a symbolic link")
        if not source.exists():
            continue
        limit = MAX_REQUIREMENTS_BYTES if name == INPUT_FILE else MAX_LOCK_BYTES
        if not source.is_file() or source.stat().st_size > limit:
            raise ValueError(f"Environment {name} is not a regular file or exceeds its size limit")
        with source.open("rb") as handle:
            resources[name] = handle.read(limit + 1)
        if len(resources[name]) > limit:
            raise ValueError(f"Environment {name} exceeds its size limit")
        try:
            text = resources[name].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Environment {name} must contain UTF-8 text") from error
        if any(ord(char) < 32 and char not in "\t\r\n" for char in text):
            raise ValueError(f"Environment {name} must contain text, not binary data")
    return resources


def _check_environment(path: Path) -> None:
    for name in (*LOCK_NAMES, STATE_FILE, ENV_DIR, ".poc-env.lock", ".poc-env-readers"):
        if (path / name).is_symlink():
            raise ValueError(f"Environment {name} must not be a symbolic link")


def _experiment(path: Path) -> Path:
    path = _directory(path)
    if not (path / "config.toml").is_file():
        raise ValueError("Select an existing experiment directory containing config.toml")
    for name in ("config.toml", ".poc-execution.lock"):
        if (path / name).is_symlink():
            raise ValueError(f"Experiment {name} must not be a symbolic link")
    _check_environment(environment_directory(path))
    return path


def _busy(active: list[dict] | None = None) -> ValueError:
    details = ""
    if active:
        labels = [
            f"{item['experiment']} ({item['kind']}, PID {item['pid'] or 'unknown'})"
            for item in active
        ]
        details = " 사용 중: " + ", ".join(labels) + "."
    return ValueError(
        "실험이 실행 중이거나 패키지 환경을 변경 중입니다. "
        "학습이 끝날 때까지 기다리거나 Jupyter에서 Kernel > Shut Down Kernel로 커널을 "
        "종료한 뒤 다시 실행하세요. 노트북 탭을 닫는 것만으로는 종료되지 않습니다." + details
    )


@contextmanager
def _file_locked(path: Path, *, timeout: float = 0):
    try:
        with FileLock(path, timeout=timeout):
            yield
    except Timeout as error:
        raise _busy() from error


@contextmanager
def _locked(path: Path):
    """Serialize environment mutations and reader registration."""
    _check_environment(path)
    with _file_locked(path / ".poc-env.lock", timeout=GATE_TIMEOUT_SECONDS):
        yield


@contextmanager
def _experiment_locked(path: Path):
    with _file_locked(path / ".poc-execution.lock"):
        yield


def _reader_metadata(lease: Path) -> dict:
    fallback = {"experiment": "unknown", "kind": "unknown", "pid": None}
    source = lease.with_suffix(".json")
    if source.is_symlink():
        raise ValueError("Environment reader metadata must not be a symbolic link")
    try:
        if source.stat().st_size > 8192:
            return fallback
        value = json.loads(source.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("experiment"), str)
            or not 0 < len(value["experiment"]) <= 200
            or any(ord(char) < 32 for char in value["experiment"])
            or not isinstance(value.get("kind"), str)
            or value["kind"] not in {"script", "kernel"}
            or type(value.get("pid")) is not int
            or value["pid"] < 1
        ):
            return fallback
        return {key: value[key] for key in ("experiment", "kind", "pid")}
    except (OSError, ValueError):
        return fallback


def _active_readers(path: Path, *, cleanup: bool = False) -> list[dict]:
    """Only an OS-backed held lease counts as active, including across processes."""
    directory = path / ".poc-env-readers"
    if not directory.exists():
        return []
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Environment reader registry must be a regular directory")
    active = []
    for lease in sorted(directory.iterdir()):
        if lease.is_symlink() or not lease.is_file():
            raise ValueError("Environment reader registry contains an invalid entry")
        if lease.suffix == ".json":
            if cleanup and not lease.with_suffix(".lock").exists():
                lease.unlink(missing_ok=True)
            continue
        if lease.suffix != ".lock":
            raise ValueError("Environment reader registry contains an invalid entry")
        try:
            with FileLock(lease, timeout=0):
                pass
        except Timeout:
            active.append(_reader_metadata(lease))
        else:
            if cleanup:
                lease.unlink(missing_ok=True)
                lease.with_suffix(".json").unlink(missing_ok=True)
    return active


def _assert_unused(path: Path) -> None:
    active = _active_readers(path, cleanup=True)
    if active:
        raise _busy(active)


def _lease(path: Path, experiment: Path | None = None, *, kind: str = "script") -> FileLock:
    directory = path / ".poc-env-readers"
    directory.mkdir(exist_ok=True)
    if directory.is_symlink():
        raise ValueError("Environment reader registry must not contain symbolic links")
    location = directory / f"{uuid.uuid4().hex}.lock"
    lease = FileLock(location, timeout=0)
    lease.acquire()
    try:
        metadata = {
            "experiment": experiment.name if experiment else path.name,
            "kind": kind,
            "pid": os.getpid(),
            "started_at": datetime.now(UTC).isoformat(),
        }
        location.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )
    except BaseException:
        lease.release()
        raise
    return lease


def _uses_common_runtime(path: Path, content: str) -> bool:
    """Only a new, unpinned scope without extra packages can use the current runtime."""
    return not _requirements(content) and not any(
        (path / name).exists() or (path / name).is_symlink()
        for name in (LOCK_FILE, MANIFEST_FILE, STATE_FILE, ENV_DIR)
    )


def _prepare(path: Path, *, relock: bool = False) -> Path:
    content = _input_content(path)
    if _uses_common_runtime(path, content):
        return Path(sys.executable)
    if not relock and all((path / name).is_file() for name in (LOCK_FILE, MANIFEST_FILE)):
        lock, manifest = _read_locked(path, content, _baseline())
        if _ready(path, manifest) and _installed_matches(path, manifest, lock):
            return _python(path / ENV_DIR)
    _assert_unused(path)
    return _sync(path, content, relock=relock, force_install=False)


@contextmanager
def _running(experiment: Path):
    with _environment_guidance(experiment), _experiment_locked(experiment):
        path = environment_directory(experiment)
        with _locked(path):
            python = _prepare(path)
            lease = _lease(path, experiment)
        try:
            yield path, python
        finally:
            lease.release()


def _requirement(value: str) -> Requirement:
    if not isinstance(value, str) or "\n" in value or "\r" in value or len(value) > 2048:
        raise ValueError("Use one PyPI package requirement per line")
    try:
        requirement = Requirement(value.strip())
    except InvalidRequirement as error:
        raise ValueError(
            "Use a PyPI package name with optional extras and version constraints"
        ) from error
    if requirement.url or canonicalize_name(requirement.name) == "sixletters-poc":
        raise ValueError(
            "Local paths, URLs, and replacing the shared sixletters package are unsupported"
        )
    return requirement


def _requirements(content: str) -> dict[str, str]:
    if len(content.encode("utf-8")) > MAX_REQUIREMENTS_BYTES:
        raise ValueError("requirements.in exceeds 64 KiB")
    result = {}
    for line in content.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = _requirement(line)
        name = canonicalize_name(requirement.name)
        if name in result:
            raise ValueError(f"Duplicate package in requirements.in: {name}")
        result[name] = str(requirement)
    return result


def _input_content(path: Path) -> str:
    source = path / INPUT_FILE
    if not source.exists():
        return ""
    if not source.is_file() or source.stat().st_size > MAX_REQUIREMENTS_BYTES:
        raise ValueError("requirements.in must be a text file smaller than 64 KiB")
    content = source.read_text(encoding="utf-8")
    _requirements(content)
    return content


def _baseline() -> dict:
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = canonicalize_name(distribution.metadata["Name"] or "")
        if name and name != "sixletters-poc":
            packages[name] = str(Version(distribution.version))
    python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
    }
    packages = dict(sorted(packages.items()))
    identity = {"python": python, "packages": packages}
    return {**identity, "fingerprint": _digest(json.dumps(identity, sort_keys=True))}


def _torch_backend(baseline: dict) -> str | None:
    torch = baseline["packages"].get("torch")
    if not torch or Version(torch).local is None:
        return None
    local = Version(torch).local
    if local == "cu126":
        return "cu126"
    if local == "cpu":
        return "cpu"
    raise ValueError(
        f"Unsupported shared PyTorch build {local}; use the repository CPU or CUDA 12.6 environment"
    )


def _uv_binary() -> str:
    executable = shutil.which("uv")
    if not executable:
        raise ValueError(
            f"uv {UV_VERSION} is required; install it in the project Python environment"
        )
    result = subprocess.run([executable, "--version"], capture_output=True, text=True, check=False)
    if result.returncode or result.stdout.split()[:2] != ["uv", UV_VERSION]:
        raise ValueError(f"uv {UV_VERSION} is required for experiment environment locks")
    return executable


def _uv_environment() -> dict[str, str]:
    # Package sources are explicit; inherited project settings must not redirect an installation.
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("UV_", "PIP_"))
        and key not in {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"}
    }
    if "UV_CACHE_DIR" in os.environ:
        environment["UV_CACHE_DIR"] = os.environ["UV_CACHE_DIR"]
    environment["UV_PYTHON_DOWNLOADS"] = "never"
    environment["UV_LINK_MODE"] = "copy"
    return environment


def _run_uv(executable: str, arguments: list[str], cwd: Path) -> None:
    result = subprocess.run(
        [executable, "--no-config", "--no-progress", *arguments],
        cwd=cwd,
        env=_uv_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        raise ValueError(
            "Experiment environment setup failed. Check the package conflict or installation "
            "message above; the previous environment and lock files were kept."
        )


def _python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _validate_lock(content: str, baseline: dict) -> None:
    if len(content.encode("utf-8")) > MAX_LOCK_BYTES:
        raise ValueError("requirements.lock exceeds 1 MiB")
    packages = {}
    for line in content.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = _requirement(line)
        specifiers = list(requirement.specifier)
        if (
            requirement.marker is not None
            or requirement.extras
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
            or "*" in specifiers[0].version
        ):
            raise ValueError(
                "requirements.lock must contain exact package versions; use env sync --relock"
            )
        name = canonicalize_name(requirement.name)
        if name in packages:
            raise ValueError("requirements.lock contains duplicate package versions")
        packages[name] = str(Version(specifiers[0].version))
    for name, version in baseline["packages"].items():
        if packages.get(name) != version:
            raise ValueError(
                f"The lock must preserve shared {name}=={version}; use env sync --relock"
            )


def _manifest(content: str, lock: str, baseline: dict) -> dict:
    return {
        "schema_version": 1,
        "uv_version": UV_VERSION,
        "python": baseline["python"],
        "base_packages": baseline["packages"],
        "base_fingerprint": baseline["fingerprint"],
        "requirements_sha256": _digest(content),
        "lock_sha256": _digest(lock),
        "torch_backend": _torch_backend(baseline),
    }


def _read_locked(path: Path, content: str, baseline: dict) -> tuple[str, dict]:
    try:
        source = path / LOCK_FILE
        manifest_path = path / MANIFEST_FILE
        if source.stat().st_size > MAX_LOCK_BYTES or manifest_path.stat().st_size > MAX_LOCK_BYTES:
            raise ValueError("Experiment environment lock files exceed 1 MiB")
        lock = source.read_text(encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != _manifest(content, lock, baseline):
            raise ValueError("Environment identity changed")
        _validate_lock(lock, baseline)
    except (OSError, ValueError) as error:
        raise EnvironmentRefreshError(LOCK_REFRESH_REASON, "--relock") from error
    return lock, manifest


def _state(
    manifest: dict,
    shared_paths: list[str] | None = None,
    shared_search_paths: list[str] | None = None,
) -> str:
    return (
        json.dumps(
            {
                "base_fingerprint": manifest["base_fingerprint"],
                "lock_sha256": manifest["lock_sha256"],
                **(
                    {"shared_paths": shared_paths, "shared_search_paths": shared_search_paths}
                    if shared_paths is not None
                    else {}
                ),
            },
            sort_keys=True,
        )
        + "\n"
    )


def _shared_paths() -> list[str]:
    """Use this interpreter's package directories, never PYTHONPATH or another environment."""
    return sorted({str(Path(sysconfig.get_path(key)).resolve()) for key in ("purelib", "platlib")})


def _site_paths(python: Path) -> list[Path]:
    return _interpreter_paths(python)[0]


def _interpreter_paths(
    python: Path,
    shared_paths: list[str] | None = None,
) -> tuple[list[Path], list[Path]]:
    setup = (
        "import site; " + "; ".join(f"site.addsitedir({value!r})" for value in shared_paths) + "; "
        if shared_paths
        else ""
    )
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            setup
            + "import json,sys,sysconfig; print(json.dumps({'sites':[sysconfig.get_path(k) for k in ('purelib','platlib')],'search':sys.path}))",
        ],
        env=_uv_environment(),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    values = json.loads(result.stdout)
    return tuple(
        sorted({Path(value).resolve() for value in values[key]}) for key in ("sites", "search")
    )


def _shared_packages(paths: list[str]) -> dict[str, str]:
    packages = {}
    try:
        for distribution in importlib.metadata.distributions(path=paths):
            name = canonicalize_name(distribution.metadata.get("Name", ""))
            if name == "sixletters-poc":
                continue
            if not name or name in packages:
                raise ValueError("Shared runtime contains missing or duplicate package metadata")
            packages[name] = str(Version(distribution.version))
    except OSError as error:
        raise ValueError("Shared runtime package metadata is missing or unreadable") from error
    return packages


def _overlay_paths(path: Path, manifest: dict) -> list[str] | None:
    """Validate local linkage separately from portable, scientific package records."""
    source = path / STATE_FILE
    if source.stat().st_size > MAX_LOCK_BYTES:
        raise ValueError("Invalid environment state")
    state = json.loads(source.read_text(encoding="utf-8"))
    if state == json.loads(_state(manifest)):
        return None  # Already installed full environments remain supported.
    paths = state.get("shared_paths") if isinstance(state, dict) else None
    search_paths = state.get("shared_search_paths") if isinstance(state, dict) else None
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(value, str) for value in paths)
        or paths != sorted(set(paths))
        or not isinstance(search_paths, list)
        or any(
            not isinstance(value, str) or not Path(value).is_absolute() for value in search_paths
        )
        or search_paths != sorted(set(search_paths))
        or state != json.loads(_state(manifest, paths, search_paths))
    ):
        raise ValueError("Invalid shared runtime link")
    for value in paths:
        directory = Path(value)
        if (
            not directory.is_absolute()
            or not directory.is_dir()
            or str(directory.resolve()) != value
            or directory.is_relative_to(path / ENV_DIR)
            or "\n" in value
            or "\r" in value
        ):
            raise ValueError("Shared runtime package directory is missing or changed")
    # A kernel is already inside the personal interpreter; its recorded shared paths
    # are checked against the saved package identity instead of its own site-packages.
    if Path(sys.prefix).resolve() != (path / ENV_DIR).resolve() and paths != _shared_paths():
        raise ValueError("Shared runtime package directory changed; run env sync --repair")
    if _shared_packages(paths) != manifest["base_packages"]:
        raise EnvironmentRefreshError(LOCK_REFRESH_REASON, "--relock")
    return paths


def _module_roots(directory: Path) -> dict[str, Path]:
    roots = {}
    for entry in directory.iterdir():
        if entry.name.endswith((".dist-info", ".egg-info")) or entry.name in {
            "__pycache__",
            "_virtualenv.py",
        }:
            continue
        if entry.is_dir() and entry.name.isidentifier():
            roots[entry.name] = entry
        elif entry.suffix in {".py", ".so", ".pyd"}:
            roots[entry.name.split(".")[0]] = entry
    return roots


def _shadows_shared(personal: Path, shared: Path) -> bool:
    local, common = _module_roots(personal), _module_roots(shared)
    for name in local.keys() & common.keys():
        left, right = local[name], common[name]
        if (
            not left.is_dir()
            or not right.is_dir()
            or any(left.glob("__init__.*"))
            or any(right.glob("__init__.*"))
            or _shadows_shared(left, right)
        ):
            return True
    return False


def _shared_pth(paths: list[str]) -> str:
    # addsitedir also loads the trusted runtime's existing .pth hooks (editable
    # installs and namespace packages), which a plain sys.path append would omit.
    return "import site; " + "; ".join(f"site.addsitedir({value!r})" for value in paths) + "\n"


def _overlay_intact(path: Path, shared_paths: list[str]) -> bool:
    personal_paths, search_paths = _interpreter_paths(_python(path / ENV_DIR))
    shared_search_paths = json.loads((path / STATE_FILE).read_text(encoding="utf-8"))[
        "shared_search_paths"
    ]
    for directory in personal_paths:
        link = directory / SHARED_PTH
        if (
            not directory.is_relative_to((path / ENV_DIR).resolve())
            or link.is_symlink()
            or link.read_text(encoding="utf-8") != _shared_pth(shared_paths)
            or "sixletters" in _module_roots(directory)
            or any(_shadows_shared(directory, Path(shared)) for shared in shared_paths)
        ):
            return False
    # Wheel .pth hooks may add editable/namespace paths. Allow those paths, but
    # reject shared-module replacements there just as in personal site-packages.
    for directory in search_paths:
        if str(directory) not in shared_search_paths and directory.is_dir():
            if any(_shadows_shared(directory, Path(shared)) for shared in shared_paths):
                return False
    return True


def _ready(path: Path, manifest: dict) -> bool:
    try:
        config = (path / ENV_DIR / "pyvenv.cfg").read_text(encoding="utf-8").lower()
        shared_paths = _overlay_paths(path, manifest)
        return (
            _python(path / ENV_DIR).is_file()
            and "include-system-site-packages = false" in config
            and (shared_paths is None or _overlay_intact(path, shared_paths))
        )
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return False


def _installed_matches(path: Path, manifest: dict, lock: str) -> bool:
    """Inspect the selected interpreter, independently of the command's Python path."""
    code = (
        "import importlib.metadata as m,json,platform,sys; "
        "print(json.dumps({'python':{'implementation':platform.python_implementation(),"
        "'version':platform.python_version(),'platform':sys.platform,'machine':platform.machine()},"
        "'packages':[[d.metadata['Name'],d.version] for d in m.distributions() if d.metadata['Name']]}))"
    )
    try:
        result = subprocess.run(
            [str(_python(path / ENV_DIR)), "-I", "-c", code],
            env=_uv_environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode or len(result.stdout) > MAX_LOCK_BYTES:
            return False
        observed = json.loads(result.stdout)
        packages = {}
        for name, version in observed["packages"]:
            name = canonicalize_name(name)
            if name == "sixletters-poc":
                continue
            if name in packages:
                return False
            packages[name] = str(Version(version))
        expected = {}
        for line in lock.splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                requirement = _requirement(line)
                expected[canonicalize_name(requirement.name)] = str(
                    Version(next(iter(requirement.specifier)).version)
                )
        return observed["python"] == manifest["python"] and packages == expected
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        StopIteration,
        subprocess.TimeoutExpired,
    ):
        return False


def _env_command(experiment: Path, action: str, *options: str) -> str:
    command = ["uv", "run", "--project", "Global", "poc"]
    command.extend(["env", action, experiment.name])
    owner = _member_directory(experiment)
    if owner is not None:
        command.extend(["--owner", owner.name])
    return shlex.join([*command, *options])


def _refresh_status(status: dict, experiment: Path, reason: str, option: str) -> dict:
    active = status["active"]
    return {
        **status,
        "state": "update-needed",
        "reason": reason + (" " + str(_busy(active)) if active else ""),
        "next_command": (
            _env_command(experiment, "status")
            if active
            else _env_command(experiment, "sync", option)
        ),
    }


def env_status(experiment_dir: Path) -> dict:
    """Describe package readiness and active users without installing or changing packages."""
    experiment = _experiment(experiment_dir)
    path = environment_directory(experiment)
    status = {
        "scope": "isolated" if path == experiment else "shared",
        "path": path,
        "python": _python(path / ENV_DIR),
        "state": "install-needed",
        "reason": "추가 패키지를 아직 설치하지 않았습니다.",
        "active": [],
        "next_command": _env_command(experiment, "sync"),
    }
    gate = FileLock(path / ".poc-env.lock", timeout=0)
    try:
        gate.acquire()
    except Timeout:
        return {
            **status,
            "state": "update-needed",
            "reason": "패키지 환경을 준비하거나 변경 중입니다. 완료될 때까지 기다린 뒤 상태를 확인하세요.",
            "next_command": _env_command(experiment, "status"),
        }
    try:
        status["active"] = _active_readers(path)
        try:
            content = _input_content(path)
        except ValueError:
            return {
                **status,
                "state": "update-needed",
                "reason": "requirements.in의 패키지 목록을 읽을 수 없습니다. 내용을 수정한 뒤 상태를 확인하세요.",
                "next_command": None,
            }
        if _uses_common_runtime(path, content):
            return {
                **status,
                "python": Path(sys.executable),
                "state": "ready",
                "reason": "추가 패키지가 없어 설치된 공통 환경을 사용합니다.",
                "next_command": None,
            }
        if not any((path / name).exists() for name in (LOCK_FILE, MANIFEST_FILE)):
            return status
        try:
            lock, manifest = _read_locked(path, content, _baseline())
        except EnvironmentRefreshError as error:
            return _refresh_status(status, experiment, error.reason, error.option)

        if not _ready(path, manifest):
            return _refresh_status(status, experiment, INSTALL_REPAIR_REASON, "--repair")
        if not _installed_matches(path, manifest, lock):
            return _refresh_status(status, experiment, INSTALL_REPAIR_REASON, "--repair")
        return {
            **status,
            "state": "ready",
            "reason": "설치된 Python과 패키지 버전이 기록과 일치합니다.",
            "next_command": None,
        }
    finally:
        gate.release()


def _publish(path: Path, staged: Path, contents: dict[str, str]) -> None:
    """Swap the completed environment and then publish a matching manifest last."""
    previous = {
        name: (path / name).read_bytes() if (path / name).exists() else None for name in contents
    }
    target = path / ENV_DIR
    backup = staged / "previous"
    had_environment = target.exists()
    if had_environment:
        target.rename(backup)
    try:
        (staged / ENV_DIR).rename(target)
        for name, content in contents.items():
            temporary = staged / name
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path / name)
    except BaseException:
        if target.exists():
            shutil.rmtree(target)
        if had_environment:
            backup.rename(target)
        for name, content in previous.items():
            if content is None:
                (path / name).unlink(missing_ok=True)
            else:
                (path / name).write_bytes(content)
        raise


def _sync(path: Path, content: str, *, relock: bool, force_install: bool) -> Path:
    if _uses_common_runtime(path, content):
        _active_readers(path, cleanup=True)
        return Path(sys.executable)
    baseline = _baseline()
    backend = _torch_backend(baseline)
    backend_args = ["--torch-backend", backend] if backend else []
    first_lock = not any((path / name).exists() for name in (LOCK_FILE, MANIFEST_FILE))
    regenerate = relock or first_lock
    if not regenerate:
        lock, manifest = _read_locked(path, content, baseline)
        if (
            not force_install
            and _ready(path, manifest)
            and _installed_matches(path, manifest, lock)
        ):
            return _python(path / ENV_DIR)
    _assert_unused(path)
    executable = _uv_binary()
    with tempfile.TemporaryDirectory(prefix=".poc-env-", dir=path) as temporary:
        staged = Path(temporary)
        if regenerate:
            inputs = staged / INPUT_FILE
            inputs.write_text(content, encoding="utf-8")
            base = staged / "base.txt"
            base.write_text(
                "".join(f"{name}=={version}\n" for name, version in baseline["packages"].items()),
                encoding="utf-8",
            )
            output = staged / LOCK_FILE
            # Retain already locked extra versions when they still satisfy edited constraints.
            old_lock = path / LOCK_FILE
            if old_lock.is_file() and old_lock.stat().st_size <= MAX_LOCK_BYTES:
                try:
                    previous_lock = old_lock.read_text(encoding="utf-8")
                    _validate_lock(previous_lock, {"packages": {}})
                except ValueError:
                    pass
                else:
                    output.write_text(previous_lock, encoding="utf-8")
            _run_uv(
                executable,
                [
                    "pip",
                    "compile",
                    str(base),
                    str(inputs),
                    "--python",
                    sys.executable,
                    "--no-header",
                    "--no-annotate",
                    "--no-sources",
                    "--output-file",
                    str(output),
                    *backend_args,
                ],
                staged,
            )
            lock = output.read_text(encoding="utf-8")
            _validate_lock(lock, baseline)
            manifest = _manifest(content, lock, baseline)
        else:
            (staged / LOCK_FILE).write_text(lock, encoding="utf-8")
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if len(manifest_text.encode("utf-8")) > MAX_LOCK_BYTES:
            raise ValueError("environment.json exceeds 1 MiB")
        _run_uv(
            executable,
            [
                "venv",
                "--no-project",
                "--relocatable",
                "--python",
                sys.executable,
                str(staged / ENV_DIR),
            ],
            staged,
        )
        shared_paths = _shared_paths()
        if _shared_packages(shared_paths) != baseline["packages"]:
            raise ValueError("Shared runtime package directories do not match its saved identity")
        personal_paths, base_search = _interpreter_paths(_python(staged / ENV_DIR), shared_paths)
        shared_search_paths = sorted(
            str(directory)
            for directory in base_search
            if not directory.is_relative_to((staged / ENV_DIR).resolve())
        )
        extras = staged / "additional-requirements.txt"
        extras.write_text(
            "".join(
                line + "\n"
                for line in lock.splitlines()
                if line.strip()
                and not line.lstrip().startswith("#")
                and canonicalize_name(_requirement(line).name) not in baseline["packages"]
            ),
            encoding="utf-8",
        )
        _run_uv(
            executable,
            [
                "pip",
                "sync",
                str(extras),
                "--python",
                str(_python(staged / ENV_DIR)),
                "--allow-empty-requirements",
                *backend_args,
            ],
            staged,
        )
        for directory in personal_paths:
            if not directory.is_relative_to((staged / ENV_DIR).resolve()):
                raise ValueError("Personal package directory must remain inside its environment")
            link = directory / SHARED_PTH
            if link.exists() or link.is_symlink():
                raise ValueError("An additional package conflicts with the shared runtime link")
            link.write_text(_shared_pth(shared_paths), encoding="utf-8")
        state = _state(manifest, shared_paths, shared_search_paths)
        (staged / STATE_FILE).write_text(state, encoding="utf-8")
        if not _ready(staged, manifest) or not _installed_matches(staged, manifest, lock):
            raise ValueError(
                "Additional packages shadow shared modules or differ from the saved lock. "
                "The previous environment and package records were kept."
            )
        _publish(
            path,
            staged,
            {
                INPUT_FILE: content,
                LOCK_FILE: lock,
                STATE_FILE: state,
                MANIFEST_FILE: manifest_text,
            },
        )
    python = _python(path / ENV_DIR)
    _refresh_scope_kernels(path, python)
    return python


def env_sync(experiment_dir: Path, *, relock: bool = False, repair: bool = False) -> Path:
    """Install the existing lock; an incompatible lock requires an explicit relock."""
    experiment = _experiment(experiment_dir)
    with _environment_guidance(experiment), _experiment_locked(experiment):
        path = environment_directory(experiment)
        with _locked(path):
            if relock or repair:
                _assert_unused(path)
            return _sync(path, _input_content(path), relock=relock, force_install=repair)


def env_add(experiment_dir: Path, specs: list[str]) -> Path:
    """Add or replace package constraints and atomically install a newly resolved lock."""
    if not specs:
        raise ValueError("Specify at least one package to add")
    additions = _requirements("\n".join(str(_requirement(value)) for value in specs))
    experiment = _experiment(experiment_dir)
    with _environment_guidance(experiment), _experiment_locked(experiment):
        path = environment_directory(experiment)
        with _locked(path):
            original = _input_content(path)
            requirements = _requirements(original)
            updated = {**requirements, **additions}
            if updated == requirements:
                return _sync(path, original, relock=False, force_install=False)
            _assert_unused(path)
            content = "".join(f"{value}\n" for _, value in sorted(updated.items()))
            return _sync(path, content, relock=True, force_install=True)


def env_remove(experiment_dir: Path, names: list[str]) -> Path:
    """Remove personal direct requirements; shared packages always remain available."""
    if not names:
        raise ValueError("Specify at least one package to remove")
    requested = []
    for value in names:
        requirement = _requirement(value)
        if requirement.specifier or requirement.extras or requirement.marker:
            raise ValueError("Remove packages by name without a version or extras")
        requested.append(canonicalize_name(requirement.name))
    experiment = _experiment(experiment_dir)
    with _environment_guidance(experiment), _experiment_locked(experiment):
        path = environment_directory(experiment)
        with _locked(path):
            _assert_unused(path)
            requirements = _requirements(_input_content(path))
            for name in requested:
                if name not in requirements:
                    raise ValueError(f"{name} is not a personal requirement in this environment")
                del requirements[name]
            content = "".join(f"{value}\n" for _, value in sorted(requirements.items()))
            return _sync(path, content, relock=True, force_install=True)


def _write_selection(path: Path, scope: str, resources: dict[str, bytes] | None = None) -> None:
    replacements = {SCOPE_FILE: f'scope = "{scope}"\n'.encode()}
    if resources is not None:
        replacements = {**{name: resources.get(name) for name in LOCK_NAMES}, **replacements}
    previous = {
        name: (path / name).read_bytes() if (path / name).exists() else None
        for name in replacements
    }
    try:
        # Publish the selector last so partially written records cannot become the active scope.
        for name, content in replacements.items():
            if content is None:
                (path / name).unlink(missing_ok=True)
                continue
            with tempfile.NamedTemporaryFile(dir=path, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
            try:
                temporary.replace(path / name)
            finally:
                temporary.unlink(missing_ok=True)
    except BaseException:
        for name, content in previous.items():
            if content is None:
                (path / name).unlink(missing_ok=True)
            else:
                (path / name).write_bytes(content)
        raise


def env_isolate(experiment_dir: Path) -> Path:
    """Copy current package records into this experiment, leaving its member unchanged."""
    experiment = _experiment(experiment_dir)
    with _experiment_locked(experiment):
        active = environment_directory(experiment)
        with _locked(active):
            resources = environment_resources(experiment)
            # An old, inactive local environment is retained; its state must match the new lock
            # before it can be reused. Never copy a virtual environment between directories.
            _check_environment(experiment)
            _write_selection(experiment, "isolated", resources)
    _refresh_registered_kernel(experiment)
    return experiment


def _check_share_requirements(experiment: Path, owner: Path) -> None:
    previous = _requirements(_input_content(experiment))
    if not previous:
        return
    content = _input_content(owner)
    try:
        lock, _ = _read_locked(owner, content, _baseline())
    except ValueError as error:
        raise ValueError(
            "The member environment does not have a compatible lock. "
            "Keep this experiment isolated, or add its required packages to a shared experiment "
            "before running env share again."
        ) from error
    packages = {}
    for line in lock.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            requirement = _requirement(line)
            packages[canonicalize_name(requirement.name)] = next(
                iter(requirement.specifier)
            ).version
    direct = _requirements(content)
    for value in previous.values():
        requirement = _requirement(value)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        name = canonicalize_name(requirement.name)
        version = packages.get(name)
        extras = _requirement(direct[name]).extras if name in direct else set()
        if (
            version is None
            or not requirement.specifier.contains(version, prereleases=True)
            or not requirement.extras.issubset(extras)
        ):
            raise ValueError(
                f"The member environment does not satisfy {requirement}. "
                "Keep this experiment isolated, or add a compatible package to a shared "
                "experiment before running env share again."
            )


def env_share(experiment_dir: Path) -> Path:
    """Adopt a compatible member environment without modifying its packages."""
    experiment = _experiment(experiment_dir)
    owner = _member_directory(experiment)
    if owner is None:
        raise ValueError("Shared environments require Members/<owner>/experiments/<name>")
    with _experiment_locked(experiment):
        active = environment_directory(experiment)
        with _locked(owner):
            if active != owner:
                with _locked(active):
                    _check_share_requirements(active, owner)
            _write_selection(experiment, "shared")
    _refresh_registered_kernel(experiment)
    return owner


def execute_script(experiment_dir: Path, script: Path, args: list[str]) -> int:
    """Run any Python file inside one experiment, preserving arguments and MLflow settings."""
    path = _experiment(experiment_dir)
    script = Path(script)
    script = script if script.is_absolute() else path / script
    absolute = script.absolute()
    if (
        not absolute.is_relative_to(path)
        or any(part.is_symlink() for part in (absolute, *absolute.parents) if part != path)
        or not absolute.resolve().is_relative_to(path)
        or absolute.suffix != ".py"
        or not absolute.is_file()
        or any(part.startswith(".") for part in absolute.relative_to(path).parts)
    ):
        raise ValueError(
            "Choose a .py script inside this experiment, outside hidden environment folders"
        )
    script = absolute.resolve()
    with _running(path) as (active, python):
        root = project_root(path)
        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        _set_runtime_environment(environment, active, python)
        sources = str(root / "Global/src")
        environment["PYTHONPATH"] = sources + (
            os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        )
        from .execution import ENVIRONMENT_KEY, invocation

        environment[ENVIRONMENT_KEY] = json.dumps(invocation(path, script, args))
        return subprocess.run(
            [str(python), str(script), *args], cwd=root, env=environment, check=False
        ).returncode


def _set_runtime_environment(environment: dict[str, str], active: Path, python: Path) -> None:
    if python == _python(active / ENV_DIR):
        environment["VIRTUAL_ENV"] = str(active / ENV_DIR)
    elif sys.prefix != sys.base_prefix:
        environment["VIRTUAL_ENV"] = sys.prefix
    else:
        environment.pop("VIRTUAL_ENV", None)
    environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")


def _runtime_prefix(active: Path, python: Path) -> Path:
    return active / ENV_DIR if python == _python(active / ENV_DIR) else Path(sys.prefix)


def jupyter_data_path(root: Path) -> Path:
    """Keep persistent kernel registrations separate for each checkout and runtime."""
    identity = {
        "root": str(Path(root).resolve()),
        "base_fingerprint": _baseline()["fingerprint"],
        "prefix": str(Path(sys.prefix).resolve()),
        "version": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
    }
    return Path(root) / ".poc-cache/jupyter" / _digest(json.dumps(identity, sort_keys=True))[:20]


def _kernel_spec_path(experiment: Path) -> Path:
    root = project_root(experiment)
    relative = experiment.relative_to(root)
    name = "sixletters-" + _digest(str(root) + "\n" + relative.as_posix())[:24]
    return jupyter_data_path(root) / "kernels" / name / "kernel.json"


def _refresh_scope_kernels(active: Path, python: Path) -> None:
    """Update only existing registrations after installation, under the environment gate."""
    if active.parent.name == "Members":
        directory = active / "experiments"
        experiments = sorted(directory.iterdir()) if directory.is_dir() else []
    else:
        experiments = [active]
    for experiment in experiments:
        if (
            _member_directory(experiment) is None
            or experiment.is_symlink()
            or not (experiment / "config.toml").is_file()
        ):
            continue
        try:
            selected = environment_directory(experiment)
        except ValueError:
            continue
        if selected == active:
            specification = _kernel_spec_path(experiment)
            if specification.exists() or specification.is_symlink():
                try:
                    _write_kernel_spec(experiment, python)
                except (OSError, ValueError) as error:
                    raise ValueError(
                        "패키지 설치는 완료했지만 기존 노트북 커널을 갱신하지 못했습니다. "
                        f"원인: {error}\n다음: {_env_command(experiment, 'kernel')}"
                    ) from error


def _refresh_registered_kernel(experiment: Path) -> None:
    if _member_directory(experiment) is None:
        return
    specification = _kernel_spec_path(experiment)
    if specification.exists() or specification.is_symlink():
        try:
            env_kernel(experiment)
        except EnvironmentRefreshError:
            raise
        except (OSError, ValueError) as error:
            raise ValueError(
                "패키지 환경 범위는 변경했지만 노트북 커널을 준비하지 못했습니다. "
                f"원인: {error}\n다음: {_env_command(experiment, 'kernel')}"
            ) from error


def env_kernel(experiment_dir: Path, *, relock: bool = False) -> tuple[str, Path]:
    """Synchronize one environment and register it without copying server credentials."""
    path = _experiment(experiment_dir)
    root = project_root(path)
    relative = path.relative_to(root)
    if (
        len(relative.parts) != 4
        or relative.parts[0] != "Members"
        or relative.parts[2] != "experiments"
    ):
        raise ValueError("Notebook kernels require a Members/<owner>/experiments/<name> folder")
    if not relock:
        active = environment_directory(path)
        with _locked(active):
            users = _active_readers(active)
            if any(user["experiment"] == path.name and user["kind"] == "kernel" for user in users):
                destination = _kernel_spec_path(path)
                try:
                    if destination.is_symlink() or destination.stat().st_size > 8192:
                        raise ValueError("Invalid kernel registration")
                    saved = json.loads(destination.read_text(encoding="utf-8"))
                    content = _input_content(active)
                    if _uses_common_runtime(active, content):
                        python = Path(sys.executable)
                        ready = True
                    else:
                        lock, manifest = _read_locked(active, content, _baseline())
                        python = _python(active / ENV_DIR)
                        ready = _ready(active, manifest) and _installed_matches(
                            active, manifest, lock
                        )
                    expected = {
                        "argv": [
                            str(python),
                            "-m",
                            "sixletters.personal_env",
                            "--kernel",
                            str(path),
                            "--connection-file",
                            "{connection_file}",
                            "--runtime-prefix",
                            str(_runtime_prefix(active, python)),
                        ],
                        "display_name": f"SixSeven · {relative.parts[1]} / {relative.parts[3]}",
                        "language": "python",
                        "env": {"PYTHONPATH": str(root / "Global/src")},
                    }
                    if saved == expected and ready:
                        return expected["display_name"], destination
                except (OSError, ValueError):
                    pass
                raise _busy(users)
    with (
        _environment_guidance(path),
        _experiment_locked(path),
        _locked(environment_directory(path)),
    ):
        python = _prepare(environment_directory(path), relock=relock)
        return _write_kernel_spec(path, python)


def _write_kernel_spec(path: Path, python: Path) -> tuple[str, Path]:
    root = project_root(path)
    relative = path.relative_to(root)
    display = f"SixSeven · {relative.parts[1]} / {relative.parts[3]}"
    directory = _kernel_spec_path(path).parent
    if any(part.is_symlink() for part in (directory, *directory.parents)):
        raise ValueError("Jupyter kernel cache must not contain symbolic links")
    directory.mkdir(parents=True, exist_ok=True)
    spec = {
        "argv": [
            str(python),
            "-m",
            "sixletters.personal_env",
            "--kernel",
            str(path),
            "--connection-file",
            "{connection_file}",
            "--runtime-prefix",
            str(_runtime_prefix(environment_directory(path), python)),
        ],
        "display_name": display,
        "language": "python",
        "env": {"PYTHONPATH": str(root / "Global/src")},
    }
    destination = directory / "kernel.json"
    if destination.is_symlink():
        raise ValueError("Jupyter kernel specification must not be a symbolic link")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(spec, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return display, destination


def launch_kernel(
    experiment_dir: Path, connection_file: Path, *, runtime_prefix: Path | None = None
) -> None:
    """Hold the environment lock until shutdown, including idle notebook time."""
    path = _experiment(experiment_dir)
    with _environment_guidance(path), _experiment_locked(path):
        active = environment_directory(path)
        with _locked(active):
            common = _uses_common_runtime(active, _input_content(active))
            expected_prefix = runtime_prefix if common else active / ENV_DIR
            if (
                expected_prefix is None
                or Path(sys.prefix).resolve() != expected_prefix.resolve()
                or (common and Path(sys.prefix).resolve() == (active / ENV_DIR).resolve())
            ):
                raise ValueError(
                    "이 실험에 등록된 SixSeven 커널을 선택하세요. "
                    "패키지 환경 범위가 바뀌었다면 커널을 다시 연결하세요.\n"
                    f"다음: {_env_command(path, 'kernel')}"
                )
            lease = _kernel_lease(active, path)
        try:
            _start_kernel(path, active, connection_file)
        finally:
            lease.release()


def _kernel_lease(active: Path, experiment: Path | None = None) -> FileLock:
    content = _input_content(active)
    if _uses_common_runtime(active, content):
        return _lease(active, experiment, kind="kernel")
    try:
        manifest = json.loads((active / MANIFEST_FILE).read_text(encoding="utf-8"))
        lock = (active / LOCK_FILE).read_text(encoding="utf-8")
        matches = manifest["requirements_sha256"] == _digest(content) and manifest[
            "lock_sha256"
        ] == _digest(lock)
    except (OSError, ValueError, KeyError) as error:
        raise EnvironmentRefreshError(
            LOCK_REFRESH_REASON, "--relock", experiment=experiment
        ) from error
    if not matches:
        raise EnvironmentRefreshError(LOCK_REFRESH_REASON, "--relock", experiment=experiment)
    try:
        _overlay_paths(active, manifest)
    except EnvironmentRefreshError as error:
        raise EnvironmentRefreshError(error.reason, error.option, experiment=experiment) from error
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise EnvironmentRefreshError(
            INSTALL_REPAIR_REASON, "--repair", experiment=experiment
        ) from error
    if not _ready(active, manifest):
        raise EnvironmentRefreshError(INSTALL_REPAIR_REASON, "--repair", experiment=experiment)
    return _lease(active, experiment, kind="kernel")


def _start_kernel(path: Path, active: Path, connection_file: Path) -> None:
    root = project_root(path)
    # Notebook cells are interactive, not a previous script invocation from the server.
    from .execution import ENVIRONMENT_KEY

    os.environ.pop(ENVIRONMENT_KEY, None)
    os.environ.pop("PYTHONHOME", None)
    _set_runtime_environment(os.environ, active, Path(sys.executable))
    os.environ["PYTHONPATH"] = str(root / "Global/src")
    os.chdir(root)
    sys.path.insert(0, str(path))
    from ipykernel.kernelapp import IPKernelApp

    application = IPKernelApp.instance()
    application.initialize(["-f", str(connection_file)])
    application.start()


def _kernel_main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Launch a registered personal experiment kernel")
    parser.add_argument("--kernel", required=True, type=Path)
    parser.add_argument("--connection-file", required=True, type=Path)
    parser.add_argument("--runtime-prefix", type=Path)
    arguments = parser.parse_args()
    try:
        launch_kernel(
            arguments.kernel, arguments.connection_file, runtime_prefix=arguments.runtime_prefix
        )
    except (OSError, ValueError, ImportError) as error:
        print(f"노트북 환경 준비 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_kernel_main())
