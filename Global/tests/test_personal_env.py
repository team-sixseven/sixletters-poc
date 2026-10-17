"""Exercise actual uv environments with tiny local wheels, without a package registry."""

import importlib.metadata
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from filelock import FileLock

from sixletters import personal_env as pe


def _wheel(directory: Path, name: str, version: str, *, dependencies=(), extra_files=None) -> None:
    module = name.replace("-", "_")
    metadata = f"{module}-{version}.dist-info"
    files = {
        f"{module}.py": f"VALUE = {version!r}\ndef main():\n    print(VALUE)\n",
        f"{metadata}/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {value}\n" for value in dependencies)
        ),
        f"{metadata}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{metadata}/entry_points.txt": f"[console_scripts]\n{module} = {module}:main\n",
    }
    files.update(extra_files or {})
    files[f"{metadata}/RECORD"] = (
        "".join(f"{path},,\n" for path in files) + f"{metadata}/RECORD,,\n"
    )
    with zipfile.ZipFile(directory / f"{module}-{version}-py3-none-any.whl", "w") as archive:
        for path, content in files.items():
            archive.writestr(path, content)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("SIXLETTERS_COMMAND_CONTEXT", "host")
    monkeypatch.setenv("SIXLETTERS_COMMAND_GPU", "cpu")
    root = tmp_path.resolve()
    (root / "Global/src").mkdir(parents=True)
    (root / "Global/pyproject.toml").touch()
    (root / "Global/src/shared_sentinel.py").write_text("VALUE = 'shared-source'\n")
    wheelhouse = root / "wheels"
    wheelhouse.mkdir()
    _wheel(wheelhouse, "poc-env-base", "1.0")
    _wheel(wheelhouse, "poc-env-extra", "1.0")
    _wheel(wheelhouse, "poc-env-extra", "2.0")
    _wheel(wheelhouse, "poc-env-conflict", "1.0", dependencies=["poc-env-base==2.0"])
    baseline = pe._baseline()
    baseline["packages"] = {"poc-env-base": "1.0"}
    baseline["fingerprint"] = "base-v1"
    monkeypatch.setattr(pe, "_baseline", lambda: baseline)
    shared = root / "shared-site-packages"
    shared.mkdir()
    with zipfile.ZipFile(wheelhouse / "poc_env_base-1.0-py3-none-any.whl") as archive:
        archive.extractall(shared)
    monkeypatch.setattr(pe, "_shared_paths", lambda: [str(shared)])
    run_uv = pe._run_uv

    def offline(executable, arguments, cwd):
        if arguments[0] == "pip":
            arguments = [*arguments, "--no-index", "--find-links", str(wheelhouse), "--offline"]
        return run_uv(executable, arguments, cwd)

    monkeypatch.setattr(pe, "_run_uv", offline)

    def experiment(name="one"):
        path = root / "Members/alice/experiments" / name
        path.mkdir(parents=True)
        (path / "config.toml").write_text('[experiment]\nname="test"\n')
        (path / pe.SCOPE_FILE).write_text('scope = "isolated"\n')
        return path

    return root, experiment, baseline


def _version(python, package):
    return subprocess.check_output(
        [
            str(python),
            "-c",
            f"import importlib.metadata; print(importlib.metadata.version({package!r}))",
        ],
        text=True,
    ).strip()


def test_actual_install_isolates_experiments_and_does_not_change_base(isolated):
    _, experiment, _ = isolated
    first, second = experiment("first"), experiment("second")
    before = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    first_python = pe.env_add(first, ["poc-env-extra==1.0"])
    second_python = pe.env_add(second, ["poc-env-extra==2.0"])
    assert _version(first_python, "poc-env-extra") == "1.0"
    assert _version(second_python, "poc-env-extra") == "2.0"
    assert _version(first_python, "poc-env-base") == "1.0"
    after = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    assert before == after
    config = (first / ".venv/pyvenv.cfg").read_text()
    assert "include-system-site-packages = false" in config
    assert str(first) not in (first / pe.MANIFEST_FILE).read_text()
    pe.env_remove(first, ["poc-env-extra"])
    missing = subprocess.run([str(first_python), "-c", "import poc_env_extra"], capture_output=True)
    assert missing.returncode != 0
    assert _version(second_python, "poc-env-extra") == "2.0"


def test_arbitrary_script_executes_with_args_shared_source_helpers_and_mlflow(
    isolated, monkeypatch
):
    root, experiment, _ = isolated
    path = experiment()
    python = pe.env_add(path, ["poc-env-extra==2.0"])
    (path / "helper.py").write_text("VALUE = 'own-helper'\n")
    (path / "fit_random_forest.py").write_text(
        "import json, os, sys, shared_sentinel, helper, poc_env_extra\n"
        "from pathlib import Path\n"
        "Path('execution.json').write_text(json.dumps({"
        "'args': sys.argv[1:], 'helper': helper.VALUE, 'shared': shared_sentinel.VALUE, "
        "'extra': poc_env_extra.VALUE, 'tracking': os.getenv('MLFLOW_TRACKING_URI'), "
        "'python': sys.executable, 'cwd': str(Path.cwd())}))\n"
        "raise SystemExit(7)\n"
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example.invalid:5000")
    assert pe.execute_script(path, Path("fit_random_forest.py"), ["--label", "a b"]) == 7
    output = json.loads((root / "execution.json").read_text())
    assert output == {
        "args": ["--label", "a b"],
        "helper": "own-helper",
        "shared": "shared-source",
        "extra": "2.0",
        "tracking": "http://example.invalid:5000",
        "python": str(python),
        "cwd": str(root),
    }
    # Installed entry points continue to work after the staging venv has been renamed.
    command = python.parent / ("poc_env_extra.exe" if os.name == "nt" else "poc_env_extra")
    assert subprocess.check_output([str(command)], text=True).strip() == "2.0"


def test_failed_resolution_keeps_previous_environment_and_files(isolated):
    _, experiment, _ = isolated
    path = experiment()
    python = pe.env_add(path, ["poc-env-extra==1.0"])
    before = {name: (path / name).read_bytes() for name in (*pe.LOCK_NAMES, pe.STATE_FILE)}
    with pytest.raises(ValueError, match="setup failed"):
        pe.env_add(path, ["poc-env-conflict"])
    assert {name: (path / name).read_bytes() for name in before} == before
    assert _version(python, "poc-env-extra") == "1.0"
    assert list(path.glob(".poc-env-*")) == [path / pe.STATE_FILE]


def test_sync_reuses_pin_and_changed_input_requires_explicit_relock(isolated):
    _, experiment, _ = isolated
    path = experiment()
    python = pe.env_add(path, ["poc-env-extra==1.0"])
    (path / pe.INPUT_FILE).write_text("poc-env-extra>=1.0\n")
    with pytest.raises(ValueError, match="--relock"):
        pe.env_sync(path)
    pe.env_sync(path, relock=True)
    assert _version(python, "poc-env-extra") == "1.0"
    (path / pe.INPUT_FILE).write_text("poc-env-extra==2.0\n")
    pe.env_sync(path, relock=True)
    assert _version(python, "poc-env-extra") == "2.0"


def test_base_change_rejected_until_explicit_relock(isolated):
    _, experiment, baseline = isolated
    path = experiment()
    pe.env_add(path, ["poc-env-base==1.0"])
    baseline["fingerprint"] = "changed-runtime"
    with pytest.raises(ValueError, match="--relock"):
        pe.env_sync(path)
    pe.env_sync(path, relock=True)
    assert (
        json.loads((path / pe.MANIFEST_FILE).read_text())["base_fingerprint"] == "changed-runtime"
    )


def test_execution_holds_lock_during_child_process(isolated, monkeypatch):
    _, experiment, _ = isolated
    path = experiment()
    pe.env_add(path, ["poc-env-base==1.0"])
    (path / "fit.py").write_text("pass\n")
    real_run = pe.subprocess.run

    def assert_locked(command, **kwargs):
        if str(command[0]) == str(pe._python(path / pe.ENV_DIR)):
            for operation in (
                lambda: pe.env_sync(path),
                lambda: pe.env_add(path, ["poc-env-extra"]),
                lambda: pe.env_remove(path, ["poc-env-extra"]),
            ):
                with pytest.raises(ValueError, match="실행 중"):
                    operation()
        return real_run(command, **kwargs)

    monkeypatch.setattr(pe.subprocess, "run", assert_locked)
    assert pe.execute_script(path, Path("fit.py"), []) == 0
    assert (path / ".poc-env.lock").exists()


@pytest.mark.parametrize(
    "spec",
    [
        "-r other.txt",
        "--index-url https://example.invalid",
        "../wheel.whl",
        "/tmp/a",
        "pkg @ https://user:secret@example.invalid/a.whl",
        "sixletters-poc==1",
        "a\nb",
    ],
)
def test_rejects_unsupported_requirement_sources(isolated, spec):
    _, experiment, _ = isolated
    path = experiment()
    with pytest.raises(ValueError):
        pe.env_add(path, [spec])
    assert not (path / pe.INPUT_FILE).exists()
    assert not (path / pe.ENV_DIR).exists()


@pytest.mark.parametrize(
    "name",
    [
        ".venv",
        "requirements.in",
        "requirements.lock",
        "environment.json",
        ".poc-env-state.json",
        ".poc-env.lock",
        "config.toml",
    ],
)
def test_rejects_symlink_targets(isolated, name):
    root, experiment, _ = isolated
    path = experiment()
    target = root / "outside"
    target.write_text("unchanged")
    (path / name).unlink(missing_ok=True)
    (path / name).symlink_to(target)
    with pytest.raises(ValueError, match="symbolic"):
        pe.env_sync(path)
    assert target.read_text() == "unchanged"


@pytest.mark.parametrize(
    "script", ["../sibling.py", "../../outside.py", ".venv/test.py", "train.sh"]
)
def test_script_paths_rejected_before_install(isolated, script):
    _, experiment, _ = isolated
    path = experiment()
    with pytest.raises(ValueError, match="script"):
        pe.execute_script(path, Path(script), [])
    assert not (path / pe.ENV_DIR).exists()


def test_external_script_symlink_rejected(isolated):
    root, experiment, _ = isolated
    path = experiment()
    external = root / "external.py"
    external.write_text("pass\n")
    (path / "fit.py").symlink_to(external)
    with pytest.raises(ValueError, match="script"):
        pe.execute_script(path, Path("fit.py"), [])


def test_malformed_lock_cannot_install_directives(isolated):
    _, experiment, baseline = isolated
    path = experiment()
    pe.env_sync(path)
    lock = "--index-url https://user:secret@example.invalid\n"
    (path / pe.LOCK_FILE).write_text(lock)
    (path / pe.MANIFEST_FILE).write_text(json.dumps(pe._manifest("", lock, baseline)))
    with pytest.raises(ValueError, match="--relock"):
        pe.env_sync(path)


@pytest.mark.parametrize(
    "torch,expected", [("2.13.0+cu126", "cu126"), ("2.13.0+cpu", "cpu"), ("2.13.0", None)]
)
def test_torch_backend_preserves_build(torch, expected):
    assert pe._torch_backend({"packages": {"torch": torch}}) == expected


def test_torch_unsupported_backend_is_not_silently_replaced():
    with pytest.raises(ValueError, match="Unsupported"):
        pe._torch_backend({"packages": {"torch": "2.13.0+cu128"}})


def test_uv_environment_ignores_inherited_package_routing(monkeypatch):
    for name in (
        "UV_PROJECT_ENVIRONMENT",
        "UV_INDEX_URL",
        "PIP_INDEX_URL",
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    ):
        monkeypatch.setenv(name, "should-not-be-used")
    monkeypatch.setenv("UV_CACHE_DIR", "/tmp/cache")
    monkeypatch.setenv("UV_LINK_MODE", "hardlink")
    environment = pe._uv_environment()
    assert environment["UV_CACHE_DIR"] == "/tmp/cache"
    assert environment["UV_PYTHON_DOWNLOADS"] == "never"
    assert environment["UV_LINK_MODE"] == "copy"
    assert all(value != "should-not-be-used" for value in environment.values())


def test_locked_environment_is_not_modified(isolated, monkeypatch):
    monkeypatch.setattr(pe, "GATE_TIMEOUT_SECONDS", 0)
    _, experiment, _ = isolated
    path = experiment()
    with FileLock(path / ".poc-env.lock"):
        with pytest.raises(ValueError, match="실행 중"):
            pe.env_sync(path)
    assert not (path / pe.ENV_DIR).exists()


def test_invalid_removed_package_does_not_install(isolated):
    _, experiment, _ = isolated
    path = experiment()
    for value in ("poc-env-extra==1.0", "missing"):
        with pytest.raises(ValueError):
            pe.env_remove(path, [value])
    assert not (path / pe.ENV_DIR).exists()


def test_failed_install_keeps_previous_environment_and_files(isolated, monkeypatch):
    _, experiment, _ = isolated
    path = experiment()
    python = pe.env_add(path, ["poc-env-extra==1.0"])
    before = {name: (path / name).read_bytes() for name in (*pe.LOCK_NAMES, pe.STATE_FILE)}
    real_run = pe._run_uv

    def fail_sync(executable, arguments, cwd):
        if arguments[:2] == ["pip", "sync"]:
            raise ValueError("simulated download failure")
        return real_run(executable, arguments, cwd)

    monkeypatch.setattr(pe, "_run_uv", fail_sync)
    with pytest.raises(ValueError, match="download failure"):
        pe.env_add(path, ["poc-env-extra==2.0"])
    assert {name: (path / name).read_bytes() for name in before} == before
    assert _version(python, "poc-env-extra") == "1.0"


def test_execution_reuses_ready_environment_without_resolving_or_installing(isolated, monkeypatch):
    _, experiment, _ = isolated
    path = experiment()
    pe.env_add(path, ["poc-env-base==1.0"])
    (path / "fit.py").write_text("import poc_env_base\n")

    def unexpected(*args, **kwargs):
        pytest.fail("A ready environment must not contact a package registry or run uv")

    monkeypatch.setattr(pe, "_run_uv", unexpected)
    monkeypatch.setattr(pe, "_uv_binary", unexpected)
    assert pe.execute_script(path, Path("fit.py"), []) == 0


def test_sync_restores_deleted_environment_from_exact_lock(isolated, monkeypatch):
    import shutil

    _, experiment, _ = isolated
    path = experiment()
    python = pe.env_add(path, ["poc-env-extra==1.0"])
    lock = (path / pe.LOCK_FILE).read_bytes()
    shutil.rmtree(path / pe.ENV_DIR)
    real_run = pe._run_uv

    def no_compile(executable, arguments, cwd):
        assert arguments[:2] != ["pip", "compile"]
        return real_run(executable, arguments, cwd)

    monkeypatch.setattr(pe, "_run_uv", no_compile)
    assert pe.env_sync(path) == python
    assert _version(python, "poc-env-extra") == "1.0"
    assert (path / pe.LOCK_FILE).read_bytes() == lock


def test_oversized_environment_manifest_is_rejected_before_install(isolated, monkeypatch):
    _, experiment, _ = isolated
    path = experiment()
    monkeypatch.setattr(pe, "MAX_LOCK_BYTES", 180)
    with pytest.raises(ValueError, match="environment.json exceeds"):
        pe.env_add(path, ["poc-env-base==1.0"])
    assert not (path / pe.ENV_DIR).exists()
    assert not (path / pe.MANIFEST_FILE).exists()


def _shared(experiment, name="shared"):
    path = experiment(name)
    (path / pe.SCOPE_FILE).unlink()
    return path


def test_member_environment_reused_by_multiple_experiments_and_owners_are_separate(
    isolated, monkeypatch
):
    _, experiment, _ = isolated
    first, second = _shared(experiment, "first"), _shared(experiment, "second")
    owner = first.parent.parent
    python = pe.env_add(first, ["poc-env-extra==1.0"])
    assert python == pe._python(owner / pe.ENV_DIR)
    assert pe.environment_directory(second) == owner
    (second / "fit.py").write_text("import poc_env_extra; assert poc_env_extra.VALUE == '1.0'\n")
    real_uv = pe._run_uv
    monkeypatch.setattr(pe, "_run_uv", lambda *args: pytest.fail("must reuse prepared environment"))
    assert pe.execute_script(second, Path("fit.py"), []) == 0
    assert not (first / pe.ENV_DIR).exists()
    assert not (second / pe.LOCK_FILE).exists()
    monkeypatch.setattr(pe, "_run_uv", real_uv)
    other = owner.parent / "bob/experiments/first"
    other.mkdir(parents=True)
    (other / "config.toml").touch()
    second_python = pe.env_add(other, ["poc-env-extra==2.0"])
    assert _version(python, "poc-env-extra") == "1.0"
    assert _version(second_python, "poc-env-extra") == "2.0"


def test_isolate_copies_current_records_and_preserves_member_and_siblings(isolated):
    _, experiment, _ = isolated
    first, second = _shared(experiment, "first"), _shared(experiment, "second")
    python = pe.env_add(first, ["poc-env-extra==1.0"])
    owner = pe.environment_directory(first)
    before = pe.environment_resources(second)
    assert pe.env_isolate(first) == first
    assert pe.environment_resources(first) == before
    assert not (first / pe.ENV_DIR).exists()
    isolated_python = pe.env_sync(first)
    assert _version(isolated_python, "poc-env-extra") == "1.0"
    pe.env_add(first, ["poc-env-extra==2.0"])
    assert _version(isolated_python, "poc-env-extra") == "2.0"
    assert _version(python, "poc-env-extra") == "1.0"
    assert pe.environment_resources(second) == before
    assert pe.environment_directory(second) == owner
    with pytest.raises(ValueError, match="does not satisfy"):
        pe.env_share(first)
    assert pe.environment_directory(first) == first
    pe.env_add(first, ["poc-env-extra==1.0"])
    local_records = pe.environment_resources(first)
    assert pe.env_share(first) == owner
    assert pe.environment_directory(first) == owner
    assert {name: (first / name).read_bytes() for name in pe.LOCK_NAMES} == local_records
    assert pe.environment_resources(first) == before


def test_share_does_not_add_missing_packages_to_member(isolated):
    _, experiment, _ = isolated
    path = experiment()
    pe.env_add(path, ["poc-env-extra==1.0"])
    owner = path.parent.parent
    with pytest.raises(ValueError, match="does not have a compatible lock"):
        pe.env_share(path)
    assert not (owner / pe.LOCK_FILE).exists()
    assert pe.environment_directory(path) == path


@pytest.mark.parametrize("record", [pe.LOCK_FILE, pe.MANIFEST_FILE, pe.INPUT_FILE])
def test_legacy_personal_records_preserve_isolated_scope(isolated, record):
    _, experiment, _ = isolated
    path = _shared(experiment)
    (path / record).write_text("poc-env-extra==1.0\n" if record == pe.INPUT_FILE else "")
    assert pe.environment_directory(path) == path
    (path / pe.SCOPE_FILE).write_text('scope = "shared"\n')
    assert pe.environment_directory(path) == path.parent.parent


def test_empty_or_comments_only_legacy_input_does_not_create_separate_environment(isolated):
    _, experiment, _ = isolated
    path = _shared(experiment)
    (path / pe.INPUT_FILE).write_text("# Optional packages\n\n")
    assert pe.environment_directory(path) == path.parent.parent
    assert pe.environment_resources(path) == {}


def test_shared_execution_can_overlap_but_all_shared_mutations_are_blocked(isolated, monkeypatch):
    _, experiment, _ = isolated
    first, second = _shared(experiment, "first"), _shared(experiment, "second")
    python = pe.env_add(first, ["poc-env-extra==1.0"])
    for path in (first, second):
        (path / "fit.py").write_text("import poc_env_extra\n")
    real_run = pe.subprocess.run
    active = False

    def during_run(command, **kwargs):
        nonlocal active
        if (
            str(command[0]) == str(python)
            and str(command[1]) == str(first / "fit.py")
            and not active
        ):
            active = True
            try:
                assert pe.execute_script(second, Path("fit.py"), []) == 0
                pe.env_kernel(second)
                for action in (
                    lambda: pe.env_sync(second, repair=True),
                    lambda: pe.env_add(second, ["poc-env-extra==2.0"]),
                    lambda: pe.env_remove(second, ["poc-env-extra"]),
                    lambda: pe.execute_script(first, Path("fit.py"), []),
                ):
                    with pytest.raises(ValueError, match="실행 중"):
                        action()
            finally:
                active = False
        return real_run(command, **kwargs)

    monkeypatch.setattr(pe.subprocess, "run", during_run)
    assert pe.execute_script(first, Path("fit.py"), []) == 0
    pe.env_add(second, ["poc-env-extra==2.0"])
    assert _version(python, "poc-env-extra") == "2.0"


def test_stale_reader_leases_do_not_block_packages(isolated):
    _, experiment, _ = isolated
    path = _shared(experiment)
    owner = pe.environment_directory(path)
    directory = owner / ".poc-env-readers"
    directory.mkdir()
    (directory / "terminated.lock").touch()
    pe.env_sync(path)
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize(
    "name",
    [
        pe.INPUT_FILE,
        pe.LOCK_FILE,
        pe.MANIFEST_FILE,
        pe.ENV_DIR,
        ".poc-env-readers",
        ".poc-env.lock",
    ],
)
def test_shared_owner_symlink_targets_rejected(isolated, name):
    root, experiment, _ = isolated
    path = _shared(experiment)
    owner = pe.environment_directory(path)
    external = root / "untouched"
    external.write_text("untouched")
    (owner / name).symlink_to(external)
    with pytest.raises(ValueError, match="symbolic"):
        pe.env_sync(path)
    assert external.read_text() == "untouched"


@pytest.mark.parametrize(
    "content", ['scope = "unknown"', "scope = []", 'scope = "shared"\nother = true', "invalid"]
)
def test_invalid_scope_rejected(isolated, content):
    _, experiment, _ = isolated
    path = _shared(experiment)
    (path / pe.SCOPE_FILE).write_text(content)
    with pytest.raises(ValueError, match="scope"):
        pe.environment_directory(path)


def test_environment_snapshot_reads_only_active_text_records(isolated):
    root, experiment, _ = isolated
    path = _shared(experiment)
    owner = pe.environment_directory(path)
    (path / pe.INPUT_FILE).write_text("stale-local==1\n")
    (path / pe.SCOPE_FILE).write_text('scope = "shared"\n')
    (owner / pe.INPUT_FILE).write_text("poc-env-extra==1.0\n")
    (owner / pe.ENV_DIR).symlink_to(root / "ignored-venv")
    assert pe.environment_resources(path) == {pe.INPUT_FILE: b"poc-env-extra==1.0\n"}
    (owner / pe.MANIFEST_FILE).write_bytes(b"bad\x00binary")
    with pytest.raises(ValueError, match="binary"):
        pe.environment_resources(path)
    (owner / pe.MANIFEST_FILE).unlink()
    (owner / pe.LOCK_FILE).symlink_to(root / "outside")
    with pytest.raises(ValueError, match="symbolic"):
        pe.environment_resources(path)


def test_distinct_shared_experiments_prepare_once_and_run_concurrently(isolated, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    root, experiment, _ = isolated
    first, second = _shared(experiment, "first"), _shared(experiment, "second")
    owner = first.parent.parent
    (owner / pe.INPUT_FILE).write_text("poc-env-extra==1.0\n")
    compilations = []
    real_uv = pe._run_uv

    def count_compile(executable, arguments, cwd):
        if arguments[:2] == ["pip", "compile"]:
            compilations.append(cwd)
        return real_uv(executable, arguments, cwd)

    monkeypatch.setattr(pe, "_run_uv", count_compile)
    for path, sibling in ((first, second), (second, first)):
        (path / "fit.py").write_text(
            "from pathlib import Path\nimport time, poc_env_extra\n"
            f"Path({path.name + '.started'!r}).touch()\n"
            "deadline = time.monotonic() + 10\n"
            f"while not Path({sibling.name + '.started'!r}).exists():\n"
            "    if time.monotonic() > deadline: raise RuntimeError('other experiment blocked')\n"
            "    time.sleep(0.02)\n"
        )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(pe.execute_script, path, Path("fit.py"), []) for path in (first, second)
        ]
        assert [future.result(timeout=30) for future in futures] == [0, 0]
    assert len(compilations) == 1
    assert (root / "first.started").exists() and (root / "second.started").exists()


def test_terminated_reader_process_releases_lease_and_packages_can_change(isolated):
    import sys
    import time

    root, experiment, _ = isolated
    path = _shared(experiment)
    pe.env_sync(path)
    owner = pe.environment_directory(path)
    directory = owner / ".poc-env-readers"
    directory.mkdir(exist_ok=True)
    marker = root / "reader-ready"
    command = (
        "from filelock import FileLock\nfrom pathlib import Path\nimport time\n"
        f"lock=FileLock({str(directory / 'process.lock')!r}); lock.acquire()\n"
        f"Path({str(marker)!r}).touch()\ntime.sleep(30)\n"
    )
    child = subprocess.Popen([sys.executable, "-c", command])
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        with pytest.raises(ValueError, match="실행 중"):
            pe.env_add(path, ["poc-env-extra==1.0"])
    finally:
        child.kill()
        child.wait(timeout=10)
    python = pe.env_add(path, ["poc-env-extra==1.0"])
    assert _version(python, "poc-env-extra") == "1.0"


def test_environment_snapshot_rejects_non_utf8_and_limits(isolated, monkeypatch):
    _, experiment, _ = isolated
    path = _shared(experiment)
    owner = pe.environment_directory(path)
    source = owner / pe.MANIFEST_FILE
    source.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="UTF-8"):
        pe.environment_resources(path)
    source.write_text("{}")
    monkeypatch.setattr(pe, "MAX_LOCK_BYTES", 1)
    with pytest.raises(ValueError, match="size limit"):
        pe.environment_resources(path)


def test_repeated_add_and_sync_reuse_verified_environment(isolated, monkeypatch):
    _, experiment, _ = isolated
    path = _shared(experiment)
    python = pe.env_add(path, ["poc-env-extra==1.0"])
    owner = pe.environment_directory(path)
    records = pe.environment_resources(path)
    before = (owner / pe.ENV_DIR / "pyvenv.cfg").stat().st_mtime_ns
    monkeypatch.setattr(
        pe, "_run_uv", lambda *args: pytest.fail("verified environment should not reinstall")
    )
    assert pe.env_add(path, ["poc-env-extra==1.0"]) == python
    assert pe.env_sync(path) == python
    assert pe.environment_resources(path) == records
    assert (owner / pe.ENV_DIR / "pyvenv.cfg").stat().st_mtime_ns == before


def test_explicit_repair_reinstalls_without_resolving_locked_versions(isolated, monkeypatch):
    _, experiment, _ = isolated
    path = _shared(experiment)
    pe.env_add(path, ["poc-env-extra==1.0"])
    records = pe.environment_resources(path)
    actions = []
    real_uv = pe._run_uv

    def record(executable, arguments, cwd):
        actions.append(arguments[:2])
        return real_uv(executable, arguments, cwd)

    monkeypatch.setattr(pe, "_run_uv", record)
    pe.env_sync(path, repair=True)
    assert ["pip", "compile"] not in actions
    assert any(action[0] == "venv" for action in actions)
    assert ["pip", "sync"] in actions
    assert pe.environment_resources(path) == records


def test_status_reports_missing_ready_stale_and_changed_installed_packages(isolated):
    _, experiment, _ = isolated
    path = _shared(experiment)
    owner = pe.environment_directory(path)
    (owner / pe.INPUT_FILE).write_text("poc-env-extra==1.0\n")
    missing = pe.env_status(path)
    assert missing["state"] == "install-needed"
    assert missing["scope"] == "shared"
    assert missing["path"] == owner
    assert missing["next_command"].startswith("./poc env sync shared")
    assert missing["active"] == []
    python = pe.env_add(path, ["poc-env-extra==1.0"])
    assert pe.env_status(path)["state"] == "ready"
    metadata = next((owner / pe.ENV_DIR).rglob("poc_env_extra-1.0.dist-info/METADATA"))
    metadata.write_text(metadata.read_text().replace("Version: 1.0", "Version: 9.0"))
    drifted = pe.env_status(path)
    assert drifted["state"] == "update-needed"
    assert "--repair" in drifted["next_command"]
    pe.env_sync(path)
    assert _version(python, "poc-env-extra") == "1.0"
    assert pe.env_status(path)["state"] == "ready"
    (owner / pe.INPUT_FILE).write_text("poc-env-extra==2.0\n")
    assert "--relock" in pe.env_status(path)["next_command"]
    (owner / pe.INPUT_FILE).write_text("--invalid\n")
    assert pe.env_status(path)["state"] == "update-needed"
    assert pe.env_status(path)["next_command"] is None


def test_status_identifies_active_experiment_and_idempotent_sync_allows_other_reader(isolated):
    _, experiment, _ = isolated
    first, second = _shared(experiment, "first"), _shared(experiment, "second")
    pe.env_add(first, ["poc-env-extra==1.0"])
    with pe._running(first):
        status = pe.env_status(second)
        assert status["state"] == "ready"
        assert status["active"] == [{"experiment": "first", "kind": "script", "pid": os.getpid()}]
        pe.env_sync(second)
        pe.env_add(second, ["poc-env-extra==1.0"])
        with pytest.raises(ValueError, match="first.*script"):
            pe.env_sync(second, repair=True)
        with pytest.raises(ValueError, match="first.*script"):
            pe.env_add(second, ["poc-env-extra==2.0"])
    assert pe.env_status(second)["active"] == []


def test_scope_changes_refresh_previously_registered_kernel(isolated):
    _, experiment, _ = isolated
    path = _shared(experiment)
    owner = pe.environment_directory(path)
    pe.env_add(path, ["poc-env-extra==1.0"])
    _, specification = pe.env_kernel(path)
    assert json.loads(specification.read_text())["argv"][0] == str(pe._python(owner / pe.ENV_DIR))
    pe.env_isolate(path)
    assert (path / pe.ENV_DIR).is_dir()
    assert json.loads(specification.read_text())["argv"][0] == str(pe._python(path / pe.ENV_DIR))
    pe.env_share(path)
    assert json.loads(specification.read_text())["argv"][0] == str(pe._python(owner / pe.ENV_DIR))


def test_status_returns_promptly_while_an_environment_update_holds_gate(isolated):
    _, experiment, _ = isolated
    path = _shared(experiment)
    owner = pe.environment_directory(path)
    with FileLock(owner / ".poc-env.lock"):
        status = pe.env_status(path)
    assert status["state"] == "update-needed"
    assert "완료될 때까지" in status["reason"]
    assert status["next_command"].startswith("./poc env status shared")


def test_registering_existing_ready_kernel_is_a_read_only_noop_while_running(isolated):
    _, experiment, _ = isolated
    path = _shared(experiment)
    pe.env_sync(path)
    display, specification = pe.env_kernel(path)
    before = specification.read_bytes()
    active = pe.environment_directory(path)
    with pe._experiment_locked(path):
        with pe._locked(active):
            lease = pe._lease(active, path, kind="kernel")
        try:
            assert pe.env_kernel(path) == (display, specification)
            assert specification.read_bytes() == before
            with pytest.raises(ValueError, match="Shut Down Kernel"):
                pe.env_kernel(path, relock=True)
        finally:
            lease.release()


@pytest.mark.parametrize("content", [None, "", "# Optional packages\n\n"])
@pytest.mark.parametrize("scope", ["shared", "isolated"])
def test_new_empty_environment_executes_with_common_runtime_without_installing(
    isolated, monkeypatch, content, scope
):
    root, experiment, _ = isolated
    path = _shared(experiment) if scope == "shared" else experiment()
    active = pe.environment_directory(path)
    if content is not None:
        (active / pe.INPUT_FILE).write_text(content)
    monkeypatch.setattr(pe, "_uv_binary", lambda: pytest.fail("common runtime must not run uv"))
    (path / "fit.py").write_text(
        "import json, os, sys, shared_sentinel\n"
        "from pathlib import Path\n"
        "Path('common.json').write_text(json.dumps([sys.executable, os.getenv('VIRTUAL_ENV'), "
        "shared_sentinel.VALUE, os.getenv('SIXLETTERS_EXECUTION_JSON')]))\n"
    )
    monkeypatch.setenv("VIRTUAL_ENV", "/stale/personal/environment")
    assert pe.execute_script(path, Path("fit.py"), ["--seed", "42"]) == 0
    output = json.loads((root / "common.json").read_text())
    assert output[:3] == [
        sys.executable,
        sys.prefix if sys.prefix != sys.base_prefix else None,
        "shared-source",
    ]
    assert json.loads(output[3])["args"] == ["--seed", "42"]
    assert pe.env_sync(path) == Path(sys.executable)
    status = pe.env_status(path)
    assert status["state"] == "ready" and status["next_command"] is None
    assert status["python"] == Path(sys.executable)
    assert "공통 환경" in status["reason"]
    assert not any(
        (active / name).exists() for name in (pe.ENV_DIR, pe.LOCK_FILE, pe.MANIFEST_FILE)
    )


def test_common_reader_blocks_first_personal_install_but_allows_other_common_reads(isolated):
    _, experiment, _ = isolated
    first, second = _shared(experiment, "first"), _shared(experiment, "second")
    with pe._running(first):
        with pe._running(second) as (_, python):
            assert python == Path(sys.executable)
        assert pe.env_sync(second) == Path(sys.executable)
        with pytest.raises(ValueError, match="first.*script"):
            pe.env_add(second, ["poc-env-extra==1.0"])
        with pytest.raises(ValueError, match="실행 중"):
            with pe._running(first):
                pytest.fail("the same experiment must not overlap")
    assert _version(pe.env_add(second, ["poc-env-extra==1.0"]), "poc-env-extra") == "1.0"


def test_empty_previously_pinned_environment_is_preserved(isolated):
    _, experiment, baseline = isolated
    path = _shared(experiment)
    python = pe.env_add(path, ["poc-env-extra==1.0"])
    pe.env_remove(path, ["poc-env-extra"])
    active = pe.environment_directory(path)
    assert (active / pe.INPUT_FILE).read_text() == ""
    before = pe.environment_resources(path)
    assert pe.env_sync(path) == python
    assert _version(python, "poc-env-base") == "1.0"
    assert pe.environment_resources(path) == before
    baseline["fingerprint"] = "changed-common-runtime"
    with pytest.raises(ValueError, match="--relock"):
        pe.env_sync(path)


@pytest.mark.parametrize("record", [pe.LOCK_FILE, pe.MANIFEST_FILE])
def test_incomplete_saved_lock_is_not_bypassed_by_empty_input(isolated, record):
    _, experiment, _ = isolated
    path = _shared(experiment)
    active = pe.environment_directory(path)
    (active / record).write_text("")
    with pytest.raises(ValueError, match="--relock"):
        pe.env_sync(path)


@pytest.mark.parametrize("action", ["add", "sync", "exec", "kernel"])
def test_first_personal_install_refreshes_registered_member_kernels_only(isolated, action):
    _, experiment, _ = isolated
    first, second = _shared(experiment, "first"), _shared(experiment, "second")
    separate = experiment("isolated")
    paths = {path: pe.env_kernel(path)[1] for path in (first, second, separate)}
    assert all(json.loads(spec.read_text())["argv"][0] == sys.executable for spec in paths.values())
    if action == "add":
        python = pe.env_add(first, ["poc-env-extra==1.0"])
    else:
        (first.parent.parent / pe.INPUT_FILE).write_text("poc-env-extra==1.0\n")
        if action == "sync":
            python = pe.env_sync(first)
        elif action == "exec":
            (first / "fit.py").write_text("import poc_env_extra\n")
            assert pe.execute_script(first, Path("fit.py"), []) == 0
            python = pe._python(first.parent.parent / pe.ENV_DIR)
        else:
            _, current = pe.env_kernel(first)
            python = Path(json.loads(current.read_text())["argv"][0])
    assert python == pe._python(first.parent.parent / pe.ENV_DIR)
    for path in (first, second):
        spec = json.loads(paths[path].read_text())
        assert spec["argv"][0] == str(python)
        assert spec["argv"][-2:] == ["--runtime-prefix", str(first.parent.parent / pe.ENV_DIR)]
    assert json.loads(paths[separate].read_text())["argv"][0] == sys.executable
    assert not (separate / pe.ENV_DIR).exists()


def test_common_kernel_requires_registered_prefix_and_holds_member_lease(isolated, monkeypatch):
    _, experiment, _ = isolated
    first, second = _shared(experiment, "first"), _shared(experiment, "second")
    with pytest.raises(ValueError, match="등록된 SixSeven"):
        pe.launch_kernel(first, Path("connection.json"), runtime_prefix=Path("/wrong/runtime"))
    visited = []

    def run(path, active, connection):
        visited.append(path)
        with pytest.raises(ValueError, match="first.*kernel"):
            pe.env_add(second, ["poc-env-extra==1.0"])
        assert pe.env_status(second)["active"][0]["kind"] == "kernel"

    monkeypatch.setattr(pe, "_start_kernel", run)
    pe.launch_kernel(first, Path("connection.json"), runtime_prefix=Path(sys.prefix))
    assert visited == [first]
    assert pe.env_status(second)["active"] == []


def test_overlay_installs_only_missing_transitive_packages_and_reuses_shared_files(isolated):
    root, experiment, _ = isolated
    _wheel(root / "wheels", "poc-env-transitive", "1.0")
    _wheel(
        root / "wheels",
        "poc-env-dependent",
        "1.0",
        dependencies=["poc-env-base==1.0", "poc-env-transitive==1.0"],
    )
    path = experiment()
    python = pe.env_add(path, ["poc-env-dependent"])
    sites = pe._site_paths(python)
    assert pe._shared_packages([str(site) for site in sites]) == {
        "poc-env-dependent": "1.0",
        "poc-env-transitive": "1.0",
    }
    observed = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import json,poc_env_base,poc_env_dependent; print(json.dumps([poc_env_base.__file__,poc_env_dependent.__file__]))",
            ],
            text=True,
        )
    )
    assert Path(observed[0]).parent == root / "shared-site-packages"
    assert Path(observed[1]).is_relative_to(path / pe.ENV_DIR)
    assert "poc-env-base==1.0" in (path / pe.LOCK_FILE).read_text()
    for data in pe.environment_resources(path).values():
        assert str(root).encode() not in data


def test_overlay_processes_trusted_common_pth_hooks(isolated):
    root, experiment, _ = isolated
    external = root / "shared-editable-source"
    external.mkdir()
    (external / "shared_hook.py").write_text("VALUE = 17\n")
    (root / "shared-site-packages/editable.pth").write_text(str(external) + "\n")
    python = pe.env_add(experiment(), ["poc-env-extra==1.0"])
    assert (
        subprocess.check_output(
            [str(python), "-c", "import shared_hook; print(shared_hook.VALUE)"], text=True
        ).strip()
        == "17"
    )


def test_moved_shared_runtime_relinks_without_changing_portable_records(isolated, monkeypatch):
    import shutil

    root, experiment, _ = isolated
    path = experiment()
    python = pe.env_add(path, ["poc-env-extra==1.0"])
    records = pe.environment_resources(path)
    relocated = root / "relocated-shared"
    shutil.copytree(root / "shared-site-packages", relocated)
    monkeypatch.setattr(pe, "_shared_paths", lambda: [str(relocated)])
    assert not pe._ready(path, json.loads((path / pe.MANIFEST_FILE).read_text()))
    assert pe.env_status(path)["state"] != "ready"
    pe.env_sync(path, repair=True)
    assert pe.environment_resources(path) == records
    assert pe.env_status(path)["state"] == "ready"
    observed = subprocess.check_output(
        [str(python), "-c", "import poc_env_base; print(poc_env_base.__file__)"], text=True
    ).strip()
    assert Path(observed).parent == relocated


@pytest.mark.parametrize("metadata_backend", ["default", "backport"])
def test_missing_shared_package_cannot_publish_or_report_ready(
    isolated, monkeypatch, metadata_backend
):
    root, experiment, _ = isolated
    path = experiment()
    pe.env_add(path, ["poc-env-extra==1.0"])
    before = {name: (path / name).read_bytes() for name in (*pe.LOCK_NAMES, pe.STATE_FILE)}
    metadata = root / "shared-site-packages/poc_env_base-1.0.dist-info/METADATA"
    metadata.unlink()
    if metadata_backend == "backport":
        import importlib_metadata

        monkeypatch.setattr(
            pe.importlib.metadata, "distributions", importlib_metadata.distributions
        )
        distribution = next(importlib_metadata.distributions(path=[str(metadata.parent.parent)]))
        with pytest.raises(OSError):
            _ = distribution.metadata
    assert pe.env_status(path)["state"] != "ready"
    with pytest.raises(ValueError, match="Shared runtime"):
        pe.env_sync(path, repair=True)
    assert {name: (path / name).read_bytes() for name in before} == before


@pytest.mark.parametrize("shadow", ["module", "project-module"])
def test_shadowing_shared_runtime_is_rejected_before_atomic_publication(isolated, shadow):
    root, experiment, _ = isolated
    path = experiment()
    python = pe.env_add(path, ["poc-env-extra==1.0"])
    before = {name: (path / name).read_bytes() for name in (*pe.LOCK_NAMES, pe.STATE_FILE)}
    files = {
        "module": {"poc_env_base.py": "VALUE = 'shadow'\n"},
        "duplicate-dist": {
            "poc_env_base-1.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: poc-env-base\nVersion: 1.0\n"
        },
        "project-module": {"sixletters.py": "VALUE = 'shadow'\n"},
    }[shadow]
    _wheel(root / "wheels", "poc-env-shadow", "1.0", extra_files=files)
    with pytest.raises(ValueError, match="shadow shared modules"):
        pe.env_add(path, ["poc-env-shadow"])
    assert {name: (path / name).read_bytes() for name in before} == before
    assert _version(python, "poc-env-extra") == "1.0"


def test_namespace_packages_can_add_distinct_modules_without_shadowing(isolated):
    root, experiment, _ = isolated
    shared = root / "shared-site-packages/poc_namespace"
    shared.mkdir()
    (shared / "common.py").write_text("VALUE = 1\n")
    _wheel(
        root / "wheels",
        "poc-env-namespace",
        "1.0",
        extra_files={
            "poc_namespace/extra.py": "VALUE = 2\n",
        },
    )
    path = experiment()
    python = pe.env_add(path, ["poc-env-namespace"])
    assert (
        subprocess.check_output(
            [
                str(python),
                "-c",
                "from poc_namespace import common,extra; print(common.VALUE + extra.VALUE)",
            ],
            text=True,
        ).strip()
        == "3"
    )
    _wheel(
        root / "wheels",
        "poc-env-shadow",
        "1.0",
        extra_files={
            "poc_namespace/common.py": "VALUE = 3\n",
        },
    )
    with pytest.raises(ValueError, match="shadow shared modules"):
        pe.env_add(path, ["poc-env-shadow"])


def test_full_legacy_environment_is_reused_until_explicit_repair(isolated, monkeypatch):
    root, experiment, _ = isolated
    path = experiment()
    python = pe.env_add(path, ["poc-env-extra==1.0"])
    # Recreate an environment installed by the previous implementation.
    for site in pe._site_paths(python):
        (site / pe.SHARED_PTH).unlink()
    pe._run_uv(
        pe._uv_binary(), ["pip", "sync", str(path / pe.LOCK_FILE), "--python", str(python)], path
    )
    manifest = json.loads((path / pe.MANIFEST_FILE).read_text())
    (path / pe.STATE_FILE).write_text(pe._state(manifest))
    records = pe.environment_resources(path)
    before = (path / pe.ENV_DIR / "pyvenv.cfg").stat().st_mtime_ns
    original = pe._run_uv
    monkeypatch.setattr(
        pe, "_run_uv", lambda *args: pytest.fail("must preserve legacy environment")
    )
    assert pe.env_sync(path) == python
    assert (path / pe.ENV_DIR / "pyvenv.cfg").stat().st_mtime_ns == before
    monkeypatch.setattr(pe, "_run_uv", original)
    pe.env_sync(path, repair=True)
    assert pe.environment_resources(path) == records
    assert all(not list(site.glob("poc_env_base*")) for site in pe._site_paths(python))
    assert _version(python, "poc-env-base") == "1.0"


def test_duplicate_shared_distribution_is_detected_in_installed_personal_env(isolated):
    import shutil

    root, experiment, _ = isolated
    path = experiment()
    python = pe.env_add(path, ["poc-env-extra==1.0"])
    site = pe._site_paths(python)[0]
    shutil.copytree(
        root / "shared-site-packages/poc_env_base-1.0.dist-info",
        site / "poc_env_base-1.0.dist-info",
    )
    status = pe.env_status(path)
    assert status["state"] == "update-needed"
    assert "--repair" in status["next_command"]


def test_additional_pth_cannot_shadow_shared_modules_through_external_path(isolated):
    root, experiment, _ = isolated
    external = root / "additional-source"
    external.mkdir()
    (external / "poc_env_base.py").write_text("VALUE = 'shadow'\n")
    _wheel(
        root / "wheels",
        "poc-env-hook",
        "1.0",
        extra_files={"hook.pth": f"import sys; sys.path.insert(0, {str(external)!r})\n"},
    )
    path = experiment()
    with pytest.raises(ValueError, match="shadow shared modules"):
        pe.env_add(path, ["poc-env-hook"])
    assert not (path / pe.ENV_DIR).exists()


def test_existing_common_vendor_paths_do_not_block_additional_packages(isolated):
    root, experiment, _ = isolated
    vendor = root / "common-vendor"
    vendor.mkdir()
    (vendor / "poc_env_base.py").write_text("VALUE = 'vendor'\n")
    (root / "shared-site-packages/vendor.pth").write_text(str(vendor) + "\n")
    path = experiment()
    python = pe.env_add(path, ["poc-env-extra==1.0"])
    assert _version(python, "poc-env-base") == "1.0"
    assert pe.env_status(path)["state"] == "ready"


def test_active_reader_refresh_guidance_waits_instead_of_relocking(isolated):
    _, experiment, _ = isolated
    first, second = _shared(experiment, "first"), _shared(experiment, "second")
    pe.env_add(first, ["poc-env-extra==1.0"])
    active = pe.environment_directory(first)
    with pe._running(first):
        (active / pe.INPUT_FILE).write_text("poc-env-extra==2.0\n")
        status = pe.env_status(second)
        assert "Shut Down Kernel" in status["reason"]
        assert "--relock" not in status["next_command"]
        assert "env status second" in status["next_command"]
        with pytest.raises(ValueError, match="실행 중") as error:
            pe.env_sync(second)
        assert "--relock" not in str(error.value)
    assert "--relock" in pe.env_status(second)["next_command"]


def test_kernel_refresh_distinguishes_changed_records_from_broken_install(isolated):
    _, experiment, _ = isolated
    path = experiment()
    pe.env_add(path, ["poc-env-extra==1.0"])
    (path / pe.ENV_DIR / "pyvenv.cfg").unlink()
    with pytest.raises(pe.EnvironmentRefreshError, match="--repair") as error:
        pe._kernel_lease(path, path)
    assert "--relock" not in str(error.value)
    (path / pe.INPUT_FILE).write_text("poc-env-extra==2.0\n")
    with pytest.raises(pe.EnvironmentRefreshError, match="--relock"):
        pe._kernel_lease(path, path)


def test_kernel_and_status_both_require_relock_after_shared_package_change(isolated):
    root, experiment, baseline = isolated
    path = experiment()
    pe.env_add(path, ["poc-env-extra==1.0"])
    metadata = root / "shared-site-packages/poc_env_base-1.0.dist-info/METADATA"
    metadata.write_text(metadata.read_text().replace("Version: 1.0", "Version: 2.0"))
    baseline["packages"]["poc-env-base"] = "2.0"
    baseline["fingerprint"] = "base-v2"
    with pytest.raises(pe.EnvironmentRefreshError, match="--relock") as error:
        pe._kernel_lease(path, path)
    assert "--repair" not in str(error.value)
    status = pe.env_status(path)
    assert status["reason"] == error.value.reason
    assert status["next_command"] in str(error.value)
