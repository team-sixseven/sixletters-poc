"""Verified archives become editable experiments, without executing archived content."""

import json
import os
import shutil
import subprocess
import sys
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from test_tracking import setup_tracking  # noqa: F401

from sixletters import reproduction, tracking
from sixletters.__main__ import personal_experiment
from sixletters.comparison import compare_runs
from sixletters.contracts import _hash_json
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment
from sixletters.provenance import python_sources
from sixletters.reproduction import fetch_archive, restore_run
from sixletters.workspace import init_experiment


@pytest.fixture
def original(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    tmp_path = tmp_path.resolve()
    config = init_experiment("original", "first", root=tmp_path / "Members", script="fit.py")
    (config.parent / "fit.py").write_text("raise RuntimeError('Must never execute on restore')\n")
    settings = config.parent / "settings"
    settings.mkdir()
    (settings / "features.yaml").write_text("depth: 3\n")
    data = save_prepared_dataset(make_demo_dataset(), tmp_path / "data", synthetic=True)
    model = tmp_path / "model.py"
    model.write_text("raise RuntimeError('Must never load on restore')\n")
    experiment = Experiment(data, config, output_root=tmp_path / "artifacts")
    run = experiment.save(
        pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5}),
        model_files={"model.py": model},
        training={"model": "ConstantScore", "seed": None, "parameters": {"score": 0.5}},
    )
    return run, data, tmp_path / "Members"


def _restore(original, **options):
    run, data, root = original
    return restore_run(run, owner="restored", name="second", root=root, data=data, **options)


def _files(folder):
    return {
        path.relative_to(folder): path.read_bytes() for path in folder.rglob("*") if path.is_file()
    }


def test_restore_preserves_files_and_trace_without_loading_code_or_models(original):
    run, data, root = original
    before = _files(run)
    config = _restore(original)
    result = json.loads((run / "result.json").read_text())
    saved = tomllib.loads((run / "config.toml").read_text())
    restored = tomllib.loads(config.read_text())
    saved["experiment"].update(owner="restored", name="second")
    assert restored == saved
    assert config == root / "restored/experiments/second/config.toml"
    assert (config.parent / "fit.py").read_bytes() == (run / "source/fit.py").read_bytes()
    assert (config.parent / "settings/features.yaml").read_bytes() == (
        run / "resources/settings/features.yaml"
    ).read_bytes()
    assert (config.parent / "outputs/.restore/models/model.py").read_bytes() == (
        run / "models/model.py"
    ).read_bytes()
    assert (config.parent / "outputs/.restore/original_config.toml").read_bytes() == (
        run / "config.toml"
    ).read_bytes()
    trace_path = config.parent / "outputs/.restore/trace.json"
    trace = json.loads(trace_path.read_text())
    assert trace["original"]["owner"] == "original"
    assert trace["original"]["training"] == result["training"]
    assert trace["original"]["dataset"] == result["comparison_context"]["dataset"]
    assert trace["automatically_executed"] is False
    assert _files(run) == before
    assert not any(".restore" in name for name in python_sources(config.parent))
    # The restored experiment is a normal editable experiment, not a disconnected dump.
    experiment = Experiment(data, config, output_root=root.parent / "new-artifacts")
    following = experiment.save(
        pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5}),
        model_files={"model.py": run / "models/model.py"},
        training={"model": "ConstantScore", "seed": None, "parameters": {"score": 0.5}},
    )
    assert trace_path.is_file()
    assert not any(".restore" in str(path) for path in (following / "resources").rglob("*"))
    assert "trace.json" not in json.dumps(
        json.loads((following / "result.json").read_text())["resources"]
    )
    assert not any(".restore" in str(path) for path in (following / "source").rglob("*.py"))


def test_existing_destination_is_never_replaced(original):
    _, _, root = original
    target = root / "restored/experiments/second"
    target.mkdir(parents=True)
    marker = target / "keep.txt"
    marker.write_text("existing work")
    with pytest.raises(FileExistsError):
        _restore(original)
    assert marker.read_text() == "existing work"


@pytest.mark.parametrize("change", ["version", "split", "value", "synthetic"])
def test_wrong_common_data_is_rejected_before_creating_experiment(original, change):
    run, _, root = original
    dataset = make_demo_dataset()
    if change == "version":
        dataset.dataset_id = "synthetic-demo-v2"
    elif change == "split":
        train, validation = dataset.frame.index[[0, 108]]
        dataset.frame.loc[train, "split"] = "validation"
        dataset.frame.loc[validation, "split"] = "train"
    elif change == "value":
        dataset.frame.loc[0, "clinical_1"] = 123.0
    data = save_prepared_dataset(
        dataset, root.parent / "different-data", synthetic=change != "synthetic"
    )
    with pytest.raises(ValueError, match="--data"):
        restore_run(run, owner="restored", name="second", root=root, data=data)
    assert not (root / "restored").exists()


@pytest.mark.parametrize("bad", ["../escape", "nested/name", "bad\\name", "CON", ".hidden"])
def test_unsafe_identity_rejected(original, bad):
    run, data, root = original
    with pytest.raises(ValueError):
        restore_run(run, owner=bad, name="second", root=root, data=data)
    assert not (root.parent / "escape").exists()


def test_case_collisions_are_rejected(original):
    run, data, root = original
    (root / "Restored").mkdir()
    with pytest.raises(ValueError, match="collides"):
        restore_run(run, owner="restored", name="second", root=root, data=data)


@pytest.mark.parametrize(
    "source", ["source/fit.py", "models/model.py", "resources/settings/features.yaml"]
)
def test_tampered_bundle_refused_before_destination_created(original, source):
    run, _, root = original
    (run / source).write_text("tampered\n")
    with pytest.raises((ValueError, tracking.TrackingError)):
        _restore(original)
    assert not (root / "restored").exists()


def test_partial_basic_bundle_cannot_be_restored(original):
    run, _, root = original
    shutil.rmtree(run / "source")
    with pytest.raises((ValueError, tracking.TrackingError)):
        _restore(original)
    assert not (root / "restored").exists()


@pytest.mark.parametrize("position", ["root", "run", "file"])
def test_symlinks_are_not_followed(original, position):
    run, data, root = original
    if position == "root":
        alias = root.parent / "linked-Members"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    elif position == "run":
        alias = root.parent / "linked-run"
        alias.symlink_to(run, target_is_directory=True)
        run = alias
    else:
        source = run / "source/fit.py"
        external = root.parent / "outside.py"
        external.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(external)
    with pytest.raises((ValueError, tracking.TrackingError)):
        restore_run(run, owner="restored", name="second", root=root, data=data)


def test_copy_failure_removes_partial_destination_and_preserves_original(original, monkeypatch):
    run, _, root = original
    before = _files(run)
    copy = reproduction._copy_verified
    calls = 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("interrupted copy")
        return copy(*args)

    monkeypatch.setattr(reproduction, "_copy_verified", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        _restore(original)
    assert not (root / "restored/experiments/second").exists()
    assert _files(run) == before


def test_simultaneous_restore_does_not_overwrite_first_writers_destination(original, monkeypatch):
    copying = threading.Event()
    release = threading.Event()
    copy = reproduction._copy_verified

    def slow_copy(*args):
        copying.set()
        assert release.wait(timeout=10)
        return copy(*args)

    monkeypatch.setattr(reproduction, "_copy_verified", slow_copy)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(_restore, original)
        assert copying.wait(timeout=10)
        try:
            with pytest.raises(FileExistsError):
                _restore(original)
        finally:
            release.set()
        config = first.result(timeout=10)
    assert config.is_file()
    assert (config.parent / "fit.py").read_bytes() == (original[0] / "source/fit.py").read_bytes()


def test_changed_data_during_copy_removes_incomplete_experiment(original, monkeypatch):
    _, data, root = original
    copy = reproduction._copy_verified
    changed = False

    def change_data(*args):
        nonlocal changed
        copy(*args)
        if not changed:
            data.write_bytes(data.read_bytes() + b"\n")
            changed = True

    monkeypatch.setattr(reproduction, "_copy_verified", change_data)
    with pytest.raises(ValueError, match="공통 데이터가 변경"):
        _restore(original)
    assert not (root / "restored/experiments/second").exists()


def test_alternate_active_config_name_and_execution_trace_are_preserved(original):
    run, _, _ = original
    path = run / "result.json"
    result = json.loads(path.read_text())
    result["config_filename"] = "balanced.toml"
    result["execution"] = {
        "script": "fit.py",
        "args": ["--config", "balanced.toml"],
        "cwd": "workspace",
    }
    path.write_text(json.dumps(result))
    config = _restore(original)
    assert config.name == "balanced.toml"
    assert config.read_bytes() == config.with_name("balanced.toml").read_bytes()
    trace = json.loads((config.parent / "outputs/.restore/trace.json").read_text())
    assert trace["original"]["execution"] == result["execution"]


def test_active_alternate_config_restores_without_overwriting_original_default(original):
    _, data, root = original
    default = root / "original/experiments/first/config.toml"
    alternate = default.with_name("balanced.toml")
    alternate.write_text(default.read_text().replace("C = 1.0", "C = 0.25"))
    experiment = Experiment(data, alternate, output_root=root.parent / "other-artifacts")
    run = experiment.save(
        pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5}),
        model_files={"model.txt": root.parent / "model.py"},
        training={"model": "ConstantScore", "seed": None, "parameters": {"score": 0.5}},
    )
    assert (run / "resources/config.toml").read_bytes() == default.read_bytes()
    restored = restore_run(run, owner="restored", name="alternate", root=root, data=data)
    assert restored.name == "balanced.toml"
    inactive = tomllib.loads(restored.with_name("config.toml").read_text())
    original_inactive = tomllib.loads(default.read_text())
    original_inactive["experiment"].update(owner="restored", name="alternate")
    assert inactive == original_inactive
    assert (
        restored.parent / "outputs/.restore/original_default_config.toml"
    ).read_bytes() == default.read_bytes()
    active = tomllib.loads(restored.read_text())
    assert active["experiment"]["owner"] == "restored"
    assert active["parameters"]["C"] == 0.25
    assert personal_experiment(root.parent, "restored", "alternate") == restored.parent
    # Both configurations match the new owner; active parameters remain unchanged.
    assert Experiment(data, restored).parameters["C"] == 0.25


def test_legacy_alternate_config_collision_is_not_guessed(original):
    _, _, root = original
    original_config = root / "original/experiments/first/config.toml"
    alternate = original_config.with_name("alternate.toml")
    alternate.write_bytes(original_config.read_bytes())
    experiment = Experiment(original[1], alternate, output_root=root.parent / "legacy-artifacts")
    legacy = experiment.save(
        pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5}),
        model_files={"model.txt": root.parent / "model.py"},
        training={"model": "ConstantScore", "seed": None, "parameters": {"score": 0.5}},
    )
    path = legacy / "result.json"
    result = json.loads(path.read_text())
    result.pop("config_filename")
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="충돌"):
        restore_run(legacy, owner="restored", name="legacy", root=root, data=original[1])
    assert not (root / "restored/experiments/legacy").exists()


def test_config_toml_resource_without_experiment_identity_refuses_ambiguous_restore(original):
    _, data, root = original
    default = root / "original/experiments/first/config.toml"
    alternate = default.with_name("active.toml")
    alternate.write_bytes(default.read_bytes())
    default.write_text('[processing]\nmethod = "custom"\n')
    experiment = Experiment(data, alternate, output_root=root.parent / "alternate-artifacts")
    run = experiment.save(
        pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5}),
        model_files={"model.txt": root.parent / "model.py"},
        training={"model": "ConstantScore", "seed": None, "parameters": {"score": 0.5}},
    )
    with pytest.raises(ValueError, match="실험 설정 형식과 충돌"):
        restore_run(run, owner="restored", name="alternate", root=root, data=data)
    assert not (root / "restored/experiments/alternate").exists()


def test_legacy_default_config_without_filename_or_execution_metadata_restores(original):
    run, _, _ = original
    path = run / "result.json"
    result = json.loads(path.read_text())
    result.pop("config_filename", None)
    result.pop("execution", None)
    path.write_text(json.dumps(result))
    config = _restore(original)
    assert config.name == "config.toml"


def test_remote_archive_download_restore_and_cache_revalidation(tmp_path, setup_tracking):  # noqa: F811
    _, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    assert tracking.archive_run(run) == "uploaded"
    remote_id = next(iter(client.runs))
    cached = fetch_archive(tmp_path.resolve() / "cache", remote_id)
    assert _files(cached) == _files(run)
    assert all(name.startswith("reproduction/") for name in client.downloads)
    downloaded = len(client.downloads)
    assert fetch_archive(tmp_path.resolve() / "cache", remote_id) == cached
    assert len(client.downloads) == downloaded
    config = restore_run(
        cached,
        owner="other",
        name="restored",
        root=tmp_path.resolve() / "Members",
        data=tmp_path / "data",
    )
    assert config.is_file()
    (cached / "models/weights.txt").write_text("modified cache")
    with pytest.raises((ValueError, tracking.TrackingError)):
        fetch_archive(tmp_path.resolve() / "cache", remote_id)


def test_remote_basic_result_requires_selected_archive(tmp_path, setup_tracking):  # noqa: F811
    root, client, make_run, enable = setup_tracking
    make_run()
    enable()
    tracking.sync_runs(root)
    with pytest.raises(ValueError, match="dev archive"):
        fetch_archive(tmp_path.resolve() / "cache", next(iter(client.runs)))
    assert not client.downloads


@pytest.mark.parametrize("tamper", ["model", "manifest", "unsafe_path"])
def test_remote_corruption_never_populates_completed_cache(tmp_path, setup_tracking, tamper):  # noqa: F811
    _, client, make_run, enable = setup_tracking
    run = make_run()
    enable()
    tracking.archive_run(run)
    remote_id = next(iter(client.runs))
    manifest_path = client.root / remote_id / "reproduction/manifest.json"
    if tamper == "model":
        (manifest_path.parent / "models/weights.txt").write_text("modified remotely")
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["../escape.py" if tamper == "unsafe_path" else "models/extra.txt"] = {
            "sha256": "0" * 64,
            "bytes": 1,
        }
        manifest_path.write_text(json.dumps(manifest))
        if tamper == "unsafe_path":
            client.runs[remote_id].data.tags[tracking.PREFIX + "archive_hash"] = _hash_json(
                manifest
            )
    cache = tmp_path.resolve() / "cache"
    with pytest.raises((ValueError, tracking.TrackingError)):
        fetch_archive(cache, remote_id)
    assert not list(cache.rglob("result.json"))
    assert not (tmp_path / "escape.py").exists()


def test_restored_starter_retrains_to_identical_predictions_when_explicitly_executed(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    tmp_path = tmp_path.resolve()
    root = tmp_path / "Members"
    config = init_experiment("first", "logistic", root=root)
    data = save_prepared_dataset(make_demo_dataset(), tmp_path / "data", synthetic=True)
    output = tmp_path / "artifacts"

    def train(config):
        completed = subprocess.run(
            [
                sys.executable,
                str(config.with_name("train.py")),
                "--data",
                str(data),
                "--output-root",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "MLFLOW_TRACKING_URI": ""},
        )
        assert "Validation results saved" in completed.stdout
        settings = tomllib.loads(config.read_text())["experiment"]
        return next((output / settings["owner"] / settings["name"]).iterdir())

    original = train(config)
    restored = restore_run(original, owner="second", name="copied", root=root, data=data)
    copied = train(restored)
    pd.testing.assert_frame_equal(
        pd.read_csv(original / "predictions.csv"), pd.read_csv(copied / "predictions.csv")
    )
    before = json.loads((original / "result.json").read_text())
    after = json.loads((copied / "result.json").read_text())
    assert before["metrics"] == after["metrics"]


def test_restored_input_groups_remain_comparable_after_retraining(tmp_path):
    root = tmp_path.resolve()
    data = save_prepared_dataset(make_demo_dataset(), root / "data", synthetic=True)
    output = root / "artifacts"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("MLFLOW_", "SIXLETTERS_EXECUTION_"))
    }

    def train(config):
        subprocess.run(
            [
                sys.executable,
                str(config.with_name("train.py")),
                "--data",
                str(data),
                "--output-root",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        settings = tomllib.loads(config.read_text())["experiment"]
        return next((output / settings["owner"] / settings["name"]).iterdir())

    original = [
        train(init_experiment("Alice", group, feature_set=group, root=root / "Members"))
        for group in ("clinical", "ecg", "combined")
    ]
    before = compare_runs(original, root / "original-inputs.csv", mode="inputs")
    restored_runs = []
    for run in original:
        config = restore_run(
            run, owner="Bob", name=run.parent.name, root=root / "Members", data=data
        )
        trace = json.loads((config.parent / "outputs/.restore/trace.json").read_text())
        assert trace["original"]["run_id"] == run.name
        restored = train(config)
        pd.testing.assert_frame_equal(
            pd.read_csv(run / "predictions.csv"), pd.read_csv(restored / "predictions.csv")
        )
        restored_runs.append(restored)

    after = compare_runs(restored_runs, root / "restored-inputs.csv", mode="inputs")
    assert len(before) == len(after) == 3
    for original_run, restored_run in zip(original, restored_runs, strict=True):
        before_metadata = json.loads((original_run / "result.json").read_text())
        after_metadata = json.loads((restored_run / "result.json").read_text())
        # Restore selects an independent scope without changing training resources.
        assert before_metadata["resources"] == {
            key: value
            for key, value in after_metadata["resources"].items()
            if key != "environment.toml"
        }
        assert (restored_run / "resources/environment.toml").read_text() == 'scope = "isolated"\n'
