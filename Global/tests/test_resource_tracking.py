"""Additional settings stay local until archive and participate in fixed-method comparisons."""

import hashlib
import json
import shutil

import pandas as pd
import pytest
from test_tracking import setup_tracking  # noqa: F401

from sixletters import tracking
from sixletters.comparison import compare_runs
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment
from sixletters.provenance import source_hash
from sixletters.workspace import init_experiment


def resource_run(tmp_path, owner, *, feature_set="clinical", depth=3):
    config = init_experiment(
        owner, "exp001", root=tmp_path / "Members", script="fit_rf.py", feature_set=feature_set
    )
    folder = config.parent / "settings"
    folder.mkdir()
    (folder / "features.yaml").write_text(f"depth: {depth}\n", encoding="utf-8")
    data = tmp_path / "data"
    if not data.exists():
        save_prepared_dataset(make_demo_dataset(), data, synthetic=True)
    model = tmp_path / "weights.txt"
    model.write_text("opaque model")
    experiment = Experiment(data, config, output_root=tmp_path / "artifacts")
    return experiment.save(
        pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5}),
        model_files={"weights.txt": model},
        training={"model": "ConstantScore", "seed": None, "parameters": {"score": 0.5}},
    )


def test_basic_sync_omits_resources_selected_archive_keeps_exact_bytes(tmp_path, setup_tracking):  # noqa: F811
    root, client, _, enable = setup_tracking
    run = resource_run(tmp_path, "팀원")
    enable()
    assert tracking.sync_runs(root)["uploaded"] == 1
    assert not any("resources" in folder or "source" in folder for _, folder, _ in client.uploads)
    result, manifest = tracking._full_manifest(run)
    resource = "resources/settings/features.yaml"
    assert manifest["files"][resource] == result["resources"]["settings/features.yaml"]
    assert "source/fit_rf.py" in manifest["files"]
    assert "source/train.py" not in manifest["files"]
    tracking.archive_run(run)
    remote_id = next(iter(client.runs))
    assert (client.root / remote_id / "reproduction" / resource).read_bytes() == (
        run / resource
    ).read_bytes()


@pytest.mark.parametrize("corruption", ["modify", "delete", "extra"])
def test_archive_rejects_resource_changes_while_basic_sync_does_not_read_resources(
    tmp_path, corruption
):
    run = resource_run(tmp_path, "팀원")
    resource = run / "resources/settings/features.yaml"
    if corruption == "modify":
        resource.write_text("depth: 5\n")
    elif corruption == "delete":
        resource.unlink()
    else:
        (resource.parent / "extra.yaml").write_text("unexpected: true\n")
    assert set(tracking._local_manifest(run)[1]["files"]) == set(tracking.BASIC_FILES)
    with pytest.raises(tracking.BundleError):
        tracking._full_manifest(run)


def test_resource_settings_vary_between_experiments_but_not_fixed_method_inputs(tmp_path):
    first = resource_run(tmp_path, "one")
    varied = resource_run(tmp_path, "two", depth=5)
    assert len(compare_runs([first, varied], tmp_path / "experiments.csv")) == 2
    ecg = resource_run(tmp_path, "three", feature_set="ecg", depth=5)
    with pytest.raises(ValueError, match="same training method"):
        compare_runs([first, ecg], tmp_path / "inputs.csv", mode="inputs")
    same = resource_run(tmp_path, "four", feature_set="ecg")
    # Remote basic bundles need metadata only, not reproduction files.
    for run in (first, same):
        shutil.rmtree(run / "resources")
    assert len(compare_runs([first, same], tmp_path / "same-inputs.csv", mode="inputs")) == 2


def test_legacy_runs_without_resource_metadata_still_compare_and_archive(tmp_path, setup_tracking):  # noqa: F811
    _, _, make_run, _ = setup_tracking
    first, second = make_run("one"), make_run("two")
    for run in (first, second):
        result_path = run / "result.json"
        result = json.loads(result_path.read_text())
        result.pop("resources")
        result_path.write_text(json.dumps(result))
        resources = run / "resources"
        if resources.exists():
            shutil.rmtree(resources)
        tracking._full_manifest(run)
    assert len(compare_runs([first, second], tmp_path / "legacy.csv")) == 2


@pytest.mark.parametrize("extra", [".venv/package.py", "build/helper.py", ".hidden.py"])
@pytest.mark.parametrize("folder", ["source", "shared_source"])
def test_archive_preserves_legacy_source_files_in_recorded_hash(
    setup_tracking,  # noqa: F811
    extra,
    folder,
):
    _, client, make_run, enable = setup_tracking
    run = make_run()
    saved = run / folder / extra
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_bytes(b"# Included by the original schema-3 source collector\n")
    result_path = run / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("resources")
    resources = run / "resources"
    if resources.exists():
        shutil.rmtree(resources)
    # Older completed runs hashed every non-cache Python file, including paths
    # that the current live experiment collector deliberately omits.
    result["training"]["source_hash" if folder == "source" else "shared_code_hash"] = source_hash(
        {
            path.relative_to(run / folder).as_posix(): path.read_bytes()
            for path in (run / folder).rglob("*.py")
            if "__pycache__" not in path.parts
        }
    )
    result_path.write_text(json.dumps(result))
    enable()
    assert tracking.archive_run(run) == "uploaded"
    remote_id = next(iter(client.runs))
    archived = client.root / remote_id / "reproduction" / folder / extra
    assert archived.read_bytes() == saved.read_bytes()


@pytest.mark.parametrize("resource", ["../bad.yaml", "/bad.yaml", "data/patients.json"])
def test_compare_rejects_unsafe_resource_metadata(tmp_path, resource):
    first = resource_run(tmp_path, "one")
    second = resource_run(tmp_path, "two")
    path = second / "result.json"
    result = json.loads(path.read_text())
    result["resources"] = {resource: {"sha256": hashlib.sha256(b"x").hexdigest(), "bytes": 1}}
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        compare_runs([first, second], tmp_path / "bad.csv")


@pytest.mark.parametrize("extra", [".venv/secret.py", "build/extra.py", ".hidden.py"])
@pytest.mark.parametrize("folder", ["source", "shared_source"])
def test_archive_rejects_source_files_excluded_from_snapshot_hash(tmp_path, extra, folder):
    run = resource_run(tmp_path, "member")
    added = run / folder / extra
    added.parent.mkdir(parents=True, exist_ok=True)
    added.write_text("# inserted after save\n")
    with pytest.raises(tracking.BundleError):
        tracking._full_manifest(run)
