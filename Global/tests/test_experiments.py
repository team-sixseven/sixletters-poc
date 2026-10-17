import copy
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from sixletters import experiment as experiment_module
from sixletters.comparison import compare_runs
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment
from sixletters.provenance import OPERATIONAL_SOURCE_PATHS, python_sources, shared_code_scope
from sixletters.workspace import init_experiment


def completed_run(
    root,
    *,
    owner="팀원가",
    feature_set="clinical",
    model="IndependentMethod",
    seed=42,
    features=None,
    dataset=None,
    synthetic=True,
):
    dataset = dataset if dataset is not None else make_demo_dataset()
    data = save_prepared_dataset(dataset, root / "data", synthetic=synthetic)
    config = init_experiment(owner, "exp001", root=root / "experiments")
    selected = "" if features is None else f"features = {json.dumps(features)}\n"
    config.write_text(
        f'[experiment]\nowner = "{owner}"\nname = "exp001"\n'
        f'feature_set = "{feature_set}"\nmodel = "{model}"\nseed = {seed}\n'
        f"{selected}\n[parameters]\nlearning_rate = 0.1\n",
        encoding="utf-8",
    )
    # Recording and comparing results must not execute personal training code.
    (config.parent / "train.py").write_text(
        'raise RuntimeError("Do not import the training script")\n', encoding="utf-8"
    )
    experiment = Experiment(data, config, output_root=root / "artifacts")
    inputs = experiment.validation_data()
    predictions = pd.DataFrame({"patient_id": inputs.index, "score": 0.5})
    model_file = root / "weights.npz"
    np.savez(model_file, weights=np.array([1.0, 2.0]))
    return experiment.save(predictions, model_files={"weights.npz": model_file})


def read_result(run):
    return json.loads((run / "result.json").read_text(encoding="utf-8"))


@pytest.fixture
def shared_package(tmp_path, monkeypatch):
    package = tmp_path / "shared_package"
    shutil.copytree(experiment_module.PACKAGE, package)
    monkeypatch.setattr(experiment_module, "PACKAGE", package)
    return package


def change_shared_source(package, name):
    path = package / name
    path.parent.mkdir(parents=True, exist_ok=True)
    original = path.read_bytes() if path.is_file() else b""
    path.write_bytes(original + b"\n# updated behavior\n")


def record_scope_version(run, version):
    """Build an older valid fixture from its actual saved source snapshot."""
    result = read_result(run)
    result["shared_code_scope"] = shared_code_scope(
        python_sources(run / "shared_source"), version=version
    )
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")


@pytest.mark.parametrize("name", OPERATIONAL_SOURCE_PATHS)
def test_input_comparison_allows_operational_differences_without_source_downloads(
    tmp_path, shared_package, name
):
    first = completed_run(tmp_path / "first")
    change_shared_source(shared_package, name)
    second = completed_run(tmp_path / "second", feature_set="ecg")
    one, two = read_result(first), read_result(second)
    assert one["training"]["shared_code_hash"] != two["training"]["shared_code_hash"]
    assert one["shared_code_scope"] == two["shared_code_scope"]
    # The ordinary remote bundle contains result/predictions/config only.
    for run in (first, second):
        shutil.rmtree(run / "shared_source")
        shutil.rmtree(run / "source")
    assert len(compare_runs([first, second], tmp_path / "inputs.csv", mode="inputs")) == 2


@pytest.mark.parametrize(
    "name", ["features/clinical.py", "contracts.py", "new_training_helper.py", "custom/tracking.py"]
)
def test_input_comparison_rejects_scientific_or_unknown_source_differences(
    tmp_path, shared_package, name
):
    first = completed_run(tmp_path / "first")
    change_shared_source(shared_package, name)
    second = completed_run(tmp_path / "second", feature_set="ecg")
    with pytest.raises(ValueError, match="same training method"):
        compare_runs([first, second], tmp_path / "rejected.csv", mode="inputs")
    assert not (tmp_path / "rejected.csv").exists()


@pytest.mark.parametrize("changed", [None, "tracking.py", "personal_env.py", "contracts.py"])
def test_previous_scope_keeps_its_original_comparison_rules(tmp_path, shared_package, changed):
    first = completed_run(tmp_path / "first")
    if changed:
        change_shared_source(shared_package, changed)
    second = completed_run(tmp_path / "second", feature_set="ecg")
    for run in (first, second):
        record_scope_version(run, 1)
        shutil.rmtree(run / "shared_source")
        shutil.rmtree(run / "source")
    output = tmp_path / "inputs.csv"
    if changed in {"personal_env.py", "contracts.py"}:
        with pytest.raises(ValueError, match="same training method"):
            compare_runs([first, second], output, mode="inputs")
        assert not output.exists()
    else:
        assert len(compare_runs([first, second], output, mode="inputs")) == 2


@pytest.mark.parametrize("changed", [None, "tracking.py", "personal_env.py", "contracts.py"])
@pytest.mark.parametrize("previous_first", [False, True])
def test_mixed_scope_versions_require_identical_full_sources_without_snapshots(
    tmp_path, shared_package, changed, previous_first
):
    first = completed_run(tmp_path / "first")
    if changed:
        change_shared_source(shared_package, changed)
    second = completed_run(tmp_path / "second", feature_set="ecg")
    record_scope_version(first if previous_first else second, 1)
    for run in (first, second):
        shutil.rmtree(run / "shared_source")
        shutil.rmtree(run / "source")
    output = tmp_path / "inputs.csv"
    if changed:
        with pytest.raises(ValueError, match="비교 범위 버전이 서로 달라") as error:
            compare_runs([first, second], output, mode="inputs")
        assert "training.shared_code_hash" in str(error.value)
        assert "원격 기본 결과에는 소스 사본이 없어" in str(error.value)
        assert not output.exists()
    else:
        assert len(compare_runs([first, second], output, mode="inputs")) == 2


@pytest.mark.parametrize("legacy_count", [1, 2])
@pytest.mark.parametrize("different_full_hash", [False, True])
def test_legacy_and_mixed_input_comparisons_require_identical_full_shared_sources(
    tmp_path, shared_package, legacy_count, different_full_hash
):
    first = completed_run(tmp_path / "first")
    if different_full_hash:
        change_shared_source(shared_package, "tracking.py")
    second = completed_run(tmp_path / "second", feature_set="ecg")
    for run in (first, second)[:legacy_count]:
        result = read_result(run)
        result.pop("shared_code_scope")
        result.pop("created_at")
        (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if different_full_hash:
        with pytest.raises(ValueError, match="same training method"):
            compare_runs([first, second], tmp_path / "rejected.csv", mode="inputs")
        assert not (tmp_path / "rejected.csv").exists()
    else:
        assert len(compare_runs([first, second], tmp_path / "inputs.csv", mode="inputs")) == 2


@pytest.mark.parametrize("bad_scope", ["missing", "null", "version", "bool", "excluded", "hash"])
def test_malformed_or_unknown_scope_never_weakens_comparison(tmp_path, bad_scope):
    first = completed_run(tmp_path / "first")
    second = completed_run(tmp_path / "second", feature_set="ecg")
    result = read_result(second)
    if bad_scope == "missing":
        result["shared_code_scope"].pop("version")
    elif bad_scope == "null":
        result["shared_code_scope"] = None
    elif bad_scope == "version":
        result["shared_code_scope"]["version"] = 999
    elif bad_scope == "bool":
        result["shared_code_scope"]["version"] = True
    elif bad_scope == "excluded":
        result["shared_code_scope"]["excluded"].append("evaluation.py")
    else:
        result["shared_code_scope"]["hash"] = "unverified"
    (second / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="shared code scope"):
        compare_runs([first, second], tmp_path / "rejected.csv", mode="inputs")
    assert not (tmp_path / "rejected.csv").exists()


def test_previous_scope_cannot_claim_the_new_personal_environment_exclusion(tmp_path):
    first = completed_run(tmp_path / "first")
    second = completed_run(tmp_path / "second", feature_set="ecg")
    result = read_result(second)
    result["shared_code_scope"]["version"] = 1
    (second / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="shared code scope"):
        compare_runs([first, second], tmp_path / "inputs.csv", mode="inputs")


@pytest.mark.parametrize("difference", ["source", "parameters", "seed", "environment"])
def test_scoped_input_comparison_preserves_personal_method_and_environment_checks(
    tmp_path, difference
):
    first = completed_run(tmp_path / "first")
    second = completed_run(tmp_path / "second", feature_set="ecg")
    result = read_result(second)
    if difference == "source":
        result["training"]["source_hash"] = "1" * 64
    elif difference == "parameters":
        result["training"]["parameters"]["learning_rate"] = 0.2
    elif difference == "seed":
        result["training"]["seed"] = 17
    else:
        result["software"]["packages"]["numpy"] = "changed-version"
    (second / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="same training method"):
        compare_runs([first, second], tmp_path / "rejected.csv", mode="inputs")


def test_independent_methods_features_and_seeds_compare_without_loading_models(tmp_path):
    first = completed_run(tmp_path / "first")
    second = completed_run(
        tmp_path / "second",
        owner="팀원나",
        model="NeuralNetwork",
        seed=7,
        features=["clinical_1"],
    )
    # A comparison consumes prediction evidence, not model weights or source code.
    (second / "models/weights.npz").unlink()
    summary = compare_runs([first, second], tmp_path / "summary.csv")
    assert summary.owner.tolist() == ["팀원가", "팀원나"]
    assert summary.model.tolist() == ["IndependentMethod", "NeuralNetwork"]
    assert summary.seed.tolist() == [42, 7]
    assert summary.feature_count.tolist() == [2, 1]
    assert "estimator" not in summary.columns
    assert {"patient_id", "target", "score"}.isdisjoint(summary.columns)
    assert (tmp_path / "summary.csv").is_file()


@pytest.mark.parametrize("difference", ["value", "label", "split", "definition", "synthetic"])
def test_comparison_rejects_different_shared_data_conditions(tmp_path, difference):
    dataset = make_demo_dataset()
    first = completed_run(tmp_path / "first", dataset=dataset)
    changed = copy.deepcopy(dataset)
    if difference == "value":
        changed.frame.loc[0, "clinical_1"] = 12345.0
    elif difference == "label":
        changed.frame.loc[0, "target"] = 1 - changed.frame.loc[0, "target"]
    elif difference == "split":
        changed.frame.loc[0, "split"] = "validation"
    elif difference == "definition":
        changed.target_definition = "A different prediction endpoint"
    second = completed_run(
        tmp_path / "second", dataset=changed, synthetic=difference != "synthetic"
    )
    with pytest.raises(ValueError):
        compare_runs([first, second], tmp_path / "rejected.csv")
    assert not (tmp_path / "rejected.csv").exists()


@pytest.mark.parametrize("corruption", ["metric", "patient", "target", "evaluation", "schema"])
def test_comparison_rechecks_saved_prediction_evidence(tmp_path, corruption):
    first = completed_run(tmp_path / "first")
    second = completed_run(tmp_path / "second")
    result = read_result(second)
    if corruption == "metric":
        result["metrics"]["roc_auc"] = 0.12345
    elif corruption == "evaluation":
        result["comparison_context"]["evaluation"]["code_hash"] = "changed"
    elif corruption == "schema":
        result["schema_version"] = 2
    else:
        predictions = pd.read_csv(second / "predictions.csv", dtype={"patient_id": str})
        if corruption == "patient":
            predictions.loc[0, "patient_id"] = "OTHER"
        else:
            predictions.loc[0, "target"] = 1 - predictions.loc[0, "target"]
        predictions.to_csv(second / "predictions.csv", index=False)
    (second / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError):
        compare_runs([first, second], tmp_path / "rejected.csv")
    assert not (tmp_path / "rejected.csv").exists()


def test_input_comparison_preserves_one_method_and_complete_input_groups(tmp_path):
    runs = [
        completed_run(tmp_path / group, feature_set=group)
        for group in ("clinical", "ecg", "combined")
    ]
    assert len(compare_runs(runs, tmp_path / "inputs.csv", mode="inputs")) == 3
    with pytest.raises(ValueError):
        compare_runs(runs, tmp_path / "personal.csv")
    changed = completed_run(tmp_path / "different", model="AnotherMethod")
    with pytest.raises(ValueError):
        compare_runs([changed, runs[1]], tmp_path / "changed.csv", mode="inputs")
    subset = completed_run(tmp_path / "subset", features=["clinical_1"])
    with pytest.raises(ValueError):
        compare_runs([subset, runs[1]], tmp_path / "subset.csv", mode="inputs")


def test_comparison_rejects_duplicate_runs_and_never_overwrites(tmp_path):
    first = completed_run(tmp_path / "first")
    second = completed_run(tmp_path / "second")
    with pytest.raises(ValueError):
        compare_runs([first, first], tmp_path / "duplicate.csv")
    output = tmp_path / "summary.csv"
    compare_runs([first, second], output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        compare_runs([first, second], output)
    assert output.read_bytes() == original
