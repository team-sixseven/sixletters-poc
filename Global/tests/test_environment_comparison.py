"""Package location does not change input comparisons; package contents still do."""

import json
import shutil

import pandas as pd
import pytest

from sixletters import experiment as experiment_module
from sixletters.comparison import compare_runs
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment
from sixletters.workspace import init_experiment


@pytest.fixture
def saved_runs(tmp_path, monkeypatch):
    package = tmp_path / "shared-package"
    shutil.copytree(experiment_module.PACKAGE, package)
    monkeypatch.setattr(experiment_module, "PACKAGE", package)
    data = save_prepared_dataset(make_demo_dataset(), tmp_path / "data", synthetic=True)
    model = tmp_path / "model.bin"
    model.write_bytes(b"constant model fixture")
    package_files = {
        "requirements.in": "personal-package==1.0.0\n",
        "requirements.lock": "personal-package==1.0.0\n",
        "environment.json": '{"version": "1.0.0"}\n',
    }

    def create(scopes=(None, "isolated"), change=None):
        runs = []
        for index, (scope, feature_set) in enumerate(zip(scopes, ("clinical", "ecg"), strict=True)):
            config = init_experiment(
                "Member", f"exp{index}", feature_set=feature_set, root=tmp_path / "Members"
            )
            active = config.parent if scope == "isolated" else config.parent.parent.parent
            for name, content in package_files.items():
                (active / name).write_text(content, encoding="utf-8")
            if scope is not None:
                config.with_name("environment.toml").write_text(
                    f'scope = "{scope}"\n', encoding="utf-8"
                )
            if index == 1 and change is not None:
                changed = active / change if change in package_files else config.parent / change
                changed.parent.mkdir(parents=True, exist_ok=True)
                content = package_files.get(change, "max_depth = 1\n")
                changed.write_text(content.replace("1", "2"), encoding="utf-8")
            experiment = Experiment(data, config, output_root=tmp_path / "artifacts")
            runs.append(
                experiment.save(
                    pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5}),
                    model_files={"model.bin": model},
                    training={"model": "Constant", "seed": 42, "parameters": {}},
                )
            )
        return runs

    return create


@pytest.mark.parametrize("scopes", [(None, "isolated"), ("shared", "isolated"), (None, "shared")])
def test_same_packages_compare_across_shared_and_isolated_locations(saved_runs, tmp_path, scopes):
    runs = saved_runs(scopes)
    original = [(run / "result.json").read_bytes() for run in runs]
    table = compare_runs(runs, tmp_path / "comparison.csv", mode="inputs")
    assert len(table) == 2
    assert [(run / "result.json").read_bytes() for run in runs] == original
    for run, scope in zip(runs, scopes, strict=True):
        resources = json.loads((run / "result.json").read_text())["resources"]
        assert ("environment.toml" in resources) == (scope is not None)


@pytest.mark.parametrize(
    "change", ["requirements.in", "requirements.lock", "environment.json", "configs/model.yaml"]
)
def test_package_records_and_learning_settings_still_must_match(saved_runs, tmp_path, change):
    runs = saved_runs(change=change)
    with pytest.raises(ValueError, match="same training method, settings and environment"):
        compare_runs(runs, tmp_path / "comparison.csv", mode="inputs")


def test_installed_software_still_must_match(saved_runs, tmp_path):
    runs = saved_runs()
    filename = runs[1] / "result.json"
    result = json.loads(filename.read_text())
    result["software"]["packages"]["personal-package"] = "2.0.0"
    filename.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="same training method, settings and environment"):
        compare_runs(runs, tmp_path / "comparison.csv", mode="inputs")


def test_selector_metadata_is_validated_before_identity_filtering(saved_runs, tmp_path):
    runs = saved_runs()
    filename = runs[1] / "result.json"
    result = json.loads(filename.read_text())
    result["resources"]["environment.toml"] = {"invalid": True}
    filename.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        compare_runs(runs, tmp_path / "comparison.csv", mode="inputs")
