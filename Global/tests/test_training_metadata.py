import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from sixletters.comparison import compare_runs
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.experiment import Experiment, _load_config, validate_training
from sixletters.tracking import _full_manifest, _local_manifest
from sixletters.workspace import init_experiment

TAG = "__sixletters_float__"


def metadata(parameters, *, seed=42):
    return {"model": "XGBClassifier", "seed": seed, "parameters": parameters}


def test_finite_json_parameters_are_unchanged_and_detached():
    original = metadata({"depth": 6, "enabled": True, "nested": [{"rate": 0.25}], "unset": None})
    saved = validate_training(original)
    assert saved == original
    saved["parameters"]["nested"][0]["rate"] = 0.9
    assert original["parameters"]["nested"][0]["rate"] == 0.25


def test_nonfinite_numbers_keep_type_and_numpy_scalars_are_normalized_recursively():
    original = metadata(
        {
            "missing": np.float32("nan"),
            "limits": (np.float64("inf"), float("-inf")),
            "literal": ["nan", "+inf", "-inf"],
            "nested": {"depth": np.int64(6), "rate": np.float32(0.25), "enabled": np.bool_(True)},
            "objective": np.str_("binary:logistic"),
        },
        seed=np.int32(42),
    )
    saved = validate_training(original)
    assert saved == metadata(
        {
            "missing": {TAG: "nan"},
            "limits": [{TAG: "+inf"}, {TAG: "-inf"}],
            "literal": ["nan", "+inf", "-inf"],
            "nested": {"depth": 6, "rate": 0.25, "enabled": True},
            "objective": "binary:logistic",
        }
    )
    assert math.isnan(original["parameters"]["missing"])
    assert isinstance(original["parameters"]["nested"]["depth"], np.integer)
    assert isinstance(original["parameters"]["limits"], tuple)
    # Canonical results can pass through local, upload and downloaded-run checks repeatedly.
    decoded = json.loads(json.dumps(saved, allow_nan=False))
    assert validate_training(decoded, stored=True) == saved
    assert validate_training(validate_training(decoded, stored=True), stored=True) == saved


@pytest.mark.parametrize("value", [object(), lambda: None, np.array([1]), {1}, 1j, np.complex64(1)])
def test_unsupported_values_show_exact_parameter_path(value):
    path = 'training.parameters["outer"][1]["value"]'
    with pytest.raises(ValueError, match=re.escape(path) + ": unsupported"):
        validate_training(metadata({"outer": [0, {"value": value}]}))


@pytest.mark.parametrize("container", ["list", "dict"])
def test_cycles_fail_with_parameter_path_without_mutating_input(container):
    value = [] if container == "list" else {}
    if container == "list":
        value.append(value)
        suffix = "[0]"
    else:
        value["cycle"] = value
        suffix = '["cycle"]'
    path = 'training.parameters["nested"]' + suffix
    with pytest.raises(ValueError, match=re.escape(path) + ": circular reference"):
        validate_training(metadata({"nested": value}))
    assert (value[0] if container == "list" else value["cycle"]) is value


def test_reused_objects_are_allowed_when_they_do_not_form_a_cycle():
    shared = {"depth": np.int32(4)}
    assert validate_training(metadata({"first": shared, "second": shared}))["parameters"] == {
        "first": {"depth": 4},
        "second": {"depth": 4},
    }


@pytest.mark.parametrize("value", [{TAG: "nan"}, {TAG: "not-a-float"}, {TAG: "nan", "other": 1}])
def test_runtime_user_dictionaries_cannot_impersonate_reserved_float_metadata(value):
    with pytest.raises(ValueError, match=r'training.parameters\["missing"\].*reserved'):
        validate_training(metadata({"missing": value}))


@pytest.mark.parametrize("value", [{TAG: "NaN"}, {TAG: None}, {TAG: []}, {TAG: "nan", "other": 1}])
def test_malformed_saved_float_tags_are_rejected(value):
    with pytest.raises(ValueError, match=r'training.parameters\["missing"\].*reserved'):
        validate_training(metadata({"missing": value}), stored=True)


def test_entire_parameter_dictionary_cannot_be_replaced_by_a_float_tag():
    with pytest.raises(ValueError, match="training.parameters: .*reserved"):
        validate_training(metadata({TAG: "nan"}), stored=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_saved_metadata_rejects_nonstandard_json_numbers(value):
    with pytest.raises(ValueError, match=r'training.parameters\["missing"\].*finite JSON'):
        validate_training(metadata({"missing": value}), stored=True)


def test_nonstring_dictionary_keys_cannot_collide_during_json_conversion():
    with pytest.raises(ValueError, match=r'training.parameters\["nested"\].*keys must be strings'):
        validate_training(metadata({"nested": {1: "numeric", "1": "string"}}))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_config_toml_remains_finite_only(tmp_path, value):
    config = init_experiment("Alice", "example", root=tmp_path / "Members")
    config.write_text(
        '[experiment]\nowner = "Alice"\nname = "example"\nfeature_set = "clinical"\n'
        f"[parameters]\nmissing = {value}\n"
    )
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        _load_config(config)


def test_xgboost_like_params_save_verify_and_compare_without_mutation(tmp_path):
    data = save_prepared_dataset(make_demo_dataset(), tmp_path / "data", synthetic=True)
    model = tmp_path / "export.json"
    model.write_text('{"model": "opaque example"}')
    parameters = {
        "missing": np.nan,
        "max_depth": np.int32(6),
        "n_jobs": None,
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "learning_rate": np.float64(0.1),
    }
    runs = []
    for owner in ("Alice", "Bob"):
        config = init_experiment(owner, "example", root=tmp_path / "Members")
        experiment = Experiment(data, config, output_root=tmp_path / "artifacts")
        scores = pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5})
        run = experiment.save(
            scores, model_files={"export.json": model}, training=metadata(parameters)
        )
        # Strict JSON loads the tagged value without accepting the nonstandard NaN token.
        saved = json.loads(
            (run / "result.json").read_text(),
            parse_constant=lambda value: pytest.fail(f"Nonstandard JSON token: {value}"),
        )
        assert saved["training"]["parameters"]["missing"] == {TAG: "nan"}
        assert _local_manifest(run)[0] == saved
        assert _full_manifest(run)[0] == saved
        runs.append(run)
    frame = compare_runs(runs, tmp_path / "comparison.csv")
    assert len(frame) == 2
    assert json.loads(frame.iloc[0]["parameters"])["missing"] == {TAG: "nan"}
    assert math.isnan(parameters["missing"])
    assert isinstance(parameters["max_depth"], np.integer)


def test_normalized_settings_do_not_allow_nonfinite_prediction_scores(tmp_path):
    data = save_prepared_dataset(make_demo_dataset(), tmp_path / "data", synthetic=True)
    config = init_experiment("Alice", "example", root=tmp_path / "Members")
    model = tmp_path / "weights.bin"
    model.write_bytes(b"weights")
    experiment = Experiment(data, config, output_root=tmp_path / "artifacts")
    scores = pd.DataFrame({"patient_id": experiment.validation_data().index, "score": np.nan})
    with pytest.raises(ValueError, match="finite"):
        experiment.save(
            scores, model_files={"weights.bin": model}, training=metadata({"missing": np.nan})
        )
    assert not list((tmp_path / "artifacts").rglob("result.json"))
