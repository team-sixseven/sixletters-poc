"""A selective workflow must never accept missing, failed, or malformed checks."""

import itertools
import json
import os
import subprocess
import sys

import pytest

from scripts import ci_validate


def workflow(
    *,
    tests=True,
    smoke=False,
    native=None,
    docker=None,
    tests_mode=None,
    test_files=None,
    native_mode=None,
    native_tests=None,
):
    native = [] if native is None else native
    docker = [] if docker is None else docker
    return {
        "checks": {"result": "success"},
        "linux": {"result": "success" if tests else "skipped"},
        "scope": {
            "result": "success",
            "outputs": {
                "tests": json.dumps(tests),
                "tests_mode": json.dumps(tests_mode or ("full" if tests else "none")),
                "test_files": json.dumps(test_files or []),
                "native_mode": json.dumps(native_mode or ("full" if native else "none")),
                "native_tests": json.dumps(native_tests or []),
                "smoke": json.dumps(smoke),
                "native": json.dumps(native),
                "docker": json.dumps(docker),
            },
        },
        "compatibility": {"result": "success" if native else "skipped"},
        "docker": {"result": "success" if docker else "skipped"},
    }


@pytest.mark.parametrize(
    "needs",
    [
        workflow(tests=False),  # Documentation still requires the checks job.
        workflow(),  # Ordinary code runs Linux tests only.
        workflow(tests_mode="personal"),
        workflow(smoke=True),
        workflow(native=ci_validate.NATIVE),
        workflow(smoke=True, native=ci_validate.NATIVE, docker=sorted(ci_validate.DOCKER)),
    ],
)
def test_accepts_documentation_linux_and_full_coverage(needs):
    assert ci_validate.validate(needs) is None


@pytest.mark.parametrize(
    "docker",
    [
        list(subset)
        for size in range(4)
        for subset in itertools.combinations(["workspace", "recovery", "tracking"], size)
    ],
)
def test_accepts_each_docker_subset_only_when_its_job_has_the_matching_result(docker):
    needs = workflow(docker=docker)
    ci_validate.validate(needs)
    needs["docker"]["result"] = "skipped" if docker else "success"
    with pytest.raises(ValueError, match="docker must report"):
        ci_validate.validate(needs)


@pytest.mark.parametrize("job", ["scope", "checks", "linux", "compatibility", "docker"])
@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", None, "pending"])
def test_required_jobs_must_succeed(job, result):
    needs = workflow(native=ci_validate.NATIVE, docker=["tracking"])
    needs[job]["result"] = result
    with pytest.raises(ValueError):
        ci_validate.validate(needs)


@pytest.mark.parametrize("job", ["linux", "compatibility", "docker"])
@pytest.mark.parametrize("result", ["success", "failure", "cancelled", None])
def test_unselected_jobs_must_be_skipped(job, result):
    needs = workflow(tests=False)
    needs[job]["result"] = result
    with pytest.raises(ValueError, match=f"{job} must report skipped"):
        ci_validate.validate(needs)


@pytest.mark.parametrize("job", ["scope", "checks", "linux", "compatibility", "docker"])
def test_missing_job_or_result_is_not_a_success(job):
    needs = workflow()
    del needs[job]["result"]
    with pytest.raises(ValueError):
        ci_validate.validate(needs)
    del needs[job]
    with pytest.raises(ValueError):
        ci_validate.validate(needs)


@pytest.mark.parametrize(
    "name",
    [
        "tests",
        "tests_mode",
        "test_files",
        "native_mode",
        "native_tests",
        "smoke",
        "native",
        "docker",
    ],
)
@pytest.mark.parametrize("value", [None, True, False, [], {}, "", "not JSON"])
def test_outputs_must_be_json_strings(name, value):
    needs = workflow()
    needs["scope"]["outputs"][name] = value
    with pytest.raises(ValueError, match=f"scope.outputs.{name}"):
        ci_validate.validate(needs)


@pytest.mark.parametrize(
    "name",
    [
        "tests",
        "tests_mode",
        "test_files",
        "native_mode",
        "native_tests",
        "smoke",
        "native",
        "docker",
    ],
)
def test_every_scope_output_is_required_even_when_all_extra_jobs_are_skipped(name):
    needs = workflow(tests=False)
    del needs["scope"]["outputs"][name]
    with pytest.raises(ValueError, match=f"scope.outputs.{name}"):
        ci_validate.validate(needs)


@pytest.mark.parametrize("name", ["tests", "smoke"])
@pytest.mark.parametrize("value", [0, 1, "false", "true", None, [], {}])
def test_boolean_outputs_cannot_be_truthy_strings_or_numbers(name, value):
    needs = workflow()
    needs["scope"]["outputs"][name] = json.dumps(value)
    with pytest.raises(ValueError, match="JSON booleans"):
        ci_validate.validate(needs)


@pytest.mark.parametrize(
    "value",
    [
        "windows-2025",
        None,
        False,
        {},
        ["windows-2025"],
        ["macos-26"],
        ["windows-2025", "windows-2025"],
        ["windows-2025", "macos-26", "ubuntu-24.04"],
        [None],
        [[]],
    ],
)
def test_native_output_must_be_the_complete_known_matrix_or_empty(value):
    needs = workflow()
    needs["scope"]["outputs"]["native"] = json.dumps(value)
    with pytest.raises(ValueError, match="full native matrix"):
        ci_validate.validate(needs)


@pytest.mark.parametrize(
    "value",
    ["tracking", None, False, {}, ["unknown"], ["tracking", "tracking"], [None], [[]]],
)
def test_docker_output_rejects_unknown_duplicate_and_nonstring_entries(value):
    needs = workflow()
    needs["scope"]["outputs"]["docker"] = json.dumps(value)
    with pytest.raises(ValueError, match="unique known Docker checks"):
        ci_validate.validate(needs)


@pytest.mark.parametrize(
    "scope", [{"smoke": True}, {"native": ci_validate.NATIVE}, {"docker": ["workspace"]}]
)
def test_extended_checks_cannot_bypass_linux_tests(scope):
    with pytest.raises(ValueError, match="require Linux tests"):
        ci_validate.validate(workflow(tests=False, **scope))


@pytest.mark.parametrize("needs", [None, [], "success", {}, {"unexpected": {}}])
def test_needs_shape_is_required(needs):
    with pytest.raises(ValueError, match="exactly scope"):
        ci_validate.validate(needs)


def test_new_dependency_cannot_be_silently_ignored():
    needs = workflow()
    needs["another_job"] = {"result": "failure"}
    with pytest.raises(ValueError, match="exactly scope"):
        ci_validate.validate(needs)


@pytest.mark.parametrize("job", ["scope", "checks", "linux", "compatibility", "docker"])
def test_job_shape_must_be_an_object(job):
    needs = workflow()
    needs[job] = "success"
    with pytest.raises(ValueError, match="result object"):
        ci_validate.validate(needs)


@pytest.mark.parametrize("outputs", [None, "", [], True])
def test_outputs_shape_is_required(outputs):
    needs = workflow()
    needs["scope"]["outputs"] = outputs
    with pytest.raises(ValueError, match="scope.outputs is missing or invalid"):
        ci_validate.validate(needs)


@pytest.mark.parametrize("payload", [None, "not JSON", "null", "{}", "[]"])
def test_command_rejects_missing_or_invalid_payload_without_dependencies(payload):
    result = run_command(payload)
    assert result.returncode == 1
    assert "CI validation failed:" in result.stderr


def run_command(payload):
    env = os.environ.copy()
    env.pop("NEEDS_JSON", None)
    if payload is not None:
        env["NEEDS_JSON"] = payload
    return subprocess.run(
        [sys.executable, "-S", ci_validate.__file__],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_command_exits_zero_for_successful_scope():
    result = run_command(json.dumps(workflow()))
    assert result.returncode == 0
    assert "All required CI checks passed" in result.stdout
    assert result.stderr == ""


def test_command_rejects_failed_required_job_without_printing_untrusted_payload():
    needs = workflow(native=ci_validate.NATIVE)
    secret_marker = "private-value-not-for-logs"
    needs["compatibility"]["result"] = secret_marker
    result = run_command(json.dumps(needs))
    assert result.returncode == 1
    assert "compatibility must report success" in result.stderr
    assert secret_marker not in result.stdout + result.stderr


@pytest.mark.parametrize("value", [None, True, False, 1, [], {}, "unknown", "none"])
def test_test_mode_cannot_reduce_or_disagree_with_selected_tests(value):
    needs = workflow()
    needs["scope"]["outputs"]["tests_mode"] = json.dumps(value)
    with pytest.raises(ValueError, match="tests_mode"):
        ci_validate.validate(needs)


@pytest.mark.parametrize(
    "extra", [{"smoke": True}, {"docker": ["workspace"]}, {"native": ci_validate.NATIVE}]
)
def test_personal_mode_cannot_replace_full_tests_for_integrations(extra):
    with pytest.raises(ValueError, match="full regression"):
        ci_validate.validate(workflow(tests_mode="personal", **extra))


def test_focused_linux_and_native_require_explicit_test_files():
    needs = workflow(
        tests_mode="focused",
        test_files=["Global/tests/test_personal_env.py"],
        native=ci_validate.NATIVE,
        native_mode="focused",
        native_tests=["Global/tests/test_personal_env.py"],
    )
    ci_validate.validate(needs)
    needs["linux"]["result"] = "skipped"
    with pytest.raises(ValueError, match="linux must report success"):
        ci_validate.validate(needs)


@pytest.mark.parametrize(
    "field, mode", [("test_files", "tests_mode"), ("native_tests", "native_mode")]
)
@pytest.mark.parametrize(
    "files",
    [
        [],
        ["--ignore=Global/tests"],
        ["Global/tests/../secret.py"],
        ["Global/tests/test_x.py", "Global/tests/test_x.py"],
        None,
    ],
)
def test_focused_file_list_cannot_be_empty_or_arbitrary(field, mode, files):
    needs = workflow(native=ci_validate.NATIVE)
    needs["scope"]["outputs"][mode] = json.dumps("focused")
    needs["scope"]["outputs"][field] = json.dumps(files)
    with pytest.raises(ValueError, match=field):
        ci_validate.validate(needs)


def test_nonfocused_modes_cannot_smuggle_test_filters():
    needs = workflow()
    needs["scope"]["outputs"]["test_files"] = json.dumps(["Global/tests/test_data_contract.py"])
    with pytest.raises(ValueError, match="test_files"):
        ci_validate.validate(needs)


def test_no_platform_jobs_requires_none_mode():
    needs = workflow(native_mode="full")
    with pytest.raises(ValueError, match="native_mode must match"):
        ci_validate.validate(needs)
