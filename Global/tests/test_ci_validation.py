"""A selective workflow must never accept missing, failed, or malformed checks."""

import itertools
import json
import os
import subprocess
import sys

import pytest

from scripts import ci_validate


def workflow(*, tests=True, smoke=False, native=None, docker=None, tests_mode=None):
    native = [] if native is None else native
    docker = [] if docker is None else docker
    return {
        "checks": {
            "result": "success",
            "outputs": {
                "tests": json.dumps(tests),
                "tests_mode": json.dumps(tests_mode or ("full" if tests else "none")),
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


@pytest.mark.parametrize("job", ["checks", "compatibility", "docker"])
@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", None, "pending"])
def test_required_jobs_must_succeed(job, result):
    needs = workflow(native=ci_validate.NATIVE, docker=["tracking"])
    needs[job]["result"] = result
    with pytest.raises(ValueError):
        ci_validate.validate(needs)


@pytest.mark.parametrize("job", ["compatibility", "docker"])
@pytest.mark.parametrize("result", ["success", "failure", "cancelled", None])
def test_unselected_jobs_must_be_skipped(job, result):
    needs = workflow(tests=False)
    needs[job]["result"] = result
    with pytest.raises(ValueError, match=f"{job} must report skipped"):
        ci_validate.validate(needs)


@pytest.mark.parametrize("job", ["checks", "compatibility", "docker"])
def test_missing_job_or_result_is_not_a_success(job):
    needs = workflow()
    del needs[job]["result"]
    with pytest.raises(ValueError):
        ci_validate.validate(needs)
    del needs[job]
    with pytest.raises(ValueError):
        ci_validate.validate(needs)


@pytest.mark.parametrize("name", ["tests", "tests_mode", "smoke", "native", "docker"])
@pytest.mark.parametrize("value", [None, True, False, [], {}, "", "not JSON"])
def test_outputs_must_be_json_strings(name, value):
    needs = workflow()
    needs["checks"]["outputs"][name] = value
    with pytest.raises(ValueError, match=f"checks.outputs.{name}"):
        ci_validate.validate(needs)


@pytest.mark.parametrize("name", ["tests", "tests_mode", "smoke", "native", "docker"])
def test_every_scope_output_is_required_even_when_all_extra_jobs_are_skipped(name):
    needs = workflow(tests=False)
    del needs["checks"]["outputs"][name]
    with pytest.raises(ValueError, match=f"checks.outputs.{name}"):
        ci_validate.validate(needs)


@pytest.mark.parametrize("name", ["tests", "smoke"])
@pytest.mark.parametrize("value", [0, 1, "false", "true", None, [], {}])
def test_boolean_outputs_cannot_be_truthy_strings_or_numbers(name, value):
    needs = workflow()
    needs["checks"]["outputs"][name] = json.dumps(value)
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
    needs["checks"]["outputs"]["native"] = json.dumps(value)
    with pytest.raises(ValueError, match="full native matrix"):
        ci_validate.validate(needs)


@pytest.mark.parametrize(
    "value",
    ["tracking", None, False, {}, ["unknown"], ["tracking", "tracking"], [None], [[]]],
)
def test_docker_output_rejects_unknown_duplicate_and_nonstring_entries(value):
    needs = workflow()
    needs["checks"]["outputs"]["docker"] = json.dumps(value)
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
    with pytest.raises(ValueError, match="exactly checks"):
        ci_validate.validate(needs)


def test_new_dependency_cannot_be_silently_ignored():
    needs = workflow()
    needs["another_job"] = {"result": "failure"}
    with pytest.raises(ValueError, match="exactly checks"):
        ci_validate.validate(needs)


@pytest.mark.parametrize("job", ["checks", "compatibility", "docker"])
def test_job_shape_must_be_an_object(job):
    needs = workflow()
    needs[job] = "success"
    with pytest.raises(ValueError, match="result object"):
        ci_validate.validate(needs)


@pytest.mark.parametrize("outputs", [None, "", [], True])
def test_outputs_shape_is_required(outputs):
    needs = workflow()
    needs["checks"]["outputs"] = outputs
    with pytest.raises(ValueError, match="checks.outputs is missing or invalid"):
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
    needs["checks"]["outputs"]["tests_mode"] = json.dumps(value)
    with pytest.raises(ValueError, match="tests_mode"):
        ci_validate.validate(needs)


@pytest.mark.parametrize(
    "extra", [{"smoke": True}, {"docker": ["workspace"]}, {"native": ci_validate.NATIVE}]
)
def test_personal_mode_cannot_replace_full_tests_for_integrations(extra):
    with pytest.raises(ValueError, match="full regression"):
        ci_validate.validate(workflow(tests_mode="personal", **extra))
