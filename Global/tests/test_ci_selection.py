"""Selective checks still fail closed at the required gate."""

import json

import pytest

from scripts.ci_scope import select_scope
from scripts.ci_validate import validate


@pytest.mark.parametrize(
    "paths,expected",
    [
        (["README.md"], []),
        (["Members/alice/code.py"], []),
        (["Global/src/sixletters/upload.py"], ["tracking"]),
        (["Global/src/sixletters/personal_env.py"], ["workspace"]),
        (["Global/uv.lock"], ["tracking", "workspace"]),
        (["unknown.file"], ["tracking", "workspace"]),
    ],
)
def test_selects_changed_behavior(paths, expected):
    assert select_scope(paths)["docker"] == expected


def test_full_check_covers_both_services_and_platforms():
    scope = select_scope([], full=True)
    assert scope["docker"] == ["tracking", "workspace"]
    assert len(scope["native"]) == 2


def needs(scope):
    result = {
        "scope": {"result": "success", "outputs": {k: json.dumps(v) for k, v in scope.items()}},
        "checks": {"result": "success"},
    }
    for job, key in [("linux", "tests"), ("compatibility", "native"), ("docker", "docker")]:
        result[job] = {"result": "success" if scope[key] else "skipped"}
    return result


@pytest.mark.parametrize("status", ["failure", "cancelled", "skipped", None])
def test_required_job_never_silently_skipped(status):
    value = needs(select_scope([], full=True))
    value["docker"]["result"] = status
    with pytest.raises(ValueError):
        validate(value)


def test_docs_allow_only_unselected_jobs_to_skip():
    validate(needs(select_scope(["README.md"])))
