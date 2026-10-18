"""A cached layer is optional, but the image under test must match the checkout."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import ci_docker_cache as cache

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "paths, expected",
    [([path], ["workspace", "tracking"]) for path in sorted(cache.SHARED_INPUTS)]
    + [
        (["Global/Dockerfile"], ["workspace"]),
        (["Global/Dockerfile.tracking"], ["tracking"]),
        (["Global/scripts/start_tracking_server.py"], ["tracking"]),
        (["Global/Dockerfile", "Global/Dockerfile.tracking"], ["workspace", "tracking"]),
        (["Global/src/sixletters/experiment.py", "Members/alice/train.py", "README.md"], []),
    ],
)
def test_shared_cache_selection_matches_only_reusable_build_inputs(paths, expected):
    assert cache.shared_cache_images(paths) == expected


@pytest.mark.parametrize(
    "ref, event_name, expected",
    [
        ("refs/heads/ci-benchmark/24-slim-base", "workflow_dispatch", ["workspace", "tracking"]),
        ("refs/heads/main", "workflow_dispatch", []),
        ("refs/heads/feature", "workflow_dispatch", []),
        ("refs/pull/24/merge", "pull_request", []),
        ("refs/heads/ci-benchmark/24-slim-base", "pull_request_target", []),
    ],
)
def test_shared_cache_updates_are_restricted_to_trusted_develop(ref, event_name, expected):
    assert (
        cache.warm_cache_scope(ROOT, {"GITHUB_REF": ref, "GITHUB_EVENT_NAME": event_name})
        == expected
    )


def test_shared_cache_push_uses_entire_pushed_diff_and_includes_removed_inputs(
    tmp_path, monkeypatch
):
    before, after = "1" * 40, "2" * 40
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"before": before, "after": after}), encoding="utf-8")
    calls = []

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(stdout="README.md\0Global/Dockerfile.tracking\0")

    monkeypatch.setattr(cache.subprocess, "run", run)
    assert cache.warm_cache_scope(
        tmp_path,
        {
            "GITHUB_REF": "refs/heads/ci-benchmark/24-slim-base",
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_EVENT_PATH": str(event),
        },
    ) == ["tracking"]
    assert calls[0][0] == ["git", "diff", "--name-only", "-z", "--no-renames", before, after, "--"]
    assert calls[0][1]["cwd"] == tmp_path
    assert calls[0][1]["check"] is True


@pytest.mark.parametrize(
    "before, after, expected",
    [("0" * 40, "1" * 40, ["workspace", "tracking"]), ("1" * 40, "0" * 40, [])],
)
def test_new_and_deleted_develop_pushes_do_not_require_a_missing_revision(
    tmp_path, monkeypatch, before, after, expected
):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"before": before, "after": after}), encoding="utf-8")
    monkeypatch.setattr(cache.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected diff"))
    assert (
        cache.warm_cache_scope(
            tmp_path,
            {
                "GITHUB_REF": "refs/heads/ci-benchmark/24-slim-base",
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_EVENT_PATH": str(event),
            },
        )
        == expected
    )


def test_cache_workflow_is_independent_and_cannot_write_from_pr_events():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/docker-cache.yml").read_text(encoding="utf-8")
    )
    events = workflow[True]  # PyYAML's YAML 1.1 loader treats the GitHub key `on` as true.
    assert set(events) == {"push", "workflow_dispatch"}
    assert events["push"]["branches"] == ["ci-benchmark/24-slim-base"]
    assert set(events["push"]["paths"]) == set.union(*cache.CACHE_INPUTS.values())
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["scope"]["if"] == "github.ref == 'refs/heads/ci-benchmark/24-slim-base'"
    assert workflow["jobs"]["warm"]["needs"] == "scope"
    assert workflow["jobs"]["warm"]["strategy"]["fail-fast"] is False

    steps = workflow["jobs"]["warm"]["steps"]
    builders = [step for step in steps if step.get("uses", "").startswith("docker/build-push")]
    assert len(builders) == 2
    assert builders[0]["with"]["target"] == "cpu-dependencies"
    for step, kind in zip(builders, ("workspace", "tracking"), strict=True):
        settings = step["with"]
        assert settings["context"] == "."
        assert settings["cache-from"] == f"type=gha,scope={kind}-shared-v1"
        assert settings["cache-to"] == f"type=gha,scope={kind}-shared-v1,mode=max,timeout=2m"
        assert "push" not in settings and "tags" not in settings and "load" not in settings

    main = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    pins = {
        step["uses"]
        for job in main["jobs"].values()
        for step in job.get("steps", [])
        if "uses" in step
    }
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "uses" in step:
                assert step["uses"] in pins
    assert "warm" not in main["jobs"]["final_gate"]["needs"]


@pytest.fixture
def build_context(tmp_path):
    for filename in (
        ".dockerignore",
        "Global/pyproject.toml",
        "Global/uv.lock",
        "Global/.python-version",
        "Global/Dockerfile",
        "Global/Dockerfile.tracking",
        "Global/src/sixletters/__init__.py",
        "Global/scripts/start_workspace.py",
        "Global/scripts/start_tracking_server.py",
    ):
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename, encoding="utf-8")
    return tmp_path


def test_fingerprint_tracks_image_inputs_but_not_local_data(build_context):
    workspace = cache.build_fingerprint(build_context, "workspace")
    tracking = cache.build_fingerprint(build_context, "tracking")
    (build_context / ".env").write_text("LOCAL_ONLY=1")
    data = build_context / "Global/data/private.csv"
    data.parent.mkdir()
    data.write_text("local data")
    assert cache.build_fingerprint(build_context, "workspace") == workspace
    assert cache.build_fingerprint(build_context, "tracking") == tracking
    (build_context / "Global/src/sixletters/__init__.py").write_text("changed source")
    assert cache.build_fingerprint(build_context, "workspace") != workspace
    assert cache.build_fingerprint(build_context, "tracking") == tracking
    (build_context / "Global/uv.lock").write_text("changed dependencies")
    assert cache.build_fingerprint(build_context, "tracking") != tracking


def test_default_local_execution_does_not_require_prebuilt_images(monkeypatch):
    monkeypatch.setattr(cache.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected Docker"))
    assert (
        cache.prepare_prebuilt_image(Path("unused"), "workspace", "test:cpu", environment={})
        is False
    )


@pytest.mark.parametrize("image", ["", "sixseven-ci:cpu", "sha256:short"])
def test_prebuilt_mode_rejects_missing_or_mutable_image_ids(build_context, image):
    with pytest.raises(ValueError, match="immutable ID"):
        cache.prepare_prebuilt_image(
            build_context,
            "workspace",
            "isolated-test:cpu",
            environment={"POC_DOCKER_PREBUILT": "1", "POC_DOCKER_WORKSPACE_IMAGE": image},
        )


@pytest.mark.parametrize("invalid", ["missing", "different-id", "stale-source", "no-label"])
def test_prebuilt_mode_fails_instead_of_silently_testing_wrong_image(
    build_context, monkeypatch, invalid
):
    image = "sha256:" + "a" * 64
    expected = cache.build_fingerprint(build_context, "workspace")
    calls = []

    def run(arguments, **kwargs):
        calls.append(arguments)
        if invalid == "missing":
            raise subprocess.CalledProcessError(1, arguments, stderr="No such image")
        return SimpleNamespace(
            stdout=json.dumps(
                [
                    {
                        "Id": "sha256:" + "b" * 64 if invalid == "different-id" else image,
                        "Config": {
                            "Labels": (
                                None
                                if invalid == "no-label"
                                else {
                                    cache.INPUT_LABEL: "stale"
                                    if invalid == "stale-source"
                                    else expected
                                }
                            )
                        },
                    }
                ]
            )
        )

    monkeypatch.setattr(cache.subprocess, "run", run)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        cache.prepare_prebuilt_image(
            build_context,
            "workspace",
            "isolated-test:cpu",
            environment={"POC_DOCKER_PREBUILT": "1", "POC_DOCKER_WORKSPACE_IMAGE": image},
        )
    assert len(calls) == 1  # No retagging or fallback build after a failed identity check.


@pytest.mark.parametrize("kind", ["workspace", "tracking"])
def test_verified_image_is_retagged_into_isolated_test_namespace(build_context, monkeypatch, kind):
    image = "sha256:" + "a" * 64
    calls = []

    def run(arguments, **kwargs):
        calls.append(arguments)
        return SimpleNamespace(
            stdout=json.dumps(
                [
                    {
                        "Id": image,
                        "Config": {
                            "Labels": {
                                cache.INPUT_LABEL: cache.build_fingerprint(build_context, kind)
                            }
                        },
                    }
                ]
            )
        )

    monkeypatch.setattr(cache.subprocess, "run", run)
    assert cache.prepare_prebuilt_image(
        build_context,
        kind,
        "isolated-test:image",
        environment={"POC_DOCKER_PREBUILT": "1", f"POC_DOCKER_{kind.upper()}_IMAGE": image},
    )
    assert calls == [
        ["docker", "image", "inspect", image],
        ["docker", "image", "tag", image, "isolated-test:image"],
    ]
