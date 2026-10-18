"""A cached layer is optional, but the image under test must match the checkout."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import ci_docker_cache as cache


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
