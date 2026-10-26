"""Check Docker test checkout isolation without starting or requiring Docker."""

from docker_support import copy_workspace


def test_workspace_copy_excludes_existing_results_and_keeps_execution_files(tmp_path):
    source = tmp_path / "developer checkout"
    workspace = tmp_path / "isolated checkout"
    required = {
        "compose.yaml": "services: {}\n",
        "dev": "#!/bin/sh\n",
        "Global/Dockerfile": "FROM python:3.12\n",
        "Global/pyproject.toml": "[project]\n",
        "Global/uv.lock": "version = 1\n",
        "Global/src/sixletters/__main__.py": "# shared commands\n",
        "Global/scripts/start_workspace.py": "# workspace startup\n",
        "Global/notebooks/00_environment_check.ipynb": '{"cells": []}\n',
        "Members/README.md": "# Personal workspace\n",
    }
    local_files = {
        "Global/artifacts/member/exp001/existing-run/result.json": b'{"run_id":"existing-run"}',
        "Global/artifacts/member/exp001/existing-run/models/model.pkl": b"synthetic model bytes",
        "Global/artifacts/member/exp001/existing-run/predictions.csv": b"patient_id,score\n",
        "Global/artifacts/.remote/cached-run/result.json": b"{}",
        "Global/data/prepared/music/dataset.json": b"{}",
        "Global/reports/draft.md": b"local report",
        "Members/member/experiments/exp001/train.py": b"# personal experiment",
        ".env": b"MLFLOW_TRACKING_PASSWORD=synthetic-test-secret",
        ".poc-cache/cache.txt": b"local cache",
    }
    for filename, content in {**required, **local_files}.items():
        path = source / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)

    copy_workspace(source, workspace)

    for filename, content in required.items():
        assert (workspace / filename).read_text() == content
    for filename, content in local_files.items():
        assert not (workspace / filename).exists()
        assert (source / filename).read_bytes() == content
    assert not (workspace / "Global/artifacts").exists()
