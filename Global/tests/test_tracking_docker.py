"""Opt-in real authentication, artifact proxy and persistence checks."""

import base64
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
import pytest
from docker_timing import command_stage, stage

from scripts.ci_docker_cache import prepare_prebuilt_image
from sixletters.comparison import compare_runs
from sixletters.data.demo import make_demo_dataset
from sixletters.data.prepared import save_prepared_dataset
from sixletters.workspace import init_experiment

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1",
    reason="Set RUN_DOCKER_TESTS=1 on a Docker host to check the real tracking server",
)


@pytest.fixture(scope="module")
def tracking_server(tmp_path_factory):
    source = Path(__file__).resolve().parents[2]
    workspace = tmp_path_factory.mktemp("tracking-docker")
    (workspace / "Global/scripts").mkdir(parents=True)
    for filename in (
        "Global/Dockerfile.tracking",
        "compose.tracking.yaml",
        "Global/pyproject.toml",
        "Global/uv.lock",
        "Global/.python-version",
        ".dockerignore",
        "Global/scripts/start_tracking_server.py",
    ):
        shutil.copy2(source / filename, workspace / filename)
    namespace = f"sixletters-tracking-test-{uuid.uuid4().hex[:12]}"
    compose_path = workspace / "compose.tracking.yaml"
    compose_path.write_text(
        compose_path.read_text().replace("sixseven-poc:tracking", f"{namespace}:tracking")
    )
    prebuilt_start = prepare_prebuilt_image(source, "tracking", f"{namespace}:tracking")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    admin = "test-admin"
    password = secrets.token_urlsafe(24)
    environment = {
        **os.environ,
        "COMPOSE_PROJECT_NAME": namespace,
        "POC_MLFLOW_PORT": str(port),
        "POC_MLFLOW_BIND_HOST": "127.0.0.1",
        "MLFLOW_AUTH_ADMIN_USERNAME": admin,
        "MLFLOW_AUTH_ADMIN_PASSWORD": password,
        "MLFLOW_FLASK_SERVER_SECRET_KEY": secrets.token_urlsafe(48),
        "MLFLOW_SERVER_ALLOWED_HOSTS": "localhost:*,127.0.0.1:*,tracking:5000",
    }
    for name in ("COMPOSE_PROFILES", "COMPOSE_FILE", "MLFLOW_TRACKING_TOKEN"):
        environment.pop(name, None)
    compose = ["docker", "compose", "-f", "compose.tracking.yaml"]

    def run(*arguments, timeout=180):
        nonlocal prebuilt_start
        if prebuilt_start and arguments[:1] == ("up",):
            arguments = tuple("--no-build" if item == "--build" else item for item in arguments)
            prebuilt_start = False
        with command_stage("tracking", arguments):
            completed = subprocess.run(
                [*compose, *arguments],
                cwd=workspace,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
            )
            assert completed.returncode == 0, (
                f"Compose {arguments!r} failed\n{completed.stdout}\n{completed.stderr}"
            )
        return completed.stdout

    uri = f"http://127.0.0.1:{port}"

    def request(path, data=None, *, credentials=(admin, password), method=None, raw=None):
        headers = {}
        if credentials:
            token = base64.b64encode(":".join(credentials).encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        if data is not None:
            raw = json.dumps(data).encode()
            headers["Content-Type"] = "application/json"
        with urlopen(
            Request(uri + path, data=raw, headers=headers, method=method), timeout=30
        ) as response:
            body = response.read()
            return (
                json.loads(body)
                if body and response.headers.get_content_type() == "application/json"
                else body
            )

    try:
        run("up", "--build", "-d", "--wait", timeout=900)
        yield {
            "workspace": workspace,
            "run": run,
            "uri": uri,
            "request": request,
            "admin": admin,
            "password": password,
            "environment": environment,
        }
    finally:
        # Only this isolated test project's volume is deleted.
        subprocess.run(
            [*compose, "down", "--volumes", "--remove-orphans"],
            cwd=workspace,
            env=environment,
            capture_output=True,
            timeout=90,
            check=False,
        )
        subprocess.run(
            ["docker", "image", "rm", f"{namespace}:tracking"],
            capture_output=True,
            timeout=60,
            check=False,
        )


def test_tracking_requires_auth_shares_experiments_and_persists_artifacts(tracking_server):
    server = tracking_server
    request = server["request"]
    with pytest.raises(HTTPError) as failure:
        request("/api/2.0/mlflow/experiments/search", {"max_results": 10}, credentials=None)
    assert failure.value.code == 401
    user = ("test-member", secrets.token_urlsafe(20))
    request("/api/2.0/mlflow/users/create", {"username": user[0], "password": user[1]})
    experiment = request("/api/2.0/mlflow/experiments/create", {"name": "shared-results"})
    experiment_id = experiment["experiment_id"]
    run = request("/api/2.0/mlflow/runs/create", {"experiment_id": experiment_id}, credentials=user)
    run_id = run["run"]["info"]["run_id"]
    artifact_uri = run["run"]["info"]["artifact_uri"]
    assert artifact_uri.startswith("mlflow-artifacts:/")
    artifact_path = artifact_uri.removeprefix("mlflow-artifacts:") + "/proof.txt"
    request(
        "/api/2.0/mlflow-artifacts/artifacts" + artifact_path,
        method="PUT",
        raw=b"persistent result artifact",
        credentials=user,
    )
    with pytest.raises(HTTPError) as failure:
        request(
            "/api/2.0/mlflow/users/create",
            {"username": "blocked", "password": secrets.token_urlsafe(20)},
            credentials=user,
        )
    assert failure.value.code == 403
    with pytest.raises(HTTPError) as failure:
        request(
            "/api/2.0/mlflow/experiments/delete", {"experiment_id": experiment_id}, credentials=user
        )
    assert failure.value.code == 403

    server["run"]("down")
    server["environment"]["MLFLOW_AUTH_ADMIN_PASSWORD"] = ""
    server["run"]("up", "-d", "--wait")
    saved = request(f"/api/2.0/mlflow/runs/get?run_id={run_id}", credentials=user)
    assert saved["run"]["info"]["run_id"] == run_id
    assert (
        request("/api/2.0/mlflow-artifacts/artifacts" + artifact_path, credentials=user)
        == b"persistent result artifact"
    )
    container_id = server["run"]("ps", "-q", "tracking").strip()
    assert container_id
    configs = json.loads(server["run"]("config", "--format", "json"))
    volumes = configs["services"]["tracking"]["volumes"]
    assert len(volumes) == 1
    assert volumes[0]["type"] == "volume"
    assert volumes[0]["target"] == "/var/lib/mlflow"


def test_deleted_result_requires_restore_without_creating_another_run(
    tracking_server, tmp_path, monkeypatch
):
    from mlflow.entities import ViewType
    from mlflow.tracking import MlflowClient

    from sixletters import tracking
    from sixletters.experiment import Experiment

    server = tracking_server
    monkeypatch.setenv("MLFLOW_TRACKING_URI", server["uri"])
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", server["admin"])
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", server["password"])
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "deleted-result-recovery")
    data = save_prepared_dataset(make_demo_dataset(), tmp_path / "data", synthetic=True)
    config = init_experiment("member", "exp001", root=tmp_path / "Members")
    experiment = Experiment(data, config, output_root=tmp_path / "artifacts")
    model = tmp_path / "weights.txt"
    model.write_text("synthetic model")
    run = experiment.save(
        pd.DataFrame({"patient_id": experiment.validation_data().index, "score": 0.5}),
        model_files={"weights.txt": model},
        training={"model": "ConstantScore", "seed": None, "parameters": {"score": 0.5}},
    )
    root = tmp_path / "artifacts"
    assert tracking.sync_runs(root)["uploaded"] == 1
    client = MlflowClient(tracking_uri=server["uri"])
    remote_id = tracking.list_remote_runs()[0]["run_id"]
    experiment_id = client.get_run(remote_id).info.experiment_id
    client.delete_run(remote_id)
    # A fresh checkout must discover deletion even without its local upload receipt.
    next((root / ".tracking").glob(f"*/{run.name}.json")).unlink()
    summary = tracking.sync_runs(root)
    assert summary["already_synced"] == summary["uploaded"] == 0
    assert summary["pending"] == 1
    assert summary["issues"][0]["code"] == "remote_deleted"
    assert tracking.list_remote_runs() == []
    assert len(client.search_runs([experiment_id], run_view_type=ViewType.ALL)) == 1
    assert client.get_run(remote_id).info.lifecycle_stage == "deleted"
    client.restore_run(remote_id)
    recovered = tracking.sync_runs(root, retry_only=True)
    assert recovered["already_synced"] == 1
    assert recovered["pending"] == 0
    assert tracking.list_remote_runs()[0]["run_id"] == remote_id


def test_six_independent_training_processes_sync_and_compare(
    tracking_server, tmp_path, monkeypatch
):
    from mlflow.tracking import MlflowClient

    from sixletters.reproduction import fetch_archive, restore_run
    from sixletters.tracking import BundleError, fetch_runs, list_remote_runs, sync_runs

    server = tracking_server
    experiment_name = "six-independent-training-processes"
    owners = [f"팀원{index + 1}" for index in range(6)]
    workspaces = []
    credentials = []
    for index, owner in enumerate(owners):
        credential = (f"training-member-{index}", secrets.token_urlsafe(24))
        server["request"](
            "/api/2.0/mlflow/users/create",
            {"username": credential[0], "password": credential[1]},
        )
        credentials.append(credential)
        checkout = tmp_path / owner
        checkout.mkdir()
        (checkout / "Global").mkdir()
        (checkout / "Global/pyproject.toml").write_text('[project]\nname="sixletters-poc"\n')
        save_prepared_dataset(make_demo_dataset(), checkout / "Global/data", synthetic=True)
        config = init_experiment(owner, "exp001", root=checkout / "Members")
        # Each contributor changes their own model settings; the common split stays fixed.
        config.write_text(config.read_text().replace("C = 1.0", f"C = {(index + 1) / 10}"))
        if index == 0:
            config.with_name("전처리 설정.yaml").write_text(
                "imputation: median\n", encoding="utf-8"
            )
        workspaces.append(checkout)

    def train(index):
        checkout = workspaces[index]
        username, password = credentials[index]
        environment = {
            **os.environ,
            "MLFLOW_TRACKING_URI": server["uri"],
            "MLFLOW_TRACKING_USERNAME": username,
            "MLFLOW_TRACKING_PASSWORD": password,
            "MLFLOW_EXPERIMENT_NAME": experiment_name,
            "MLFLOW_HTTP_REQUEST_TIMEOUT": "10",
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0",
            "PYTHONIOENCODING": "utf-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        environment.pop("MLFLOW_TRACKING_TOKEN", None)
        result = subprocess.run(
            [
                sys.executable,
                str(checkout / "Members" / owners[index] / "experiments/exp001/train.py"),
                "--data",
                str(checkout / "Global/data"),
            ],
            cwd=checkout,
            env=environment,
            text=True,
            capture_output=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        assert "upload pending" not in result.stderr, result.stderr
        assert password not in result.stdout + result.stderr
        markers = list((checkout / "Global/artifacts").glob("*/*/*/result.json"))
        assert len(markers) == 1
        assert not (checkout / "Global/artifacts" / ".tracking").exists()
        # Direct Python processes have no workspace worker. Once training exits,
        # a separate sync command publishes the persisted completion marker.
        synced = subprocess.run(
            [
                sys.executable,
                "-m",
                "sixletters",
                "sync",
                "--root",
                str(checkout / "Global/artifacts"),
            ],
            cwd=checkout,
            env=environment,
            text=True,
            capture_output=True,
            timeout=180,
        )
        assert synced.returncode == 0, synced.stderr
        assert password not in synced.stdout + synced.stderr
        return markers[0].parent

    with stage("tracking.six_member_train_and_sync"), ThreadPoolExecutor(max_workers=6) as pool:
        local_runs = list(pool.map(train, range(6)))
    for key, value in {
        "MLFLOW_TRACKING_URI": server["uri"],
        "MLFLOW_TRACKING_USERNAME": credentials[0][0],
        "MLFLOW_TRACKING_PASSWORD": credentials[0][1],
        "MLFLOW_EXPERIMENT_NAME": experiment_name,
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    client = MlflowClient(tracking_uri=server["uri"])
    experiment = client.get_experiment_by_name(experiment_name)
    remote_runs = client.search_runs([experiment.experiment_id])
    assert len(remote_runs) == 6
    assert {run.data.tags["sixletters.owner"] for run in remote_runs} == set(owners)
    assert all(run.info.status == "FINISHED" for run in remote_runs)
    assert len({run.data.params["training.C"] for run in remote_runs}) == 6
    for run in remote_runs:
        assert {file.path for file in client.list_artifacts(run.info.run_id, "bundle")} == {
            "bundle/result.json",
            "bundle/predictions.csv",
            "bundle/config.toml",
            "bundle/manifest.json",
        }
        assert not client.list_artifacts(run.info.run_id, "reproduction")

    # Repeated synchronization must not duplicate any contributor's completed run.
    for workspace in workspaces:
        summary = sync_runs(workspace / "Global/artifacts")
        assert summary["already_synced"] == 1
        assert summary["uploaded"] == summary["invalid"] == summary["pending"] == 0
    assert len(client.search_runs([experiment.experiment_id])) == 6
    with stage("tracking.download_results"):
        remote_dirs = fetch_runs(tmp_path / "reviewer" / "artifacts")
    assert len(remote_dirs) == 6
    comparison_path = tmp_path / "six-members.csv"
    with stage("tracking.compare_results"):
        compared = subprocess.run(
            [
                sys.executable,
                "-m",
                "sixletters",
                "compare",
                "--remote",
                "--root",
                str(tmp_path / "reviewer" / "artifacts"),
                "--output",
                str(comparison_path),
            ],
            cwd=workspaces[0],
            input="1 2 3 4 5 6\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        assert compared.returncode == 0, compared.stderr
    assert "95%" in compared.stdout
    comparison = pd.read_csv(comparison_path)
    assert set(comparison.owner) == set(owners)
    local = compare_runs(local_runs, tmp_path / "local-reference.csv")
    assert (
        comparison.set_index("owner").roc_auc.sort_index().tolist()
        == local.set_index("owner").roc_auc.sort_index().tolist()
    )
    uncertainty_columns = [
        "patient_count",
        "positive_count",
        "negative_count",
        "roc_auc_ci_lower",
        "roc_auc_ci_upper",
        "average_precision_ci_lower",
        "average_precision_ci_upper",
        "brier_score_ci_lower",
        "brier_score_ci_upper",
    ]
    pd.testing.assert_frame_equal(
        comparison.set_index("owner")[uncertainty_columns].sort_index(),
        local.set_index("owner")[uncertainty_columns].sort_index(),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert comparison.scd_count.isna().all()  # Synthetic target is not clinical SCD.
    for item in list_remote_runs():
        row = comparison.set_index("owner").loc[item["owner"]]
        interval = item["validation_summary"]
        assert interval["patients"] == row.patient_count
        assert interval["positive"] == row.positive_count
        assert interval["intervals"]["roc_auc"]["lower"] == pytest.approx(row.roc_auc_ci_lower)
    assert not list((tmp_path / "reviewer").rglob("model.pkl"))

    # A selected run is expanded in place; cached comparison inputs remain valid.
    selected = next(run for run in remote_runs if run.data.tags["sixletters.owner"] == owners[0])
    with stage("tracking.archive_model"):
        archived = subprocess.run(
            [sys.executable, "-m", "sixletters", "archive", "--run", str(local_runs[0])],
            cwd=workspaces[0],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert archived.returncode == 0, archived.stderr
    assert len(client.search_runs([experiment.experiment_id])) == 6
    assert client.get_run(selected.info.run_id).data.tags["sixletters.archive_status"] == "complete"
    assert len(client.get_metric_history(selected.info.run_id, "roc_auc")) == 1
    # A different authenticated teammate downloads the complete archive through
    # the real artifact proxy, then restores it into their own experiment folder.
    recipient = workspaces[-1]
    recipient_name = owners[-1]
    username, password = credentials[-1]
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", username)
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", password)
    with stage("tracking.download_archive"):
        downloaded = fetch_archive(recipient / "Global/artifacts/.restored", selected.info.run_id)
    original = json.loads((local_runs[0] / "result.json").read_text())
    assert "전처리 설정.yaml" in original["resources"]
    assert downloaded != local_runs[0]
    assert downloaded.name == original["run_id"]
    for local in local_runs[0].rglob("*"):
        if local.is_file():
            assert (
                downloaded / local.relative_to(local_runs[0])
            ).read_bytes() == local.read_bytes()
    assert (
        fetch_archive(recipient / "Global/artifacts/.restored", selected.info.run_id) == downloaded
    )
    restored_name = "restored-exp001"
    with stage("tracking.restore_model"):
        restored_config = restore_run(
            downloaded,
            owner=recipient_name,
            name=restored_name,
            root=recipient / "Members",
            data=recipient / "Global/data",
        )
    assert (
        restored_config.parent
        == recipient / "Members" / recipient_name / "experiments" / restored_name
    )
    assert (restored_config.parent / "outputs/.restore/models/model.pkl").read_bytes() == (
        local_runs[0] / "models/model.pkl"
    ).read_bytes()
    # Download and restore must not run either saved code or the pickled model.
    assert not (recipient / "Global/artifacts" / recipient_name / restored_name).exists()
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(
                None,
                (str(Path(__file__).resolve().parents[1] / "src"), os.environ.get("PYTHONPATH")),
            )
        ),
        "PYTHONIOENCODING": "utf-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    environment.pop("SIXLETTERS_EXECUTION_JSON", None)
    with stage("tracking.retrain_restored_model"):
        retrained = subprocess.run(
            [
                sys.executable,
                str(restored_config.with_name("train.py")),
                "--config",
                str(restored_config),
                "--data",
                str(recipient / "Global/data"),
            ],
            cwd=recipient,
            env=environment,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=180,
        )
        assert retrained.returncode == 0, retrained.stderr
    assert password not in retrained.stdout + retrained.stderr
    markers = list(
        (recipient / "Global/artifacts" / recipient_name / restored_name).glob("*/result.json")
    )
    assert len(markers) == 1
    reproduced = json.loads(markers[0].read_text())
    assert reproduced["owner"] == recipient_name
    assert reproduced["experiment"] == restored_name
    assert reproduced["run_id"] != original["run_id"]
    assert markers[0].parent.name == reproduced["run_id"]
    assert reproduced["comparison_context"] == original["comparison_context"]
    assert reproduced["training"]["parameters"] == original["training"]["parameters"]
    expected_predictions = (
        pd.read_csv(
            local_runs[0] / "predictions.csv",
            dtype={"patient_id": str},
            float_precision="round_trip",
        )
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    reproduced_predictions = (
        pd.read_csv(
            markers[0].with_name("predictions.csv"),
            dtype={"patient_id": str},
            float_precision="round_trip",
        )
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(expected_predictions, reproduced_predictions, check_exact=True)
    # Retraining is a new local run; no implicit server upload or duplicate remote run.
    assert len(client.search_runs([experiment.experiment_id])) == 6
    for remote in remote_runs:
        if remote.info.run_id != selected.info.run_id:
            assert not client.list_artifacts(remote.info.run_id, "reproduction")
    assert sync_runs(workspaces[0] / "Global/artifacts")["already_synced"] == 1
    assert fetch_runs(tmp_path / "reviewer" / "artifacts") == remote_dirs

    # A duplicate identity must not make the other five contributors unavailable.
    duplicate = client.create_run(experiment.experiment_id, tags=selected.data.tags)
    client.set_terminated(duplicate.info.run_id, status="FINISHED")
    issues = []
    healthy = list_remote_runs(issues=issues)
    assert len(healthy) == 5
    assert issues[0]["code"] == "duplicate_remote"
    assert set(issues[0]["remote_run_ids"]) == {selected.info.run_id, duplicate.info.run_id}
    assert owners[0] not in {run["owner"] for run in healthy}
    # Refresh remote eligibility while reusing the six already verified bundles.
    # A cached duplicate must still be excluded before local comparison.
    paths = fetch_runs(tmp_path / "reviewer" / "artifacts", issues=[])
    assert set(paths) < set(remote_dirs)
    assert {path.name for path in paths} == {
        remote.info.run_id for remote in remote_runs if remote.info.run_id != selected.info.run_id
    }
    assert len(compare_runs(paths, tmp_path / "healthy-comparison.csv")) == 5
    with pytest.raises(BundleError, match="duplicate"):
        fetch_runs(tmp_path / "reviewer" / "artifacts", run_ids=[duplicate.info.run_id], issues=[])
    assert len(client.search_runs([experiment.experiment_id])) == 7
