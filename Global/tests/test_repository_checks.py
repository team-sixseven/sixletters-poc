import json
import subprocess
from pathlib import Path

import pytest

from scripts.check_repository import MIB, main, violations


def test_rejects_forced_add_of_data_model_and_secrets():
    for filename in [
        "Global/data/raw/patients.csv",
        "Global/artifacts/run/split.json",
        ".env",
        "model.onnx",
        ".poc.local.toml",
    ]:
        assert violations(Path(filename), b"test")


def test_allows_documentation_and_aggregate_results():
    for filename in [
        "Global/data/README.md",
        "Global/artifacts/README.md",
        "Global/reports/run/metrics.json",
    ]:
        assert violations(Path(filename), b"{}") == []


@pytest.mark.parametrize(
    "header,delimiter,encoding",
    [
        ("Patient ID", ";", "utf-8"),
        ("patient_id", ",", "utf-8-sig"),
        (" PATIENT-ID ", "\t", "utf-8"),
        ("Subject ID", ",", "utf-16"),
        ("participant_id", ";", "utf-32"),
        ("Ｐａｔｉｅｎｔ　ＩＤ", ",", "utf-8"),
        ("환자 ID", ",", "utf-8"),
    ],
)
def test_rejects_patient_csv_headers_in_member_and_report_folders(header, delimiter, encoding):
    content = f'"{header}"{delimiter}target{delimiter}score\nsynthetic{delimiter}0{delimiter}0.2\n'
    for name in ("Members/member/notebooks/renamed.CSV", "Global/reports/summary.csv"):
        problems = violations(Path(name), content.encode(encoding))
        assert len(problems) == 1
        assert "patient identifier columns" in problems[0]


def test_rejects_patient_csv_with_spreadsheet_preamble_and_tsv_extension():
    assert violations(
        Path("Global/reports/patients.csv"), b"\xef\xbb\xbf\nsep=;\nAge;Patient ID\n60;synthetic\n"
    )
    assert violations(Path("Members/member/predictions.tsv"), b"target\tpatient_id\n0\tsynthetic\n")


def test_aggregate_csv_and_identifier_mentions_in_values_remain_trackable():
    assert (
        violations(
            Path("Global/reports/metrics.csv"),
            b"model,patient_count,positive_count,auroc\nlogistic,144,14,0.65\n",
        )
        == []
    )
    assert (
        violations(Path("Global/reports/column_notes.csv"), b"column,usage\npatient_id,excluded\n")
        == []
    )


def test_repository_check_rejects_small_patient_csv_added_outside_data_folders(
    tmp_path, monkeypatch, capsys
):
    repository_ignore = Path(__file__).resolve().parents[2] / ".gitignore"
    (tmp_path / ".gitignore").write_bytes(repository_ignore.read_bytes())
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    paths = [
        Path("Members/member/notebooks/subject-info.csv"),
        Path("Global/reports/predictions.csv"),
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Patient ID;Age;Exit;Cause\nsynthetic;60;1;1\n")
    subprocess.run(["git", "add", "."], check=True)
    assert main() == 1
    output = capsys.readouterr().out
    for path in paths:
        assert f"{path}: CSV/TSV with patient identifier columns" in output


@pytest.mark.parametrize(
    "filename,content,encoding",
    [
        ("renamed.csv", "index,target,score\nP0001,1,0.8\n", "utf-8"),
        ("renamed.tsv", "record\ttarget\nP0001\t1\n", "utf-16"),
        ("records.csv", 'MUSIC export\nrecord;target\n" P0001 ";1\n', "utf-8-sig"),
        ("records.json", '[{"record": "P0001", "score": 0.8}]', "utf-8"),
        ("records.json", '{"P0001": {"score": 0.8}}', "utf-32"),
        ("records.JSON", '{"nested": {"ids": ["P0001"]}}', "utf-16"),
        ("records.json", r'{"record": "\u0050\u0030\u0030\u0030\u0031"}', "utf-8"),
        ("records.jsonl", '{"record": "P0001"}\n{"record": "P0002"}\n', "utf-8"),
        ("records.ndjson", '{"record": "P0001"}\n', "utf-8"),
        ("records.txt", "Patient export\nindex,target\nP0001,1\n", "utf-8"),
        ("records.txt", "P0001\nP0002\n", "utf-16"),
        ("records.txt", "record target score\nP0001 1 0.8\n", "utf-8"),
        ("records.txt", '"P0001"   1   8e-1\n', "utf-16"),
        ("records.txt", "record=P0001\nscore=0.8\n", "utf-8"),
        ("records.txt", 'record: "P0001"\nscore: 0.8\n', "utf-8"),
        ("records.txt", '{"record": "P0001", "score": 0.8}', "utf-8"),
    ],
)
def test_known_patient_values_block_text_exports_in_every_folder(filename, content, encoding):
    for folder in ("Members/member/notebooks", "Global/reports", "Global/tests/fixtures"):
        problems = violations(Path(folder) / filename, content.encode(encoding))
        assert len(problems) == 1
        assert "patient identifier values" in problems[0]
        assert "P0001" not in problems[0]


def test_title_before_patient_header_is_rejected_even_without_known_id_values():
    problems = violations(
        Path("Global/reports/renamed.csv"),
        "MUSIC export\n환자 ID;target\nsynthetic;1\n".encode(),
    )
    assert len(problems) == 1
    assert "patient identifier columns" in problems[0]


@pytest.mark.parametrize(
    "filename,content",
    [
        ("Global/reports/summary.csv", "model,patients,scd,auroc\nlogistic,144,14,0.65\n"),
        ("Members/member/config.json", '{"seed": 42, "model": "xgboost"}'),
        ("Global/reports/summary.json", '{"patients": 144, "scd": 14, "auroc": 0.65}'),
        ("Global/reports/notes.txt", "A patient identifier such as P0001 is an example.\n"),
        ("Global/reports/column_notes.csv", "column\npatient_id\n"),
        ("Global/reports/column_notes.csv", 'column,usage\nrecord,"P0001 format example"\n'),
        ("Members/member/config.json", '{"id_format": "P0001-style identifiers"}'),
        ("Members/member/config.json", '{"code": "XP0001", "run": "P00010"}'),
        ("Global/reports/notes.csv", "run,score\nP0001_run,0.65\n"),
        ("Global/docs/example.md", "Example patient ID: `P0001`.\n"),
        ("Members/member/train.py", 'example = {"patient_id": "P0001"}\n'),
        ("Global/tests/test_example.py", 'assert example_id == "P0001"\n'),
    ],
)
def test_aggregate_settings_and_source_examples_remain_trackable(filename, content):
    assert violations(Path(filename), content.encode()) == []


def test_clean_notebook_source_can_show_a_synthetic_identifier_example():
    notebook = {"cells": [{"cell_type": "code", "source": ['example_id = "P0001"'], "outputs": []}]}
    assert (
        violations(Path("Members/member/notebooks/example.ipynb"), json.dumps(notebook).encode())
        == []
    )


def test_git_add_does_not_bypass_value_guard_or_print_patient_values(tmp_path, monkeypatch, capsys):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    path = Path("Global/reports/renamed.json")
    path.parent.mkdir(parents=True)
    path.write_text('[{"index": "P0001", "target": 1, "score": 0.8}]')
    subprocess.run(["git", "add", str(path)], check=True)
    assert main() == 1
    output = capsys.readouterr().out
    assert str(path) in output
    assert "patient identifier values" in output
    assert "P0001" not in output


def test_only_root_environment_templates_with_blank_secrets_are_allowed():
    for name in (".env.example", ".env.tracking.example"):
        assert violations(Path(name), b"# Account setup\nMLFLOW_TRACKING_PASSWORD=\n") == []
        assert violations(Path(name), b"MLFLOW_TRACKING_PASSWORD=real-value\n")
        assert violations(Path(name), b"MLFLOW_FLASK_SERVER_SECRET_KEY=real-value\n")
    for name in (".env.local.example", "nested/.env.example", ".env.tracking"):
        assert violations(Path(name), b"MLFLOW_TRACKING_PASSWORD=\n")


def test_rejects_notebook_output_and_allows_clean_notebook():
    notebook = {
        "cells": [{"cell_type": "code", "outputs": [{"text": "patient"}], "execution_count": 1}]
    }
    assert violations(Path("Members/member/notebooks/test.ipynb"), json.dumps(notebook).encode())
    notebook["cells"][0].update(outputs=[], execution_count=None)
    assert (
        violations(Path("Members/member/notebooks/test.ipynb"), json.dumps(notebook).encode()) == []
    )


@pytest.mark.parametrize(
    "suffix", ["pdf", "png", "jpg", "jpeg", "svg", "webp", "docx", "pptx", "PDF"]
)
def test_report_documents_and_images_have_bounded_larger_limit(suffix):
    path = Path(f"Global/reports/final/제출.{suffix}")
    assert violations(path, b"x" * (20 * MIB)) == []
    problems = violations(path, b"x" * (20 * MIB + 1))
    assert len(problems) == 1
    assert "20 MiB" in problems[0]


@pytest.mark.parametrize(
    "name",
    [
        "Members/member/experiments/figure.png",
        "report.pdf",
        "reports-large/report.pdf",
        "Global/reports/table.csv",
        "Global/reports/metrics.json",
        "Global/reports/report.md",
    ],
)
def test_other_files_keep_one_mib_limit(name):
    assert violations(Path(name), b"x" * MIB) == []
    assert "1 MiB" in violations(Path(name), b"x" * (MIB + 1))[0]


@pytest.mark.parametrize(
    "name",
    [
        "Global/reports/model.pt",
        "Global/reports/model.pkl",
        "Global/reports/.env",
        "Global/data/report.pdf",
        "Global/artifacts/report.png",
    ],
)
def test_report_policy_does_not_relax_forbidden_files(name):
    assert violations(Path(name), b"small")


def test_tracked_report_size_rule_is_enforced_by_repository_check(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    path = Path("Global/reports/final/report.pdf")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * (MIB + 1))
    subprocess.run(["git", "add", str(path)], check=True)
    assert main() == 0
    path.write_bytes(b"x" * (20 * MIB + 1))
    assert main() == 1
    assert "Global/reports/final/report.pdf: file exceeds the 20 MiB" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["data/patients.csv", "artifacts/run/result.json"])
def test_legacy_local_directories_still_reject_forced_add(name):
    assert violations(Path(name), b"sensitive")


def test_repository_check_covers_root_files_when_called_from_global(tmp_path, monkeypatch, capsys):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    nested = tmp_path / "Global"
    nested.mkdir()
    secret = tmp_path / ".env"
    secret.write_bytes(b"SECRET=value")
    subprocess.run(["git", "-C", str(tmp_path), "add", ".env"], check=True)
    monkeypatch.chdir(nested)
    assert main() == 1
    assert ".env: environment secrets must not be tracked" in capsys.readouterr().out


GENERATED_EXPERIMENT_PATHS = [
    "Members/member/experiments/exp001/outputs/model.json",
    "Members/member/experiments/exp001/outputs/metrics.csv",
    "Members/member/experiments/exp001/outputs/.hidden/config.txt",
    "Members/member/experiments/exp001/nested/Outputs/model.JSON",
    "Members/member/experiments/exp001/models/model.json",
    "Members/member/experiments/exp001/models/.hidden/model.json",
    "Members/member/experiments/exp001/nested/Models/nested/model.JSON",
    "experiments/member/exp001/outputs/model.json",
    "experiments/member/exp001/models/model.json",
    "Members/member/experiments/exp001/model.ubj",
    "model.UBJ",
]
TRACKABLE_EXPERIMENT_PATHS = [
    "Members/member/experiments/exp001/models/network.py",
    "Members/member/experiments/exp001/nested/Models/network.py",
    "Members/member/experiments/exp001/configs/model.json",
    "Members/member/experiments/exp001/preprocessing.JSON",
    "Members/member/experiments/exp001/fit_model.py",
]


@pytest.mark.parametrize("name", GENERATED_EXPERIMENT_PATHS)
def test_generated_experiment_files_are_rejected_even_when_force_added(name):
    assert violations(Path(name), b"{}")


@pytest.mark.parametrize("name", TRACKABLE_EXPERIMENT_PATHS)
def test_model_source_and_ordinary_json_settings_remain_trackable(name):
    assert violations(Path(name), b"{}") == []


def test_git_add_skips_generated_outputs_and_repository_guard_catches_force_add(
    tmp_path, monkeypatch, capsys
):
    repository_ignore = Path(__file__).resolve().parents[2] / ".gitignore"
    (tmp_path / ".gitignore").write_bytes(repository_ignore.read_bytes())
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    for name in GENERATED_EXPERIMENT_PATHS + TRACKABLE_EXPERIMENT_PATHS:
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")

    subprocess.run(["git", "add", "."], check=True)
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0"))
    assert not tracked.intersection(GENERATED_EXPERIMENT_PATHS)
    assert set(TRACKABLE_EXPERIMENT_PATHS) <= tracked
    assert main() == 0

    subprocess.run(["git", "add", "--force", "--", *GENERATED_EXPERIMENT_PATHS], check=True)
    assert main() == 1
    output = capsys.readouterr().out
    for name in GENERATED_EXPERIMENT_PATHS:
        assert f"{name}: " in output
