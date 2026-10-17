"""Synthetic-only checks of MUSIC parsing, provenance, and common preparation."""

import csv
import hashlib
import io
from pathlib import Path
from urllib.error import URLError

import pandas as pd
import pytest

from sixletters.data import music
from sixletters.data.prepared import load_prepared_dataset


def synthetic_rows():
    """Create fictional patients; no real MUSIC patient row is included in tests."""
    rows = []
    for index in range(80):
        row = dict.fromkeys(music.EXPECTED_HEADERS, "0")
        row.update(
            {
                "Patient ID": f"SYN{index:03}",
                "Exit of the study": "" if index < 50 else "3",
                "Cause of death": "0" if index < 50 else "3",
                "Follow-up period from enrollment (days)": "2000",
                "days_4years": "1460",
                "Age": str(40 + index % 40),
                "NYHA class": "1",
                "HF etiology - Diagnosis": "1",
                "Body Mass Index (Kg/m2)": "24,75",
                "QRS duration (ms)": "90,5",
                "QT interval (ms)": "400",
                "QT corrected ": "9999",
                "Average RR (ms) ": "800,5",
                "Average RR (ms)": "1234,5",
                "PR interval (ms)": "999",
                "SDNN (ms)": "100,5",
                "SDANN (ms)": "90,5",
                "RMSSD (ms)": "30,5",
                "pNN50 (%)": "15,5",
            }
        )
        rows.append(row)
    rows[0]["Age"] = ">89"
    rows[1]["SDNN (ms)"] = ""
    rows[2]["Holter  rhythm "] = "1"
    for index, cause in enumerate(("1", "6", "7"), start=4):
        rows[index].update({"Exit of the study": "3", "Cause of death": cause})
    for status in ("1", "2"):
        row = rows[3].copy()
        row.update({"Patient ID": f"EXCLUDED{status}", "Exit of the study": status})
        rows.append(row)
    return rows


def csv_bytes(rows, headers=None):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=headers or music.EXPECTED_HEADERS, delimiter=";", extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def parse_rows(tmp_path, rows, cohort="basic"):
    source = tmp_path / "synthetic.csv"
    source.write_bytes(csv_bytes(rows))
    return music._build_dataset(source, cohort)


def aliases(manifest):
    return {item["source_column"]: item["alias"] for item in manifest}


@pytest.fixture
def synthetic_sources(tmp_path, monkeypatch):
    payloads = {
        "subject-info.csv": csv_bytes(synthetic_rows()),
        "subject-info_codes.csv": b"Synthetic test codebook\n",
        "subject-info_definitions.csv": b"Synthetic test definitions\n",
    }
    monkeypatch.setattr(
        music,
        "SOURCE_FILES",
        {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()},
    )
    calls = []

    def fetch(url, **kwargs):
        address = url.full_url if hasattr(url, "full_url") else str(url)
        calls.append(address)
        response = io.BytesIO(payloads[address.rsplit("/", 1)[-1]])
        response.status = 200
        response.headers = {"Content-Length": str(len(response.getvalue()))}
        return response

    monkeypatch.setattr("urllib.request.urlopen", fetch)
    root = tmp_path / "팀 프로젝트"
    (root / "Members").mkdir(parents=True)
    (root / "Global").mkdir()
    (root / "Global/pyproject.toml").write_text('[project]\nname = "sixletters-poc"\n')
    return (root, payloads, calls)


def test_basic_preparation_preserves_values_and_excludes_followup_from_inputs(tmp_path):
    dataset, _, manifest = parse_rows(tmp_path, synthetic_rows())
    by_source = aliases(manifest)
    frame = dataset.frame.set_index("patient_id")
    assert len(frame) == 80
    assert frame.target.value_counts().to_dict() == {0: 50, 1: 30}
    assert frame.loc["SYN050", "target"] == 1
    assert frame.loc[["SYN004", "SYN005", "SYN006"], "target"].eq(0).all()
    assert pd.isna(frame.loc["SYN000", by_source["Age"]])
    assert frame.loc["SYN001", by_source["Body Mass Index (Kg/m2)"]] == 24.75
    assert frame.loc["SYN001", by_source["Average RR (ms) "]] == 800.5
    assert by_source["Average RR (ms)"] not in frame
    assert by_source["QT interval (ms)"] in dataset.feature_groups["ecg"]
    assert len(dataset.feature_groups["ecg"]) == 5
    for source in (
        "Follow-up period from enrollment (days)",
        "days_4years",
        "Exit of the study",
        "Cause of death",
        "PR interval (ms)",
        "QT corrected ",
        "Normalized Troponin",
        "cigarettes /year",
    ):
        assert by_source[source] not in frame
    assert {"patient_id", "target", "split"}.isdisjoint(
        column for group in dataset.feature_groups.values() for column in group
    )
    for column in dataset.categorical_features:
        assert frame[column].dropna().map(lambda value: isinstance(value, str)).all()


def test_hrv_requires_complete_features_and_sinus_rhythm_on_same_cohort(tmp_path):
    dataset, _, manifest = parse_rows(tmp_path, synthetic_rows(), "hrv")
    frame = dataset.frame.set_index("patient_id")
    assert set(frame.index) == {f"SYN{i:03}" for i in range(80)} - {"SYN001", "SYN002"}
    assert len(dataset.feature_groups["ecg"]) == 9
    by_source = aliases(manifest)
    for source in ("SDNN (ms)", "SDANN (ms)", "RMSSD (ms)", "pNN50 (%)"):
        assert by_source[source] in dataset.feature_groups["ecg"]
        assert frame[by_source[source]].notna().all()


@pytest.mark.parametrize("cohort", ["basic", "hrv"])
def test_manifest_keeps_unresolved_units_explicit_without_inventing_source_meanings(
    tmp_path, cohort
):
    rows = synthetic_rows()
    rows[0]["Creatinine (?mol/L)"] = "123,45"
    dataset, _, manifest = parse_rows(tmp_path, rows, cohort)
    by_source = {item["source_column"]: item for item in manifest}
    creatinine = by_source["Creatinine (?mol/L)"]
    assert creatinine["alias"] == "creatinine"
    assert creatinine["role"] == "clinical"
    assert "unit is unresolved" in creatinine["note"]
    assert "without unit conversion" in creatinine["note"]
    assert dataset.frame.set_index("patient_id").loc["SYN000", "creatinine"] == 123.45
    smoking = by_source["cigarettes /year"]
    assert smoking["role"] == "excluded"
    assert "unit is unresolved" in smoking["note"]
    assert "malformed" not in smoking["note"]
    holter_rr = by_source["Average RR (ms)"]
    assert holter_rr["role"] == "excluded"
    assert "ECG-section" in holter_rr["note"]
    assert "resting" not in holter_rr["note"]
    assert by_source["Average RR (ms) "]["alias"] == "ecg_average_rr_ms"


def test_splits_and_fingerprint_are_identical_after_input_rows_are_reversed(tmp_path):
    first, _, _ = parse_rows(tmp_path, synthetic_rows())
    second, _, _ = parse_rows(tmp_path, synthetic_rows()[::-1])
    assert first.fingerprint() == second.fingerprint()
    pd.testing.assert_frame_equal(first.frame, second.frame)
    counts = first.frame.groupby(["split", "target"]).size()
    assert len(counts) == 6
    assert (counts > 0).all()
    assert first.frame.patient_id.is_unique


def test_too_few_scd_events_for_three_splits_fail_instead_of_producing_invalid_evaluation(tmp_path):
    rows = [row for row in synthetic_rows() if row["Cause of death"] != "3"]
    rows += synthetic_rows()[50:54]
    with pytest.raises(ValueError, match="both target classes"):
        parse_rows(tmp_path, rows)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Cause of death", "2"),
        ("Exit of the study", "3"),
        ("Patient ID", " SYN000"),
        ("Patient ID", "SYN001"),
        ("Age", "not-a-number"),
        ("ECG rhythm ", "987"),
        ("Diabetes (yes=1)", "2"),
        ("QRS duration (ms)", "Infinity"),
    ],
)
def test_invalid_endpoints_identities_and_selected_features_fail(tmp_path, column, value):
    rows = synthetic_rows()
    rows[0][column] = value
    with pytest.raises(ValueError):
        parse_rows(tmp_path, rows)


def test_unknown_cohort_and_missing_header_fail(tmp_path):
    with pytest.raises(ValueError):
        parse_rows(tmp_path, synthetic_rows(), "unreviewed")
    headers = list(music.EXPECTED_HEADERS)
    headers[headers.index("Average RR (ms) ")] = "Average RR (ms)"
    source = tmp_path / "duplicate-header.csv"
    source.write_bytes(csv_bytes(synthetic_rows(), headers))
    with pytest.raises(ValueError):
        music._build_dataset(source, "basic")


def test_download_then_offline_reuse_is_identical_and_cohorts_stay_separate(synthetic_sources):
    root, _, calls = synthetic_sources
    basic = music.prepare_music(root)
    original = {
        path.relative_to(basic.parent): path.read_bytes() for path in basic.parent.iterdir()
    }
    assert len(calls) == 3
    assert music.prepare_music(root, offline=True) == basic
    assert len(calls) == 3
    assert original == {
        path.relative_to(basic.parent): path.read_bytes() for path in basic.parent.iterdir()
    }
    hrv = music.prepare_music(root, cohort="hrv", offline=True)
    assert basic.parent != hrv.parent
    assert basic.read_bytes() == original[Path("dataset.json")]
    first, _ = load_prepared_dataset(basic)
    second, _ = load_prepared_dataset(hrv)
    assert first.fingerprint()["cohort_hash"] != second.fingerprint()["cohort_hash"]


def test_offline_never_downloads_missing_cache(synthetic_sources):
    root, _, calls = synthetic_sources
    with pytest.raises((ValueError, OSError)):
        music.prepare_music(root, offline=True)
    assert not calls
    assert not music.get_music_destination(root).exists()


@pytest.mark.parametrize("offline", [False, True])
def test_changed_source_cache_is_rejected_without_overwriting(synthetic_sources, offline):
    root, _, calls = synthetic_sources
    output = music.prepare_music(root)
    before = output.read_bytes()
    source = next((root / "Global/data" / "raw").rglob("subject-info.csv"))
    source.write_bytes(source.read_bytes() + b"tampered")
    with pytest.raises((ValueError, OSError)):
        music.prepare_music(root, offline=offline)
    assert source.read_bytes().endswith(b"tampered")
    assert output.read_bytes() == before
    assert len(calls) == 3


def test_modified_prepared_output_is_not_silently_replaced(synthetic_sources):
    root, _, _ = synthetic_sources
    output = music.prepare_music(root)
    output.write_text("user changes\n", encoding="utf-8")
    with pytest.raises((ValueError, OSError)):
        music.prepare_music(root, offline=True)
    assert output.read_text(encoding="utf-8") == "user changes\n"


def test_changed_split_policy_cannot_reuse_or_overwrite_existing_version(
    synthetic_sources, monkeypatch
):
    root, _, _ = synthetic_sources
    output = music.prepare_music(root)
    before = output.read_bytes()
    monkeypatch.setattr(music, "SPLIT_SEED", 123)
    with pytest.raises(ValueError, match="not overwritten"):
        music.prepare_music(root, offline=True)
    assert output.read_bytes() == before


def test_existing_empty_output_directory_is_preserved(synthetic_sources):
    root, _, _ = synthetic_sources
    destination = music.get_music_destination(root)
    destination.mkdir(parents=True)
    with pytest.raises(ValueError, match="not overwritten"):
        music.prepare_music(root)
    assert destination.is_dir()
    assert not list(destination.iterdir())


@pytest.mark.parametrize("failure", ["network", "truncated", "oversized"])
def test_bad_download_never_becomes_trusted_cache(synthetic_sources, monkeypatch, failure):
    root, payloads, _ = synthetic_sources

    def fetch(url, **kwargs):
        if failure == "network":
            raise URLError("synthetic network outage")
        content = payloads["subject-info.csv"][:-10]
        if failure == "oversized":
            content = b"x" * (music.MAX_DOWNLOAD_BYTES + 1)
        response = io.BytesIO(content)
        response.status = 200
        response.headers = {"Content-Length": str(len(content))}
        return response

    monkeypatch.setattr("urllib.request.urlopen", fetch)
    with pytest.raises((ValueError, OSError)):
        music.prepare_music(root)
    assert not music.get_music_destination(root).exists()
    assert not list((root / "Global/data" / "raw").rglob("subject-info.csv"))
    assert not list((root / "Global/data" / "raw").rglob(".music-download-*"))
