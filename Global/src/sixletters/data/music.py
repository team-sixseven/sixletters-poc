"""Pinned MUSIC CSV preparation with a versioned patient-level comparison protocol.

Preparation standardizes interpretation, never fits imputation, scaling or a model.
The observed-follow-up SCD endpoint is an experimental comparison policy, not a
fixed-horizon risk estimate or a clinically validated prediction specification.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from sixletters.contracts import Dataset

BASE_URL = "https://physionet.org/files/music-sudden-cardiac-death/1.0.1/"
SOURCE_FILES = {
    "subject-info.csv": "dae6975035fbc38e36bb57305021b31205f62e4e7af4b213cceeb781367fdd45",
    "subject-info_codes.csv": "d800b36fe0c5628bce5c6ce12ceb281f398276cedce562870ffd657fb79f5a36",
    "subject-info_definitions.csv": "129d6f222a5897e592611ae4ec2fc9caf1d2f00a03bb264fa139b4bfabe6650f",
}
MAX_DOWNLOAD_BYTES = 2_000_000
DOWNLOAD_TIMEOUT_SECONDS = 30
PROTOCOL_VERSION = 1
SPLIT_SEED = 42
EXPECTED_HEADERS = (
    "Patient ID",
    "Follow-up period from enrollment (days)",
    "days_4years",
    "Exit of the study",
    "Cause of death",
    "Age",
    "Gender (male=1)",
    "Weight (kg)",
    "Height (cm)",
    "Body Mass Index (Kg/m2)",
    "NYHA class",
    "Diastolic blood  pressure (mmHg)",
    "Systolic blood pressure (mmHg)",
    "HF etiology - Diagnosis",
    "Diabetes (yes=1)",
    "History of dyslipemia (yes=1)",
    "Peripheral vascular disease (yes=1)",
    "History of hypertension (yes=1)",
    "Prior Myocardial Infarction (yes=1)",
    "Prior implantable device",
    "Prior Revascularization",
    "Syncope",
    "daily smoking (cigarretes/day)",
    "smoke-free time (years)",
    "cigarettes /year",
    "alcohol consumption (standard units)",
    "Albumin (g/L)",
    "ALT or GPT (IU/L)",
    "AST or GOT (IU/L)",
    "Normalized Troponin",
    "Total Cholesterol (mmol/L)",
    "Creatinine (?mol/L)",
    "Gamma-glutamil transpeptidase (IU/L)",
    "Glucose (mmol/L)",
    "Hemoglobin (g/L)",
    "HDL (mmol/L)",
    "Potassium (mEq/L)",
    "LDL (mmol/L)",
    "Sodium (mEq/L)",
    "Pro-BNP (ng/L)",
    "Protein (g/L)",
    "T3 (pg/dL)",
    "T4 (ng/L)",
    "Troponin (ng/mL)",
    "TSH (mIU/L)",
    "Urea (mg/dL)",
    "Signs of pulmonary venous hypertension (yes=1)",
    "Cardiothoracic ratio",
    "Left atrial size (mm)",
    "Right ventricle contractility (altered=1)",
    "Right ventricle end-diastolic diameter (mm)",
    "LVEF (%)",
    "Mitral valve insufficiency ",
    "Mitral flow pattern",
    "Left ventricular posterior wall thickness (mm)",
    "Septal thickness (mm)",
    "Left ventricle end-diastolic diameter (mm)",
    "Left ventricle end-systolic diameter (mm)",
    "Hig-resolution ECG available",
    "ECG rhythm ",
    "Q-waves (necrosis, yes=1)",
    "PR interval (ms)",
    "QRS duration (ms)",
    "QRS > 120 ms ",
    "QT interval (ms)",
    "QT corrected ",
    "Average RR (ms) ",
    "Left ventricular hypertrophy (yes=1)",
    "Intraventricular conduction disorder",
    "Holter available",
    "Holter onset (hh:mm:ss)",
    "Holter  rhythm ",
    "minimum RR (ms) ",
    "Average RR (ms)",
    "maximum RR (ms)",
    "RR range (ms)",
    "Number of ventricular premature beats in 24h",
    "Extrasystole couplets ",
    "Ventricular Extrasystole",
    "Ventricular Tachycardia",
    "Number of ventricular premature contractions per hour",
    "Non-sustained ventricular tachycardia (CH>10)",
    "Number of supraventricular premature beats in 24h",
    "Paroxysmal supraventricular tachyarrhythmia",
    "Longest RR pause (ms)",
    "Bradycardia",
    "SDNN (ms)",
    "SDANN (ms)",
    "RMSSD (ms)",
    "pNN50 (%)",
    "Calcium channel blocker (yes=1)",
    "Diabetes medication (yes=1)",
    "Amiodarone (yes=1)",
    "Angiotensin-II receptor blocker (yes=1)",
    "Anticoagulants/antitrombotics  (yes=1)",
    "Betablockers (yes=1)",
    "Digoxin (yes=1)",
    "Loop diuretics (yes=1)",
    "Spironolactone (yes=1)",
    "Statins (yes=1)",
    "Hidralazina (yes=1)",
    "ACE inhibitor (yes=1)",
    "Nitrovasodilator (yes=1)",
)

# Exact source positions preserve distinctions such as the two Average RR headers.
# These aliases are stable protocol fields, not automatically stripped raw headers.
CLINICAL_ALIASES = {
    6: "age_years",
    7: "sex_male",
    8: "weight_kg",
    9: "height_cm",
    10: "body_mass_index",
    11: "nyha_class",
    12: "diastolic_bp_mmhg",
    13: "systolic_bp_mmhg",
    14: "hf_etiology",
    15: "diabetes",
    16: "history_dyslipemia",
    17: "peripheral_vascular_disease",
    18: "history_hypertension",
    19: "prior_myocardial_infarction",
    20: "prior_implantable_device",
    21: "prior_revascularization",
    22: "syncope",
    23: "daily_smoking_cigarettes",
    24: "smoke_free_years",
    26: "alcohol_standard_units",
    27: "albumin_g_l",
    28: "alt_iu_l",
    29: "ast_iu_l",
    31: "total_cholesterol_mmol_l",
    32: "creatinine",
    33: "ggt_iu_l",
    34: "glucose_mmol_l",
    35: "hemoglobin_g_l",
    36: "hdl_mmol_l",
    37: "potassium_meq_l",
    38: "ldl_mmol_l",
    39: "sodium_meq_l",
    40: "pro_bnp_ng_l",
    41: "protein_g_l",
    42: "t3_pg_dl",
    43: "t4_ng_l",
    44: "troponin_ng_ml",
    45: "tsh_miu_l",
    46: "urea_mg_dl",
    47: "pulmonary_venous_hypertension",
    48: "cardiothoracic_ratio",
    49: "left_atrial_size_mm",
    50: "right_ventricle_contractility_altered",
    51: "right_ventricle_end_diastolic_diameter_mm",
    52: "lvef_percent",
    53: "mitral_valve_insufficiency",
    54: "mitral_flow_pattern",
    55: "left_ventricular_posterior_wall_thickness_mm",
    56: "septal_thickness_mm",
    57: "left_ventricle_end_diastolic_diameter_mm",
    58: "left_ventricle_end_systolic_diameter_mm",
    91: "med_calcium_channel_blocker",
    92: "med_diabetes",
    93: "med_amiodarone",
    94: "med_angiotensin_ii_receptor_blocker",
    95: "med_anticoagulants_antithrombotics",
    96: "med_beta_blockers",
    97: "med_digoxin",
    98: "med_loop_diuretics",
    99: "med_spironolactone",
    100: "med_statins",
    101: "med_hydralazine",
    102: "med_ace_inhibitor",
    103: "med_nitrovasodilator",
}
BASIC_ECG_ALIASES = {
    60: "ecg_rhythm",
    61: "ecg_q_waves",
    63: "ecg_qrs_duration_ms",
    65: "ecg_qt_interval_ms",
    67: "ecg_average_rr_ms",
}
HRV_ALIASES = {
    87: "hrv_sdnn_ms",
    88: "hrv_sdann_ms",
    89: "hrv_rmssd_ms",
    90: "hrv_pnn50_percent",
}
CATEGORICAL_CODES = {
    "nyha_class": frozenset({"1", "2", "3", "4"}),
    "hf_etiology": frozenset(str(value) for value in range(1, 10)),
    "prior_implantable_device": frozenset({"0", "1", "2", "3", "4", "5", "6", "8"}),
    "prior_revascularization": frozenset({"0", "1", "2", "3"}),
    "syncope": frozenset({"0", "1", "2"}),
    "mitral_valve_insufficiency": frozenset({"0", "1", "2", "3", "4"}),
    "mitral_flow_pattern": frozenset({"0", "1", "2", "3", "9"}),
    "ecg_rhythm": frozenset({"0", "1", "2", "3"}),
}
_BINARY_POSITIONS = frozenset({7, 15, 16, 17, 18, 19, 47, 50, 61, *range(91, 104)})
_NUMERIC_PATTERN = re.compile(r"[+-]?(?:[0-9]+(?:[.,][0-9]+)?|[.,][0-9]+)(?:[eE][+-]?[0-9]+)?")


def _validate_cohort(cohort: str) -> None:
    if cohort not in {"basic", "hrv"}:
        raise ValueError("MUSIC cohort must be basic or hrv")


def get_music_destination(root: Path, cohort: str = "basic") -> Path:
    """Return the versioned output directory, without creating it."""
    _validate_cohort(cohort)
    return Path(root) / "Global/data" / "prepared" / f"music-scd-{cohort}-v1"


def _check_path(path: Path) -> None:
    """Do not traverse user-created links when downloading or publishing data."""
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError(f"MUSIC paths must not contain symlinks: {candidate}")


def _source_hash(path: Path) -> str:
    _check_path(path)
    if not path.is_file():
        raise ValueError(f"MUSIC source must be a regular file: {path.name}")
    if path.stat().st_size > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"MUSIC source exceeds the size limit: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_source(path: Path, expected_sha256: str) -> None:
    if _source_hash(path) != expected_sha256:
        raise ValueError(f"MUSIC source SHA-256 mismatch: {path.name}; file was not overwritten")


def _download_source(url: str, destination: Path, expected_sha256: str) -> None:
    """Download a bounded, verified file and publish it without replacing any file."""
    _check_path(destination)
    if destination.exists():
        _verify_source(destination, expected_sha256)
        return
    temporary = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "sixletters-poc/0.1"})
        with (
            urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
            tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=".music-download-", delete=False
            ) as handle,
        ):
            temporary = Path(handle.name)
            total = 0
            digest = hashlib.sha256()
            while chunk := response.read(64 * 1024):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"MUSIC download exceeds the size limit: {destination.name}")
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"MUSIC download SHA-256 mismatch: {destination.name}")
        _check_path(destination)
        try:
            # A hard link publishes a complete file atomically, and cannot replace one.
            os.link(temporary, destination)
        except FileExistsError:
            _verify_source(destination, expected_sha256)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _number(value: str, column: str) -> float:
    value = value.strip()
    if not value:
        return float("nan")
    if not _NUMERIC_PATTERN.fullmatch(value):
        raise ValueError(f"MUSIC numeric value cannot be parsed in {column}")
    result = float(value.replace(",", "."))
    if not math.isfinite(result):
        raise ValueError(f"MUSIC numeric value must be finite in {column}")
    return result


def _feature_value(raw: str, position: int, alias: str) -> float | str:
    value = raw.strip()
    if alias in CATEGORICAL_CODES:
        if not value:
            return float("nan")
        if value not in CATEGORICAL_CODES[alias]:
            raise ValueError(f"MUSIC categorical code is unknown in {alias}")
        return value
    if alias == "age_years" and value == ">89":
        return float("nan")
    number = _number(value, alias)
    if position in _BINARY_POSITIONS and not math.isnan(number) and number not in {0.0, 1.0}:
        raise ValueError(f"MUSIC binary value must be 0 or 1 in {alias}")
    return number


def _column_manifest(cohort: str) -> list[dict]:
    ecg = BASIC_ECG_ALIASES | (HRV_ALIASES if cohort == "hrv" else {})
    rows = []
    for position, source_column in enumerate(EXPECTED_HEADERS, start=1):
        alias = (CLINICAL_ALIASES | ecg).get(position)
        role = "excluded"
        note = "Not selected in this versioned feature policy."
        if position in CLINICAL_ALIASES:
            role, note = "clinical", "Missing values are retained; no fitted preprocessing."
        elif position in ecg:
            role, note = "ecg", "Existing CSV measurement; no waveform extraction."
        elif position == 1:
            role, note = "identifier", "Patient identity only; never a model input."
        elif position in {2, 3, 4, 5}:
            role, note = "outcome", "Endpoint or follow-up metadata; never a model input."
        elif position in {59, 70, 71, 72}:
            role, note = "metadata", "Availability, acquisition or cohort metadata only."
        if position == 6:
            note = "The privacy-bounded value >89 is missing, not an exact age."
        elif position == 25:
            note = "Held out: smoking unit is unresolved."
        elif position == 30:
            note = "Held out: normalization definition is not established for this protocol."
        elif position == 32:
            note = "Source unit is unresolved (?mol/L); preserve values without unit conversion."
        elif position == 62:
            note = "Held out: PR value 999 has unresolved missing/sentinel meaning."
        elif position == 66:
            note = "Held out: QT correction formula unresolved; raw QT is selected instead."
        elif position == 74:
            note = "Holter Average RR differs from ECG-section Average RR; not selected."
        elif position == 81:
            note = "Held out: multiple-comma numeric values require source clarification."
        elif position in HRV_ALIASES and cohort == "basic":
            note = "Available only in the separate sinus-rhythm, complete-HRV cohort."
        rows.append(
            {
                "source_column": source_column,
                "source_position": position,
                "alias": alias,
                "role": role,
                "note": note,
            }
        )
    return rows


def _assign_splits(frame: pd.DataFrame) -> pd.Series:
    assignments = {}
    for target in (0, 1):
        ids = frame.loc[frame["target"].eq(target), "patient_id"].tolist()
        ids.sort(
            key=lambda patient_id: (
                hashlib.sha256(f"{SPLIT_SEED}:{target}:{patient_id}".encode()).hexdigest(),
                patient_id,
            )
        )
        train_end, validation_size = len(ids) * 70 // 100, len(ids) * 15 // 100
        if min(train_end, validation_size, len(ids) - train_end - validation_size) < 1:
            raise ValueError("MUSIC cohort needs both target classes in all three splits")
        for index, patient_id in enumerate(ids):
            assignments[patient_id] = (
                "train"
                if index < train_end
                else "validation"
                if index < train_end + validation_size
                else "test"
            )
    return frame["patient_id"].map(assignments)


def _build_dataset(raw_path: Path, cohort: str) -> tuple[Dataset, dict, list[dict]]:
    """Interpret already verified source CSV; inputs are allowlisted, outcomes never leak."""
    import pandas as pd

    from sixletters.contracts import Dataset

    _validate_cohort(cohort)
    with raw_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        if tuple(next(reader, [])) != EXPECTED_HEADERS:
            raise ValueError("MUSIC source headers do not match the pinned 1.0.1 schema")
        raw_rows = list(reader)
    ecg = BASIC_ECG_ALIASES | (HRV_ALIASES if cohort == "hrv" else {})
    features = CLINICAL_ALIASES | ecg
    rows, seen = [], set()
    excluded_exit = {"1": 0, "2": 0}
    excluded_hrv, bounded_age = 0, 0
    for row in raw_rows:
        if len(row) != len(EXPECTED_HEADERS):
            raise ValueError("MUSIC source row has an incorrect number of columns")
        patient_id = row[0]
        if not patient_id or patient_id != patient_id.strip() or patient_id in seen:
            raise ValueError("MUSIC patient IDs must be unique, nonblank and unpadded")
        seen.add(patient_id)
        exit_code, cause = row[3].strip(), row[4].strip()
        if (
            exit_code not in {"", "0", "1", "2", "3"}
            or cause not in {"0", "1", "3", "6", "7"}
            or ((exit_code == "3") != (cause != "0"))
        ):
            raise ValueError("MUSIC exit and cause-of-death codes are unknown or inconsistent")
        if exit_code in excluded_exit:
            excluded_exit[exit_code] += 1
            continue
        if cohort == "hrv":
            hrv_values = [
                _number(row[position - 1], alias) for position, alias in HRV_ALIASES.items()
            ]
            if row[71].strip() != "0" or not all(math.isfinite(value) for value in hrv_values):
                excluded_hrv += 1
                continue
        values = {
            alias: _feature_value(row[position - 1], position, alias)
            for position, alias in features.items()
        }
        bounded_age += row[5].strip() == ">89"
        rows.append({"patient_id": patient_id, "target": int(cause == "3"), **values})
    if not rows:
        raise ValueError("MUSIC cohort contains no eligible patients")
    frame = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    frame["split"] = _assign_splits(frame)
    # All-null numerical fields must remain numeric even in small test fixtures.
    for alias in features.values():
        if alias not in CATEGORICAL_CODES:
            frame[alias] = frame[alias].astype(float)
    dataset = Dataset(
        frame=frame,
        feature_groups={"clinical": tuple(CLINICAL_ALIASES.values()), "ecg": tuple(ecg.values())},
        dataset_id=f"music-1.0.1-scd-{cohort}-v1",
        target_definition=(
            f"music-scd-{cohort}-v1: Cause of death=3 during observed follow-up is 1; "
            "survivors (Exit blank/0,Cause=0) and non-SCD deaths (Exit=3,Cause=1/6/7) "
            "are 0; exclude lost follow-up and transplant (Exit=1/2); "
            + (
                "require Holter rhythm=0 and finite SDNN/SDANN/RMSSD/pNN50; "
                if cohort == "hrv"
                else ""
            )
            + "not a fixed-four-year or time-to-event endpoint"
        ),
        categorical_features=tuple(
            alias for alias in features.values() if alias in CATEGORICAL_CODES
        ),
    )
    dataset.validate()
    report = {
        "source_rows": len(raw_rows),
        "included_patients": len(frame),
        "scd_patients": int(frame["target"].sum()),
        "excluded_lost_followup": excluded_exit["1"],
        "excluded_transplant": excluded_exit["2"],
        "excluded_hrv_criteria": excluded_hrv,
        "bounded_age_values_set_missing": bounded_age,
        "feature_counts": {group: len(names) for group, names in dataset.feature_groups.items()},
        "missing_feature_values": {
            alias: int(frame[alias].isna().sum()) for alias in features.values()
        },
        "split_counts": {
            split: {
                "patients": int(frame["split"].eq(split).sum()),
                "scd": int(frame.loc[frame["split"].eq(split), "target"].sum()),
            }
            for split in ("train", "validation", "test")
        },
        "warnings": [
            "Observed follow-up differs among patients; this binary endpoint is experimental.",
            "Lost-followup and transplant exclusions may introduce selection bias.",
            "Missing values remain missing; fit imputers and scalers on training patients only.",
            "Source measurement validity and clinical usefulness are not established by parsing.",
            "Feature policy is versioned; excluded ambiguous fields are not silently repaired.",
        ],
        "fingerprint": dataset.fingerprint(),
    }
    if cohort == "hrv":
        report["warnings"].append(
            "HRV uses a selected sinus-rhythm, complete-case cohort; compare only within this cohort."
        )
    return dataset, report, _column_manifest(cohort)


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _verify_outputs(destination: Path, expected: Path) -> None:
    _check_path(destination)
    if not destination.is_dir():
        raise ValueError("MUSIC output path must be a directory; it was not overwritten")
    actual_names = {path.name for path in destination.iterdir()}
    expected_names = {path.name for path in expected.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            "MUSIC output files are missing or unexpected; directory was not overwritten"
        )
    for expected_file in expected.iterdir():
        actual = destination / expected_file.name
        _check_path(actual)
        if not actual.is_file() or actual.stat().st_size != expected_file.stat().st_size:
            raise ValueError(f"MUSIC output mismatch: {actual.name}; file was not overwritten")
        if actual.read_bytes() != expected_file.read_bytes():
            raise ValueError(f"MUSIC output mismatch: {actual.name}; file was not overwritten")


def prepare_music(root: Path, *, cohort: str = "basic", offline: bool = False) -> Path:
    """Download verified CSVs and create or verify an identical, versioned dataset bundle.

    Existing source and output files are never replaced. Offline mode requires all
    three pinned source files in the raw cache. All generated files are deterministic.
    """
    from sixletters.data.prepared import save_prepared_dataset

    _validate_cohort(cohort)
    root = Path(root).absolute()
    raw_dir = root / "Global/data" / "raw" / "music-1.0.1"
    destination = get_music_destination(root, cohort)
    for path in (raw_dir, destination):
        _check_path(path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, digest in SOURCE_FILES.items():
        path = raw_dir / filename
        _check_path(path)
        if path.exists():
            _verify_source(path, digest)
        elif offline:
            raise FileNotFoundError(f"MUSIC offline cache is missing {filename}; run online first")
        else:
            _download_source(BASE_URL + filename, path, digest)
    dataset, report, columns = _build_dataset(raw_dir / "subject-info.csv", cohort)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".music-prepare-", dir=destination.parent))
    staging = staging_root / "bundle"
    try:
        save_prepared_dataset(dataset, staging, synthetic=False)
        _write_json(staging / "report.json", report)
        _write_json(staging / "column_manifest.json", columns)
        _write_json(
            staging / "source_manifest.json",
            {
                "dataset": "MUSIC 1.0.1",
                "homepage": "https://physionet.org/content/music-sudden-cardiac-death/1.0.1/",
                "license": "Open Data Commons Open Database License v1.0",
                "files": [
                    {"filename": filename, "url": BASE_URL + filename, "sha256": digest}
                    for filename, digest in SOURCE_FILES.items()
                ],
            },
        )
        _write_json(
            staging / "protocol.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "dataset_id": dataset.dataset_id,
                "cohort": cohort,
                "target_definition": dataset.target_definition,
                "status": "experimental comparison protocol; confirm endpoint before final analysis",
                "source_csv": {"delimiter": ";", "decimal_separator": ",", "missing": "empty"},
                "split": {
                    "seed": SPLIT_SEED,
                    "unit": "patient_id",
                    "stratify_by": "target",
                    "algorithm": "sort each class by SHA256('42:{target}:{patient_id}'), then ID",
                    "train": "floor(class_size * 70 / 100)",
                    "validation": "floor(class_size * 15 / 100)",
                    "test": "remaining patients",
                },
                "feature_groups": dataset.feature_groups,
                "categorical_codes": {
                    alias: sorted(CATEGORICAL_CODES[alias])
                    for alias in dataset.categorical_features
                },
                "preprocessing": {
                    "age_gt_89": "missing; do not substitute an exact age",
                    "ambiguous_fields": "excluded; see column_manifest.json",
                    "missing_values": "retained",
                    "imputation_scaling_feature_selection": "fit only on the training split",
                },
                "evaluation": "select experiments on validation; keep test for final evaluation",
            },
        )
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["patient_id", "target", "split"])
        writer.writerows(
            dataset.frame[["patient_id", "target", "split"]].itertuples(index=False, name=None)
        )
        (staging / "split_manifest.csv").write_text(
            buffer.getvalue(), encoding="utf-8", newline="\n"
        )
        _check_path(destination)
        if destination.exists():
            _verify_outputs(destination, staging)
        else:
            try:
                os.rename(staging, destination)
            except OSError:
                if not destination.exists():
                    raise
                _verify_outputs(destination, staging)
    finally:
        shutil.rmtree(staging_root)
    return destination / "dataset.json"
