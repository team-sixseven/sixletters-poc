"""Four predefined clinical input sets, shared by all clinical estimators."""

MODEL_1 = [
    "Gender (male=1)",
    "Age",
    "LVEF (%)",
    "NYHA class",
]

MODEL_2 = MODEL_1 + [
    "eGFR",
    "Ischemic etiology",
    "Albumin (g/L)",
    "Diabetes (yes=1)",
]

MODEL_3 = MODEL_2 + [
    "Left atrial size (mm)",
    "Left ventricle end-diastolic diameter (mm)",
    "Left ventricle end-systolic diameter (mm)",
    "Left ventricular posterior wall thickness (mm)",
    "Septal thickness (mm)",
    "Right ventricle contractility (altered=1)",
    "Right ventricle end-diastolic diameter (mm)",
    "Mitral valve insufficiency",
    "Prior Myocardial Infarction (yes=1)",
    "Systolic blood pressure (mmHg)",
    "Diastolic blood  pressure (mmHg)",
    "Pro-BNP (ng/L)",
    "Cardiothoracic ratio",
]

MODEL_4 = MODEL_3 + [
    "cigarettes /year",
    "alcohol consumption (standard units)",
]

FEATURE_VERSIONS = {1: MODEL_1, 2: MODEL_2, 3: MODEL_3, 4: MODEL_4}


def get_feature_columns(model_version: int) -> list[str]:
    """Return a copy in the declared input order; reject invalid version values."""
    if isinstance(model_version, bool) or not isinstance(model_version, int):
        raise ValueError("model_version must be an integer from 1 to 4")
    if model_version not in FEATURE_VERSIONS:
        raise ValueError("model_version must be an integer from 1 to 4")
    return list(FEATURE_VERSIONS[model_version])
