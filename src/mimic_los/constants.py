from __future__ import annotations

DEFAULT_EARLY_WINDOW_HOURS = 24

LABELS = ["SHORT", "MEDIUM", "LONG"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

NUMERIC_FEATURES = [
    "anchor_age",
    "acuity",
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "pain",
    "ed_dwell_minutes",
    "med_count",
    "radiology_note_count",
]

CATEGORICAL_FEATURES = [
    "gender",
    "arrival_transport",
    "admission_type",
    "admission_location",
    "insurance",
    "language",
    "marital_status",
    "race",
]

RAW_TEXT_FIELDS = [
    "chiefcomplaint",
    "med_names",
    "med_categories",
    "prior_diagnosis_summary",
    "radiology_findings",
    "radiology_impression",
    "narrative",
    "narrative_source",
    "narrative_template",
    "narrative_llm",
]

STRUCTURED_FEATURE_LABELS = {
    "anchor_age": "Age",
    "acuity": "Acuity",
    "temperature": "Temperature",
    "heartrate": "Heart rate",
    "resprate": "Respiratory rate",
    "o2sat": "O2 saturation",
    "sbp": "Systolic BP",
    "dbp": "Diastolic BP",
    "pain": "Pain score",
    "ed_dwell_minutes": "ED dwell",
    "med_count": "Medication count",
    "radiology_note_count": "Radiology notes",
    "gender": "Gender",
    "arrival_transport": "Arrival transport",
    "admission_type": "Admission type",
    "admission_location": "Admission location",
    "insurance": "Insurance",
    "language": "Language",
    "marital_status": "Marital status",
    "race": "Race",
}
