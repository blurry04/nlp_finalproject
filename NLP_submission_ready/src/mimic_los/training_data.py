from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import CATEGORICAL_FEATURES, LABEL_TO_ID, NUMERIC_FEATURES, RAW_TEXT_FIELDS


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value).strip()


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_tagged_text(record: dict[str, Any]) -> str:
    sections: list[str] = []
    if _text(record.get("chiefcomplaint")):
        sections.append(f"[CHIEF_COMPLAINT] {_text(record.get('chiefcomplaint'))}")
    if _text(record.get("prior_diagnosis_summary")):
        sections.append(f"[PRIOR_DIAGNOSES] {_text(record.get('prior_diagnosis_summary'))}")
    if _text(record.get("med_categories")):
        sections.append(f"[MED_RECON] {_text(record.get('med_categories'))}")
    if _text(record.get("radiology_findings")):
        sections.append(f"[FINDINGS] {_text(record.get('radiology_findings'))}")
    if _text(record.get("radiology_impression")):
        sections.append(f"[IMPRESSION] {_text(record.get('radiology_impression'))}")
    return " ".join(sections).strip()


def build_tier2a_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    narrative = _text(record.get("narrative"))
    tagged = build_tagged_text(record)
    if narrative:
        parts.append(f"[NARRATIVE] {narrative}")
    if tagged:
        parts.append(tagged)
    return " ".join(parts).strip()


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "subject_id": _text(record.get("subject_id")),
        "hadm_id": _text(record.get("hadm_id")),
        "stay_id": _text(record.get("stay_id")),
        "los_bucket": _text(record.get("los_bucket")),
        "label_id": LABEL_TO_ID[_text(record.get("los_bucket"))],
        "tagged_text": build_tagged_text(record),
        "tier2a_text": build_tier2a_text(record),
        "narrative": _text(record.get("narrative")),
    }
    for feature in NUMERIC_FEATURES:
        compact[feature] = _number(record.get(feature))
    for feature in CATEGORICAL_FEATURES:
        compact[feature] = _text(record.get(feature))
    for feature in RAW_TEXT_FIELDS:
        compact[feature] = _text(record.get(feature))
    return compact


def assign_subject_splits(source_jsonl: Path, seed: int = 42) -> dict[str, str]:
    subject_ids = sorted({_text(record.get("subject_id")) for record in iter_jsonl(source_jsonl) if _text(record.get("subject_id"))})
    rng = np.random.default_rng(seed)
    rng.shuffle(subject_ids)
    n = len(subject_ids)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_ids = set(subject_ids[:n_train])
    val_ids = set(subject_ids[n_train : n_train + n_val])
    assignments: dict[str, str] = {}
    for subject_id in subject_ids:
        if subject_id in train_ids:
            assignments[subject_id] = "train"
        elif subject_id in val_ids:
            assignments[subject_id] = "val"
        else:
            assignments[subject_id] = "test"
    return assignments


def build_structured_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def load_split_frame(path: Path, limit: int | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(iter_jsonl(path), start=1):
        if columns is not None:
            record = {column: record.get(column) for column in columns}
        rows.append(record)
        if limit is not None and idx >= limit:
            break
    return pd.DataFrame(rows)


def _write_jsonl(path: Path, records: Iterator[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_training_data(
    source_jsonl: Path,
    output_dir: Path,
    seed: int = 42,
    limit: int | None = None,
) -> dict[str, Any]:
    source_jsonl = source_jsonl.resolve()
    output_dir = output_dir.resolve()
    raw_dir = output_dir / "raw_splits"
    prepared_dir = output_dir / "prepared_splits"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    split_by_subject = assign_subject_splits(source_jsonl, seed=seed)
    raw_paths = {
        split: raw_dir / f"{split}.jsonl"
        for split in ("train", "val", "test")
    }
    writers = {
        split: raw_paths[split].open("w", encoding="utf-8")
        for split in raw_paths
    }
    counts = {"train": 0, "val": 0, "test": 0}
    try:
        for idx, record in enumerate(iter_jsonl(source_jsonl), start=1):
            compact = compact_record(record)
            split = split_by_subject.get(compact["subject_id"], "train")
            writers[split].write(json.dumps(compact, ensure_ascii=False) + "\n")
            counts[split] += 1
            if limit is not None and idx >= limit:
                break
    finally:
        for handle in writers.values():
            handle.close()

    train_frames: list[pd.DataFrame] = []
    for chunk in pd.read_json(raw_paths["train"], lines=True, chunksize=4096):
        train_frames.append(chunk[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    train_frame = pd.concat(train_frames, ignore_index=True)
    preprocessor = build_structured_preprocessor()
    preprocessor.fit(train_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    preprocessor_path = output_dir / "structured_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    prepared_paths = {
        split: prepared_dir / f"{split}.jsonl"
        for split in ("train", "val", "test")
    }
    structured_dim = None
    for split in ("train", "val", "test"):
        out_path = prepared_paths[split]
        with out_path.open("w", encoding="utf-8") as out_handle:
            for chunk in pd.read_json(raw_paths[split], lines=True, chunksize=4096):
                transformed = preprocessor.transform(chunk[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
                if hasattr(transformed, "toarray"):
                    transformed = transformed.toarray()
                transformed = np.asarray(transformed, dtype=np.float32)
                structured_dim = int(transformed.shape[1])
                records = chunk.to_dict(orient="records")
                for row_idx, row in enumerate(records):
                    row["structured_vector"] = transformed[row_idx].tolist()
                    out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "source_jsonl": str(source_jsonl),
        "seed": seed,
        "split_counts": counts,
        "raw_split_paths": {key: str(value) for key, value in raw_paths.items()},
        "prepared_split_paths": {key: str(value) for key, value in prepared_paths.items()},
        "structured_preprocessor_path": str(preprocessor_path),
        "structured_dim": structured_dim,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
