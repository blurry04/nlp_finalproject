from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import DEFAULT_EARLY_WINDOW_HOURS
from .ollama_client import OllamaClient
from .text_utils import (
    build_gemma_prompt,
    build_verification_prompt,
    extract_radiology_sections,
    narrative_template,
    safe_preview,
)


def _parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _first_non_empty(values: pd.Series) -> str:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


def _join_unique(values: pd.Series, limit: int = 12) -> str:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
        if len(items) >= limit:
            break
    return "; ".join(items)


def _csv_preview_row(record: dict[str, Any], csv_preview_width: int) -> dict[str, Any]:
    return {
        "subject_id": record.get("subject_id", ""),
        "hadm_id": record.get("hadm_id", ""),
        "stay_id": record.get("stay_id", ""),
        "admittime": str(record.get("admittime", "")),
        "los_days": record.get("los_days", ""),
        "los_bucket": record.get("los_bucket", ""),
        "narrative_source": record.get("narrative_source", ""),
        "chiefcomplaint": safe_preview(str(record.get("chiefcomplaint", "")), csv_preview_width),
        "prior_diagnosis_summary": safe_preview(str(record.get("prior_diagnosis_summary", "")), csv_preview_width),
        "radiology_findings": safe_preview(str(record.get("radiology_findings", "")), csv_preview_width),
        "radiology_impression": safe_preview(str(record.get("radiology_impression", "")), csv_preview_width),
        "narrative": safe_preview(str(record.get("narrative", "")), csv_preview_width),
        "narrative_template": safe_preview(str(record.get("narrative_template", "")), csv_preview_width),
        "narrative_llm": safe_preview(str(record.get("narrative_llm", "")), csv_preview_width),
    }


def _recover_existing_jsonl(jsonl_path: Path) -> tuple[int, dict[str, int]]:
    if not jsonl_path.exists():
        return 0, {}
    valid_offsets: list[int] = []
    source_counts: dict[str, int] = defaultdict(int)
    offset = 0
    with jsonl_path.open("rb") as handle:
        for raw_line in handle:
            next_offset = offset + len(raw_line)
            if not raw_line.strip():
                offset = next_offset
                continue
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except Exception:
                break
            source_counts[str(record.get("narrative_source", "") or "unknown")] += 1
            valid_offsets.append(next_offset)
            offset = next_offset
    if valid_offsets:
        last_valid_offset = valid_offsets[-1]
        current_size = jsonl_path.stat().st_size
        if last_valid_offset != current_size:
            with jsonl_path.open("rb+") as handle:
                handle.truncate(last_valid_offset)
    else:
        jsonl_path.write_text("", encoding="utf-8")
    return len(valid_offsets), dict(source_counts)


def _rewrite_csv_from_jsonl(
    jsonl_path: Path,
    csv_path: Path,
    csv_fieldnames: list[str],
    csv_preview_width: int,
) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames, quoting=csv.QUOTE_MINIMAL)
        csv_writer.writeheader()
        if not jsonl_path.exists():
            return
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as jsonl_file:
            for line in jsonl_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                csv_writer.writerow(_csv_preview_row(record, csv_preview_width))


def _select_edstays_for_admissions(edstays: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    admission_times = admissions[["subject_id", "hadm_id", "admittime"]].dropna(subset=["subject_id", "hadm_id", "admittime"])
    ranked = edstays.merge(admission_times, on=["subject_id", "hadm_id"], how="inner")
    ranked["contains_admittime"] = (
        ranked["intime"].notna()
        & ranked["outtime"].notna()
        & (ranked["intime"] <= ranked["admittime"])
        & (ranked["outtime"] >= ranked["admittime"])
    )
    ranked["outtime_gap_hours"] = (ranked["admittime"] - ranked["outtime"]).abs().dt.total_seconds() / 3600.0
    ranked["intime_gap_hours"] = (ranked["admittime"] - ranked["intime"]).abs().dt.total_seconds() / 3600.0
    ranked["outtime_gap_hours"] = ranked["outtime_gap_hours"].fillna(float("inf"))
    ranked["intime_gap_hours"] = ranked["intime_gap_hours"].fillna(float("inf"))
    ranked = ranked.sort_values(
        ["hadm_id", "contains_admittime", "outtime_gap_hours", "intime_gap_hours", "intime"],
        ascending=[True, False, True, True, True],
    )
    ranked = ranked.drop_duplicates(subset=["hadm_id"], keep="first")
    return ranked.drop(columns=["admittime", "contains_admittime", "outtime_gap_hours", "intime_gap_hours"])


def load_base_tables(data_dir: Path) -> pd.DataFrame:
    patients = pd.read_csv(
        data_dir / "patients.csv",
        usecols=["subject_id", "gender", "anchor_age"],
        dtype={"subject_id": "string", "gender": "string", "anchor_age": "Float64"},
    )

    admissions = pd.read_csv(
        data_dir / "admissions.csv",
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "hospital_expire_flag",
            "admission_type",
            "admission_location",
            "insurance",
            "language",
            "marital_status",
            "race",
        ],
        dtype="string",
    )
    admissions["admittime"] = _parse_dt(admissions["admittime"])
    admissions["dischtime"] = _parse_dt(admissions["dischtime"])

    edstays = pd.read_csv(
        data_dir / "edstays.csv",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "arrival_transport"],
        dtype="string",
    )
    edstays["intime"] = _parse_dt(edstays["intime"])
    edstays["outtime"] = _parse_dt(edstays["outtime"])
    edstays = edstays.dropna(subset=["hadm_id", "stay_id"])
    edstays = _select_edstays_for_admissions(edstays, admissions)

    triage = pd.read_csv(
        data_dir / "triage.csv",
        usecols=[
            "stay_id",
            "temperature",
            "heartrate",
            "resprate",
            "o2sat",
            "sbp",
            "dbp",
            "pain",
            "acuity",
            "chiefcomplaint",
        ],
        dtype="string",
    )
    triage = triage.drop_duplicates(subset=["stay_id"], keep="first")

    df = admissions.merge(patients, on="subject_id", how="left")
    df = df.merge(edstays, on=["subject_id", "hadm_id"], how="left")
    df = df.merge(triage, on="stay_id", how="left")
    df["ed_dwell_minutes"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 60.0

    los_days = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400.0
    df["los_days"] = los_days
    df["los_bucket"] = pd.cut(
        los_days,
        bins=[-float("inf"), 3, 7, float("inf")],
        labels=["SHORT", "MEDIUM", "LONG"],
        right=False,
    ).astype("string")
    return df


def aggregate_medrecon(
    data_dir: Path,
    stay_ids: set[str] | None = None,
    window_lookup: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    med = pd.read_csv(
        data_dir / "medrecon.csv",
        usecols=["stay_id", "charttime", "name", "etcdescription"],
        dtype="string",
    )
    med = med.dropna(subset=["stay_id"])
    if stay_ids:
        med = med[med["stay_id"].isin(stay_ids)]
    if window_lookup:
        med["charttime"] = _parse_dt(med["charttime"])
        start_lookup = {stay_id: bounds[0] for stay_id, bounds in window_lookup.items()}
        end_lookup = {stay_id: bounds[1] for stay_id, bounds in window_lookup.items()}
        med["window_start"] = med["stay_id"].map(start_lookup)
        med["window_end"] = med["stay_id"].map(end_lookup)
        med = med[
            med["window_start"].notna()
            & med["window_end"].notna()
            & med["charttime"].notna()
            & (med["charttime"] >= med["window_start"])
            & (med["charttime"] <= med["window_end"])
        ]
    grouped = med.groupby("stay_id", dropna=False).agg(
        med_count=("name", lambda s: int(s.notna().sum())),
        med_names=("name", _join_unique),
        med_categories=("etcdescription", _join_unique),
    )
    return grouped.reset_index()


def aggregate_radiology(
    data_dir: Path,
    admissions_df: pd.DataFrame,
    observation_window_hours: int,
) -> pd.DataFrame:
    windows = admissions_df[["hadm_id", "admittime", "intime"]].dropna(subset=["hadm_id", "admittime"]).copy()
    windows["window_start"] = windows["intime"].where(windows["intime"].notna(), windows["admittime"])
    windows["window_end"] = windows["admittime"] + pd.Timedelta(hours=observation_window_hours)
    window_lookup = {
        str(row.hadm_id): (row.window_start, row.window_end)
        for row in windows.itertuples(index=False)
        if pd.notna(row.hadm_id) and pd.notna(row.window_start) and pd.notna(row.window_end)
    }

    collected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chunks = pd.read_csv(
        data_dir / "radiology.csv",
        usecols=["hadm_id", "charttime", "text"],
        dtype="string",
        chunksize=5000,
    )
    for chunk in chunks:
        chunk = chunk.dropna(subset=["hadm_id", "charttime", "text"])
        chunk["charttime"] = _parse_dt(chunk["charttime"])
        for row in chunk.itertuples(index=False):
            hadm_id = str(row.hadm_id)
            if hadm_id not in window_lookup or pd.isna(row.charttime):
                continue
            start, end = window_lookup[hadm_id]
            if row.charttime < start or row.charttime > end:
                continue
            sections = extract_radiology_sections(str(row.text))
            collected[hadm_id].append(
                {
                    "charttime": row.charttime,
                    "findings": sections["findings"],
                    "impression": sections["impression"],
                }
            )

    rows: list[dict[str, Any]] = []
    for hadm_id, notes in collected.items():
        notes = sorted(notes, key=lambda item: item["charttime"])
        findings = " ".join(note["findings"] for note in notes if note["findings"]).strip()
        impression = " ".join(note["impression"] for note in notes if note["impression"]).strip()
        rows.append(
            {
                "hadm_id": hadm_id,
                "radiology_note_count": len(notes),
                "radiology_findings": findings,
                "radiology_impression": impression,
            }
        )
    return pd.DataFrame(rows)


def _summarize_prior_titles(titles: list[str], limit: int = 6) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for title in titles:
        clean = str(title or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
        if len(ordered) >= limit:
            break
    return "; ".join(ordered)


def aggregate_prior_diagnoses(data_dir: Path, admissions_df: pd.DataFrame) -> pd.DataFrame:
    admissions_lookup = (
        admissions_df[["subject_id", "hadm_id", "admittime"]]
        .dropna(subset=["subject_id", "hadm_id", "admittime"])
        .drop_duplicates(subset=["hadm_id"])
        .copy()
    )
    admissions_lookup["seq"] = range(len(admissions_lookup))

    diagnoses = pd.read_csv(
        data_dir / "diagnoses_icd.csv",
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        dtype="string",
    )
    title_lookup = pd.read_csv(
        data_dir / "d_icd_diagnoses.csv",
        usecols=["icd_code", "icd_version", "long_title"],
        dtype="string",
    )

    diagnoses = diagnoses.merge(title_lookup, on=["icd_code", "icd_version"], how="left")
    diagnoses = diagnoses.merge(
        admissions_lookup.rename(columns={"admittime": "diagnosis_admittime"}),
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    diagnoses["seq_num_int"] = pd.to_numeric(diagnoses["seq_num"], errors="coerce").fillna(9999)
    diagnoses = diagnoses.sort_values(["subject_id", "diagnosis_admittime", "seq_num_int", "long_title"])

    hadm_to_titles: dict[str, list[str]] = {}
    for row in diagnoses.itertuples(index=False):
        hadm = str(row.hadm_id)
        title = str(row.long_title or "").strip()
        if not title:
            continue
        titles = hadm_to_titles.setdefault(hadm, [])
        if title not in titles:
            titles.append(title)

    subject_rows = admissions_lookup.sort_values(["subject_id", "admittime", "seq"])
    prior_map: dict[str, str] = {}
    current_subject = None
    accumulated_titles: list[str] = []
    accumulated_seen: set[str] = set()
    for row in subject_rows.itertuples(index=False):
        subject_id = str(row.subject_id)
        hadm_id = str(row.hadm_id)
        if subject_id != current_subject:
            current_subject = subject_id
            accumulated_titles = []
            accumulated_seen = set()
        prior_map[hadm_id] = _summarize_prior_titles(accumulated_titles)
        for title in hadm_to_titles.get(hadm_id, []):
            if title not in accumulated_seen:
                accumulated_seen.add(title)
                accumulated_titles.append(title)

    rows = [{"hadm_id": hadm_id, "prior_diagnosis_summary": summary} for hadm_id, summary in prior_map.items() if summary]
    return pd.DataFrame(rows)


def _serialize_record(
    row: dict[str, Any],
    llm_mode: str,
    llm_client: OllamaClient | None,
    observation_window_hours: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in dict(row).items():
        if pd.isna(value):
            record[key] = ""
        elif isinstance(value, pd.Timestamp):
            record[key] = value.isoformat(sep=" ")
        else:
            record[key] = value
    template_text = narrative_template(record, observation_window_hours=observation_window_hours)
    record["narrative"] = template_text
    record["narrative_source"] = "deterministic"
    record["narrative_template"] = template_text
    record["narrative_llm"] = ""
    if llm_mode == "gemma" and llm_client is not None:
        prompt = build_gemma_prompt(record, template_text)
        try:
            candidate = llm_client.generate(prompt, temperature=0.1, num_predict=180)
            record["narrative_llm"] = candidate
            verdict_prompt = build_verification_prompt(record, candidate, template_text)
            verdict = llm_client.generate(verdict_prompt, temperature=0.0, num_predict=4).strip().upper()
            if verdict == "SUPPORTED":
                record["narrative"] = candidate
                record["narrative_source"] = "llm_verified"
            else:
                record["narrative_source"] = "deterministic_fallback"
        except Exception as exc:
            record["narrative_llm"] = ""
            record["llm_error"] = str(exc)
            record["narrative_source"] = "deterministic_error_fallback"
    return record


def build_narrative_dataset(
    data_dir: Path,
    output_dir: Path,
    llm_mode: str = "gemma",
    llm_model: str = "gemma3:4b",
    limit: int | None = None,
    csv_preview_width: int = 240,
    observation_window_hours: int = DEFAULT_EARLY_WINDOW_HOURS,
    resume_if_exists: bool = False,
) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_tables(data_dir)
    candidate_rows = int(len(base))
    base = base.dropna(subset=["hadm_id", "admittime", "dischtime", "los_bucket"])
    eligible_rows = int(len(base))
    expire_mask = base["hospital_expire_flag"].fillna("0").astype("string").str.strip().eq("1")
    excluded_hospital_expire_flag = int(expire_mask.sum())
    base = base.loc[~expire_mask].copy()
    base = base.sort_values(["subject_id", "admittime"])
    if limit is not None:
        base = base.head(limit)
    stay_ids = {str(value) for value in base["stay_id"].dropna().tolist()}
    window_lookup = {}
    for row in base[["stay_id", "intime", "admittime"]].dropna(subset=["stay_id", "admittime"]).itertuples(index=False):
        window_start = row.intime if pd.notna(row.intime) else row.admittime
        window_end = row.admittime + pd.Timedelta(hours=observation_window_hours)
        window_lookup[str(row.stay_id)] = (window_start, window_end)
    med = aggregate_medrecon(data_dir, stay_ids=stay_ids, window_lookup=window_lookup)
    rad = aggregate_radiology(data_dir, base, observation_window_hours=observation_window_hours)
    prior_dx = aggregate_prior_diagnoses(data_dir, base)

    df = base.merge(med, on="stay_id", how="left")
    df = df.merge(rad, on="hadm_id", how="left")
    df = df.merge(prior_dx, on="hadm_id", how="left")
    df = df.drop(columns=["hospital_expire_flag", "disposition"], errors="ignore")
    df["med_count"] = df["med_count"].fillna(0).astype(int)
    for column in [
        "med_names",
        "med_categories",
        "prior_diagnosis_summary",
        "radiology_findings",
        "radiology_impression",
        "chiefcomplaint",
        "arrival_transport",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
        "race",
    ]:
        if column in df.columns:
            df[column] = df[column].fillna("")
    llm_client = OllamaClient(model=llm_model) if llm_mode == "gemma" else None

    jsonl_path = output_dir / "narrative_dataset.jsonl"
    csv_path = output_dir / "narrative_dataset.csv"
    summary_path = output_dir / "run_summary.json"

    csv_fieldnames = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "admittime",
        "los_days",
        "los_bucket",
        "narrative_source",
        "chiefcomplaint",
        "prior_diagnosis_summary",
        "radiology_findings",
        "radiology_impression",
        "narrative",
        "narrative_template",
        "narrative_llm",
    ]
    resumed_from_records = 0
    narrative_source_counts: dict[str, int] = defaultdict(int)
    if resume_if_exists and jsonl_path.exists():
        resumed_from_records, recovered_counts = _recover_existing_jsonl(jsonl_path)
        narrative_source_counts.update(recovered_counts)
        _rewrite_csv_from_jsonl(jsonl_path, csv_path, csv_fieldnames, csv_preview_width)
    records_written = resumed_from_records
    started_at = time.perf_counter()
    rows_to_write = df.to_dict(orient="records")
    if resumed_from_records:
        rows_to_write = rows_to_write[resumed_from_records:]

    with (
        jsonl_path.open("a" if resumed_from_records else "w", encoding="utf-8") as jsonl_file,
        csv_path.open("a" if resumed_from_records else "w", encoding="utf-8", newline="") as csv_file,
    ):
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames, quoting=csv.QUOTE_MINIMAL)
        if not resumed_from_records:
            csv_writer.writeheader()
        for row in rows_to_write:
            record = _serialize_record(
                row,
                llm_mode=llm_mode,
                llm_client=llm_client,
                observation_window_hours=observation_window_hours,
            )
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            csv_writer.writerow(_csv_preview_row(record, csv_preview_width))
            records_written += 1
            narrative_source = str(record.get("narrative_source", "") or "unknown")
            narrative_source_counts[narrative_source] += 1
            if records_written % 1000 == 0:
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                rate_per_minute = (records_written / elapsed) * 60.0
                print(
                    f"[progress] records_written={records_written} rate_per_min={rate_per_minute:.2f} "
                    f"llm_verified={narrative_source_counts.get('llm_verified', 0)} "
                    f"fallback={narrative_source_counts.get('deterministic_fallback', 0) + narrative_source_counts.get('deterministic_error_fallback', 0)}"
                )
                jsonl_file.flush()
                csv_file.flush()

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "candidate_rows": candidate_rows,
        "eligible_rows_after_required_fields": eligible_rows,
        "excluded_hospital_expire_flag": excluded_hospital_expire_flag,
        "observation_window_hours": observation_window_hours,
        "window_start_basis": "ed_intime_if_available_else_admittime",
        "resumed_from_records": resumed_from_records,
        "records_written": records_written,
        "narrative_source_counts": dict(narrative_source_counts),
        "llm_mode": llm_mode,
        "llm_model": llm_model if llm_mode == "gemma" else None,
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
