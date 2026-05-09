from __future__ import annotations

import re
from typing import Any

from .constants import DEFAULT_EARLY_WINDOW_HOURS


SECTION_PATTERN = re.compile(
    r"(?P<header>FINDINGS|IMPRESSION|IMPRESSIONS)\s*:\s*(?P<body>.*?)(?=(?:\n[A-Z][A-Z\s/]{2,30}\s*:)|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def compact_text(text: str) -> str:
    return normalize_whitespace(text).replace("\n", " ")


def _display_value(value: Any, decimals: int = 1) -> str:
    if value in ("", None):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return f"{number:.{decimals}f}".rstrip("0").rstrip(".")


def extract_radiology_sections(note_text: str) -> dict[str, str]:
    matches = {m.group("header").upper(): compact_text(m.group("body")) for m in SECTION_PATTERN.finditer(note_text or "")}
    findings = matches.get("FINDINGS", "")
    impression = matches.get("IMPRESSION", "") or matches.get("IMPRESSIONS", "")
    return {"findings": findings, "impression": impression}


def summarize_medications(med_names: str, med_categories: str, med_count: int) -> str:
    if med_count <= 0:
        return "No home medications were documented at arrival."
    parts = [f"{med_count} home medications were documented at arrival"]
    if med_names:
        parts.append(f"including {med_names}")
    if med_categories:
        parts.append(f"with categories such as {med_categories}")
    return ", ".join(parts) + "."


def structured_highlight_sentence(highlights: list[dict[str, Any]]) -> str:
    phrases = [str(item.get("evidence_text", "")).strip() for item in highlights if str(item.get("evidence_text", "")).strip()]
    if not phrases:
        return ""
    return "The highest-signal structured admission features were " + "; ".join(phrases) + "."


def _has_text(value: Any) -> bool:
    return bool(str(value or "").strip())


def _join_phrases(phrases: list[str]) -> str:
    phrases = [phrase.strip() for phrase in phrases if phrase and phrase.strip()]
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def _is_sparse_record(row: dict[str, Any]) -> bool:
    signal_count = 0
    if _has_text(row.get("chiefcomplaint")):
        signal_count += 1
    if _has_text(row.get("radiology_findings")) or _has_text(row.get("radiology_impression")):
        signal_count += 1
    if int(row.get("med_count", 0) or 0) > 0:
        signal_count += 1
    vitals_present = sum(
        1
        for key in ("acuity", "heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature", "pain")
        if _has_text(row.get(key))
    )
    if vitals_present >= 2:
        signal_count += 1
    return signal_count <= 1


def narrative_template(
    row: dict[str, Any],
    structured_highlights: list[dict[str, Any]] | None = None,
    observation_window_hours: int = DEFAULT_EARLY_WINDOW_HOURS,
) -> str:
    pieces: list[str] = []
    sparse_record = _is_sparse_record(row)
    age = _display_value(row.get("anchor_age"), decimals=0)
    gender = row.get("gender")
    arrival_transport = row.get("arrival_transport")
    chief_complaint = row.get("chiefcomplaint")

    intro = ["A"]
    if age:
        intro.append(f"{age}-year-old")
    if gender:
        intro.append("female" if str(gender).upper().startswith("F") else "male")
    intro.append("patient")
    if arrival_transport:
        intro.append(f"arrived via {str(arrival_transport).lower()}")
    if chief_complaint:
        intro.append(f"with chief complaint of {chief_complaint}")
    pieces.append(" ".join(intro).strip().rstrip(".") + ".")

    acuity = _display_value(row.get("acuity"), decimals=0)
    vitals = []
    for label, key, decimals in [
        ("HR", "heartrate", 0),
        ("RR", "resprate", 0),
        ("O2 sat", "o2sat", 0),
        ("SBP", "sbp", 0),
        ("DBP", "dbp", 0),
        ("temperature", "temperature", 1),
        ("pain score", "pain", 0),
    ]:
        value = _display_value(row.get(key), decimals=decimals)
        if value:
            vitals.append(f"{label} {value}")
    if acuity and vitals:
        pieces.append(f"Triage acuity was {acuity}; recorded vitals included " + ", ".join(vitals) + ".")
    elif acuity:
        pieces.append(f"Triage acuity was {acuity}.")
    elif vitals:
        pieces.append("Recorded vitals included " + ", ".join(vitals) + ".")

    if sparse_record:
        admin_context: list[str] = []
        admission_type = str(row.get("admission_type", "") or "").strip()
        admission_location = str(row.get("admission_location", "") or "").strip()
        insurance = str(row.get("insurance", "") or "").strip()
        language = str(row.get("language", "") or "").strip()
        marital_status = str(row.get("marital_status", "") or "").strip()
        race = str(row.get("race", "") or "").strip()
        ed_dwell_minutes = _display_value(row.get("ed_dwell_minutes"), decimals=0)

        if admission_type or admission_location:
            admission_bits: list[str] = []
            if admission_type:
                admission_bits.append(f"admission type {admission_type}")
            if admission_location:
                admission_bits.append(f"from {admission_location.lower()}")
            admin_context.append("Administrative intake identified " + " ".join(admission_bits))
        if ed_dwell_minutes:
            admin_context.append(f"recorded ED dwell time was {ed_dwell_minutes} minutes")
        background_bits = _join_phrases(
            [
                f"insurance {insurance}" if insurance else "",
                f"language {language}" if language else "",
                f"marital status {marital_status.lower()}" if marital_status else "",
                f"race {race.lower()}" if race else "",
            ]
        )
        if background_bits:
            admin_context.append("Background registration fields listed " + background_bits)
        if admin_context:
            pieces.append(". ".join(admin_context) + ".")

    prior_diagnosis_summary = str(row.get("prior_diagnosis_summary", "") or "").strip()
    if prior_diagnosis_summary:
        pieces.append("Prior admission history documented diagnoses such as " + prior_diagnosis_summary + ".")

    highlight_sentence = structured_highlight_sentence(structured_highlights or [])
    if highlight_sentence:
        pieces.append(highlight_sentence)

    med_sentence = summarize_medications(
        med_names=str(row.get("med_names", "") or ""),
        med_categories=str(row.get("med_categories", "") or ""),
        med_count=int(row.get("med_count", 0) or 0),
    )
    pieces.append(med_sentence)

    note_count = _display_value(row.get("radiology_note_count"), decimals=0)
    if row.get("radiology_findings"):
        prefix = f"Radiology from arrival through the first {observation_window_hours} hours described"
        if note_count:
            prefix = f"{prefix} findings from {note_count} note(s)"
        pieces.append(f"{prefix}: {row['radiology_findings']}.")
    if row.get("radiology_impression"):
        pieces.append(f"Radiology impression: {row['radiology_impression']}.")

    if sparse_record:
        missing_sources: list[str] = []
        if not _has_text(row.get("stay_id")):
            missing_sources.append("a linked ED stay")
        vitals_present = any(_has_text(row.get(key)) for key in ("acuity", "heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature", "pain"))
        if not vitals_present and not _has_text(row.get("chiefcomplaint")):
            missing_sources.append("triage details")
        if int(row.get("med_count", 0) or 0) <= 0:
            missing_sources.append("home medication reconciliation")
        if not (_has_text(row.get("radiology_findings")) or _has_text(row.get("radiology_impression"))):
            missing_sources.append(f"radiology text within the first {observation_window_hours} hours")
        if missing_sources:
            pieces.append("Source tables did not include " + _join_phrases(missing_sources) + " for this admission.")

    return compact_text(" ".join(pieces))


def tier2_handoff_narrative(
    row: dict[str, Any],
    structured_highlights: list[dict[str, Any]],
    observation_window_hours: int = DEFAULT_EARLY_WINDOW_HOURS,
) -> str:
    return narrative_template(
        row,
        structured_highlights=structured_highlights,
        observation_window_hours=observation_window_hours,
    )


def build_gemma_prompt(row: dict[str, Any], template_text: str) -> str:
    sparse_clause = ""
    if _is_sparse_record(row):
        sparse_clause = """
- This is a sparse record. Keep it to 1-2 short sentences.
- Do not add generic filler such as "presented for evaluation", "assessment is pending", "further workup is pending", or "being assessed".
- If the draft is already minimal and factual, keep it almost unchanged.
""".strip()
    prompt = f"""
You are polishing a deterministic clinical intake narrative for machine learning input.

Rules:
- Use only information already present in the draft.
- Do not invent diagnoses, timelines, labs, treatments, or outcomes.
- Keep it to 3-5 sentences.
- Keep the tone clinical and compact.
- Preserve uncertainty exactly as written when the source text is uncertain.
- Do not mention length of stay or discharge.
- If you are not fully confident, keep the draft almost unchanged.
{sparse_clause}

Deterministic draft:
{template_text}

Return only the final rewritten narrative.
""".strip()
    return prompt


def build_verification_prompt(row: dict[str, Any], candidate_text: str, template_text: str) -> str:
    prompt = f"""
You are a strict factual verifier for a clinical narrative rewrite.

Check whether the candidate narrative is fully supported by the deterministic reference.

Rules:
- Mark UNSUPPORTED if the candidate adds any diagnosis, treatment, timeline, severity claim, or outcome not explicitly present in the reference.
- Mark UNSUPPORTED if it introduces speculation or strengthens a claim beyond the reference.
- Mark SUPPORTED only if every claim is grounded in the reference.

Deterministic reference:
{template_text}

Candidate:
{candidate_text}

Reply with exactly one token:
SUPPORTED
or
UNSUPPORTED
""".strip()
    return prompt


def safe_preview(text: str, width: int) -> str:
    text = compact_text(text)
    if len(text) <= width:
        return text
    return text[: width - 3].rstrip() + "..."
