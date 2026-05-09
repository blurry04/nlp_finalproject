"""Run Tier 2A inference for one patient record.

This command predicts against rows already materialized by the project dataset
builder. For a completely new admission, first create the same 24-hour patient
snapshot/narrative row, then point ``--source-jsonl`` at that file.

Example:
    python -m src.mimic_los.predict_patient --hadm-id 28861371 --write-context
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from .constants import CATEGORICAL_FEATURES, LABELS, NUMERIC_FEATURES, RAW_TEXT_FIELDS
from .explanation_engine import ACTION_BUNDLES
from .tier2_training import HeadTailCollator, LateFusionClinicalBert, Tier2Config
from .training_data import build_tier2a_text, iter_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "outputs_15k_gemma_24h_prior_dx" / "narrative_dataset.jsonl"
DEFAULT_RUN_DIR = ROOT / "tier_runs_15k_gemma_24h_prior_dx" / "seed_62"
DEFAULT_CONTEXT_DIR = ROOT / "outputs_15k_gemma_24h_prior_dx" / "explanations"


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


def _ids_match(record: dict[str, Any], args: argparse.Namespace) -> bool:
    checks = []
    if args.hadm_id:
        checks.append(_text(record.get("hadm_id")) == str(args.hadm_id))
    if args.subject_id:
        checks.append(_text(record.get("subject_id")) == str(args.subject_id))
    if args.stay_id:
        checks.append(_text(record.get("stay_id")) == str(args.stay_id))
    return bool(checks) and all(checks)


def _find_record(source_jsonl: Path, args: argparse.Namespace) -> dict[str, Any]:
    for record in iter_jsonl(source_jsonl):
        if _ids_match(record, args):
            return record
    identifiers = {
        "hadm_id": args.hadm_id,
        "subject_id": args.subject_id,
        "stay_id": args.stay_id,
    }
    raise SystemExit(
        "No matching row found in "
        f"{source_jsonl}. Identifiers: {json.dumps({k: v for k, v in identifiers.items() if v})}"
    )


def _inference_record(record: dict[str, Any], preprocessor: Any) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "subject_id": _text(record.get("subject_id")),
        "hadm_id": _text(record.get("hadm_id")),
        "stay_id": _text(record.get("stay_id")),
        "los_bucket": _text(record.get("los_bucket")) or "UNKNOWN",
        "tier2a_text": build_tier2a_text(record),
        "narrative": _text(record.get("narrative")),
    }
    for feature in NUMERIC_FEATURES:
        compact[feature] = _number(record.get(feature))
    for feature in CATEGORICAL_FEATURES:
        compact[feature] = _text(record.get(feature))
    for feature in RAW_TEXT_FIELDS:
        compact[feature] = _text(record.get(feature))

    frame = pd.DataFrame([{feature: compact.get(feature) for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES}])
    structured = preprocessor.transform(frame)
    if hasattr(structured, "toarray"):
        structured = structured.toarray()
    compact["structured_vector"] = np.asarray(structured, dtype=np.float32)[0].tolist()
    compact["label_id"] = 0
    compact["text"] = compact["tier2a_text"]
    return compact


def _load_result_config(result_path: Path) -> Tier2Config:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return Tier2Config(**result["best_result"]["config"])


def _load_shap_features(run_dir: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    target = (
        _text(record.get("subject_id")),
        _text(record.get("hadm_id")),
        _text(record.get("stay_id")),
    )
    tier0_dir = run_dir / "results" / "tier0"
    for split in ("test", "val", "train"):
        path = tier0_dir / f"predictions_{split}.jsonl"
        if not path.exists():
            continue
        for prediction in iter_jsonl(path):
            key = (
                _text(prediction.get("subject_id")),
                _text(prediction.get("hadm_id")),
                _text(prediction.get("stay_id")),
            )
            if key == target:
                return list(prediction.get("shap_top_features", []))
    return []


def predict_record(record: dict[str, Any], run_dir: Path, device_name: str = "auto") -> dict[str, Any]:
    prepared_dir = run_dir / "prepared_data"
    result_dir = run_dir / "results" / "tier2a"
    metadata = json.loads((prepared_dir / "metadata.json").read_text(encoding="utf-8"))
    preprocessor = joblib.load(prepared_dir / "structured_preprocessor.joblib")
    result_path = result_dir / "tier2a_late_fusion_results.json"
    config = _load_result_config(result_path)
    model_name = json.loads(result_path.read_text(encoding="utf-8"))["model_name"]
    tokenizer_dir = result_dir / "tier2a_late_fusion_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir if tokenizer_dir.exists() else model_name)

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    prepared = _inference_record(record, preprocessor)
    collator = HeadTailCollator(tokenizer=tokenizer, max_length=512, with_structured=True)
    batch = collator([prepared])
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
        if key not in {"labels", "subject_ids", "hadm_ids", "stay_ids"}
    }

    model = LateFusionClinicalBert(
        model_name=model_name,
        structured_dim=int(metadata["structured_dim"]),
        num_labels=len(LABELS),
        dropout=config.dropout,
        freeze_layers=config.freeze_layers,
    ).to(device)
    state = torch.load(result_dir / "tier2a_late_fusion_best.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(**batch)
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

    pred_idx = int(np.argmax(probabilities))
    prediction = LABELS[pred_idx]
    return {
        "hadm_id": prepared["hadm_id"],
        "subject_id": prepared["subject_id"],
        "stay_id": prepared["stay_id"],
        "prediction": prediction,
        "actual": prepared.get("los_bucket", "UNKNOWN"),
        "correct": prepared.get("los_bucket") == prediction,
        "confidence_raw": float(probabilities[pred_idx]),
        "calibrated_confidence": float(probabilities[pred_idx]),
        "probabilities": {label: float(probabilities[idx]) for idx, label in enumerate(LABELS)},
        "narrative": prepared.get("narrative", ""),
        "shap_top_features": _load_shap_features(run_dir, record),
        "recommended_actions": ACTION_BUNDLES.get(prediction, {}),
        "discharge_summary_posthoc": None,
        "source_tier": "tier2a_live_inference",
        "model_artifact": str((result_dir / "tier2a_late_fusion_best.pt").resolve()),
        "source_record": {
            "narrative_source": record.get("narrative_source", ""),
            "llm_model": record.get("llm_model", ""),
        },
    }


def _write_context(context: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"patient_{context['hadm_id']}.json"
    path.write_text(json.dumps(context, indent=2), encoding="utf-8")

    index_path = output_dir / "patient_index.json"
    index: list[dict[str, Any]] = []
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    index = [row for row in index if str(row.get("hadm_id")) != str(context["hadm_id"])]
    index.insert(
        0,
        {
            "hadm_id": context["hadm_id"],
            "prediction": context["prediction"],
            "actual": context["actual"],
            "correct": context["correct"],
            "source_tier": context["source_tier"],
        },
    )
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict LOS for one patient using the trained Tier 2A model.")
    parser.add_argument("--hadm-id", help="Hospital admission ID to predict.")
    parser.add_argument("--subject-id", help="Subject ID to match. Can be combined with hadm/stay ID.")
    parser.add_argument("--stay-id", help="ED stay ID to match. Can be combined with hadm/subject ID.")
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE, help="Dataset JSONL containing patient rows.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Trained seed run directory.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--write-context", action="store_true", help="Write prediction into dashboard explanations folder.")
    parser.add_argument("--context-dir", type=Path, default=DEFAULT_CONTEXT_DIR, help="Dashboard explanation context folder.")
    parser.add_argument("--output-json", type=Path, help="Optional path for the full prediction JSON.")
    args = parser.parse_args()
    if not any([args.hadm_id, args.subject_id, args.stay_id]):
        parser.error("Provide at least one identifier: --hadm-id, --subject-id, or --stay-id.")
    return args


def main() -> None:
    args = parse_args()
    record = _find_record(args.source_jsonl.resolve(), args)
    prediction = predict_record(record, args.run_dir.resolve(), device_name=args.device)

    if args.write_context:
        prediction["dashboard_context_path"] = str(_write_context(prediction, args.context_dir.resolve()))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(prediction, indent=2), encoding="utf-8")

    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
