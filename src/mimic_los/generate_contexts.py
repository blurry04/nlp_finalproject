"""Generate explanation-ready patient JSON files from real model artifacts.

Usage:
    python -m src.mimic_los.generate_contexts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import LABELS
from .explanation_engine import ACTION_BUNDLES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "outputs_15k_gemma_24h_prior_dx" / "narrative_dataset.jsonl"
DEFAULT_WORK_DIR = ROOT / "tier_runs_15k_gemma_24h_prior_dx"
DEFAULT_OUTPUT = ROOT / "outputs_15k_gemma_24h_prior_dx" / "explanations"
TIER_PRIORITY = ["tier2a", "tier2b", "tier1", "tier0"]


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("subject_id", "")).strip(),
        str(record.get("hadm_id", "")).strip(),
        str(record.get("stay_id", "")).strip(),
    )


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _load_dataset_map(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {_record_key(record): record for record in _iter_jsonl(path)}


def _clear_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.glob("patient_*.json"):
        path.unlink()
    index_path = output_dir / "patient_index.json"
    if index_path.exists():
        index_path.unlink()


def _seed_dirs(work_dir: Path) -> list[Path]:
    if not work_dir.exists():
        return []
    seed_dirs = sorted(path for path in work_dir.iterdir() if path.is_dir() and path.name.startswith("seed_"))
    if seed_dirs:
        return seed_dirs
    if (work_dir / "results").exists():
        return [work_dir]
    return []


def _prediction_path(seed_dir: Path, tier: str) -> Path:
    return seed_dir / "results" / tier / "predictions_test.jsonl"


def _result_path(seed_dir: Path, tier: str) -> Path:
    filenames = {
        "tier0": "tier0_results.json",
        "tier1": "tier1_results.json",
        "tier2a": "tier2a_late_fusion_results.json",
        "tier2b": "tier2b_early_fusion_results.json",
    }
    return seed_dir / "results" / tier / filenames[tier]


def _tier_metric(seed_dir: Path, tier: str) -> float:
    path = _result_path(seed_dir, tier)
    if not path.exists():
        return -1.0
    data = json.loads(path.read_text(encoding="utf-8"))
    if tier in {"tier0", "tier1"}:
        return float(data.get("test_metrics", {}).get("macro_f1", -1.0))
    return float(data.get("best_result", {}).get("test_metrics", {}).get("macro_f1", -1.0))


def _pick_best_artifact_bundle(work_dir: Path) -> tuple[Path, str, list[dict[str, Any]]] | None:
    best_rank: tuple[int, float, str, str] | None = None
    best_choice: tuple[Path, str, list[dict[str, Any]]] | None = None
    for seed_dir in _seed_dirs(work_dir):
        for priority, tier in enumerate(TIER_PRIORITY):
            path = _prediction_path(seed_dir, tier)
            if not path.exists():
                continue
            rows = _load_jsonl(path)
            if not rows:
                continue
            metric = _tier_metric(seed_dir, tier)
            rank = (-priority, metric, seed_dir.name, tier)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_choice = (seed_dir, tier, rows)
    if best_choice is None:
        return None
    return best_choice


def _load_handoff_narratives(seed_dir: Path) -> dict[tuple[str, str, str], str]:
    handoff_path = seed_dir / "handoff_data" / "test.jsonl"
    if not handoff_path.exists():
        return {}
    narratives: dict[tuple[str, str, str], str] = {}
    for record in _iter_jsonl(handoff_path):
        narratives[_record_key(record)] = str(record.get("tier2_narrative", "") or "")
    return narratives


def _load_tier0_shap(seed_dir: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    path = _prediction_path(seed_dir, "tier0")
    if not path.exists():
        return {}
    return {
        _record_key(record): list(record.get("shap_top_features", []))
        for record in _iter_jsonl(path)
    }


def _context_from_prediction(
    prediction: dict[str, Any],
    dataset_map: dict[tuple[str, str, str], dict[str, Any]],
    shap_map: dict[tuple[str, str, str], list[dict[str, Any]]],
    handoff_narratives: dict[tuple[str, str, str], str],
    selected_tier: str,
) -> dict[str, Any]:
    key = _record_key(prediction)
    dataset_row = dataset_map.get(key, {})
    narrative = str(dataset_row.get("narrative", "") or "")
    if selected_tier == "tier2b":
        narrative = handoff_narratives.get(key, narrative)
    confidence = float(prediction.get("confidence_raw", 0.0) or 0.0)
    shap_top_features = list(prediction.get("shap_top_features", [])) or shap_map.get(key, [])
    return {
        "hadm_id": int(float(prediction.get("hadm_id", 0) or 0)),
        "subject_id": str(prediction.get("subject_id", "")),
        "narrative": narrative,
        "prediction": str(prediction.get("predicted", "")),
        "actual": str(prediction.get("actual", "")),
        "correct": bool(prediction.get("correct", False)),
        "confidence_raw": confidence,
        "calibrated_confidence": confidence,
        "probabilities": dict(prediction.get("probabilities", {})),
        "shap_top_features": shap_top_features,
        "recommended_actions": ACTION_BUNDLES.get(str(prediction.get("predicted", "")), {}),
        "discharge_summary_posthoc": None,
        "source_tier": selected_tier,
    }


def _pick_representative_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for label in LABELS:
        candidates = [item for item in contexts if item["prediction"] == label and item["actual"] == label and item["correct"]]
        if not candidates:
            continue
        chosen = max(candidates, key=lambda item: float(item.get("confidence_raw", 0.0)))
        selected.append(chosen)

    wrong = [item for item in contexts if not item.get("correct")]
    wrong = sorted(wrong, key=lambda item: float(item.get("confidence_raw", 0.0)), reverse=True)
    for item in wrong[:2]:
        if item["hadm_id"] not in {row["hadm_id"] for row in selected}:
            selected.append(item)
    return selected


def generate(
    dataset_path: Path | None = None,
    work_dir: Path | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    dataset_path = (dataset_path or DEFAULT_DATASET).resolve()
    work_dir = (work_dir or DEFAULT_WORK_DIR).resolve()
    output_dir = (output_dir or DEFAULT_OUTPUT).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_output_dir(output_dir)

    artifact_bundle = _pick_best_artifact_bundle(work_dir)
    if artifact_bundle is None or not dataset_path.exists():
        return []

    seed_dir, selected_tier, predictions = artifact_bundle
    dataset_map = _load_dataset_map(dataset_path)
    shap_map = _load_tier0_shap(seed_dir)
    handoff_narratives = _load_handoff_narratives(seed_dir)
    contexts = [
        _context_from_prediction(
            prediction=prediction,
            dataset_map=dataset_map,
            shap_map=shap_map,
            handoff_narratives=handoff_narratives,
            selected_tier=selected_tier,
        )
        for prediction in predictions
    ]
    contexts = _pick_representative_contexts(contexts)

    for context in contexts:
        path = output_dir / f"patient_{context['hadm_id']}.json"
        path.write_text(json.dumps(context, indent=2), encoding="utf-8")

    index = [
        {
            "hadm_id": item["hadm_id"],
            "prediction": item["prediction"],
            "actual": item["actual"],
            "correct": item["correct"],
            "source_tier": item["source_tier"],
        }
        for item in contexts
    ]
    (output_dir / "patient_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return contexts


if __name__ == "__main__":
    generated = generate()
    print(f"Generated {len(generated)} explanation contexts in {DEFAULT_OUTPUT}")
