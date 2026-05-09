from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LOS pipeline through Tier 2.")
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=Path("outputs/narrative_dataset.jsonl"),
        help="Canonical narrative dataset JSONL.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("tier_runs"),
        help="Directory for prepared splits, checkpoints, and reports.",
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "tier0", "handoff", "tier1", "tier2a", "tier2b", "tier2", "all"],
        default="all",
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--seeds",
        default="42,52,62",
        help="Comma-separated random seeds for the experimental protocol.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional single-seed override for quick experiments.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for split preparation.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for Tier 2 train samples.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional cap for Tier 2 validation and test samples.")
    parser.add_argument(
        "--clinicalbert-model",
        default="emilyalsentzer/Bio_ClinicalBERT",
        help="Hugging Face model name for Tier 2A and Tier 2B.",
    )
    return parser.parse_args()


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seed is not None:
        return [args.seed]
    seeds = [int(chunk.strip()) for chunk in str(args.seeds).split(",") if chunk.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _metric_value(summary: dict[str, Any], path: str) -> float | None:
    current: Any = summary
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    return float(current)


def _aggregate_metrics(per_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    metric_paths = {
        "tier0": {
            "macro_f1": "tier0.test_metrics.macro_f1",
            "weighted_f1": "tier0.test_metrics.weighted_f1",
            "weighted_ovr_auroc": "tier0.test_metrics.weighted_ovr_auroc",
        },
        "tier1": {
            "macro_f1": "tier1.test_metrics.macro_f1",
            "weighted_f1": "tier1.test_metrics.weighted_f1",
            "weighted_ovr_auroc": "tier1.test_metrics.weighted_ovr_auroc",
        },
        "tier2a": {
            "macro_f1": "tier2a.best_result.test_metrics.macro_f1",
            "weighted_f1": "tier2a.best_result.test_metrics.weighted_f1",
            "weighted_ovr_auroc": "tier2a.best_result.test_metrics.weighted_ovr_auroc",
        },
        "tier2b": {
            "macro_f1": "tier2b.best_result.test_metrics.macro_f1",
            "weighted_f1": "tier2b.best_result.test_metrics.weighted_f1",
            "weighted_ovr_auroc": "tier2b.best_result.test_metrics.weighted_ovr_auroc",
        },
    }
    aggregate: dict[str, Any] = {}
    for tier_name, tier_metrics in metric_paths.items():
        bucket: dict[str, Any] = {}
        for metric_name, path in tier_metrics.items():
            values = [value for value in (_metric_value(summary, path) for summary in per_seed.values()) if value is not None]
            if not values:
                continue
            bucket[metric_name] = {
                "values": values,
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
            }
        if bucket:
            aggregate[tier_name] = bucket
    return aggregate


def _run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    work_dir = args.work_dir.resolve()
    seed_dir = work_dir / f"seed_{seed}"
    prepared_dir = seed_dir / "prepared_data"
    handoff_dir = seed_dir / "handoff_data"
    results_dir = seed_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"seed": seed}
    _log(f"seed={seed}: starting stage={args.stage} in {seed_dir}")

    if args.stage in {"prepare", "all"}:
        from src.mimic_los.training_data import prepare_training_data

        _log(f"seed={seed}: prepare started")
        summary["prepare"] = prepare_training_data(
            source_jsonl=args.source_jsonl.resolve(),
            output_dir=prepared_dir,
            seed=seed,
            limit=args.limit,
        )
        _log(f"seed={seed}: prepare completed")

    if args.stage in {"tier0", "all"}:
        if not (prepared_dir / "metadata.json").exists():
            raise FileNotFoundError("Prepared data not found. Run with --stage prepare or --stage all first.")
        from src.mimic_los.baseline_training import train_tier0_structured

        _log(f"seed={seed}: tier0 started")
        summary["tier0"] = train_tier0_structured(
            prepared_dir=prepared_dir,
            output_dir=results_dir / "tier0",
            seed=seed,
        )
        _log(f"seed={seed}: tier0 completed")

    if args.stage in {"handoff", "all"}:
        if not (prepared_dir / "metadata.json").exists():
            raise FileNotFoundError("Prepared data not found. Run with --stage prepare or --stage all first.")
        if not (results_dir / "tier0" / "tier0_results.json").exists():
            raise FileNotFoundError("Tier 0 results not found. Run with --stage tier0 or --stage all first.")
        from src.mimic_los.baseline_training import build_tier2_handoff

        _log(f"seed={seed}: handoff started")
        summary["handoff"] = build_tier2_handoff(
            prepared_dir=prepared_dir,
            tier0_dir=results_dir / "tier0",
            output_dir=handoff_dir,
        )
        _log(f"seed={seed}: handoff completed")

    if args.stage in {"tier1", "all"}:
        if not (prepared_dir / "metadata.json").exists():
            raise FileNotFoundError("Prepared data not found. Run with --stage prepare or --stage all first.")
        from src.mimic_los.baseline_training import train_tier1_sparse_text

        _log(f"seed={seed}: tier1 started")
        summary["tier1"] = train_tier1_sparse_text(
            prepared_dir=prepared_dir,
            output_dir=results_dir / "tier1",
            seed=seed,
        )
        _log(f"seed={seed}: tier1 completed")

    if args.stage in {"tier2", "tier2a", "tier2b", "all"}:
        if not (prepared_dir / "metadata.json").exists():
            raise FileNotFoundError("Prepared data not found. Run with --stage prepare or --stage all first.")
        from src.mimic_los.tier2_training import train_tier2a_late_fusion, train_tier2b_early_fusion

        if args.stage in {"tier2", "tier2a", "all"}:
            _log(f"seed={seed}: tier2a started")
            summary["tier2a"] = train_tier2a_late_fusion(
                prepared_dir=prepared_dir,
                output_dir=results_dir / "tier2a",
                seed=seed,
                model_name=args.clinicalbert_model,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
            )
            _log(f"seed={seed}: tier2a completed")
        if args.stage in {"tier2", "tier2b", "all"}:
            if not (handoff_dir / "metadata.json").exists():
                raise FileNotFoundError("Tier 2 handoff data not found. Run with --stage handoff or --stage all first.")
            _log(f"seed={seed}: tier2b started")
            summary["tier2b"] = train_tier2b_early_fusion(
                prepared_dir=handoff_dir,
                output_dir=results_dir / "tier2b",
                seed=seed,
                model_name=args.clinicalbert_model,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
            )
            _log(f"seed={seed}: tier2b completed")

    summary_path = results_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"seed={seed}: wrote summary to {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args)
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    _log(
        f"run started stage={args.stage} source={args.source_jsonl.resolve()} "
        f"work_dir={work_dir} seeds={seeds}"
    )

    per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        per_seed[seed] = _run_seed(args, seed)

    aggregate = {
        "stage": args.stage,
        "source_jsonl": str(args.source_jsonl.resolve()),
        "work_dir": str(work_dir),
        "seeds": seeds,
        "aggregate_metrics": _aggregate_metrics(per_seed),
        "seed_summary_paths": {
            str(seed): str((work_dir / f"seed_{seed}" / "results" / "run_summary.json").resolve())
            for seed in seeds
        },
    }
    aggregate_path = work_dir / "aggregate_summary.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    _log(f"run completed; wrote aggregate summary to {aggregate_path}")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
