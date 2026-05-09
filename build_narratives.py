from pathlib import Path
import argparse
import json

from src.mimic_los.narrative_dataset import build_narrative_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a narrative LOS dataset from local MIMIC CSVs.")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing the CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for output files.")
    parser.add_argument(
        "--llm-mode",
        choices=["off", "gemma"],
        default="gemma",
        help="Use Gemma via local Ollama to rewrite deterministic narratives.",
    )
    parser.add_argument("--llm-model", default="gemma3:4b", help="Ollama model name.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of admissions to export.")
    parser.add_argument(
        "--csv-preview-width",
        type=int,
        default=240,
        help="Max characters to keep in long text fields for the CSV companion file.",
    )
    parser.add_argument(
        "--early-window-hours",
        type=int,
        default=24,
        help="Use data from ED arrival if available through this many hours after admission.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing JSONL output without regenerating completed rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_narrative_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        llm_mode=args.llm_mode,
        llm_model=args.llm_model,
        limit=args.limit,
        csv_preview_width=args.csv_preview_width,
        observation_window_hours=args.early_window_hours,
        resume_if_exists=args.resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
