# MIMIC LOS Narrative Builder

This workspace includes:

- a leakage-aware dataset builder that joins the local MIMIC-style CSV files and writes a physical narrative training file
- a Tier 0 to Tier 2 code pipeline scaffold with an explicit Tier 0 to Tier 2B handoff
- a UI-first explainability demo in Streamlit

Training code is present, but no training artifacts are kept in the repo right now.

Primary output:

- `JSONL` for model training and large-text safety
- `CSV` companion for quick inspection in Excel / notebooks

Current scope stops at the Tier 2 pipeline structure and the explainability UI:

- structured features
- chief complaint / medication / radiology text
- deterministic narrative template
- optional local LLM rewrite using Ollama `gemma3:4b`
- Tier 0 `XGBoost + SHAP-style contribution export + top-K structured feature handoff`
- explanation context generation from real model artifacts only
- Streamlit demo UI

# Submission Package Contents

This submission-ready package contains only the core project materials needed for review.

Included:
- Streamlit application (`app/`)
- Core source code (`src/`)
- Main training/build scripts
- Requirements file and README

Excluded intentionally:
- Raw CSV datasets
- Model checkpoints and training outputs
- Local LLM/Ollama files
- Logs
- Node modules and other large build artifacts


## Files

- `build_narratives.py`: CLI entrypoint
- `src/mimic_los/narrative_dataset.py`: join + label + narrative pipeline
- `src/mimic_los/ollama_client.py`: local Ollama wrapper
- `src/mimic_los/text_utils.py`: radiology parsing + narrative formatting helpers
- `src/mimic_los/training_data.py`: patient-level split preparation
- `src/mimic_los/baseline_training.py`: Tier 0 / Tier 1 trainers + Tier 2B handoff builder
- `src/mimic_los/tier2_training.py`: Tier 2A / Tier 2B trainers
- `src/mimic_los/explanation_engine.py`: rule-based clinical Q&A fallback
- `src/mimic_los/generate_contexts.py`: explanation context builder
- `app/streamlit_demo.py`: final UI demo
- `train_tiers.py`: Tier 0 to Tier 2 pipeline scaffold

## Install

```bash
pip install -r requirements.txt
```

## Build a sample with Gemma

```bash
python build_narratives.py --data-dir . --output-dir outputs --limit 100 --llm-mode gemma
```

## Build the full dataset without LLM rewriting

This is the fastest path for generating the full physical training file:

```bash
python build_narratives.py --data-dir . --output-dir outputs --llm-mode off
```

## Build the full dataset with Gemma

This is much slower because each admission becomes a local model call:

```bash
python build_narratives.py --data-dir . --output-dir outputs --llm-mode gemma
```

## Outputs

The builder writes:

- `outputs/narrative_dataset.jsonl`
- `outputs/narrative_dataset.csv`
- `outputs/run_summary.json`

Each row includes:

- IDs: `subject_id`, `hadm_id`, `stay_id`
- LOS target: `los_days`, `los_bucket`
- structured features used before discharge
- source text fields
- `narrative`: canonical storyline; in Gemma mode this uses the LLM only if a verification pass marks it supported, otherwise it falls back
- `narrative_source`: `llm_verified` or deterministic fallback status
- `narrative_llm`: raw Gemma rewrite for inspection

The builder excludes admissions with `hospital_expire_flag=1` and keeps radiology text to the first 48 hours after admission.

## Tier Pipeline

The training scaffold is stage-based and seed-aware:

```bash
python train_tiers.py --stage prepare --seeds 42,52,62
python train_tiers.py --stage tier0 --seeds 42,52,62
python train_tiers.py --stage handoff --seeds 42,52,62
python train_tiers.py --stage tier1 --seeds 42,52,62
python train_tiers.py --stage tier2 --seeds 42,52,62
```

`--stage all` runs the full order:

1. prepare patient-level splits
2. train Tier 0 structured XGBoost
3. export Tier 0 contribution artifacts and build the Tier 2B handoff narratives
4. train Tier 1 sparse text
5. train Tier 2A late fusion and Tier 2B early fusion

Each seed writes to `tier_runs/seed_<seed>/...`, and `train_tiers.py` also writes `tier_runs/aggregate_summary.json` with mean and standard deviation across seeds.

## Explainability UI

Generate explanation contexts:

```bash
python -m src.mimic_los.generate_contexts
```

Run the demo app:

```bash
streamlit run app/streamlit_demo.py
```

The app:

- loads generated patient contexts from `outputs/explanations/` when real tier artifacts exist
- falls back to built-in demo patients if no real context files exist yet
- tries a local Ollama model first and falls back to the rule engine if no compatible model is available

## Leakage guardrails

The builder excludes known leakage fields from training features:

- patient `dod`
- admission `deathtime`
- admission `hospital_expire_flag`
- admission `discharge_location`
- diagnoses / ICD tables as model input

`admittime` and `dischtime` are used only to compute LOS labels and radiology filtering windows.
