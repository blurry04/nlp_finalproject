from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mimic_los.explanation_engine import ACTION_BUNDLES, answer_question
from src.mimic_los.generate_contexts import generate
from src.mimic_los.ollama_client import OllamaClient
from src.mimic_los.predict_patient import _find_record, _write_context, predict_record


EXPLANATIONS_DIR = ROOT / "outputs_15k_gemma_24h_prior_dx" / "explanations"
FULL_DETERMINISTIC_SOURCE = ROOT / "outputs_full_deterministic_24h_prior_dx" / "narrative_dataset.jsonl"
FULL_DETERMINISTIC_RUN_DIR = ROOT / "tier_runs_full_deterministic_24h_prior_dx_tier2a" / "seed_42"
GEMMA_15K_SOURCE = ROOT / "outputs_15k_gemma_24h_prior_dx" / "narrative_dataset.jsonl"
GEMMA_15K_RUN_DIR = ROOT / "tier_runs_15k_gemma_24h_prior_dx" / "seed_62"
MAX_SELECTOR_PATIENTS = 5

DEMO_PATIENTS = [
    {
        "hadm_id": 26840593,
        "subject_id": "demo-a",
        "narrative": (
            "89-year-old female presented to the ED as a walk-in with abdominal pain, nausea, "
            "hypotension, and abdominal distention. Triage acuity 3. Vitals: HR 95, BP 101/58, "
            "O2 97%, Temp 98.2. Current medications include metoprolol, furosemide, omeprazole, "
            "lactulose, cholecalciferol, with 9 medications total. Radiology impression: small "
            "bilateral pleural effusions, bibasilar atelectasis, distended loops of bowel."
        ),
        "prediction": "LONG",
        "actual": "LONG",
        "correct": True,
        "confidence_raw": 0.87,
        "calibrated_confidence": 0.79,
        "probabilities": {"SHORT": 0.05, "MEDIUM": 0.08, "LONG": 0.87},
        "shap_top_features": [
            {"feature": "Radiology: pleural effusions", "value": 0.22},
            {"feature": "Age: 89", "value": 0.17},
            {"feature": "Med count: 9", "value": 0.12},
            {"feature": "Acuity: 3", "value": 0.08},
            {"feature": "Has radiology", "value": 0.07},
            {"feature": "HR: 95", "value": 0.03},
            {"feature": "BP: 101/58", "value": 0.02},
            {"feature": "O2: 97%", "value": -0.04},
            {"feature": "Temp: 98.2", "value": -0.03},
        ],
        "recommended_actions": ACTION_BUNDLES["LONG"],
        "discharge_summary_posthoc": None,
    },
    {
        "hadm_id": 99999999,
        "subject_id": "demo-b",
        "narrative": (
            "45-year-old female walked into the ED with abdominal pain and nausea. Triage acuity 3. "
            "Vitals: HR 88, BP 124/78, O2 98%, Temp 98.6. Current medications include omeprazole "
            "and sertraline (2 medications total)."
        ),
        "prediction": "SHORT",
        "actual": "LONG",
        "correct": False,
        "confidence_raw": 0.71,
        "calibrated_confidence": 0.62,
        "probabilities": {"SHORT": 0.71, "MEDIUM": 0.19, "LONG": 0.10},
        "shap_top_features": [
            {"feature": "No radiology", "value": -0.18},
            {"feature": "O2: 98%", "value": -0.12},
            {"feature": "Med count: 2", "value": -0.09},
            {"feature": "Age: 45", "value": -0.07},
            {"feature": "Acuity: 3", "value": -0.04},
            {"feature": "CC: abdominal pain", "value": 0.04},
        ],
        "recommended_actions": ACTION_BUNDLES["SHORT"],
        "discharge_summary_posthoc": (
            "Synthetic post-hoc note: later complications and operative findings prolonged the stay "
            "after the initial early prediction window."
        ),
    },
]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {background: linear-gradient(135deg, #f7f1e6 0%, #fbfaf6 45%, #edf5ef 100%);}
        h1 {letter-spacing:-0.04em;}
        .hero {border:1px solid #ded6c8; border-radius:24px; padding:1.1rem 1.25rem; background:rgba(255,253,248,0.9); box-shadow:0 18px 45px rgba(40,32,19,0.08); margin-bottom:1rem;}
        .hero-title {font-size:1.55rem; font-weight:800; letter-spacing:-0.03em; color:#152019;}
        .hero-sub {font-size:0.95rem; color:#5f6b63; margin-top:0.3rem; max-width:900px;}
        .small-label {font-size:0.74rem; color:#66736a; margin-bottom:0.18rem; text-transform:uppercase; letter-spacing:0.06em;}
        .metric-card {border:1px solid #ded6c8; border-radius:18px; padding:0.9rem 0.95rem; background:#fffdf8; box-shadow:0 8px 22px rgba(40,32,19,0.05);}
        .metric-value {font-size:1.55rem; font-weight:800; line-height:1.12;}
        .prediction-long {color: #9f2f2f;}
        .prediction-medium {color: #9a6700;}
        .prediction-short {color: #1f6f43;}
        .section-card {border:1px solid #ded6c8; border-radius:22px; padding:1.05rem 1.1rem; background:#fffdf8; margin-bottom:1rem; box-shadow:0 10px 28px rgba(40,32,19,0.05);}
        .narrative-box {border:1px solid #d7cbbb; border-radius:18px; padding:1.05rem 1.1rem; background:#fff8ec; line-height:1.85; color:#111827; font-size:1.02rem; max-height:360px; overflow-y:auto;}
        .narrative-box p {margin:0 0 0.85rem 0;}
        .evidence-grid {display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:0.7rem; margin-top:0.5rem;}
        .evidence-card {border:1px solid #e2dacd; border-radius:16px; padding:0.85rem; background:#faf8f1;}
        .evidence-head {font-weight:750; color:#1f2f25; font-size:0.96rem; margin-bottom:0.25rem;}
        .evidence-body {font-size:0.86rem; color:#5f6b63; line-height:1.45;}
        .action-box {border:1px solid #7fb06d; border-radius:18px; padding:0.95rem 1rem; background:#eef8e7; margin-top:0.85rem;}
        .action-head {font-weight:800; color:#2d6d2a; margin-bottom:0.35rem;}
        .action-step {font-size:0.95rem; color:#285f26; margin:0.2rem 0;}
        .shap-wrap {margin-top: 0.25rem;}
        .shap-row {display:flex; align-items:center; gap:0.65rem; margin:0.34rem 0;}
        .shap-name {width:170px; text-align:right; color:#526157; font-size:0.84rem; flex-shrink:0;}
        .shap-bg {flex:1; height:18px; border-radius:999px; background:#efede7; position:relative; overflow:hidden;}
        .shap-bar-pos {position:absolute; left:50%; height:100%; background:#df5a58; border-radius:999px;}
        .shap-bar-neg {position:absolute; right:50%; height:100%; background:#4f86d9; border-radius:999px;}
        .shap-val {width:52px; font-size:0.8rem; color:#526157; text-align:right;}
        .legend {display:flex; gap:1rem; margin-top:0.5rem; font-size:0.78rem; color:#5f6b63;}
        .legend-dot {display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:0.3rem;}
        .panel-title {font-size:1.08rem; font-weight:800; color:#17211b; margin-bottom:0.75rem; letter-spacing:-0.02em;}
        .subtle-tag {padding:0.24rem 0.65rem; border-radius:999px; background:#eef1ec; color:#415147; font-size:0.76rem; font-weight:650;}
        .source-pill {display:inline-block; padding:0.25rem 0.55rem; border:1px solid #d7cbbb; border-radius:999px; margin:0.15rem 0.25rem 0.15rem 0; background:#fffaf0; color:#526157; font-size:0.78rem;}
        .chat-note {border:1px solid #d7cbbb; border-radius:16px; background:#fffaf0; padding:0.8rem; color:#526157; font-size:0.9rem; line-height:1.45; margin-bottom:0.8rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_patients() -> list[dict[str, Any]]:
    index_path = EXPLANATIONS_DIR / "patient_index.json"
    if not index_path.exists():
        try:
            generate(
                dataset_path=ROOT / "outputs_15k_gemma_24h_prior_dx" / "narrative_dataset.jsonl",
                work_dir=ROOT / "tier_runs_15k_gemma_24h_prior_dx",
                output_dir=EXPLANATIONS_DIR,
            )
        except Exception:
            pass
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        patients: list[dict[str, Any]] = []
        for entry in index:
            path = EXPLANATIONS_DIR / f"patient_{entry['hadm_id']}.json"
            if path.exists():
                patients.append(json.loads(path.read_text(encoding="utf-8")))
        if patients:
            return patients[:MAX_SELECTOR_PATIENTS]
    return DEMO_PATIENTS[:MAX_SELECTOR_PATIENTS]


def model_badge() -> tuple[str, str | None]:
    selected = OllamaClient.pick_available_model(["gemma3:4b", "llama3.1:8b", "qwen3.5:latest"])
    if selected:
        return f"{selected} / rule engine", selected
    return "rule engine only", None


def ask_model(question: str, ctx: dict[str, Any], model_name: str | None) -> tuple[str, str]:
    if not model_name:
        return answer_question(question, ctx), "Rule engine"
    try:
        client = OllamaClient(model=model_name)
        prompt = (
            "You are explaining a hospital length-of-stay prediction to a clinical user.\n"
            "Use only the facts provided below. Do not invent diagnoses, treatments, or future events.\n"
            "The Bio_ClinicalBERT model produced the prediction and probabilities; you are only translating "
            "the saved model output into a readable explanation. Be concise, concrete, and mention the "
            "specific contributing factors.\n\n"
            f"Patient narrative:\n{ctx.get('narrative', '')}\n\n"
            f"Prediction: {ctx.get('prediction', 'UNKNOWN')}\n"
            f"Actual: {ctx.get('actual', 'UNKNOWN')}\n"
            f"Confidence raw: {ctx.get('confidence_raw', 0):.2f}\n"
            f"Confidence calibrated: {ctx.get('calibrated_confidence', 0):.2f}\n"
            f"Top features: {json.dumps(ctx.get('shap_top_features', []))}\n"
            f"Action bundle: {json.dumps(ctx.get('recommended_actions', {}))}\n"
            f"Post-hoc note: {ctx.get('discharge_summary_posthoc', '')}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return client.generate(prompt, temperature=0.1), model_name
    except Exception:
        return answer_question(question, ctx), "Rule engine"


def narrative_paragraphs(text: str) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return ["No narrative available."]
    markers = [
        "Administrative intake",
        "Background registration",
        "Prior admission history",
        "The highest-signal",
        "Radiology from",
        "Radiology impression:",
        "Source tables",
    ]
    for marker in markers:
        text = text.replace(f" {marker}", f"\n\n{marker}")
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def render_narrative(text: str) -> None:
    paragraphs = "".join(f"<p>{escape(part)}</p>" for part in narrative_paragraphs(text))
    st.markdown(f'<div class="narrative-box">{paragraphs}</div>', unsafe_allow_html=True)


def render_source_pills(ctx: dict[str, Any]) -> None:
    source = ctx.get("source_tier", "tier2a")
    record_source = ctx.get("source_record") or {}
    narrative_source = record_source.get("narrative_source") or ctx.get("narrative_source") or "saved context"
    llm_model = record_source.get("llm_model") or ctx.get("llm_model") or "not stored"
    pills = [
        f"Prediction source: {source}",
        "Model: Tier 2A Bio_ClinicalBERT late fusion",
        f"Narrative source: {narrative_source}",
        f"Narrative LLM: {llm_model}",
    ]
    html = "".join(f'<span class="source-pill">{escape(pill)}</span>' for pill in pills)
    st.markdown(html, unsafe_allow_html=True)


def render_evidence_cards(ctx: dict[str, Any]) -> None:
    features = list(ctx.get("shap_top_features", []))[:4]
    if not features:
        return
    cards = []
    for item in features:
        feature = str(item.get("feature", "Unknown feature"))
        evidence = str(item.get("evidence_text", "") or "Evidence comes from the early patient snapshot.")
        value = float(item.get("value", 0.0) or 0.0)
        direction = "pushes LONG" if value >= 0 else "pushes SHORT"
        cards.append(
            '<div class="evidence-card">'
            f'<div class="evidence-head">{escape(feature)} ({value:+.2f})</div>'
            f'<div class="evidence-body">{escape(direction)}. {escape(evidence)}</div>'
            "</div>"
        )
    st.markdown(f'<div class="evidence-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_metric_card(label: str, value: str, css_class: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-label">{label}</div>
            <div class="metric-value {css_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shap(features: list[dict[str, Any]]) -> None:
    if not features:
        st.info("No feature attribution data available yet.")
        return
    max_abs = max(abs(float(item.get("value", 0.0))) for item in features) or 1.0
    html = ['<div class="shap-wrap">']
    for item in features:
        feature = str(item.get("feature", "Unknown"))
        short_name = feature.split(":", 1)[0]
        value = float(item.get("value", 0.0))
        width = max(int((abs(value) / max_abs) * 48), 2)
        bar_class = "shap-bar-pos" if value >= 0 else "shap-bar-neg"
        bar_style = f"width:{width}%;"
        html.append(
            f'<div class="shap-row"><div class="shap-name">{short_name}</div>'
            f'<div class="shap-bg"><div class="{bar_class}" style="{bar_style}"></div></div>'
            f'<div class="shap-val">{value:+.2f}</div></div>'
        )
    html.append(
        '<div class="legend">'
        '<div><span class="legend-dot" style="background:#df5a58"></span>Pushes LONG</div>'
        '<div><span class="legend-dot" style="background:#4f86d9"></span>Pushes SHORT</div>'
        "</div>"
    )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_probabilities(probabilities: dict[str, Any]) -> None:
    colors = {"SHORT": "#60c19d", "MEDIUM": "#f4a340", "LONG": "#df5a58"}
    for label in ["SHORT", "MEDIUM", "LONG"]:
        prob = float(probabilities.get(label, 0.0))
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:0.6rem;margin:0.2rem 0;">
                <div style="width:68px;font-size:0.85rem;">{label}</div>
                <div style="flex:1;height:16px;border-radius:999px;background:#efede7;overflow:hidden;">
                    <div style="width:{prob * 100:.1f}%;height:100%;background:{colors[label]};border-radius:999px;"></div>
                </div>
                <div style="width:48px;text-align:right;font-size:0.82rem;">{prob * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_actions(ctx: dict[str, Any]) -> None:
    bundle = ctx.get("recommended_actions") or ACTION_BUNDLES.get(ctx.get("prediction", ""))
    if not bundle:
        return
    lines = [
        f'<div class="action-box"><div class="action-head">{bundle["bundle"]} ({bundle["owner"]})</div>'
    ]
    for step in bundle.get("steps", []):
        lines.append(f'<div class="action-step">- {step}</div>')
    lines.append("</div>")
    st.markdown("".join(lines), unsafe_allow_html=True)


def patient_button_label(patient: dict[str, Any], index: int) -> str:
    base = f"Patient {chr(65 + index)}"
    suffix = "correct" if patient.get("correct") else "wrong"
    return f"{base} ({suffix})"


def ensure_chat_state(patient_id: str) -> None:
    if "chat_by_patient" not in st.session_state:
        st.session_state.chat_by_patient = {}
    if patient_id not in st.session_state.chat_by_patient:
        st.session_state.chat_by_patient[patient_id] = [
            {
                "role": "assistant",
                "content": "Ask about this patient's prediction, reliability, or recommended actions.",
                "source": "System",
            }
        ]


def _write_live_prediction(prediction: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest_live_prediction.json"
    latest_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
    patient_path = output_dir / f"patient_{prediction['hadm_id']}.json"
    patient_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
    return patient_path


def run_dashboard_inference(hadm_id: str, model_source: str) -> dict[str, Any]:
    class _Args:
        def __init__(self, hadm_id: str):
            self.hadm_id = hadm_id
            self.subject_id = None
            self.stay_id = None

    if model_source == "Full deterministic Tier 2A (seed 42)":
        source_jsonl = FULL_DETERMINISTIC_SOURCE
        run_dir = FULL_DETERMINISTIC_RUN_DIR
    else:
        source_jsonl = GEMMA_15K_SOURCE
        run_dir = GEMMA_15K_RUN_DIR

    record = _find_record(source_jsonl.resolve(), _Args(str(hadm_id).strip()))
    prediction = predict_record(record, run_dir.resolve(), device_name="cuda")
    prediction["dashboard_context_path"] = str(_write_live_prediction(prediction, EXPLANATIONS_DIR.resolve()))
    return prediction


def main() -> None:
    st.set_page_config(page_title="LOS explainability demo", layout="wide")
    inject_css()

    patients = load_patients()
    engine_label, local_model = model_badge()

    if "selected_patient_idx" not in st.session_state:
        st.session_state.selected_patient_idx = 0
    if "live_hadm_id" not in st.session_state:
        st.session_state.live_hadm_id = ""
    if "live_model_source" not in st.session_state:
        st.session_state.live_model_source = "15K Gemma Tier 2A"
    if "live_patient" not in st.session_state:
        st.session_state.live_patient = None

    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">LOS clinical explainability dashboard</div>
          <div class="hero-sub">
            Predicts early hospital length-of-stay risk with the trained Tier 2A clinical model,
            then uses grounded patient context for readable explanations and action planning.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1], gap="large")
    base_selected_patient = patients[st.session_state.selected_patient_idx]
    selected_patient = st.session_state.live_patient or base_selected_patient
    patient_key = str(selected_patient.get("hadm_id", "demo"))
    ensure_chat_state(patient_key)

    with left:
        st.markdown('<div class="section-card"><div class="panel-title">Live patient inference</div>', unsafe_allow_html=True)
        st.caption("Enter a hospital admission ID to run inference directly from the dashboard and add the result to the patient list.")
        infer_cols = st.columns([1.2, 1.3, 0.8])
        with infer_cols[0]:
            hadm_input = st.text_input("HADM ID", key="live_hadm_id", placeholder="e.g. 28861371")
        with infer_cols[1]:
            model_source = st.selectbox(
                "Prediction source",
                ["15K Gemma Tier 2A", "Full deterministic Tier 2A (seed 42)"],
                key="live_model_source",
            )
        with infer_cols[2]:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            if st.button("Run prediction", use_container_width=True):
                hadm = hadm_input.strip()
                if not hadm:
                    st.error("Enter a valid HADM ID first.")
                else:
                    with st.spinner("Running patient inference..."):
                        try:
                            result = run_dashboard_inference(hadm, model_source)
                            st.session_state.live_patient = result
                            ensure_chat_state(str(result.get("hadm_id")))
                            st.success(
                                f"Loaded HADM ID {result.get('hadm_id')} with prediction {result.get('prediction')}."
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Inference failed: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card"><div class="panel-title">Patient selector</div>', unsafe_allow_html=True)
        tab_cols = st.columns(len(patients))
        for idx, patient in enumerate(patients):
            if tab_cols[idx].button(
                patient_button_label(patient, idx),
                key=f"patient_btn_{idx}",
                use_container_width=True,
            ):
                st.session_state.selected_patient_idx = idx
                st.session_state.live_patient = None
                selected_patient = patients[idx]
                patient_key = str(selected_patient.get("hadm_id", "demo"))
                ensure_chat_state(patient_key)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.live_patient is not None:
            live_cols = st.columns([3, 1])
            with live_cols[0]:
                st.info(
                    f"Viewing live inference for HADM ID {selected_patient.get('hadm_id')} "
                    f"using {st.session_state.live_model_source}."
                )
            with live_cols[1]:
                if st.button("Back to demo patients", use_container_width=True):
                    st.session_state.live_patient = None
                    st.rerun()

        result_cols = st.columns(4)
        pred_class = f"prediction-{selected_patient['prediction'].lower()}"
        renderers = [
            ("Prediction", selected_patient["prediction"], pred_class),
            ("Actual", selected_patient["actual"], ""),
            ("Confidence", f"{(selected_patient.get('confidence_raw', 0) * 100):.0f}%", ""),
            ("Result", "Correct" if selected_patient.get("correct") else "Misclassified", ""),
        ]
        for col, (label, value, css_class) in zip(result_cols, renderers):
            with col:
                render_metric_card(label, value, css_class)

        st.markdown('<div class="section-card"><div class="panel-title">Model and data provenance</div>', unsafe_allow_html=True)
        render_source_pills(selected_patient)
        st.markdown("</div>", unsafe_allow_html=True)

        tab_summary, tab_narrative, tab_model = st.tabs(["Clinical summary", "Readable narrative", "Model details"])
        with tab_summary:
            st.markdown('<div class="section-card"><div class="panel-title">Key evidence</div>', unsafe_allow_html=True)
            render_evidence_cards(selected_patient)
            render_actions(selected_patient)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_narrative:
            st.markdown('<div class="section-card"><div class="panel-title">Narrative used by the model/explainer</div>', unsafe_allow_html=True)
            render_narrative(selected_patient.get("narrative", ""))
            st.caption("This text is split for readability. It is still grounded in the patient context file.")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_model:
            st.markdown('<div class="section-card"><div class="panel-title">Probability profile</div>', unsafe_allow_html=True)
            render_probabilities(selected_patient.get("probabilities", {}))
            st.markdown("<br><div class='panel-title'>Feature contributions</div>", unsafe_allow_html=True)
            render_shap(selected_patient.get("shap_top_features", []))
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        title_left, title_right = st.columns([3, 2])
        with title_left:
            st.markdown('<div class="panel-title">Explanation chat</div>', unsafe_allow_html=True)
        with title_right:
            st.markdown(
                f"<div style='display:flex;justify-content:flex-end;'><span class='subtle-tag'>{engine_label}</span></div>",
                unsafe_allow_html=True,
            )

        predicted_label = selected_patient.get("prediction", "this class")
        quick_questions = [
            f"Why is this predicted {predicted_label}?",
            "What would change the prediction?",
            "Is this prediction reliable?",
            "What actions should we take?",
            "What happened after the prediction window?",
        ]
        qq_cols = st.columns(1)
        for question in quick_questions:
            if qq_cols[0].button(question, key=f"quick_{patient_key}_{question}", use_container_width=True):
                response, source = ask_model(question, selected_patient, local_model)
                st.session_state.chat_by_patient[patient_key].append({"role": "user", "content": question, "source": "User"})
                st.session_state.chat_by_patient[patient_key].append({"role": "assistant", "content": response, "source": source})
                st.rerun()

        for message in st.session_state.chat_by_patient[patient_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("source") and message["role"] == "assistant":
                    st.caption(f"Source: {message['source']}")

        user_input = st.chat_input("Ask about this patient...")
        if user_input:
            response, source = ask_model(user_input, selected_patient, local_model)
            st.session_state.chat_by_patient[patient_key].append({"role": "user", "content": user_input, "source": "User"})
            st.session_state.chat_by_patient[patient_key].append({"role": "assistant", "content": response, "source": source})
            st.rerun()


if __name__ == "__main__":
    main()
