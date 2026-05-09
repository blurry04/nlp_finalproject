"""Rule-based explanation engine for LOS prediction UI flows."""

from __future__ import annotations

from typing import Any


ACTION_BUNDLES = {
    "LONG": {
        "bundle": "Complex discharge activation",
        "owner": "Case manager + social work",
        "steps": [
            "Assign case manager within 6 hours of admission",
            "Initiate post-acute referral by day 1 noon",
            "Place PT/OT evaluation order for morning of day 1",
            "Review discharge barriers daily in multidisciplinary rounds",
        ],
    },
    "MEDIUM": {
        "bundle": "Clinical pathway enrollment",
        "owner": "Service line lead + bedside nursing",
        "steps": [
            "Enroll in the disease-specific care pathway on day 0",
            "Track milestones against the expected inpatient timeline",
            "Escalate if pathway milestones slip by more than 24 hours",
        ],
    },
    "SHORT": {
        "bundle": "Discharge-by-noon protocol",
        "owner": "Hospitalist + bed management",
        "steps": [
            "Document an expected discharge date immediately",
            "Pre-stage discharge medications and instructions",
            "Target discharge before noon if the clinical course stays stable",
        ],
    },
}


def _confidence_text(ctx: dict[str, Any]) -> str:
    raw = float(ctx.get("confidence_raw") or 0.0)
    calibrated = ctx.get("calibrated_confidence")
    calibrated = float(calibrated) if calibrated not in (None, "") else None
    base = f"Raw confidence is {raw * 100:.0f}%."
    if calibrated is not None:
        base += f" Calibrated confidence is {calibrated * 100:.0f}%."
        if calibrated < raw - 0.03:
            base += " Calibration reduces the score, so the model was somewhat overconfident."
    final_score = calibrated if calibrated is not None else raw
    if final_score < 0.60:
        base += " This sits in a low-confidence range and should be treated as uncertain."
    elif final_score < 0.75:
        base += " This is a moderate-confidence prediction."
    else:
        base += " This is a high-confidence prediction."
    return base


def _top_driver_lines(ctx: dict[str, Any], top_k: int = 3) -> list[str]:
    lines: list[str] = []
    for item in ctx.get("shap_top_features", [])[:top_k]:
        feature = item.get("feature", "unknown feature")
        value = float(item.get("value", 0.0))
        direction = "toward LONG" if value > 0 else "toward SHORT"
        lines.append(f"{feature} ({value:+.2f}, pushes {direction})")
    return lines


def explain_prediction(ctx: dict[str, Any]) -> str:
    pred = ctx.get("prediction", "UNKNOWN")
    actual = ctx.get("actual", "UNKNOWN")
    correct = bool(ctx.get("correct"))
    if correct:
        intro = f"The model predicted {pred}, and that matches the observed LOS class."
    else:
        intro = f"The model predicted {pred}, but the observed LOS class was {actual}."
    drivers = _top_driver_lines(ctx)
    if not drivers:
        return intro + " No SHAP-style feature attributions are available yet."
    body = "The strongest drivers were:\n"
    body += "\n".join(f"- {line}" for line in drivers)
    return intro + "\n\n" + body


def explain_counterfactual(ctx: dict[str, Any]) -> str:
    features = ctx.get("shap_top_features", [])
    if not features:
        return "No feature attributions are available yet, so I cannot name the strongest counterfactual driver."
    strongest = features[0]
    feature = strongest.get("feature", "unknown feature")
    value = float(strongest.get("value", 0.0))
    if value > 0:
        return (
            f"The largest push toward a longer stay is {feature} ({value:+.2f}). "
            f"If that factor were absent, milder, or replaced by a more reassuring finding, "
            f"the prediction would likely move toward MEDIUM or SHORT."
        )
    return (
        f"The largest push toward a shorter stay is {feature} ({value:+.2f}). "
        f"If that factor were replaced by a more concerning signal, the prediction would likely move upward."
    )


def explain_reliability(ctx: dict[str, Any]) -> str:
    return _confidence_text(ctx)


def explain_actions(ctx: dict[str, Any]) -> str:
    pred = ctx.get("prediction", "UNKNOWN")
    bundle = ACTION_BUNDLES.get(pred)
    if not bundle:
        return "No action bundle is defined for this prediction."
    lines = [
        f"Recommended bundle: {bundle['bundle']}",
        f"Owner: {bundle['owner']}",
        "",
        "Suggested steps:",
    ]
    lines.extend(f"{idx}. {step}" for idx, step in enumerate(bundle["steps"], start=1))
    return "\n".join(lines)


def explain_what_happened(ctx: dict[str, Any]) -> str:
    discharge_summary = ctx.get("discharge_summary_posthoc")
    if not discharge_summary:
        return (
            "No post-hoc discharge summary is loaded for this patient. "
            "In the full workflow this field is only used after prediction to explain misses, never as model input."
        )
    pred = ctx.get("prediction", "UNKNOWN")
    actual = ctx.get("actual", "UNKNOWN")
    return (
        f"Prediction was {pred}; observed LOS class was {actual}.\n\n"
        f"Post-hoc summary:\n{str(discharge_summary)[:1200]}"
    )


def explain_summary(ctx: dict[str, Any]) -> str:
    lines = [
        explain_prediction(ctx),
        "",
        explain_reliability(ctx),
        "",
        explain_actions(ctx),
    ]
    return "\n".join(lines)


HANDLERS = {
    "why": explain_prediction,
    "driver": explain_prediction,
    "predicted": explain_prediction,
    "what would change": explain_counterfactual,
    "counterfactual": explain_counterfactual,
    "reliable": explain_reliability,
    "confidence": explain_reliability,
    "trust": explain_reliability,
    "action": explain_actions,
    "recommend": explain_actions,
    "take": explain_actions,
    "what happened": explain_what_happened,
    "post-hoc": explain_what_happened,
    "discharge": explain_what_happened,
    "summary": explain_summary,
}


def answer_question(question: str, ctx: dict[str, Any]) -> str:
    q = question.lower().strip()
    for keyword, handler in HANDLERS.items():
        if keyword in q:
            return handler(ctx)
    return (
        "I can help with:\n"
        "- Why this LOS class was predicted\n"
        "- What would change the prediction\n"
        "- Whether the prediction looks reliable\n"
        "- Recommended clinical actions\n"
        "- What happened afterward in post-hoc review"
    )
