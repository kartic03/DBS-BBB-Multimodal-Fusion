"""
Structured Prompts for Groq LLM Clinical Recommendations
==========================================================
"""

SYSTEM_PROMPT = """You are a clinical neurology AI assistant specializing in Parkinson's disease \
and Deep Brain Stimulation (DBS). Given patient biomarker data and ML model \
predictions, generate a concise, evidence-based clinical recommendation about \
DBS candidacy and expected response. Base reasoning on provided features only. \
Output 100-150 words. Format: [Prediction][Key Drivers][Recommendation]."""

USER_PROMPT_TEMPLATE = """Patient Profile:
- Fusion model DBS response probability: {prob:.2%} ({label})
- Model confidence: {confidence}
Top LFP Neural Signal Drivers:
{lfp_shap_summary}
Top BBB/Inflammation Drivers:
{bbb_shap_summary}
Patient context: UPDRS-III = {updrs_score:.1f}, Disease duration: {years:.0f} years
Generate clinical DBS recommendation based on above."""


def format_shap_summary(feature_names, shap_values, top_k=5):
    """Format top SHAP features as human-readable summary."""
    abs_shap = [(abs(v), n, v) for n, v in zip(feature_names, shap_values)]
    abs_shap.sort(reverse=True)

    lines = []
    for _, name, val in abs_shap[:top_k]:
        direction = "HIGH" if val > 0 else "LOW"
        clean_name = name.replace("lfp_", "").replace("bbb_", "").replace("_", " ").title()
        lines.append(f"- {clean_name}: {direction} (SHAP {val:+.3f})")
    return "\n".join(lines)


def build_patient_prompt(prob, label, confidence, lfp_shap_summary,
                         bbb_shap_summary, updrs_score, years):
    """Build the full user prompt for a patient."""
    return USER_PROMPT_TEMPLATE.format(
        prob=prob,
        label=label,
        confidence=confidence,
        lfp_shap_summary=lfp_shap_summary,
        bbb_shap_summary=bbb_shap_summary,
        updrs_score=updrs_score,
        years=years,
    )
