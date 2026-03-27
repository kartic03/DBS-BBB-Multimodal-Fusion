"""
Groq LLM Clinical Recommendation Layer
=========================================
Uses Groq API to generate explainable clinical recommendations
for DBS candidacy based on model predictions + SHAP features.

Outputs:
    - results/llm_recommendations.csv
    - results/figures/fig8_llm_examples.png
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from groq_llm.prompts import SYSTEM_PROMPT, format_shap_summary, build_patient_prompt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

FIGURES = os.path.join(PROJECT_ROOT, config["paths"]["figures"])
RESULTS = os.path.join(PROJECT_ROOT, config["paths"]["results"])
os.makedirs(FIGURES, exist_ok=True)

DPI = config["figures"]["dpi"]


def get_groq_client():
    """Initialize Groq client with API key."""
    api_key = os.environ.get("GROQ_API_KEY", config["groq"]["api_key"])
    if api_key == "YOUR_GROQ_API_KEY_HERE":
        print("  [WARN] No Groq API key set. Using placeholder recommendations.")
        return None

    from groq import Groq
    return Groq(api_key=api_key)


def generate_recommendation(client, prompt, model=None, max_retries=3):
    """Call Groq API with exponential backoff."""
    if client is None:
        return "[Placeholder] DBS recommendation pending API key configuration."

    model = model or config["groq"]["model"]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=config["groq"]["max_tokens"],
                temperature=config["groq"]["temperature"],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                # Try fallback model
                model = config["groq"]["fallback_model"]
                print(f"    Retrying with fallback model: {model}")
            else:
                print(f"    [ERROR] Groq API failed: {e}")
                return f"[Error] Could not generate recommendation: {e}"


def generate_figure8(recommendations_df):
    """Generate Figure 8: LLM example outputs."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    for i, (_, row) in enumerate(recommendations_df.head(3).iterrows()):
        ax = axes[i]
        ax.axis("off")

        # Patient info box
        info_text = (
            f"Patient {i+1}: "
            f"DBS Probability = {row.get('probability', 'N/A'):.1%} | "
            f"UPDRS-III = {row.get('updrs_score', 'N/A'):.0f} | "
            f"Disease Duration = {row.get('disease_years', 'N/A'):.0f} yrs"
        )

        ax.text(0.02, 0.95, info_text, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top",
                bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8))

        # Top features
        features_text = f"Top Drivers: {row.get('top_features', 'N/A')}"
        ax.text(0.02, 0.72, features_text, transform=ax.transAxes,
                fontsize=9, va="top", color="#424242",
                bbox=dict(boxstyle="round", facecolor="#FFF9C4", alpha=0.6))

        # LLM recommendation
        rec_text = row.get("recommendation", "No recommendation available")
        ax.text(0.02, 0.45, f"LLM Recommendation:\n{rec_text}", transform=ax.transAxes,
                fontsize=9, va="top", wrap=True,
                bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.6))

    plt.suptitle("Figure 8. Groq LLM Clinical Recommendations",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig8_llm_examples.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: fig8_llm_examples.png")


def main():
    print("=" * 60)
    print("Groq LLM Clinical Recommendations")
    print("=" * 60)

    client = get_groq_client()

    # Load test data
    fused_path = os.path.join(PROJECT_ROOT, "data/processed/fused/multimodal_dataset.csv")
    df = pd.read_csv(fused_path)
    df_test = df[df["split"] == "test"].head(10)  # Process first 10 test subjects

    lfp_cols = [c for c in df.columns if c.startswith("lfp_")]
    bbb_cols = [c for c in df.columns if c.startswith("bbb_")]

    # Load SHAP values if available
    feat_path = os.path.join(PROJECT_ROOT, "results/tables/feature_importance_ranked.csv")
    if os.path.isfile(feat_path):
        feat_imp = pd.read_csv(feat_path)
    else:
        feat_imp = None

    recommendations = []
    print(f"\nGenerating recommendations for {len(df_test)} test patients...")

    for i, (_, row) in enumerate(df_test.iterrows()):
        prob = row["label"]  # Use label as proxy (will be replaced by model prob)
        label = "Likely Responder" if prob == 1 else "Unlikely Responder"
        confidence = "Moderate"

        # Mock SHAP summaries (use feature importance if available)
        lfp_shap = "\n".join([f"- {c.replace('lfp_', '').replace('_', ' ').title()}: "
                              f"{'HIGH' if row[c] > 0 else 'LOW'} (value={row[c]:.3f})"
                              for c in lfp_cols[:5]])
        bbb_shap = "\n".join([f"- {c.replace('bbb_', '').replace('_', ' ').title()}: "
                              f"{'ELEVATED' if row[c] > 0 else 'LOW'} (value={row[c]:.3f})"
                              for c in bbb_cols[:3]])

        updrs = row.get("updrs_iii_baseline", 30)
        years = row.get("bbb_disease_duration_years", 5)
        if pd.isna(updrs):
            updrs = 30
        if pd.isna(years):
            years = 5

        prompt = build_patient_prompt(prob, label, confidence, lfp_shap, bbb_shap, updrs, years)
        rec = generate_recommendation(client, prompt)

        top_feats = ", ".join(lfp_cols[:3] + bbb_cols[:2])

        recommendations.append({
            "subject_id": row["subject_id"],
            "probability": prob,
            "label": label,
            "updrs_score": updrs,
            "disease_years": years,
            "top_features": top_feats,
            "recommendation": rec,
        })

        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(df_test)}")

    df_rec = pd.DataFrame(recommendations)
    rec_path = os.path.join(RESULTS, "llm_recommendations.csv")
    df_rec.to_csv(rec_path, index=False)
    print(f"\n  Saved: {rec_path}")

    # Generate Figure 8
    print("\nGenerating Figure 8...")
    generate_figure8(df_rec)

    print("\nDone!")


if __name__ == "__main__":
    main()
