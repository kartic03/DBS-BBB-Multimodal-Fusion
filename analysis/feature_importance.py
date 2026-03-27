"""
Feature Importance Analysis (SHAP)
====================================
Computes SHAP values for XGBoost and fusion model.
Generates beeswarm plots, waterfall plots, and modality contribution table.

Outputs:
    - results/figures/shap_beeswarm.png
    - results/figures/shap_waterfall_*.png
    - results/tables/modality_contribution.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import yaml
import torch
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

FIGURES = os.path.join(PROJECT_ROOT, config["paths"]["figures"])
TABLES = os.path.join(PROJECT_ROOT, "results/tables")
CHECKPOINTS = os.path.join(PROJECT_ROOT, config["paths"]["checkpoints"])
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

DPI = config["figures"]["dpi"]
plt.rcParams.update({
    "font.family": config["figures"]["font_family"],
    "font.size": config["figures"]["font_size"],
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    print("=" * 60)
    print("SHAP Feature Importance Analysis")
    print("=" * 60)

    # Load fused dataset
    fused_path = os.path.join(PROJECT_ROOT, "data/processed/fused/multimodal_dataset.csv")
    df = pd.read_csv(fused_path)

    lfp_cols = [c for c in df.columns if c.startswith("lfp_")]
    bbb_cols = [c for c in df.columns if c.startswith("bbb_")]
    feature_cols = lfp_cols + bbb_cols

    df_test = df[df["split"] == "test"]
    df_train = df[df["split"] == "train"]

    X_train = df_train[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)
    y_train = df_train["label"].values
    y_test = df_test["label"].values

    # --- XGBoost SHAP (Early Fusion model for interpretability) ---
    print("\n[1/3] XGBoost SHAP analysis...")
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", random_state=config["model"]["seed"], n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)

    # Clean feature names for display
    display_names = [c.replace("lfp_", "LFP: ").replace("bbb_", "BBB: ") for c in feature_cols]

    # Beeswarm plot (top 20)
    print("  Generating beeswarm plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=display_names,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    beeswarm_path = os.path.join(FIGURES, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {beeswarm_path}")

    # Waterfall plots for 3 example patients
    print("  Generating waterfall plots...")
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_test,
        feature_names=display_names,
    )

    for i, idx in enumerate([0, len(X_test) // 2, -1]):
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation[idx], max_display=15, show=False)
        plt.tight_layout()
        wf_path = os.path.join(FIGURES, f"shap_waterfall_patient{i+1}.png")
        plt.savefig(wf_path, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {wf_path}")

    # --- Modality contribution (Table 3) ---
    print("\n[2/3] Computing modality contributions...")
    shap_abs = np.abs(shap_values)
    mean_shap = shap_abs.mean(axis=0)

    lfp_indices = [i for i, c in enumerate(feature_cols) if c.startswith("lfp_")]
    bbb_indices = [i for i, c in enumerate(feature_cols) if c.startswith("bbb_")]

    lfp_total = mean_shap[lfp_indices].sum()
    bbb_total = mean_shap[bbb_indices].sum()
    total = lfp_total + bbb_total

    modality_df = pd.DataFrame({
        "Modality": ["LFP Neural Signals", "BBB Biomarkers", "Total"],
        "Mean |SHAP|": [lfp_total, bbb_total, total],
        "Contribution (%)": [
            lfp_total / total * 100,
            bbb_total / total * 100,
            100.0,
        ],
    })
    mod_path = os.path.join(TABLES, "modality_contribution.csv")
    modality_df.to_csv(mod_path, index=False)
    print(f"  LFP contribution: {lfp_total/total*100:.1f}%")
    print(f"  BBB contribution: {bbb_total/total*100:.1f}%")
    print(f"  Saved: {mod_path}")

    # --- Top features per modality ---
    print("\n[3/3] Top features by modality...")
    feat_importance = pd.DataFrame({
        "feature": feature_cols,
        "display_name": display_names,
        "mean_abs_shap": mean_shap,
        "modality": ["LFP" if c.startswith("lfp_") else "BBB" for c in feature_cols],
    }).sort_values("mean_abs_shap", ascending=False)

    feat_path = os.path.join(TABLES, "feature_importance_ranked.csv")
    feat_importance.to_csv(feat_path, index=False)

    print("\n  Top 10 LFP features:")
    for _, row in feat_importance[feat_importance["modality"] == "LFP"].head(10).iterrows():
        print(f"    {row['display_name']:40s} SHAP={row['mean_abs_shap']:.4f}")

    print("\n  Top 10 BBB features:")
    for _, row in feat_importance[feat_importance["modality"] == "BBB"].head(10).iterrows():
        print(f"    {row['display_name']:40s} SHAP={row['mean_abs_shap']:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
