"""
Statistical Analysis
=====================
Group comparisons, correlation analysis, and statistical tests for the paper.

Outputs:
    - results/tables/group_comparison.csv (Mann-Whitney + FDR)
    - results/figures/correlation_heatmap.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for _f in ['/home/kartic/miniforge3/fonts/arial.ttf', '/home/kartic/miniforge3/fonts/arialbd.ttf',
           '/home/kartic/miniforge3/fonts/ariali.ttf', '/home/kartic/miniforge3/fonts/arialbi.ttf']:
    if os.path.isfile(_f):
        fm.fontManager.addfont(_f)
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import yaml
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

FIGURES = os.path.join(PROJECT_ROOT, config["paths"]["figures"])
TABLES = os.path.join(PROJECT_ROOT, "results/tables")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

DPI = config["figures"]["dpi"]
plt.rcParams.update({
    "font.family": config["figures"]["font_family"],
    "font.size": config["figures"]["font_size"],
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def rank_biserial(U, n1, n2):
    """Compute rank-biserial correlation from Mann-Whitney U."""
    return 1 - (2 * U) / (n1 * n2)


def main():
    print("=" * 60)
    print("Statistical Analysis")
    print("=" * 60)

    # Load fused dataset
    fused_path = os.path.join(PROJECT_ROOT, "data/processed/fused/multimodal_dataset.csv")
    df = pd.read_csv(fused_path)

    lfp_cols = [c for c in df.columns if c.startswith("lfp_")]
    bbb_cols = [c for c in df.columns if c.startswith("bbb_")]
    feature_cols = lfp_cols + bbb_cols

    # --- 1. Group comparison: PD vs Healthy ---
    print("\n[1/3] Group comparison (PD vs Healthy)...")
    group_pd = df[df["label"] == 1]
    group_hc = df[df["label"] == 0]

    comparison_results = []
    for col in feature_cols:
        vals_pd = group_pd[col].dropna().values
        vals_hc = group_hc[col].dropna().values

        if len(vals_pd) < 3 or len(vals_hc) < 3:
            continue

        U, p = stats.mannwhitneyu(vals_pd, vals_hc, alternative="two-sided")
        d = cohens_d(vals_pd, vals_hc)
        r = rank_biserial(U, len(vals_pd), len(vals_hc))

        comparison_results.append({
            "Feature": col,
            "PD_mean": np.mean(vals_pd),
            "PD_std": np.std(vals_pd),
            "HC_mean": np.mean(vals_hc),
            "HC_std": np.std(vals_hc),
            "U_statistic": U,
            "p_value": p,
            "cohens_d": d,
            "rank_biserial_r": r,
        })

    df_comp = pd.DataFrame(comparison_results)

    # FDR correction (Benjamini-Hochberg)
    reject, p_corrected, _, _ = multipletests(df_comp["p_value"], method="fdr_bh")
    df_comp["p_corrected_fdr"] = p_corrected
    df_comp["significant_fdr"] = reject

    # Significance stars
    def sig_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return "ns"

    df_comp["significance"] = df_comp["p_corrected_fdr"].apply(sig_stars)
    df_comp = df_comp.sort_values("p_corrected_fdr")

    comp_path = os.path.join(TABLES, "group_comparison.csv")
    df_comp.to_csv(comp_path, index=False)
    print(f"  Significant features (FDR<0.05): {df_comp['significant_fdr'].sum()} / {len(df_comp)}")
    print(f"  Saved: {comp_path}")

    print("\n  Top 10 most significant features:")
    for _, row in df_comp.head(10).iterrows():
        print(f"    {row['Feature']:40s} p={row['p_corrected_fdr']:.2e} d={row['cohens_d']:+.3f} {row['significance']}")

    # --- 2. Spearman correlation heatmap ---
    print("\n[2/3] Spearman correlation analysis...")

    # Select top features from each modality for readable heatmap
    top_lfp = df_comp[df_comp["Feature"].str.startswith("lfp_")].head(10)["Feature"].tolist()
    top_bbb = df_comp[df_comp["Feature"].str.startswith("bbb_")].head(10)["Feature"].tolist()
    top_features = top_lfp + top_bbb

    if len(top_features) > 2:
        corr_data = df[top_features].dropna()
        corr_matrix = corr_data.corr(method="spearman")

        # Clean names for display
        clean_names = [c.replace("lfp_", "LFP: ").replace("bbb_", "BBB: ")[:25] for c in top_features]
        corr_matrix.index = clean_names
        corr_matrix.columns = clean_names

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Spearman ρ"},
        )
        ax.set_title("Cross-Modal Feature Correlation (Spearman)", fontsize=12, fontweight="bold")
        plt.tight_layout()

        heatmap_path = os.path.join(FIGURES, "correlation_heatmap.png")
        plt.savefig(heatmap_path, dpi=DPI, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {heatmap_path}")
    else:
        print("  [WARN] Not enough features for correlation heatmap")

    # --- 3. Dataset characteristics table (Table 1) ---
    print("\n[3/3] Dataset characteristics (Table 1)...")

    table1_rows = []
    table1_rows.append({
        "Characteristic": "N",
        "PD (n={})".format(len(group_pd)): str(len(group_pd)),
        "HC (n={})".format(len(group_hc)): str(len(group_hc)),
        "p-value": "",
    })

    # Add key BBB features to Table 1
    key_features = {
        "bbb_q_albumin": "Q-albumin",
        "bbb_il6": "IL-6 (pg/mL)",
        "bbb_nfl": "NfL (pg/mL)",
        "bbb_crp": "CRP (mg/L)",
        "bbb_tnf_alpha": "TNF-α (pg/mL)",
        "bbb_alpha_synuclein": "α-Synuclein (pg/mL)",
        "bbb_tau": "Tau (pg/mL)",
    }

    for col, name in key_features.items():
        if col not in df.columns:
            continue
        pd_vals = group_pd[col].dropna()
        hc_vals = group_hc[col].dropna()
        _, p = stats.mannwhitneyu(pd_vals, hc_vals, alternative="two-sided")

        table1_rows.append({
            "Characteristic": name,
            f"PD (n={len(group_pd)})": f"{pd_vals.mean():.2f} ± {pd_vals.std():.2f}",
            f"HC (n={len(group_hc)})": f"{hc_vals.mean():.2f} ± {hc_vals.std():.2f}",
            "p-value": f"{p:.4f}",
        })

    df_table1 = pd.DataFrame(table1_rows)
    table1_path = os.path.join(TABLES, "table1_demographics.csv")
    df_table1.to_csv(table1_path, index=False)
    print(f"  Saved: {table1_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
