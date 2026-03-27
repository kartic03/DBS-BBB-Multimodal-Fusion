"""
Data Fusion Pipeline
=====================
Merges LFP features + BBB biomarker features into a unified multimodal
dataset for the cross-attention fusion model.

For PESD subjects (simulated LFP): synthetically assigns BBB biomarker
profiles drawn from matched distributions by disease severity.

Outputs:
    - data/processed/fused/multimodal_dataset.csv
    - data/splits/train_val_test_splits.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

# ============================================================
# Load config
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

SEED = config["model"]["seed"]
np.random.seed(SEED)
N_FOLDS = config["training"]["n_folds"]
TEST_SIZE = config["training"]["test_size"]
VAL_SIZE = config["training"]["val_size"]

DATA_PROC = os.path.join(PROJECT_ROOT, config["paths"]["data_processed"])
os.makedirs(os.path.join(DATA_PROC, "fused"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "data", "splits"), exist_ok=True)


def main():
    print("=" * 60)
    print("Data Fusion Pipeline")
    print("=" * 60)

    # --- Load LFP tabular features ---
    lfp_path = os.path.join(DATA_PROC, "lfp_features", "lfp_tabular_features.csv")
    if not os.path.isfile(lfp_path):
        print(f"[ERROR] LFP features not found: {lfp_path}")
        print("  Run preprocessing/lfp_preprocessing.py first.")
        sys.exit(1)

    print("\n[1/4] Loading LFP features...")
    df_lfp = pd.read_csv(lfp_path)
    print(f"  LFP: {df_lfp.shape[0]} subjects, {df_lfp.shape[1]} columns")

    # --- Load BBB features ---
    bbb_path = os.path.join(DATA_PROC, "bbb_features", "bbb_features.csv")
    if not os.path.isfile(bbb_path):
        print(f"[ERROR] BBB features not found: {bbb_path}")
        print("  Run preprocessing/bbb_feature_extraction.py first.")
        sys.exit(1)

    print("\n[2/4] Loading BBB features...")
    df_bbb = pd.read_csv(bbb_path)
    print(f"  BBB: {df_bbb.shape[0]} subjects, {df_bbb.shape[1]} columns")

    # --- Match and merge ---
    print("\n[3/4] Fusing modalities...")

    # Pair LFP and BBB subjects by label (random matching, since different datasets)
    lfp_meta = ["subject_id", "label", "source", "n_epochs"]
    lfp_feature_cols = [c for c in df_lfp.columns if c not in lfp_meta]

    bbb_meta = ["subject_id", "label", "group", "dbs_responder",
                "updrs_iii_baseline", "updrs_pct_improvement",
                "has_imputed", "data_source"]
    bbb_feature_cols = [c for c in df_bbb.columns if c not in bbb_meta]

    # Prefix feature columns to avoid collision
    lfp_rename = {c: f"lfp_{c}" for c in lfp_feature_cols}
    bbb_rename = {c: f"bbb_{c}" for c in bbb_feature_cols}

    lfp_feat_cols_new = [f"lfp_{c}" for c in lfp_feature_cols]
    bbb_feat_cols_new = [f"bbb_{c}" for c in bbb_feature_cols]

    # Match by label: pair PD LFP with PD BBB, HC LFP with HC BBB (vectorized)
    fused_parts = []

    for label_val in [0, 1]:
        lfp_group = df_lfp[df_lfp["label"] == label_val].reset_index(drop=True)
        bbb_group = df_bbb[df_bbb["label"] == label_val].reset_index(drop=True)

        if len(lfp_group) == 0 or len(bbb_group) == 0:
            print(f"  [WARN] No data for label={label_val} in one modality")
            continue

        n_lfp = len(lfp_group)
        bbb_indices = np.random.choice(len(bbb_group), size=n_lfp,
                                        replace=len(bbb_group) < n_lfp)

        # Build LFP side
        lfp_part = lfp_group[["subject_id", "label", "source"]].copy()
        for col in lfp_feature_cols:
            lfp_part[f"lfp_{col}"] = lfp_group[col].values

        # Build BBB side (sampled)
        bbb_sampled = bbb_group.iloc[bbb_indices].reset_index(drop=True)
        for col in bbb_feature_cols:
            lfp_part[f"bbb_{col}"] = bbb_sampled[col].values

        # Add BBB metadata
        lfp_part["dbs_responder"] = bbb_sampled["dbs_responder"].values if "dbs_responder" in bbb_sampled.columns else label_val
        lfp_part["updrs_iii_baseline"] = bbb_sampled["updrs_iii_baseline"].values if "updrs_iii_baseline" in bbb_sampled.columns else np.nan
        lfp_part["updrs_pct_improvement"] = bbb_sampled["updrs_pct_improvement"].values if "updrs_pct_improvement" in bbb_sampled.columns else np.nan

        fused_parts.append(lfp_part)

    df_fused = pd.concat(fused_parts, ignore_index=True)
    print(f"  Fused dataset: {df_fused.shape[0]} subjects, {df_fused.shape[1]} columns")
    print(f"    LFP features: {len(lfp_feat_cols_new)}")
    print(f"    BBB features: {len(bbb_feat_cols_new)}")
    print(f"    Labels: {df_fused['label'].value_counts().to_dict()}")

    # --- Create train/val/test splits ---
    print("\n[4/4] Creating stratified splits...")

    # Use DBS responder as stratification target for PD, label for all
    # First: hold out test set (15%)
    stratify_col = df_fused["label"].values

    from sklearn.model_selection import train_test_split

    idx_trainval, idx_test = train_test_split(
        np.arange(len(df_fused)),
        test_size=TEST_SIZE,
        stratify=stratify_col,
        random_state=SEED,
    )
    # Split train/val from trainval
    stratify_trainval = stratify_col[idx_trainval]
    relative_val_size = VAL_SIZE / (1 - TEST_SIZE)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=relative_val_size,
        stratify=stratify_trainval,
        random_state=SEED,
    )

    df_fused["split"] = "train"
    df_fused.loc[idx_val, "split"] = "val"
    df_fused.loc[idx_test, "split"] = "test"

    # Also create 5-fold CV indices for training
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df_fused["cv_fold"] = -1
    train_mask = df_fused["split"] == "train"
    train_df = df_fused[train_mask]
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
        actual_indices = train_df.index[val_idx]
        df_fused.loc[actual_indices, "cv_fold"] = fold

    # Save
    fused_path = os.path.join(DATA_PROC, "fused", "multimodal_dataset.csv")
    df_fused.to_csv(fused_path, index=False)
    print(f"  Saved: {df_fused.shape} → {fused_path}")

    # Save split summary
    split_summary = df_fused.groupby("split")["label"].value_counts().unstack(fill_value=0)
    print(f"\n  Split distribution:")
    print(split_summary.to_string())

    splits_path = os.path.join(PROJECT_ROOT, "data", "splits", "train_val_test_splits.csv")
    df_fused[["subject_id", "label", "source", "split", "cv_fold"]].to_csv(splits_path, index=False)
    print(f"  Split indices: → {splits_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
