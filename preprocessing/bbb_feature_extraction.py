"""
BBB Feature Extraction Pipeline
================================
Extracts Blood-Brain Barrier permeability and neuroinflammation biomarker
features from PPMI data. If PPMI data is not yet available, generates
synthetic BBB data matching published distributions.

Reference ranges: Lindblom et al. 2021, PPMI published norms.

Outputs:
    - data/processed/bbb_features/bbb_features.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

# ============================================================
# Load config
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

SEED = config["model"]["seed"]
np.random.seed(SEED)
N_FEATURES = config["bbb"]["n_features"]
IMPUTE_K = config["bbb"]["impute_k"]

DATA_RAW = os.path.join(PROJECT_ROOT, config["paths"]["data_raw"])
DATA_PROC = os.path.join(PROJECT_ROOT, config["paths"]["data_processed"])
os.makedirs(os.path.join(DATA_PROC, "bbb_features"), exist_ok=True)


# ============================================================
# PPMI Data Loading (when real data is available)
# ============================================================

def load_ppmi_real(ppmi_dir):
    """Load real PPMI CSV files and extract BBB/inflammation features.

    Uses baseline (BL) visit data. Merges biospecimen biomarkers, blood
    chemistry, demographics, PD diagnosis history, and MDS-UPDRS Part III.

    Returns DataFrame with subject_id, features, and labels, or None if
    required files are missing.
    """
    import gc

    # --- Locate files (match any date suffix) ---
    def find_file(prefix):
        for f in os.listdir(ppmi_dir):
            if f.startswith(prefix) and f.endswith(".csv"):
                return os.path.join(ppmi_dir, f)
        return None

    biospec_path = find_file("Current_Biospecimen_Analysis_Results")
    blood_path = find_file("Blood_Chemistry")
    updrs_path = find_file("MDS-UPDRS_Part_III")
    demo_path = find_file("Demographics")
    dx_path = find_file("PD_Diagnosis_History")

    if biospec_path is None or updrs_path is None:
        print("  [WARN] Required PPMI files not found — falling back to synthetic")
        return None

    print("  Loading real PPMI data...")

    # ==========================================================
    # 1. Biospecimen Analysis — CSF & blood biomarkers (long format)
    # ==========================================================
    # This file is large (~177 MB). Read in chunks, filter to BL visit
    # and our target biomarkers to keep memory manageable.
    target_tests = {
        # CSF biomarkers
        "CSF Alpha-synuclein": "alpha_synuclein",
        "ABeta 1-42": "amyloid_beta",
        "NfL": "nfl",
        "BD tTau": "tau",
        "GFAP": "gfap",
        "S100": "s100",
        "IL-6": "il6",
        "CSF Albumin": "csf_albumin",
        "Plasma Albumin": "plasma_albumin",
        "IgG": "csf_igg",
        "Serum IGF-1": "serum_igf1",
    }

    print(f"  Reading biospecimen file (chunked)...")
    biomarker_rows = []
    reader = pd.read_csv(
        biospec_path,
        usecols=["PATNO", "COHORT", "CLINICAL_EVENT", "TESTNAME", "TESTVALUE"],
        low_memory=False,
        chunksize=100000,
        dtype={"PATNO": str, "TESTVALUE": str},
    )
    for chunk in reader:
        bl = chunk[chunk["CLINICAL_EVENT"] == "BL"]
        matched = bl[bl["TESTNAME"].isin(target_tests.keys())]
        if len(matched) > 0:
            biomarker_rows.append(matched)
        gc.collect()

    if not biomarker_rows:
        print("  [WARN] No baseline biomarker data found")
        return None

    bio_df = pd.concat(biomarker_rows, ignore_index=True)
    print(f"  Biospecimen: {len(bio_df)} rows at BL for target tests")

    # Convert TESTVALUE to numeric
    bio_df["TESTVALUE"] = pd.to_numeric(bio_df["TESTVALUE"], errors="coerce")

    # Pivot: one row per PATNO, columns = biomarker names
    bio_df["feature_name"] = bio_df["TESTNAME"].map(target_tests)
    bio_pivot = bio_df.groupby(["PATNO", "feature_name"])["TESTVALUE"].median().reset_index()
    bio_pivot = bio_pivot.pivot(index="PATNO", columns="feature_name", values="TESTVALUE").reset_index()

    # Get cohort per subject
    cohort_map = bio_df.groupby("PATNO")["COHORT"].first().to_dict()

    del bio_df, biomarker_rows
    gc.collect()
    print(f"  Pivoted biomarkers: {bio_pivot.shape}")

    # ==========================================================
    # 2. Blood Chemistry — WBC, serum albumin, total protein
    # ==========================================================
    blood_features = {}
    if blood_path:
        print(f"  Reading blood chemistry (chunked)...")
        blood_tests = {
            "WBC": "wbc",
            "Albumin-BCG": "serum_albumin",
            "Albumin-QT": "serum_albumin_qt",
            "Total Protein": "total_protein",
        }
        blood_rows = []
        reader = pd.read_csv(
            blood_path,
            usecols=["PATNO", "EVENT_ID", "LTSTNAME", "LSIRES"],
            low_memory=False,
            chunksize=100000,
            dtype={"PATNO": str},
        )
        for chunk in reader:
            sc = chunk[(chunk["EVENT_ID"] == "SC") & (chunk["LTSTNAME"].isin(blood_tests.keys()))]
            if len(sc) > 0:
                blood_rows.append(sc)
            gc.collect()

        if blood_rows:
            bl_blood = pd.concat(blood_rows, ignore_index=True)
            for test_name, feat_name in blood_tests.items():
                subset = bl_blood[bl_blood["LTSTNAME"] == test_name][["PATNO", "LSIRES"]].copy()
                subset["LSIRES"] = pd.to_numeric(subset["LSIRES"], errors="coerce")
                subset = subset.groupby("PATNO")["LSIRES"].median().reset_index()
                subset.columns = ["PATNO", feat_name]
                blood_features[feat_name] = subset
            del bl_blood
        del blood_rows
        gc.collect()

    # ==========================================================
    # 3. MDS-UPDRS Part III — motor scores (DBS response proxy)
    # ==========================================================
    print(f"  Reading MDS-UPDRS Part III...")
    updrs = pd.read_csv(
        updrs_path,
        usecols=["PATNO", "EVENT_ID", "NP3TOT"],
        low_memory=False,
        dtype={"PATNO": str},
    )
    # Baseline score
    updrs_bl = updrs[updrs["EVENT_ID"] == "BL"].groupby("PATNO")["NP3TOT"].median().reset_index()
    updrs_bl.columns = ["PATNO", "updrs_iii_baseline"]

    # Best follow-up score (proxy for treatment response)
    followup_events = [f"V{i:02d}" for i in range(1, 20)]
    updrs_fu = updrs[updrs["EVENT_ID"].isin(followup_events)]
    updrs_best = updrs_fu.groupby("PATNO")["NP3TOT"].min().reset_index()
    updrs_best.columns = ["PATNO", "updrs_iii_best_followup"]

    updrs_merged = updrs_bl.merge(updrs_best, on="PATNO", how="left")
    # Compute improvement percentage
    updrs_merged["updrs_pct_improvement"] = (
        (updrs_merged["updrs_iii_baseline"] - updrs_merged["updrs_iii_best_followup"])
        / updrs_merged["updrs_iii_baseline"].clip(lower=1)
    ).clip(0, 1)
    # DBS responder: ≥30% improvement
    updrs_merged["dbs_responder"] = (updrs_merged["updrs_pct_improvement"] >= 0.30).astype(int)

    del updrs, updrs_fu
    gc.collect()

    # ==========================================================
    # 4. Demographics — age, sex
    # ==========================================================
    if demo_path:
        print(f"  Reading demographics...")
        demo = pd.read_csv(
            demo_path,
            usecols=["PATNO", "BIRTHDT", "SEX"],
            low_memory=False,
            dtype={"PATNO": str},
        )
        demo = demo.drop_duplicates(subset="PATNO")
        # Compute approximate age (BIRTHDT is MM/YYYY format)
        demo["birth_year"] = demo["BIRTHDT"].str.extract(r"(\d{4})").astype(float)
        demo["age"] = 2026 - demo["birth_year"]
        # SEX: 0=Female, 1=Male in PPMI (but stored as 0/1)
        demo["sex"] = demo["SEX"].astype(float)
        demo = demo[["PATNO", "age", "sex"]]
    else:
        demo = pd.DataFrame(columns=["PATNO", "age", "sex"])

    # ==========================================================
    # 5. PD Diagnosis History — disease duration
    # ==========================================================
    if dx_path:
        print(f"  Reading PD diagnosis history...")
        dx = pd.read_csv(
            dx_path,
            usecols=["PATNO", "SXDT", "PDDXDT"],
            low_memory=False,
            dtype={"PATNO": str},
        )
        dx = dx.drop_duplicates(subset="PATNO")
        dx["dx_year"] = dx["PDDXDT"].str.extract(r"(\d{4})").astype(float)
        dx["sx_year"] = dx["SXDT"].str.extract(r"(\d{4})").astype(float)
        dx["disease_duration_years"] = 2026 - dx["dx_year"]
        dx = dx[["PATNO", "disease_duration_years"]]
    else:
        dx = pd.DataFrame(columns=["PATNO", "disease_duration_years"])

    # ==========================================================
    # 6. Merge everything
    # ==========================================================
    print(f"  Merging all tables...")
    df = bio_pivot.copy()

    # Merge blood chemistry features
    for feat_name, feat_df in blood_features.items():
        df = df.merge(feat_df, on="PATNO", how="left")

    # If both albumin sources exist, combine them
    if "serum_albumin" in df.columns and "serum_albumin_qt" in df.columns:
        df["serum_albumin"] = df["serum_albumin"].fillna(df["serum_albumin_qt"])
        df.drop(columns=["serum_albumin_qt"], inplace=True)

    # Merge UPDRS
    df = df.merge(updrs_merged, on="PATNO", how="left")

    # Merge demographics
    df = df.merge(demo, on="PATNO", how="left")

    # Merge diagnosis history
    df = df.merge(dx, on="PATNO", how="left")

    # ==========================================================
    # 7. Compute derived BBB markers
    # ==========================================================
    # Q-albumin = CSF albumin / plasma albumin × 1000
    if "csf_albumin" in df.columns and "plasma_albumin" in df.columns:
        df["q_albumin"] = df["csf_albumin"] / (df["plasma_albumin"] + 1e-6) * 1000
        df["bbb_disrupted"] = (df["q_albumin"] > 9).astype(int)
    elif "csf_albumin" in df.columns and "serum_albumin" in df.columns:
        # serum_albumin from blood chem is in g/dL, convert to same unit
        df["q_albumin"] = df["csf_albumin"] / (df["serum_albumin"] * 10 + 1e-6) * 1000
        df["bbb_disrupted"] = (df["q_albumin"] > 9).astype(int)

    # IgG index
    if "csf_igg" in df.columns and "q_albumin" in df.columns:
        df["igg_index"] = df["csf_igg"] / (df["q_albumin"] + 1e-6)

    # ==========================================================
    # 8. Assign labels and groups
    # ==========================================================
    df["cohort"] = df["PATNO"].map(cohort_map)
    # Label: PD=1, Control/SWEDD=0; Prodromal depends on analysis
    df["group"] = df["cohort"].map({
        "PD": "PD", "Control": "HC", "SWEDD": "HC", "Prodromal": "Prodromal"
    }).fillna("Unknown")
    # For our binary classification: PD vs non-PD
    df["label"] = (df["group"] == "PD").astype(int)

    # Fill missing disease_duration for non-PD
    df.loc[df["label"] == 0, "disease_duration_years"] = 0

    # Fill missing UPDRS for controls
    df.loc[df["label"] == 0, "updrs_iii_baseline"] = df.loc[
        df["label"] == 0, "updrs_iii_baseline"
    ].fillna(0)
    df.loc[df["label"] == 0, "updrs_pct_improvement"] = 0
    df.loc[df["label"] == 0, "dbs_responder"] = 0

    # Rename PATNO to subject_id
    df.rename(columns={"PATNO": "subject_id"}, inplace=True)
    df["subject_id"] = "ppmi_" + df["subject_id"].astype(str)

    # Drop rows with no biomarker data at all
    biomarker_cols = [c for c in df.columns if c not in [
        "subject_id", "label", "group", "cohort", "dbs_responder",
        "updrs_iii_baseline", "updrs_iii_best_followup",
        "updrs_pct_improvement", "bbb_disrupted",
    ]]
    df = df.dropna(subset=biomarker_cols, how="all")

    # Drop cohort column (not needed downstream)
    df.drop(columns=["cohort"], inplace=True, errors="ignore")

    print(f"  Final PPMI dataset: {len(df)} subjects")
    print(f"    PD: {(df['label']==1).sum()}, HC/Other: {(df['label']==0).sum()}")
    print(f"    DBS responders: {df['dbs_responder'].sum()} / {(df['label']==1).sum()} PD")
    print(f"    Features: {len(biomarker_cols)}")
    print(f"    Missingness per feature:")
    for col in biomarker_cols[:15]:
        pct = df[col].isna().mean() * 100
        print(f"      {col}: {pct:.1f}% missing")

    return df


# ============================================================
# Synthetic BBB Data Generation
# ============================================================

def generate_synthetic_bbb(n_subjects, pd_ratio=0.7):
    """Generate synthetic BBB + neuroinflammation biomarker data.

    Based on published reference ranges:
    - Lindblom et al. 2021 (BBB permeability in PD)
    - Ahn et al. 2020 (neuroinflammation markers)
    - PPMI published norms

    Args:
        n_subjects: Total number of subjects to generate
        pd_ratio: Fraction of PD subjects (rest are healthy controls)
    """
    n_pd = int(n_subjects * pd_ratio)
    n_hc = n_subjects - n_pd

    records = []

    for group, n, label in [("PD", n_pd, 1), ("HC", n_hc, 0)]:
        for i in range(n):
            subj = {}
            subj["subject_id"] = f"ppmi_{group}_{i:04d}"
            subj["label"] = label
            subj["group"] = group

            # Demographics
            subj["age"] = np.random.normal(65 if group == "PD" else 62, 8)
            subj["sex"] = np.random.choice([0, 1])  # 0=F, 1=M
            subj["disease_duration_years"] = max(0, np.random.normal(7, 4)) if group == "PD" else 0

            # --- BBB Permeability Markers ---
            # Q-albumin (CSF albumin / serum albumin × 1000)
            # Normal: 4-7, Elevated in PD: 6-12, BBB disrupted > 9
            if group == "PD":
                subj["q_albumin"] = np.random.lognormal(np.log(8.5), 0.35)
            else:
                subj["q_albumin"] = np.random.lognormal(np.log(5.5), 0.25)
            subj["bbb_disrupted"] = int(subj["q_albumin"] > 9)

            # CSF albumin (mg/L) — PD elevated
            subj["csf_albumin"] = subj["q_albumin"] * np.random.normal(42, 4) / 1000 * 1000

            # Serum albumin (g/dL)
            subj["serum_albumin"] = np.random.normal(4.2, 0.4)

            # IgG index = (CSF IgG / serum IgG) / Q-albumin
            csf_igg = np.random.lognormal(np.log(35 if group == "PD" else 28), 0.3)
            serum_igg = np.random.normal(1100, 200)
            subj["igg_index"] = (csf_igg / serum_igg) / (subj["q_albumin"] / 1000 + 1e-6)
            subj["csf_igg"] = csf_igg
            subj["serum_igg"] = serum_igg

            # --- Neuroinflammation Markers ---
            # IL-6 (pg/mL) — right-skewed, elevated in PD
            subj["il6"] = np.random.lognormal(
                np.log(3.5) if group == "PD" else np.log(1.8), 0.6
            )

            # TNF-α (pg/mL) — right-skewed
            subj["tnf_alpha"] = np.random.lognormal(
                np.log(8.5) if group == "PD" else np.log(5.2), 0.4
            )

            # CRP (mg/L) — right-skewed
            subj["crp"] = np.random.lognormal(
                np.log(3.2) if group == "PD" else np.log(1.5), 0.7
            )

            # Fibrinogen (mg/dL)
            subj["fibrinogen"] = np.random.normal(
                320 if group == "PD" else 280, 50
            )

            # WBC (×10³/μL)
            subj["wbc"] = np.random.normal(7.0, 1.8)

            # --- CSF Biomarkers ---
            # Neurofilament light chain (NfL, pg/mL) — axonal damage
            subj["nfl"] = np.random.lognormal(
                np.log(22) if group == "PD" else np.log(12), 0.5
            )

            # α-synuclein (pg/mL) — decreased in PD CSF
            subj["alpha_synuclein"] = np.random.normal(
                1200 if group == "PD" else 1800, 400
            )

            # Total tau (pg/mL)
            subj["tau"] = np.random.lognormal(
                np.log(220) if group == "PD" else np.log(180), 0.3
            )

            # Amyloid-β 1-42 (pg/mL) — decreased in neurodegeneration
            subj["amyloid_beta"] = np.random.normal(
                800 if group == "PD" else 950, 150
            )

            # --- UPDRS-III Motor Score ---
            # DBS responder: UPDRS-III drops ≥30%
            if group == "PD":
                subj["updrs_iii_baseline"] = max(10, np.random.normal(35, 12))
                # Simulate post-DBS response
                improvement = np.random.beta(2, 3)  # 0 to 1
                subj["updrs_iii_post_dbs"] = subj["updrs_iii_baseline"] * (1 - improvement)
                pct_change = improvement
                subj["updrs_pct_improvement"] = pct_change
                subj["dbs_responder"] = int(pct_change >= 0.30)
            else:
                subj["updrs_iii_baseline"] = max(0, np.random.normal(5, 3))
                subj["updrs_iii_post_dbs"] = subj["updrs_iii_baseline"]
                subj["updrs_pct_improvement"] = 0
                subj["dbs_responder"] = 0

            records.append(subj)

    df = pd.DataFrame(records)

    # Introduce ~5% missing values randomly (realistic for clinical data)
    biomarker_cols = ["q_albumin", "csf_albumin", "serum_albumin", "igg_index",
                      "csf_igg", "serum_igg", "il6", "tnf_alpha", "crp",
                      "fibrinogen", "wbc", "nfl", "alpha_synuclein", "tau", "amyloid_beta"]
    mask = np.random.random(size=(len(df), len(biomarker_cols))) < 0.05
    for i, col in enumerate(biomarker_cols):
        df.loc[mask[:, i], col] = np.nan

    return df


# ============================================================
# Feature Engineering
# ============================================================

def engineer_bbb_features(df):
    """Compute derived BBB and inflammation features."""

    # Log1p transform for right-skewed markers
    for col in ["il6", "tnf_alpha", "crp", "nfl", "tau", "gfap", "s100", "serum_igf1"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    # Composite inflammation score (z-scored sum of available inflammatory markers)
    inflamm_cols = ["il6", "crp", "tnf_alpha", "gfap", "s100"]
    existing = [c for c in inflamm_cols if c in df.columns and df[c].notna().sum() > 10]
    if existing:
        scaler = StandardScaler()
        z_scores = scaler.fit_transform(df[existing].fillna(df[existing].median()))
        df["inflammation_score"] = z_scores.sum(axis=1)

    # Q-albumin / NfL ratio (BBB leak + axonal damage interaction)
    if "q_albumin" in df.columns and "nfl" in df.columns:
        df["qalb_nfl_ratio"] = df["q_albumin"] / (df["nfl"] + 1e-6)

    # Tau / amyloid-β ratio (neurodegeneration index)
    if "tau" in df.columns and "amyloid_beta" in df.columns:
        df["tau_abeta_ratio"] = df["tau"] / (df["amyloid_beta"] + 1e-6)

    # Age-adjusted Q-albumin (Q-albumin increases with age)
    if "q_albumin" in df.columns and "age" in df.columns:
        df["q_albumin_age_adj"] = df["q_albumin"] - 0.05 * (df["age"] - 60)

    # NfL / tau ratio (differential neurodegeneration)
    if "nfl" in df.columns and "tau" in df.columns:
        df["nfl_tau_ratio"] = df["nfl"] / (df["tau"] + 1e-6)

    # GFAP / NfL ratio (astrocytic vs axonal damage)
    if "gfap" in df.columns and "nfl" in df.columns:
        df["gfap_nfl_ratio"] = df["gfap"] / (df["nfl"] + 1e-6)

    return df


# ============================================================
# Missing Data Handling
# ============================================================

def impute_missing(df, feature_cols):
    """KNN imputation for missing biomarker values."""
    imputer = KNNImputer(n_neighbors=IMPUTE_K)
    df_features = df[feature_cols].copy()

    # Track which values were imputed
    was_missing = df_features.isna()
    n_missing = was_missing.sum().sum()

    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_features),
        columns=feature_cols,
        index=df.index,
    )
    df[feature_cols] = df_imputed

    # Flag imputed rows
    df["n_imputed_values"] = was_missing.sum(axis=1)
    df["has_imputed"] = (df["n_imputed_values"] > 0).astype(int)

    print(f"  Imputed {n_missing} missing values across {was_missing.any(axis=1).sum()} subjects")
    return df


# ============================================================
# Feature Selection
# ============================================================

def select_features(df, feature_cols, label_col="dbs_responder", max_features=N_FEATURES):
    """Select top features using correlation filtering + mutual information."""

    # Step 1: Remove highly correlated features (|r| > 0.85)
    corr_matrix = df[feature_cols].corr(method="pearson").abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
    remaining = [c for c in feature_cols if c not in to_drop]
    print(f"  Correlation filter: {len(feature_cols)} → {len(remaining)} features (dropped {len(to_drop)})")

    # Step 2: Mutual information ranking
    X = df[remaining].fillna(0)
    y = df[label_col]
    mi_scores = mutual_info_classif(X, y, random_state=SEED)
    mi_df = pd.DataFrame({"feature": remaining, "mi_score": mi_scores})
    mi_df = mi_df.sort_values("mi_score", ascending=False)

    selected = mi_df.head(max_features)["feature"].tolist()
    print(f"  Mutual information: top {len(selected)} features selected")
    print(f"  Top 5: {selected[:5]}")

    return selected


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("BBB Feature Extraction Pipeline")
    print("=" * 60)

    # --- Try loading real PPMI data ---
    ppmi_dir = os.path.join(DATA_RAW, "ppmi")
    df = None
    data_source = "synthetic"

    if os.path.isdir(ppmi_dir):
        # Check for the actual PPMI CSV files (not just the dashboard xlsx)
        has_biospec = any(f.startswith("Current_Biospecimen") and f.endswith(".csv")
                         for f in os.listdir(ppmi_dir))
        has_updrs = any(f.startswith("MDS-UPDRS") and f.endswith(".csv")
                        for f in os.listdir(ppmi_dir))
        if has_biospec and has_updrs:
            print(f"\n[1/5] Loading real PPMI data...")
            df = load_ppmi_real(ppmi_dir)
            if df is not None:
                data_source = "ppmi"
                print(f"  Loaded: {len(df)} subjects ({df['label'].sum()} PD, "
                      f"{(df['label']==0).sum()} HC/Other)")
                print(f"  DBS responders: {df['dbs_responder'].sum()} / "
                      f"{df['label'].sum()} PD subjects")

    if df is None:
        # Load LFP labels to match subject count
        lfp_labels_path = os.path.join(PROJECT_ROOT, "data", "splits", "lfp_labels.csv")
        if os.path.isfile(lfp_labels_path):
            lfp_labels = pd.read_csv(lfp_labels_path)
            n_subjects = lfp_labels["subject_id"].nunique()
        else:
            n_subjects = 732  # default: 573 PD + 159 HC from PESD

        print(f"\n[1/5] Generating synthetic BBB data for {n_subjects} subjects...")
        print("  (PPMI data not yet available — using published distributions)")
        df = generate_synthetic_bbb(n_subjects)
        print(f"  Generated: {len(df)} subjects ({df['label'].sum()} PD, {(df['label']==0).sum()} HC)")
        print(f"  DBS responders: {df['dbs_responder'].sum()} / {df['label'].sum()} PD subjects")

    # --- Feature engineering ---
    print(f"\n[2/5] Engineering BBB features...")
    df = engineer_bbb_features(df)
    print(f"  Total columns: {len(df.columns)}")

    # --- Define feature columns ---
    # Exclude clinical metadata that would cause label leakage:
    # - disease_duration_years: always 0 for controls, >0 for PD (trivial separator)
    # - updrs_iii_*: motor scores directly encode disease status
    # - age, sex: demographics (keep as covariates but not BBB biomarkers)
    # - bbb_disrupted: derived binary from q_albumin (redundant)
    meta_cols = ["subject_id", "label", "group", "dbs_responder",
                 "updrs_iii_baseline", "updrs_iii_post_dbs", "updrs_pct_improvement",
                 "updrs_iii_best_followup",
                 "disease_duration_years", "age", "sex", "bbb_disrupted",
                 "n_imputed_values", "has_imputed"]
    feature_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    # --- Impute missing data ---
    print(f"\n[3/5] Imputing missing data (KNN, k={IMPUTE_K})...")
    numeric_feature_cols = [c for c in feature_cols if c in df.select_dtypes(include=[np.number]).columns]
    df = impute_missing(df, numeric_feature_cols)

    # --- Feature selection ---
    print(f"\n[4/5] Feature selection...")
    selected_features = select_features(df, numeric_feature_cols, label_col="dbs_responder")

    # --- Save outputs ---
    print(f"\n[5/5] Saving outputs...")

    # Save full feature set
    output_cols = ["subject_id", "label", "group", "dbs_responder",
                   "updrs_iii_baseline", "updrs_pct_improvement",
                   "has_imputed"] + selected_features
    output_cols = [c for c in output_cols if c in df.columns]
    df_out = df[output_cols].copy()
    df_out["data_source"] = data_source

    out_path = os.path.join(DATA_PROC, "bbb_features", "bbb_features.csv")
    df_out.to_csv(out_path, index=False)
    print(f"  BBB features: {df_out.shape} → {out_path}")

    # Save full unfiltered set too (for exploration)
    full_path = os.path.join(DATA_PROC, "bbb_features", "bbb_features_full.csv")
    df.to_csv(full_path, index=False)
    print(f"  Full features: {df.shape} → {full_path}")

    # --- Summary ---
    print(f"\n  Summary:")
    print(f"    Data source: {data_source}")
    print(f"    Subjects: {len(df_out)}")
    print(f"    Selected features: {len(selected_features)}")
    print(f"    DBS responders: {df_out['dbs_responder'].sum()} / {(df_out['label']==1).sum()} PD")

    # Print feature statistics
    print(f"\n  Feature summary (selected):")
    for feat in selected_features[:10]:
        vals = df_out[feat]
        print(f"    {feat}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
              f"range=[{vals.min():.3f}, {vals.max():.3f}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
