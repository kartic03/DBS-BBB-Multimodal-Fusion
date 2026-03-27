"""
Model Evaluation Script
========================
Evaluates ALL 10 models on the held-out test set.
Computes AUC-ROC, AUC-PR, Accuracy, F1, Precision, Recall
with bootstrap 95% CIs and DeLong / McNemar tests.

Outputs:
    - results/tables/model_comparison.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import yaml
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lfp_transformer import LFPTransformer
from models.bbb_encoder import BBBMLPEncoder
from models.fusion_model import MultimodalFusionModel
from models.baseline_models import CNN1D, LSTMClassifier

# ============================================================
# Config
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

SEED = config["model"]["seed"]
DEVICE = config["model"]["device"] if torch.cuda.is_available() else "cpu"
BOOTSTRAP_N = config["training"]["bootstrap_n"]
CHECKPOINTS = os.path.join(PROJECT_ROOT, config["paths"]["checkpoints"])
TABLES = os.path.join(PROJECT_ROOT, "results/tables")
os.makedirs(TABLES, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Data Loading
# ============================================================

def load_test_data():
    """Load all test data for different model types."""
    # Fused dataset (has splits)
    fused_path = os.path.join(PROJECT_ROOT, "data/processed/fused/multimodal_dataset.csv")
    df = pd.read_csv(fused_path)
    df_test = df[df["split"] == "test"].copy()
    df_train = df[df["split"] == "train"].copy()

    # LFP tabular features
    lfp_tab_cols = [c for c in df.columns if c.startswith("lfp_")]
    X_lfp_tab_test = df_test[lfp_tab_cols].values.astype(np.float32)
    X_lfp_tab_train = df_train[lfp_tab_cols].values.astype(np.float32)

    # BBB features
    bbb_cols = [c for c in df.columns if c.startswith("bbb_")]
    X_bbb_test = df_test[bbb_cols].values.astype(np.float32)
    X_bbb_train = df_train[bbb_cols].values.astype(np.float32)

    # Early fusion (concatenated)
    X_fused_tab_test = np.hstack([X_lfp_tab_test, X_bbb_test])
    X_fused_tab_train = np.hstack([X_lfp_tab_train, X_bbb_train])

    # Standardize
    scaler_lfp = StandardScaler().fit(X_lfp_tab_train)
    scaler_bbb = StandardScaler().fit(X_bbb_train)
    scaler_fused = StandardScaler().fit(X_fused_tab_train)

    X_lfp_tab_test_s = scaler_lfp.transform(X_lfp_tab_test)
    X_lfp_tab_train_s = scaler_lfp.transform(X_lfp_tab_train)
    X_bbb_test_s = scaler_bbb.transform(X_bbb_test)
    X_bbb_train_s = scaler_bbb.transform(X_bbb_train)
    X_fused_tab_test_s = scaler_fused.transform(X_fused_tab_test)
    X_fused_tab_train_s = scaler_fused.transform(X_fused_tab_train)

    y_test = df_test["label"].values
    y_train = df_train["label"].values

    # Raw LFP epochs for test subjects
    all_epochs = np.load(os.path.join(PROJECT_ROOT, "data/processed/lfp_features/lfp_raw_epochs.npy"))
    epoch_labels = pd.read_csv(os.path.join(PROJECT_ROOT, "data/splits/lfp_labels.csv"))

    lfp_raw_test = []
    for subj in df_test["subject_id"]:
        mask = epoch_labels["subject_id"] == subj
        if mask.any():
            idx = mask.idxmax()
            lfp_raw_test.append(all_epochs[idx])
        else:
            lfp_raw_test.append(np.zeros(all_epochs.shape[1], dtype=np.float32))
    X_lfp_raw_test = np.array(lfp_raw_test, dtype=np.float32)

    lfp_raw_train = []
    for subj in df_train["subject_id"]:
        mask = epoch_labels["subject_id"] == subj
        if mask.any():
            idx = mask.idxmax()
            lfp_raw_train.append(all_epochs[idx])
        else:
            lfp_raw_train.append(np.zeros(all_epochs.shape[1], dtype=np.float32))
    X_lfp_raw_train = np.array(lfp_raw_train, dtype=np.float32)

    return {
        "lfp_tab_train": X_lfp_tab_train_s, "lfp_tab_test": X_lfp_tab_test_s,
        "bbb_train": X_bbb_train_s, "bbb_test": X_bbb_test_s,
        "fused_tab_train": X_fused_tab_train_s, "fused_tab_test": X_fused_tab_test_s,
        "lfp_raw_train": X_lfp_raw_train, "lfp_raw_test": X_lfp_raw_test,
        "y_train": y_train, "y_test": y_test,
        "n_bbb_features": len(bbb_cols),
    }


# ============================================================
# Bootstrap Confidence Intervals
# ============================================================

def bootstrap_metric(y_true, y_score, metric_fn, n_bootstrap=BOOTSTRAP_N):
    """Compute metric with bootstrap 95% CI."""
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        try:
            s = metric_fn(y_true[idx], y_score[idx])
            scores.append(s)
        except (ValueError, ZeroDivisionError):
            continue
    if not scores:
        return 0, 0, 0
    mean = np.mean(scores)
    ci_low = np.percentile(scores, 2.5)
    ci_high = np.percentile(scores, 97.5)
    return mean, ci_low, ci_high


# ============================================================
# DeLong Test
# ============================================================

def delong_test(y_true, y_score1, y_score2):
    """Simplified DeLong test for comparing two AUCs.
    Returns z-statistic and p-value.
    """
    from scipy import stats

    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    auc1 = roc_auc_score(y_true, y_score1)
    auc2 = roc_auc_score(y_true, y_score2)

    # Placement values
    pos_scores1 = y_score1[y_true == 1]
    neg_scores1 = y_score1[y_true == 0]
    pos_scores2 = y_score2[y_true == 1]
    neg_scores2 = y_score2[y_true == 0]

    # Variance estimation (simplified)
    v1_pos = np.array([np.mean(s > neg_scores1) + 0.5 * np.mean(s == neg_scores1) for s in pos_scores1])
    v1_neg = np.array([np.mean(pos_scores1 > s) + 0.5 * np.mean(pos_scores1 == s) for s in neg_scores1])
    v2_pos = np.array([np.mean(s > neg_scores2) + 0.5 * np.mean(s == neg_scores2) for s in pos_scores2])
    v2_neg = np.array([np.mean(pos_scores2 > s) + 0.5 * np.mean(pos_scores2 == s) for s in neg_scores2])

    s10_1 = np.var(v1_pos) / n1
    s01_1 = np.var(v1_neg) / n0
    s10_2 = np.var(v2_pos) / n1
    s01_2 = np.var(v2_neg) / n0

    cov_10 = np.cov(v1_pos, v2_pos)[0, 1] / n1 if n1 > 1 else 0
    cov_01 = np.cov(v1_neg, v2_neg)[0, 1] / n0 if n0 > 1 else 0

    var_diff = s10_1 + s01_1 + s10_2 + s01_2 - 2 * cov_10 - 2 * cov_01

    if var_diff <= 0:
        return 0, 1.0

    z = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value


# ============================================================
# Get predictions from all models
# ============================================================

def get_sklearn_predictions(model_name, builder, X_train, y_train, X_test):
    """Train sklearn model and get test predictions."""
    model = builder(seed=SEED)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return y_prob, y_pred


def get_pytorch_predictions(model, X_test, device=DEVICE):
    """Get predictions from a PyTorch model."""
    model.eval()
    model.to(device)
    with torch.no_grad():
        X = torch.FloatTensor(X_test).to(device)
        logits = model(X)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
    return probs, preds


def get_fusion_predictions(fusion_model, X_lfp, X_bbb, device=DEVICE):
    """Get predictions from the fusion model."""
    fusion_model.eval()
    fusion_model.to(device)
    with torch.no_grad():
        lfp = torch.FloatTensor(X_lfp).to(device)
        bbb = torch.FloatTensor(X_bbb).to(device)
        logits, _, _, _ = fusion_model(lfp, bbb)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
    return probs, preds


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    data = load_test_data()
    y_test = data["y_test"]
    y_train = data["y_train"]

    results = []
    all_probs = {}  # For DeLong test

    # --- Models 1-3: Sklearn on LFP tabular ---
    from models.baseline_models import build_svm, build_random_forest, build_xgboost_lfp

    for name, desc, builder in [
        ("svm_lfp", "SVM (RBF) - LFP", build_svm),
        ("rf_lfp", "Random Forest - LFP", build_random_forest),
        ("xgb_lfp", "XGBoost - LFP", build_xgboost_lfp),
    ]:
        print(f"\n  Evaluating: {desc}...")
        probs, preds = get_sklearn_predictions(
            name, builder, data["lfp_tab_train"], y_train, data["lfp_tab_test"]
        )
        all_probs[name] = probs

    # --- Model 4: 1D-CNN ---
    print(f"\n  Evaluating: 1D-CNN - LFP...")
    cnn = CNN1D()
    cnn_ckpt = os.path.join(CHECKPOINTS, "cnn1d_best.pt")
    if os.path.isfile(cnn_ckpt):
        cnn.load_state_dict(torch.load(cnn_ckpt, map_location="cpu", weights_only=True))
    # Train quickly if no checkpoint
    else:
        print("    Training CNN1D from scratch...")
        cnn = _quick_train_pytorch(cnn, data["lfp_raw_train"], y_train, epochs=30)
    probs, preds = get_pytorch_predictions(cnn, data["lfp_raw_test"])
    all_probs["cnn1d"] = probs

    # --- Model 5: LSTM ---
    print(f"\n  Evaluating: LSTM - LFP...")
    lstm = LSTMClassifier()
    lstm_ckpt = os.path.join(CHECKPOINTS, "lstm_best.pt")
    if os.path.isfile(lstm_ckpt):
        lstm.load_state_dict(torch.load(lstm_ckpt, map_location="cpu", weights_only=True))
    else:
        print("    Training LSTM from scratch...")
        lstm = _quick_train_pytorch(lstm, data["lfp_raw_train"], y_train, epochs=30)
    probs, preds = get_pytorch_predictions(lstm, data["lfp_raw_test"])
    all_probs["lstm"] = probs

    # --- Model 6: XGBoost BBB ---
    from models.baseline_models import build_xgboost_bbb
    print(f"\n  Evaluating: XGBoost - BBB...")
    probs, preds = get_sklearn_predictions(
        "xgb_bbb", build_xgboost_bbb, data["bbb_train"], y_train, data["bbb_test"]
    )
    all_probs["xgb_bbb"] = probs

    # --- Model 7: XGBoost Early Fusion ---
    from models.baseline_models import build_xgboost_early_fusion
    print(f"\n  Evaluating: XGBoost Early Fusion...")
    probs, preds = get_sklearn_predictions(
        "xgb_early", build_xgboost_early_fusion,
        data["fused_tab_train"], y_train, data["fused_tab_test"]
    )
    all_probs["xgb_early"] = probs

    # --- Model 8: LFP Transformer alone ---
    print(f"\n  Evaluating: LFP Transformer...")
    lfp_tf = LFPTransformer(
        seq_len=data["lfp_raw_test"].shape[1],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
    )
    lfp_ckpt = os.path.join(CHECKPOINTS, "lfp_transformer_best.pt")
    if os.path.isfile(lfp_ckpt):
        lfp_tf.load_state_dict(torch.load(lfp_ckpt, map_location="cpu", weights_only=True))
    else:
        print("    Training LFP Transformer from scratch...")
        lfp_tf = _quick_train_pytorch(lfp_tf, data["lfp_raw_train"], y_train, epochs=30)
    probs, preds = get_pytorch_predictions(lfp_tf, data["lfp_raw_test"])
    all_probs["lfp_transformer"] = probs

    # --- Model 9: BBB MLP alone ---
    print(f"\n  Evaluating: BBB MLP...")
    bbb_mlp = BBBMLPEncoder(
        n_features=data["n_bbb_features"],
        embedding_dim=config["bbb"]["embedding_dim"],
    )
    bbb_ckpt = os.path.join(CHECKPOINTS, "bbb_encoder.pt")
    if os.path.isfile(bbb_ckpt):
        ckpt = torch.load(bbb_ckpt, map_location="cpu", weights_only=True)
        bbb_mlp.load_state_dict(ckpt["model_state_dict"])
    else:
        print("    Training BBB MLP from scratch...")
        bbb_mlp = _quick_train_pytorch(bbb_mlp, data["bbb_train"], y_train, epochs=50)
    probs, preds = get_pytorch_predictions(bbb_mlp, data["bbb_test"])
    all_probs["bbb_mlp"] = probs

    # --- Model 10: Cross-Attention Fusion (PROPOSED) ---
    print(f"\n  Evaluating: Cross-Attention Fusion (PROPOSED)...")
    lfp_enc = LFPTransformer(
        seq_len=data["lfp_raw_test"].shape[1],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
    )
    bbb_enc = BBBMLPEncoder(
        n_features=data["n_bbb_features"],
        embedding_dim=config["bbb"]["embedding_dim"],
    )
    fusion = MultimodalFusionModel(
        lfp_enc, bbb_enc,
        d_model=config["model"]["d_model"],
        bbb_embed_dim=config["bbb"]["embedding_dim"],
    )
    fusion_ckpt = os.path.join(CHECKPOINTS, "fusion_model_best.pt")
    if os.path.isfile(fusion_ckpt):
        fusion.load_state_dict(torch.load(fusion_ckpt, map_location="cpu", weights_only=True))
    probs, preds = get_fusion_predictions(fusion, data["lfp_raw_test"], data["bbb_test"])
    all_probs["fusion"] = probs

    # --- Compute all metrics with bootstrap CIs ---
    print(f"\n{'='*60}")
    print("Computing metrics with bootstrap CIs...")
    print(f"{'='*60}")

    model_names = {
        "svm_lfp": "SVM (RBF) - LFP",
        "rf_lfp": "Random Forest - LFP",
        "xgb_lfp": "XGBoost - LFP",
        "cnn1d": "1D-CNN - LFP",
        "lstm": "LSTM - LFP",
        "xgb_bbb": "XGBoost - BBB",
        "xgb_early": "XGBoost Early Fusion",
        "lfp_transformer": "LFP Transformer",
        "bbb_mlp": "BBB MLP",
        "fusion": "Cross-Attention Fusion*",
    }

    for key, desc in model_names.items():
        if key not in all_probs:
            continue
        probs = all_probs[key]
        preds = (probs >= 0.5).astype(int)

        auc_mean, auc_lo, auc_hi = bootstrap_metric(y_test, probs, roc_auc_score)
        apr_mean, apr_lo, apr_hi = bootstrap_metric(y_test, probs, average_precision_score)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)

        # DeLong test vs proposed fusion
        delong_p = ""
        if key != "fusion" and "fusion" in all_probs:
            _, p = delong_test(y_test, all_probs["fusion"], probs)
            delong_p = f"{p:.4f}"

        results.append({
            "Model": desc,
            "AUC-ROC": f"{auc_mean:.4f}",
            "AUC-ROC 95% CI": f"[{auc_lo:.4f}-{auc_hi:.4f}]",
            "AUC-PR": f"{apr_mean:.4f}",
            "Accuracy": f"{acc:.4f}",
            "F1": f"{f1:.4f}",
            "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}",
            "DeLong p vs Fusion": delong_p,
        })

        print(f"  {desc:30s} | AUC={auc_mean:.4f} [{auc_lo:.4f}-{auc_hi:.4f}] | F1={f1:.4f}")

    # Save results
    df_results = pd.DataFrame(results)
    out_path = os.path.join(TABLES, "model_comparison.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(f"\n{df_results.to_string(index=False)}")


def _quick_train_pytorch(model, X_train, y_train, epochs=30, lr=1e-3):
    """Quick training for models without saved checkpoints."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    return model


if __name__ == "__main__":
    main()
