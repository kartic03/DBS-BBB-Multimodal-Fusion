"""
Train ALL Models with 5-Fold CV
=================================
Ensures every model gets identical 5-fold CV treatment for fair comparison.

Sklearn models: SVM, Random Forest, XGBoost (LFP), XGBoost (BBB), XGBoost (Early Fusion)
PyTorch models: CNN1D, LSTM, BBB MLP (retrain with 5-fold)
(LFP Transformer and Fusion already trained separately)

Outputs:
    - results/tables/model_comparison_5fold.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import yaml
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline_models import (build_svm, build_random_forest, build_xgboost_lfp,
                                     build_xgboost_bbb, build_xgboost_early_fusion,
                                     CNN1D, LSTMClassifier)
from models.lfp_transformer import LFPTransformer, FocalLoss
from models.bbb_encoder import BBBMLPEncoder
from models.fusion_model import MultimodalFusionModel, FusionLoss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

SEED = config["model"]["seed"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = config["training"]["n_folds"]
CHECKPOINTS = os.path.join(PROJECT_ROOT, config["paths"]["checkpoints"])
TABLES = os.path.join(PROJECT_ROOT, "results/tables")
os.makedirs(TABLES, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FusionDataset(Dataset):
    def __init__(self, X_lfp, X_bbb, y):
        self.lfp = torch.FloatTensor(X_lfp)
        self.bbb = torch.FloatTensor(X_bbb)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.lfp[idx], self.bbb[idx], self.y[idx]


def train_pytorch_fold(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=15):
    """Train a PyTorch model for one fold. Returns best val AUC and test predictions function."""
    model.to(DEVICE)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = torch.amp.GradScaler() if DEVICE == "cuda" else None

    best_auc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                logits = model(X)
                loss = criterion(logits, y)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        scheduler.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE)
                probs = torch.softmax(model(X), dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.numpy())
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_auc


def get_predictions(model, X, device=DEVICE):
    """Get predictions from PyTorch model."""
    model.eval()
    model.to(device)
    with torch.no_grad():
        t = torch.FloatTensor(X).to(device)
        probs = torch.softmax(model(t), dim=1)[:, 1].cpu().numpy()
    return probs


def bootstrap_ci(y_true, y_score, metric_fn, n=1000):
    scores = []
    for _ in range(n):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_score[idx]))
        except:
            continue
    if not scores:
        return 0, 0, 0
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def main():
    print("=" * 60)
    print(f"5-Fold CV for ALL 10 Models (device: {DEVICE})")
    print("=" * 60)

    # Load all data
    fused = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/fused/multimodal_dataset.csv"))
    all_epochs = np.load(os.path.join(PROJECT_ROOT, "data/processed/lfp_features/lfp_raw_epochs.npy"))
    epoch_labels = pd.read_csv(os.path.join(PROJECT_ROOT, "data/splits/lfp_labels.csv"))

    lfp_tab_cols = [c for c in fused.columns if c.startswith("lfp_")]
    bbb_cols = [c for c in fused.columns if c.startswith("bbb_")]

    # Get one raw epoch per subject
    raw_epochs = []
    for subj in fused["subject_id"]:
        mask = epoch_labels["subject_id"] == subj
        if mask.any():
            raw_epochs.append(all_epochs[mask.idxmax()])
        else:
            raw_epochs.append(np.zeros(all_epochs.shape[1], dtype=np.float32))

    X_lfp_tab = fused[lfp_tab_cols].values.astype(np.float32)
    X_bbb = fused[bbb_cols].values.astype(np.float32)
    X_lfp_raw = np.array(raw_epochs, dtype=np.float32)
    X_fused_tab = np.hstack([X_lfp_tab, X_bbb])
    y = fused["label"].values

    print(f"  Subjects: {len(y)} (pos={y.sum()}, neg={(1-y).sum()})")
    print(f"  LFP tabular: {X_lfp_tab.shape}, BBB: {X_bbb.shape}, Raw: {X_lfp_raw.shape}")

    # Define all 10 models
    models_config = [
        ("SVM (RBF) - LFP",        "sklearn", build_svm,                 X_lfp_tab, True),
        ("Random Forest - LFP",     "sklearn", build_random_forest,       X_lfp_tab, True),
        ("XGBoost - LFP",           "sklearn", build_xgboost_lfp,         X_lfp_tab, False),
        ("1D-CNN - LFP",            "pytorch", lambda: CNN1D(),            X_lfp_raw, False),
        ("LSTM - LFP",              "pytorch", lambda: LSTMClassifier(),   X_lfp_raw, False),
        ("XGBoost - BBB",           "sklearn", build_xgboost_bbb,          X_bbb,     False),
        ("XGBoost Early Fusion",    "sklearn", build_xgboost_early_fusion, X_fused_tab, False),
        ("LFP Transformer",         "pytorch", lambda: LFPTransformer(
            seq_len=X_lfp_raw.shape[1], d_model=config["model"]["d_model"],
            n_heads=config["model"]["n_heads"], n_layers=config["model"]["n_layers"],
            dropout=config["model"]["dropout"]),                            X_lfp_raw, False),
        ("BBB MLP",                 "pytorch", lambda: BBBMLPEncoder(
            n_features=len(bbb_cols), embedding_dim=config["bbb"]["embedding_dim"],
            dropout=config["model"]["dropout"]),                            X_bbb,     True),
        ("Cross-Attention Fusion*", "fusion",  None,                        None,      False),
    ]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results = []

    for model_name, model_type, builder, X_input, needs_scale in models_config:
        print(f"\n{'='*50}")
        print(f"  {model_name}")
        print(f"{'='*50}")

        fold_aucs = []
        all_test_probs = []
        all_test_labels = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            t0 = time.time()

            if model_type == "sklearn":
                X_tr, X_te = X_input[train_idx], X_input[val_idx]
                y_tr, y_te = y[train_idx], y[val_idx]

                if needs_scale:
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr)
                    X_te = scaler.transform(X_te)

                model = builder(seed=SEED)
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_te)[:, 1]

            elif model_type == "pytorch":
                X_tr, X_te = X_input[train_idx], X_input[val_idx]
                y_tr, y_te = y[train_idx], y[val_idx]

                if needs_scale:
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr)
                    X_te = scaler.transform(X_te)

                train_ds = SimpleDataset(X_tr, y_tr)
                val_ds = SimpleDataset(X_te, y_te)
                bs = 64 if "CNN" in model_name else 32
                train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
                val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

                model = builder()
                lr = 5e-4 if "CNN" in model_name else (1e-3 if "LSTM" in model_name or "MLP" in model_name else 1e-4)
                model, _ = train_pytorch_fold(model, train_loader, val_loader, epochs=100, lr=lr, patience=15)
                probs = get_predictions(model, X_te)
                y_te = y[val_idx]

            elif model_type == "fusion":
                # Fusion needs both LFP raw + BBB
                X_lfp_tr, X_lfp_te = X_lfp_raw[train_idx], X_lfp_raw[val_idx]
                X_bbb_tr, X_bbb_te = X_bbb[train_idx], X_bbb[val_idx]
                y_tr, y_te = y[train_idx], y[val_idx]

                scaler = StandardScaler()
                X_bbb_tr_s = scaler.fit_transform(X_bbb_tr).astype(np.float32)
                X_bbb_te_s = scaler.transform(X_bbb_te).astype(np.float32)

                train_ds = FusionDataset(X_lfp_tr, X_bbb_tr_s, y_tr)
                val_ds = FusionDataset(X_lfp_te, X_bbb_te_s, y_te)
                train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

                lfp_enc = LFPTransformer(seq_len=X_lfp_raw.shape[1], d_model=config["model"]["d_model"],
                                          n_heads=config["model"]["n_heads"], n_layers=config["model"]["n_layers"])
                bbb_enc = BBBMLPEncoder(n_features=len(bbb_cols), embedding_dim=config["bbb"]["embedding_dim"])
                fusion_model = MultimodalFusionModel(lfp_enc, bbb_enc, d_model=config["model"]["d_model"],
                                                      bbb_embed_dim=config["bbb"]["embedding_dim"]).to(DEVICE)

                criterion = FusionLoss(alignment_weight=0.1)
                optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-4, weight_decay=1e-2)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

                best_auc = 0
                best_state = None
                patience_counter = 0

                for epoch in range(100):
                    fusion_model.train()
                    for lfp_b, bbb_b, y_b in train_loader:
                        lfp_b, bbb_b, y_b = lfp_b.to(DEVICE), bbb_b.to(DEVICE), y_b.to(DEVICE)
                        optimizer.zero_grad()
                        logits, le, be, _ = fusion_model(lfp_b, bbb_b)
                        loss, _, _ = criterion(logits, y_b, le, be)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
                        optimizer.step()
                    scheduler.step()

                    fusion_model.eval()
                    vp, vl = [], []
                    with torch.no_grad():
                        for lfp_b, bbb_b, y_b in val_loader:
                            lfp_b, bbb_b = lfp_b.to(DEVICE), bbb_b.to(DEVICE)
                            logits, _, _, _ = fusion_model(lfp_b, bbb_b)
                            vp.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                            vl.extend(y_b.numpy())
                    try:
                        auc = roc_auc_score(vl, vp)
                    except:
                        auc = 0.5
                    if auc > best_auc:
                        best_auc = auc
                        best_state = {k: v.cpu().clone() for k, v in fusion_model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= 15:
                            break

                if best_state:
                    fusion_model.load_state_dict(best_state)
                fusion_model.eval()
                with torch.no_grad():
                    lfp_t = torch.FloatTensor(X_lfp_te).to(DEVICE)
                    bbb_t = torch.FloatTensor(X_bbb_te_s).to(DEVICE)
                    logits, _, _, _ = fusion_model(lfp_t, bbb_t)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            try:
                fold_auc = roc_auc_score(y_te, probs)
            except:
                fold_auc = 0.5
            fold_aucs.append(fold_auc)
            all_test_probs.extend(probs)
            all_test_labels.extend(y_te)

            elapsed = time.time() - t0
            print(f"    Fold {fold+1}/{N_FOLDS}: AUC={fold_auc:.4f} ({elapsed:.1f}s)")

        # Compute metrics from all fold predictions
        all_test_probs = np.array(all_test_probs)
        all_test_labels = np.array(all_test_labels)
        all_test_preds = (all_test_probs >= 0.5).astype(int)

        auc_mean, auc_lo, auc_hi = bootstrap_ci(all_test_labels, all_test_probs, roc_auc_score)
        apr_mean, _, _ = bootstrap_ci(all_test_labels, all_test_probs, average_precision_score)

        results.append({
            "Model": model_name,
            "AUC-ROC": f"{auc_mean:.4f}",
            "AUC-ROC 95% CI": f"[{auc_lo:.4f}-{auc_hi:.4f}]",
            "AUC-PR": f"{apr_mean:.4f}",
            "Accuracy": f"{accuracy_score(all_test_labels, all_test_preds):.4f}",
            "F1": f"{f1_score(all_test_labels, all_test_preds):.4f}",
            "Precision": f"{precision_score(all_test_labels, all_test_preds, zero_division=0):.4f}",
            "Recall": f"{recall_score(all_test_labels, all_test_preds, zero_division=0):.4f}",
            "5-Fold Mean AUC": f"{np.mean(fold_aucs):.4f}",
            "5-Fold Std": f"{np.std(fold_aucs):.4f}",
        })

        print(f"  => Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    # Save
    df = pd.DataFrame(results)
    out_path = os.path.join(TABLES, "model_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"Saved: {out_path}")
    print(f"{'='*60}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
