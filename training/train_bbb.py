"""
BBB Encoder Training Script
=============================
Trains XGBoost baseline + MLP encoder on BBB features.
Uses Optuna for XGBoost hyperparameter optimization.

Outputs:
    - results/checkpoints/bbb_encoder.pt
    - results/checkpoints/xgb_bbb_baseline.json
    - results/training_log.csv (appends)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import yaml
import csv
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bbb_encoder import BBBMLPEncoder

# ============================================================
# Config
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

SEED = config["model"]["seed"]
DEVICE = config["model"]["device"] if torch.cuda.is_available() else "cpu"
N_FOLDS = config["training"]["n_folds"]

CHECKPOINTS = os.path.join(PROJECT_ROOT, config["paths"]["checkpoints"])
os.makedirs(CHECKPOINTS, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Dataset
# ============================================================

class BBBDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================
# Stage 1: XGBoost with Optuna
# ============================================================

def train_xgboost_optuna(X_train, y_train, X_val, y_val, n_trials=50):
    """Train XGBoost with Optuna hyperparameter optimization."""
    try:
        import optuna
        from xgboost import XGBClassifier
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [WARN] Optuna not installed, using default XGBoost params")
        return train_xgboost_default(X_train, y_train, X_val, y_val)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "eval_metric": "auc",
            "random_state": SEED,
            "n_jobs": -1,
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Optuna best AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Train final model with best params
    from xgboost import XGBClassifier
    best_params = study.best_params
    best_params["eval_metric"] = "auc"
    best_params["random_state"] = SEED
    best_params["n_jobs"] = -1

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return model, study.best_params


def train_xgboost_default(X_train, y_train, X_val, y_val):
    """Train XGBoost with default parameters."""
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", random_state=SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, {}


# ============================================================
# Stage 2: MLP Encoder Training
# ============================================================

def train_mlp_encoder(X_train, y_train, X_val, y_val, n_features):
    """Train BBB MLP encoder."""
    train_ds = BBBDataset(X_train, y_train)
    val_ds = BBBDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = BBBMLPEncoder(
        n_features=n_features,
        embedding_dim=config["bbb"]["embedding_dim"],
        dropout=config["model"]["dropout"],
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_auc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(100):
        # Train
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE)
                logits = model(X)
                probs = torch.softmax(logits, dim=1)[:, 1]
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
            if patience_counter >= 20:
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: AUC={auc:.4f} (best={best_auc:.4f})")

    if best_state:
        model.load_state_dict(best_state)
    return model, best_auc


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print(f"BBB Encoder Training (device: {DEVICE})")
    print("=" * 60)

    # Load data
    bbb_path = os.path.join(PROJECT_ROOT, "data/processed/bbb_features/bbb_features.csv")
    df = pd.read_csv(bbb_path)

    # Use dbs_responder as label for BBB model
    meta_cols = ["subject_id", "label", "group", "dbs_responder",
                 "updrs_iii_baseline", "updrs_pct_improvement",
                 "has_imputed", "data_source"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = df["dbs_responder"].values

    print(f"\n  Features: {X.shape}")
    print(f"  Labels: responder={y.sum()}, non-responder={(1-y).sum()}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(skf.split(X, y))  # Use first fold for train/val

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # --- Stage 1: XGBoost ---
    print(f"\n[Stage 1] XGBoost with Optuna ({50} trials)...")
    xgb_model, xgb_params = train_xgboost_optuna(
        X[train_idx], y_train, X[val_idx], y_val, n_trials=50
    )
    xgb_auc = roc_auc_score(y_val, xgb_model.predict_proba(X[val_idx])[:, 1])
    print(f"  XGBoost val AUC: {xgb_auc:.4f}")

    # Save XGBoost
    xgb_path = os.path.join(CHECKPOINTS, "xgb_bbb_baseline.json")
    xgb_model.save_model(xgb_path)
    print(f"  Saved: {xgb_path}")

    # --- Stage 2: MLP Encoder ---
    print(f"\n[Stage 2] MLP Encoder training...")
    mlp_model, mlp_auc = train_mlp_encoder(
        X_train, y_train, X_val, y_val, n_features=len(feature_cols)
    )
    print(f"  MLP val AUC: {mlp_auc:.4f}")

    # Save MLP
    mlp_path = os.path.join(CHECKPOINTS, "bbb_encoder.pt")
    torch.save({
        "model_state_dict": mlp_model.state_dict(),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }, mlp_path)
    print(f"  Saved: {mlp_path}")

    # Save scaler params for inference
    scaler_path = os.path.join(CHECKPOINTS, "bbb_scaler.npz")
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)

    print(f"\n{'='*40}")
    print(f"BBB Training Complete")
    print(f"  XGBoost AUC: {xgb_auc:.4f}")
    print(f"  MLP AUC:     {mlp_auc:.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
