"""
Deep Learning Baseline Training (CNN1D + LSTM)
================================================
Proper training for 1D-CNN and LSTM baselines with:
- 5-fold CV, early stopping, focal loss, mixed precision
- Same training protocol as LFP Transformer for fair comparison

Outputs:
    - results/checkpoints/cnn1d_best.pt
    - results/checkpoints/lstm_best.pt
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
import yaml
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.baseline_models import CNN1D, LSTMClassifier
from models.lfp_transformer import FocalLoss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

SEED = config["model"]["seed"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINTS = os.path.join(PROJECT_ROOT, config["paths"]["checkpoints"])

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class LFPDataset(Dataset):
    def __init__(self, epochs, labels):
        self.epochs = torch.FloatTensor(epochs)
        self.labels = torch.LongTensor(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.epochs[idx], self.labels[idx]


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, n = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0, 0
    all_probs, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item()
        n += 1
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    return total_loss / max(n, 1), auc


def train_model(model_class, model_name, X, y, epochs=100, lr=1e-3, patience=15, batch_size=32):
    """Full 5-fold CV training for a DL baseline."""
    print(f"\n{'='*50}")
    print(f"Training {model_name} (device: {DEVICE})")
    print(f"{'='*50}")

    n_folds = config["training"]["n_folds"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold+1}/{n_folds}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = LFPDataset(X_train, y_train)
        val_ds = LFPDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)

        model = model_class().to(DEVICE)
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        scaler = torch.amp.GradScaler() if DEVICE == "cuda" else None

        best_auc = 0
        patience_counter = 0

        for epoch in range(epochs):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
            val_loss, val_auc = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step()
            elapsed = time.time() - t0

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:3d}/{epochs} | train={train_loss:.4f} | "
                      f"val={val_loss:.4f} | AUC={val_auc:.4f} | {elapsed:.1f}s")

            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(),
                           os.path.join(CHECKPOINTS, f"{model_name}_fold{fold}.pt"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1} (best AUC: {best_auc:.4f})")
                    break

        fold_aucs.append(best_auc)
        print(f"  Fold {fold+1} best AUC: {best_auc:.4f}")

    # Save best fold as main checkpoint
    best_fold = np.argmax(fold_aucs)
    import shutil
    src = os.path.join(CHECKPOINTS, f"{model_name}_fold{best_fold}.pt")
    dst = os.path.join(CHECKPOINTS, f"{model_name}_best.pt")
    shutil.copy2(src, dst)

    print(f"\n  {model_name} Results:")
    print(f"    Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"    Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"    Saved: {dst}")
    return fold_aucs


def main():
    # Load data
    epochs_path = os.path.join(PROJECT_ROOT, "data/processed/lfp_features/lfp_raw_epochs.npy")
    labels_path = os.path.join(PROJECT_ROOT, "data/splits/lfp_labels.csv")
    X = np.load(epochs_path)
    df_labels = pd.read_csv(labels_path)
    y = df_labels["label"].values
    print(f"Data: {X.shape}, labels: {y.shape} (pos={y.sum()}, neg={(1-y).sum()})")

    # Train CNN1D
    cnn_aucs = train_model(CNN1D, "cnn1d", X, y, epochs=100, lr=5e-4, patience=15, batch_size=64)

    # Train LSTM
    lstm_aucs = train_model(LSTMClassifier, "lstm", X, y, epochs=100, lr=1e-3, patience=15, batch_size=32)

    print(f"\n{'='*50}")
    print(f"ALL DONE")
    print(f"  CNN1D:  Mean AUC = {np.mean(cnn_aucs):.4f} ± {np.std(cnn_aucs):.4f}")
    print(f"  LSTM:   Mean AUC = {np.mean(lstm_aucs):.4f} ± {np.std(lstm_aucs):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
