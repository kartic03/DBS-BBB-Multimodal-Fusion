"""
LFP Transformer Training Script
=================================
Trains the LFP Transformer on raw LFP epochs with 5-fold CV.
Uses Focal Loss, AdamW optimizer, CosineAnnealingLR scheduler.

Outputs:
    - results/checkpoints/lfp_transformer_fold{k}.pt
    - results/checkpoints/lfp_transformer_best.pt
    - results/training_log.csv
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
import csv
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lfp_transformer import LFPTransformer, FocalLoss

# ============================================================
# Config
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

SEED = config["model"]["seed"]
DEVICE = config["model"]["device"] if torch.cuda.is_available() else "cpu"
BATCH_SIZE = config["model"]["batch_size"]
EPOCHS = config["model"]["epochs"]
LR = config["model"]["lr"]
WEIGHT_DECAY = config["model"]["weight_decay"]
PATIENCE = config["model"]["patience"]
N_FOLDS = config["training"]["n_folds"]

CHECKPOINTS = os.path.join(PROJECT_ROOT, config["paths"]["checkpoints"])
os.makedirs(CHECKPOINTS, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Dataset
# ============================================================

class LFPDataset(Dataset):
    """Dataset for raw LFP epochs."""

    def __init__(self, epochs, labels):
        self.epochs = torch.FloatTensor(epochs)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.epochs[idx], self.labels[idx]


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    """Train one epoch with mixed precision."""
    model.train()
    total_loss = 0
    n_batches = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device if device != "cuda" else "cuda",
                                enabled=(device == "cuda")):
            logits = model(X)
            loss = criterion(logits, y)

        if scaler is not None:
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
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model and return loss + AUC."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_probs = []
    all_labels = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    return avg_loss, auc


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print(f"LFP Transformer Training (device: {DEVICE})")
    print("=" * 60)

    # Load data
    epochs_path = os.path.join(PROJECT_ROOT, "data/processed/lfp_features/lfp_raw_epochs.npy")
    labels_path = os.path.join(PROJECT_ROOT, "data/splits/lfp_labels.csv")

    print("\nLoading data...")
    X = np.load(epochs_path)
    df_labels = pd.read_csv(labels_path)
    y = df_labels["label"].values
    subject_ids = df_labels["subject_id"].values

    print(f"  Epochs: {X.shape}")
    print(f"  Labels: {y.shape} (pos={y.sum()}, neg={(1-y).sum()})")

    # Training log
    log_path = os.path.join(PROJECT_ROOT, "results/training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["model", "fold", "epoch", "train_loss", "val_loss", "val_auc"])

    # 5-fold CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*40}")
        print(f"Fold {fold + 1}/{N_FOLDS}")
        print(f"{'='*40}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = LFPDataset(X_train, y_train)
        val_ds = LFPDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=2, pin_memory=True)

        # Model
        model = LFPTransformer(
            seq_len=X.shape[1],
            d_model=config["model"]["d_model"],
            n_heads=config["model"]["n_heads"],
            n_layers=config["model"]["n_layers"],
            dropout=config["model"]["dropout"],
        ).to(DEVICE)

        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        scaler = torch.amp.GradScaler() if DEVICE == "cuda" else None

        best_auc = 0
        patience_counter = 0

        for epoch in range(EPOCHS):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
            val_loss, val_auc = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step()
            elapsed = time.time() - t0

            log_writer.writerow(["lfp_transformer", fold, epoch, f"{train_loss:.6f}",
                                 f"{val_loss:.6f}", f"{val_auc:.4f}"])

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                      f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                      f"val_auc={val_auc:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | "
                      f"{elapsed:.1f}s")

            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(),
                           os.path.join(CHECKPOINTS, f"lfp_transformer_fold{fold}.pt"))
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1} (best AUC: {best_auc:.4f})")
                    break

        fold_aucs.append(best_auc)
        print(f"  Fold {fold+1} best AUC: {best_auc:.4f}")

    log_file.close()

    # Save best fold model as the "best" model
    best_fold = np.argmax(fold_aucs)
    best_path = os.path.join(CHECKPOINTS, f"lfp_transformer_fold{best_fold}.pt")
    final_path = os.path.join(CHECKPOINTS, "lfp_transformer_best.pt")

    import shutil
    shutil.copy2(best_path, final_path)

    print(f"\n{'='*40}")
    print(f"Training Complete")
    print(f"{'='*40}")
    print(f"  Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"  Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"  Best fold: {best_fold} (AUC={fold_aucs[best_fold]:.4f})")
    print(f"  Saved: {final_path}")


if __name__ == "__main__":
    main()
