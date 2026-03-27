"""
Fusion Model Training Script
==============================
Two-phase training of the cross-attention multimodal fusion model:
  Phase 1: Train fusion head only (encoders frozen) — 10 epochs warmup
  Phase 2: Unfreeze all, fine-tune end-to-end — 90 epochs

Uses SMOTE for class imbalance on training data.

Outputs:
    - results/checkpoints/fusion_model_best.pt
    - results/training_log.csv (appends)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import yaml
import csv
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lfp_transformer import LFPTransformer
from models.bbb_encoder import BBBMLPEncoder
from models.fusion_model import MultimodalFusionModel, FusionLoss

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

CHECKPOINTS = os.path.join(PROJECT_ROOT, config["paths"]["checkpoints"])
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Dataset
# ============================================================

class FusionDataset(Dataset):
    """Multimodal dataset with LFP epochs + BBB features."""

    def __init__(self, lfp_epochs, bbb_features, labels):
        self.lfp = torch.FloatTensor(lfp_epochs)
        self.bbb = torch.FloatTensor(bbb_features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.lfp[idx], self.bbb[idx], self.labels[idx]


# ============================================================
# Data Loading
# ============================================================

def load_fusion_data():
    """Load and prepare multimodal data for fusion training."""
    # Load fused dataset (tabular — for splits and BBB features)
    fused_path = os.path.join(PROJECT_ROOT, "data/processed/fused/multimodal_dataset.csv")
    df = pd.read_csv(fused_path)

    # Load raw LFP epochs
    epochs_path = os.path.join(PROJECT_ROOT, "data/processed/lfp_features/lfp_raw_epochs.npy")
    lfp_labels_path = os.path.join(PROJECT_ROOT, "data/splits/lfp_labels.csv")
    all_epochs = np.load(epochs_path)
    epoch_labels = pd.read_csv(lfp_labels_path)

    # BBB feature columns
    bbb_cols = [c for c in df.columns if c.startswith("bbb_")]

    # Match: for each subject in fused dataset, grab their first LFP epoch
    # and their BBB features
    lfp_list, bbb_list, label_list, split_list = [], [], [], []

    for _, row in df.iterrows():
        subj = row["subject_id"]
        # Find matching epochs
        epoch_mask = epoch_labels["subject_id"] == subj
        if not epoch_mask.any():
            continue

        # Take first epoch for this subject
        first_epoch_idx = epoch_mask.idxmax()
        lfp_epoch = all_epochs[first_epoch_idx]

        bbb_feats = row[bbb_cols].values.astype(np.float32)
        label = int(row["label"])
        split = row["split"]

        lfp_list.append(lfp_epoch)
        bbb_list.append(bbb_feats)
        label_list.append(label)
        split_list.append(split)

    X_lfp = np.array(lfp_list, dtype=np.float32)
    X_bbb = np.array(bbb_list, dtype=np.float32)
    y = np.array(label_list, dtype=np.int64)
    splits = np.array(split_list)

    # Standardize BBB features
    train_mask = splits == "train"
    scaler = StandardScaler()
    X_bbb[train_mask] = scaler.fit_transform(X_bbb[train_mask])
    X_bbb[~train_mask] = scaler.transform(X_bbb[~train_mask])

    return X_lfp, X_bbb, y, splits, len(bbb_cols)


# ============================================================
# SMOTE
# ============================================================

def apply_smote(X_lfp, X_bbb, y):
    """Apply SMOTE to oversample minority class."""
    try:
        from imblearn.over_sampling import SMOTE
        # Flatten LFP for SMOTE, then reshape back
        n = len(y)
        X_combined = np.hstack([X_lfp.reshape(n, -1), X_bbb])
        sm = SMOTE(random_state=SEED)
        X_res, y_res = sm.fit_resample(X_combined, y)

        lfp_dim = X_lfp.shape[1]
        X_lfp_res = X_res[:, :lfp_dim].astype(np.float32)
        X_bbb_res = X_res[:, lfp_dim:].astype(np.float32)

        print(f"  SMOTE: {n} → {len(y_res)} samples "
              f"(pos={y_res.sum()}, neg={(1-y_res).sum()})")
        return X_lfp_res, X_bbb_res, y_res
    except ImportError:
        print("  [WARN] imbalanced-learn not installed, skipping SMOTE")
        return X_lfp, X_bbb, y


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for lfp, bbb, labels in loader:
        lfp, bbb, labels = lfp.to(device), bbb.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, lfp_e, bbb_e, _ = model(lfp, bbb)
        loss, ce, align = criterion(logits, labels, lfp_e, bbb_e)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_fusion(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n = 0
    all_probs, all_labels = [], []
    for lfp, bbb, labels in loader:
        lfp, bbb, labels = lfp.to(device), bbb.to(device), labels.to(device)
        logits, lfp_e, bbb_e, _ = model(lfp, bbb)
        loss, _, _ = criterion(logits, labels, lfp_e, bbb_e)
        total_loss += loss.item()
        n += 1
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    return total_loss / max(n, 1), auc


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print(f"Fusion Model Training (device: {DEVICE})")
    print("=" * 60)

    # Load data
    print("\nLoading multimodal data...")
    X_lfp, X_bbb, y, splits, n_bbb_features = load_fusion_data()

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    print(f"  LFP shape: {X_lfp.shape}")
    print(f"  BBB shape: {X_bbb.shape}")
    print(f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    # Apply SMOTE on training data
    if config["training"]["smote"]:
        X_lfp_train, X_bbb_train, y_train = apply_smote(
            X_lfp[train_mask], X_bbb[train_mask], y[train_mask]
        )
    else:
        X_lfp_train, X_bbb_train, y_train = X_lfp[train_mask], X_bbb[train_mask], y[train_mask]

    train_ds = FusionDataset(X_lfp_train, X_bbb_train, y_train)
    val_ds = FusionDataset(X_lfp[val_mask], X_bbb[val_mask], y[val_mask])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Build model with pretrained encoders
    print("\nBuilding fusion model...")
    lfp_model = LFPTransformer(
        seq_len=X_lfp.shape[1],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        dropout=config["model"]["dropout"],
    )
    bbb_model = BBBMLPEncoder(
        n_features=n_bbb_features,
        embedding_dim=config["bbb"]["embedding_dim"],
        dropout=config["model"]["dropout"],
    )

    # Load pretrained weights if available
    lfp_ckpt = os.path.join(CHECKPOINTS, "lfp_transformer_best.pt")
    if os.path.isfile(lfp_ckpt):
        lfp_model.load_state_dict(torch.load(lfp_ckpt, map_location="cpu", weights_only=True))
        print("  Loaded pretrained LFP Transformer")

    bbb_ckpt = os.path.join(CHECKPOINTS, "bbb_encoder.pt")
    if os.path.isfile(bbb_ckpt):
        ckpt = torch.load(bbb_ckpt, map_location="cpu", weights_only=True)
        bbb_model.load_state_dict(ckpt["model_state_dict"])
        print("  Loaded pretrained BBB Encoder")

    fusion = MultimodalFusionModel(
        lfp_model, bbb_model,
        d_model=config["model"]["d_model"],
        bbb_embed_dim=config["bbb"]["embedding_dim"],
        n_heads=config["model"]["n_heads"],
        dropout=config["model"]["dropout"],
    ).to(DEVICE)

    criterion = FusionLoss(alignment_weight=0.1)

    # --- Phase 1: Frozen encoders (warmup) ---
    print("\n[Phase 1] Training fusion head only (10 epochs)...")
    fusion.freeze_encoders()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, fusion.parameters()),
        lr=LR * 10, weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(10):
        t_loss = train_one_epoch(fusion, train_loader, criterion, optimizer, DEVICE)
        v_loss, v_auc = evaluate_fusion(fusion, val_loader, criterion, DEVICE)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/10 | train={t_loss:.4f} | val={v_loss:.4f} | AUC={v_auc:.4f}")

    # --- Phase 2: End-to-end fine-tuning ---
    print("\n[Phase 2] End-to-end fine-tuning (90 epochs)...")
    fusion.unfreeze_all()
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90)

    best_auc = 0
    patience_counter = 0

    # Training log
    log_path = os.path.join(PROJECT_ROOT, "results/training_log.csv")
    log_mode = "a" if os.path.isfile(log_path) else "w"
    log_file = open(log_path, log_mode, newline="")
    log_writer = csv.writer(log_file)
    if log_mode == "w":
        log_writer.writerow(["model", "fold", "epoch", "train_loss", "val_loss", "val_auc"])

    for epoch in range(90):
        t0 = time.time()
        t_loss = train_one_epoch(fusion, train_loader, criterion, optimizer, DEVICE)
        v_loss, v_auc = evaluate_fusion(fusion, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        log_writer.writerow(["fusion", 0, epoch + 10, f"{t_loss:.6f}",
                             f"{v_loss:.6f}", f"{v_auc:.4f}"])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/90 | train={t_loss:.4f} | val={v_loss:.4f} | "
                  f"AUC={v_auc:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        if v_auc > best_auc:
            best_auc = v_auc
            patience_counter = 0
            torch.save(fusion.state_dict(),
                       os.path.join(CHECKPOINTS, "fusion_model_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (best AUC: {best_auc:.4f})")
                break

    log_file.close()

    print(f"\n{'='*40}")
    print(f"Fusion Training Complete")
    print(f"  Best val AUC: {best_auc:.4f}")
    print(f"  Saved: {os.path.join(CHECKPOINTS, 'fusion_model_best.pt')}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
