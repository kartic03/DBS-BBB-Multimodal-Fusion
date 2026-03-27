"""
Baseline Models
================
All comparison models for the ablation table (Table 2 in paper).

Models 1-7: Classical/simple baselines
Models 8-9: Unimodal deep learning (ablation components)
Model 10:   Proposed cross-attention fusion (in fusion_model.py)

All models implement a unified interface for evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ============================================================
# Model 1: SVM (RBF kernel) — LFP tabular features
# ============================================================

def build_svm(seed=42):
    """SVM with RBF kernel for LFP tabular classification."""
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=seed,
        class_weight="balanced",
    )


# ============================================================
# Model 2: Random Forest — LFP tabular features
# ============================================================

def build_random_forest(seed=42):
    """Random Forest for LFP tabular classification."""
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


# ============================================================
# Model 3: XGBoost — LFP tabular features
# ============================================================

def build_xgboost_lfp(seed=42):
    """XGBoost for LFP tabular classification."""
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=seed,
        n_jobs=-1,
    )


# ============================================================
# Model 4: 1D-CNN — Raw LFP epochs
# ============================================================

class CNN1D(nn.Module):
    """1D Convolutional Neural Network for raw LFP classification.

    Architecture:
        Conv1d(1, 32, k=7) → BN → ReLU → MaxPool
        Conv1d(32, 64, k=5) → BN → ReLU → MaxPool
        Conv1d(64, 128, k=3) → BN → ReLU → AdaptiveAvgPool
        Linear(128, 64) → ReLU → Dropout → Linear(64, 2)
    """

    def __init__(self, n_classes=2, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        """x: (batch, seq_len) → logits: (batch, n_classes)"""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, seq_len)
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)


# ============================================================
# Model 5: LSTM — Raw LFP epochs
# ============================================================

class LSTMClassifier(nn.Module):
    """Bidirectional LSTM for raw LFP classification.

    Architecture:
        BiLSTM(input=1, hidden=128, layers=2)
        → take last hidden state → Linear(256, 64) → ReLU → Linear(64, 2)
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 dropout=0.3, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        """x: (batch, seq_len) → logits: (batch, n_classes)"""
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        output, (hidden, _) = self.lstm(x)
        # Concatenate forward and backward final hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.classifier(hidden)


# ============================================================
# Model 6: XGBoost — BBB features only
# ============================================================

def build_xgboost_bbb(seed=42):
    """XGBoost for BBB-only classification."""
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=seed,
        n_jobs=-1,
    )


# ============================================================
# Model 7: XGBoost Early Fusion — LFP tabular + BBB concatenated
# ============================================================

def build_xgboost_early_fusion(seed=42):
    """XGBoost on concatenated LFP tabular + BBB features."""
    return XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.6,
        eval_metric="auc",
        random_state=seed,
        n_jobs=-1,
    )


# ============================================================
# Registry
# ============================================================

SKLEARN_MODELS = {
    "svm_lfp": ("SVM (RBF) - LFP", build_svm, "lfp_tabular"),
    "rf_lfp": ("Random Forest - LFP", build_random_forest, "lfp_tabular"),
    "xgb_lfp": ("XGBoost - LFP", build_xgboost_lfp, "lfp_tabular"),
    "xgb_bbb": ("XGBoost - BBB", build_xgboost_bbb, "bbb"),
    "xgb_early_fusion": ("XGBoost Early Fusion", build_xgboost_early_fusion, "fused_tabular"),
}

PYTORCH_MODELS = {
    "cnn1d": ("1D-CNN - LFP", CNN1D, "lfp_raw"),
    "lstm": ("LSTM - LFP", LSTMClassifier, "lfp_raw"),
}


if __name__ == "__main__":
    # Quick tests
    print("=== Sklearn Models ===")
    for name, (desc, builder, input_type) in SKLEARN_MODELS.items():
        model = builder()
        print(f"  {name}: {desc} (input: {input_type})")

    print("\n=== PyTorch Models ===")
    x_raw = torch.randn(4, 2000)

    cnn = CNN1D()
    out = cnn(x_raw)
    print(f"  CNN1D: input={x_raw.shape} → output={out.shape}, "
          f"params={sum(p.numel() for p in cnn.parameters()):,}")

    lstm = LSTMClassifier()
    out = lstm(x_raw)
    print(f"  LSTM: input={x_raw.shape} → output={out.shape}, "
          f"params={sum(p.numel() for p in lstm.parameters()):,}")
