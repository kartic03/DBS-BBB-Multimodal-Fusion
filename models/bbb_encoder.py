"""
BBB Encoder Model
==================
Two-stage encoder for BBB + neuroinflammation tabular features:
  Stage 1: XGBoost classifier (standalone baseline)
  Stage 2: MLP encoder producing 64-dim embedding (for fusion model)

Input:  BBB feature vector [batch, n_features]
Output: Classification logits [batch, 2] + embedding [batch, 64]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BBBMLPEncoder(nn.Module):
    """MLP encoder for BBB tabular features.

    Architecture:
        Input(n_features) → Linear(64) → BN → ReLU → Dropout(0.3)
                          → Linear(128) → BN → ReLU → Dropout(0.3)
                          → Linear(64) → embedding

    The 64-dim embedding feeds into the fusion model.
    Optionally includes a classification head for standalone training.
    """

    def __init__(self, n_features=20, embedding_dim=64, dropout=0.3, n_classes=2):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, embedding_dim),
        )

        # Classification head (for standalone BBB-only training)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_embedding(self, x):
        """Extract embedding vector.

        Args:
            x: (batch, n_features)
        Returns:
            embedding: (batch, embedding_dim)
        """
        return self.encoder(x)

    def forward(self, x):
        """Forward pass with classification.

        Args:
            x: (batch, n_features)
        Returns:
            logits: (batch, n_classes)
        """
        embedding = self.get_embedding(x)
        logits = self.classifier(embedding)
        return logits


def build_bbb_encoder(config):
    """Build BBB MLP encoder from config dict."""
    model = BBBMLPEncoder(
        n_features=config["bbb"]["n_features"],
        embedding_dim=config["bbb"]["embedding_dim"],
        dropout=config["model"]["dropout"],
        n_classes=2,
    )
    return model


def get_xgboost_params(config=None):
    """Return default XGBoost parameters for BBB baseline."""
    return {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "auc",
        "random_state": config["model"]["seed"] if config else 42,
        "n_jobs": -1,
    }


if __name__ == "__main__":
    # Quick test
    model = BBBMLPEncoder(n_features=20, embedding_dim=64)
    x = torch.randn(4, 20)
    logits = model(x)
    embed = model.get_embedding(x)
    print(f"Input: {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Embedding: {embed.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
