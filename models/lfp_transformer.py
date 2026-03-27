"""
LFP Transformer Model
======================
Transformer encoder for raw STN-LFP epoch classification.
Architecture based on arXiv 2508.10160 (DBS Transformer).

Input:  Raw LFP epoch [batch, seq_len=2000, channels=1]
Output: Classification logits [batch, 2] + CLS embedding [batch, 256]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """Split LFP signal into patches and project to d_model."""

    def __init__(self, patch_size=40, d_model=256, in_channels=1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_size,
                              stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (batch, seq_len, channels) -> (batch, n_patches, d_model)"""
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        x = self.proj(x)         # (batch, d_model, n_patches)
        x = x.permute(0, 2, 1)  # (batch, n_patches, d_model)
        x = self.norm(x)
        return x


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (PD:Healthy ~ 2:1)."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LFPTransformer(nn.Module):
    """Transformer encoder for LFP signal classification.

    Architecture:
        - Patch embedding (40-sample patches at 1000 Hz = 40ms windows)
        - Learnable CLS token + positional encoding
        - 6-layer Transformer encoder
        - Classification head from CLS token

    Args:
        seq_len: Input sequence length (default: 2000 = 2s at 1000Hz)
        patch_size: Patch size in samples (default: 40 = 40ms)
        d_model: Transformer dimension (default: 256)
        n_heads: Number of attention heads (default: 8)
        n_layers: Number of Transformer layers (default: 6)
        dropout: Dropout rate (default: 0.1)
        n_classes: Number of output classes (default: 2)
    """

    def __init__(self, seq_len=2000, patch_size=40, d_model=256, n_heads=8,
                 n_layers=6, dropout=0.1, n_classes=2):
        super().__init__()
        self.d_model = d_model
        self.n_patches = seq_len // patch_size

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, d_model, in_channels=1)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable positional encoding (CLS + patches)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches + 1, d_model) * 0.02
        )
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_embedding(self, x):
        """Extract CLS token embedding (for fusion model).

        Args:
            x: (batch, seq_len) or (batch, seq_len, 1)
        Returns:
            CLS embedding: (batch, d_model)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)

        # Patch embedding
        x = self.patch_embed(x)  # (batch, n_patches, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_patches+1, d_model)

        # Add positional encoding
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.pos_drop(x)

        # Transformer encoder
        x = self.transformer(x)
        x = self.norm(x)

        # Return CLS token
        return x[:, 0]  # (batch, d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, seq_len) or (batch, seq_len, 1)
        Returns:
            logits: (batch, n_classes)
        """
        cls_embed = self.get_embedding(x)
        logits = self.classifier(cls_embed)
        return logits

    def freeze_backbone(self, n_freeze_layers=4):
        """Freeze first n layers of transformer for fine-tuning."""
        # Freeze patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        # Freeze first n_freeze_layers
        for i, layer in enumerate(self.transformer.layers):
            if i < n_freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Frozen {n_freeze_layers}/{len(self.transformer.layers)} layers. "
              f"Trainable: {trainable:,} / {total:,} params")


def build_lfp_transformer(config):
    """Build LFP Transformer from config dict."""
    model = LFPTransformer(
        seq_len=int(config["lfp"]["sampling_rate"] * config["lfp"]["epoch_length_sec"]),
        patch_size=40,
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        dropout=config["model"]["dropout"],
        n_classes=2,
    )
    return model


if __name__ == "__main__":
    # Quick test
    model = LFPTransformer()
    x = torch.randn(4, 2000)  # batch of 4, 2000 samples
    logits = model(x)
    embed = model.get_embedding(x)
    print(f"Input: {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Embedding: {embed.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
