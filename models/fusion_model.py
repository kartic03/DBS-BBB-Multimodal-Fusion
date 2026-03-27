"""
Cross-Attention Multimodal Fusion Model
=========================================
Fuses LFP Transformer embeddings (256-dim) with BBB MLP embeddings (64-dim)
using cross-attention, then classifies DBS response.

This is the PROPOSED model — the star of the paper.

Architecture:
    LFP embedding (256-dim) + BBB embedding (64-dim)
        → Project BBB to 256-dim
        → Cross-attention (Query=LFP, Key/Value=BBB)
        → Concatenate attended + original LFP → 512-dim
        → MLP head → P(DBS responder)

Loss: Cross-entropy + 0.1 * cosine similarity (modality alignment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lfp_transformer import LFPTransformer
from models.bbb_encoder import BBBMLPEncoder


class CrossAttentionFusion(nn.Module):
    """Cross-attention block for multimodal fusion.

    Query: LFP embedding
    Key, Value: BBB embedding (projected to same dim)
    """

    def __init__(self, embed_dim=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query, key_value):
        """
        Args:
            query: (batch, embed_dim) — LFP embedding
            key_value: (batch, embed_dim) — BBB embedding (projected)
        Returns:
            attended: (batch, embed_dim)
        """
        # Add sequence dimension for attention
        q = query.unsqueeze(1)      # (batch, 1, embed_dim)
        kv = key_value.unsqueeze(1)  # (batch, 1, embed_dim)

        # Cross-attention
        attn_out, attn_weights = self.cross_attn(q, kv, kv)
        attn_out = self.norm1(q + attn_out)

        # FFN
        ffn_out = self.ffn(attn_out)
        out = self.norm2(attn_out + ffn_out)

        return out.squeeze(1), attn_weights  # (batch, embed_dim)


class MultimodalFusionModel(nn.Module):
    """Full multimodal fusion model combining LFP Transformer + BBB MLP.

    Args:
        lfp_model: Pretrained LFPTransformer (provides 256-dim embedding)
        bbb_model: Pretrained BBBMLPEncoder (provides 64-dim embedding)
        d_model: Fusion dimension (default: 256)
        n_heads: Attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
        n_classes: Output classes (default: 2)
    """

    def __init__(self, lfp_model, bbb_model, d_model=256, bbb_embed_dim=64,
                 n_heads=8, dropout=0.1, n_classes=2):
        super().__init__()
        self.lfp_model = lfp_model
        self.bbb_model = bbb_model
        self.d_model = d_model

        # Project BBB embedding to d_model dimension
        self.bbb_proj = nn.Sequential(
            nn.Linear(bbb_embed_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Cross-attention fusion
        self.cross_attention = CrossAttentionFusion(
            embed_dim=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Classification head: concatenate attended + original LFP → 2*d_model
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self._init_fusion_weights()

    def _init_fusion_weights(self):
        """Initialize only the fusion-specific layers."""
        for module in [self.bbb_proj, self.cross_attention, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, lfp_input, bbb_input):
        """Forward pass.

        Args:
            lfp_input: (batch, seq_len) — raw LFP epoch
            bbb_input: (batch, n_features) — BBB feature vector
        Returns:
            logits: (batch, n_classes)
            lfp_embed: (batch, d_model) — for cosine loss
            bbb_embed_proj: (batch, d_model) — for cosine loss
            attn_weights: attention weights for interpretability
        """
        # Get embeddings from pretrained models
        lfp_embed = self.lfp_model.get_embedding(lfp_input)   # (batch, 256)
        bbb_embed = self.bbb_model.get_embedding(bbb_input)   # (batch, 64)

        # Project BBB to same dimension
        bbb_embed_proj = self.bbb_proj(bbb_embed)  # (batch, 256)

        # Cross-attention: LFP attends to BBB
        attended, attn_weights = self.cross_attention(lfp_embed, bbb_embed_proj)

        # Concatenate attended vector with original LFP embedding
        fused = torch.cat([lfp_embed, attended], dim=-1)  # (batch, 512)

        # Classify
        logits = self.classifier(fused)

        return logits, lfp_embed, bbb_embed_proj, attn_weights

    def freeze_encoders(self):
        """Freeze both pretrained encoders (Phase 1: train fusion head only)."""
        for param in self.lfp_model.parameters():
            param.requires_grad = False
        for param in self.bbb_model.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Encoders frozen. Trainable params: {trainable:,}")

    def unfreeze_all(self):
        """Unfreeze everything (Phase 2: end-to-end fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters())
        print(f"  All unfrozen. Trainable params: {trainable:,}")


class FusionLoss(nn.Module):
    """Combined loss: Cross-entropy + cosine alignment.

    L = CE(logits, targets) + lambda * (1 - cos_sim(lfp_embed, bbb_embed))
    """

    def __init__(self, alignment_weight=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.alignment_weight = alignment_weight

    def forward(self, logits, targets, lfp_embed, bbb_embed):
        ce_loss = self.ce(logits, targets)
        cos_sim = F.cosine_similarity(lfp_embed, bbb_embed, dim=-1).mean()
        alignment_loss = 1 - cos_sim
        total = ce_loss + self.alignment_weight * alignment_loss
        return total, ce_loss, alignment_loss


def build_fusion_model(config, lfp_model, bbb_model):
    """Build fusion model from config + pretrained encoders."""
    model = MultimodalFusionModel(
        lfp_model=lfp_model,
        bbb_model=bbb_model,
        d_model=config["model"]["d_model"],
        bbb_embed_dim=config["bbb"]["embedding_dim"],
        n_heads=config["model"]["n_heads"],
        dropout=config["model"]["dropout"],
        n_classes=2,
    )
    return model


if __name__ == "__main__":
    # Quick test
    lfp = LFPTransformer(seq_len=2000, d_model=256)
    bbb = BBBMLPEncoder(n_features=20, embedding_dim=64)
    fusion = MultimodalFusionModel(lfp, bbb, d_model=256, bbb_embed_dim=64)

    x_lfp = torch.randn(4, 2000)
    x_bbb = torch.randn(4, 20)

    logits, lfp_e, bbb_e, attn_w = fusion(x_lfp, x_bbb)
    print(f"LFP input: {x_lfp.shape}")
    print(f"BBB input: {x_bbb.shape}")
    print(f"Logits: {logits.shape}")
    print(f"LFP embed: {lfp_e.shape}")
    print(f"BBB embed proj: {bbb_e.shape}")
    print(f"Attn weights: {attn_w.shape}")

    total_params = sum(p.numel() for p in fusion.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Test loss
    criterion = FusionLoss(alignment_weight=0.1)
    targets = torch.tensor([0, 1, 1, 0])
    total, ce, align = criterion(logits, targets, lfp_e, bbb_e)
    print(f"Loss: total={total:.4f}, CE={ce:.4f}, alignment={align:.4f}")
