"""
EXP-04: Small transformer decoder for modular arithmetic grokking.

1-layer transformer decoder, d_model=128, 4 heads.
Input: 2p-dimensional one-hot vector (projected to d_model).
Output: p classes.

Minimal architecture — just enough to grok.
"""

import math
import torch
import torch.nn as nn


class GrokTransformer(nn.Module):
    """1-layer transformer decoder for modular addition."""

    def __init__(self, modulus, d_model=128, n_heads=4, dropout=0.0):
        super().__init__()
        self.modulus = modulus
        self.d_model = d_model

        # Project one-hot pair to two d_model tokens
        self.embed_a = nn.Linear(modulus, d_model)
        self.embed_b = nn.Linear(modulus, d_model)

        # Learned positional embedding for 2 positions
        self.pos_embed = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)

        # Single transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # Output head: pool over sequence, project to classes
        self.head = nn.Linear(d_model, modulus)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (B, 2p) one-hot encoded pair
        a_onehot = x[:, :self.modulus]
        b_onehot = x[:, self.modulus:]

        # Embed each operand as a token
        tok_a = self.embed_a(a_onehot).unsqueeze(1)  # (B, 1, d)
        tok_b = self.embed_b(b_onehot).unsqueeze(1)  # (B, 1, d)
        tokens = torch.cat([tok_a, tok_b], dim=1)    # (B, 2, d)
        tokens = tokens + self.pos_embed

        # Causal mask not needed for 2-token sequence with full attention
        # Decoder with self-attention only (memory = tokens)
        out = self.decoder(tokens, tokens)  # (B, 2, d)

        # Mean pool over sequence
        pooled = out.mean(dim=1)  # (B, d)
        return self.head(pooled)  # (B, modulus)


def build_model(cfg, device):
    """Build model from config."""
    model_cfg = cfg["model"]
    modulus = cfg["task"]["modulus"]
    model = GrokTransformer(
        modulus=modulus,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        dropout=model_cfg.get("dropout", 0.0),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: GrokTransformer ({n_params:,} parameters)")
    return model.to(device)
