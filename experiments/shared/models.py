"""
Standard architectures for experiments.

All models accept num_classes parameter for split-task training.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50


def get_resnet18(num_classes: int = 50, pretrained: bool = False) -> nn.Module:
    """ResNet-18 adapted for CIFAR-100 (32x32 input)."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet50(num_classes: int = 50, pretrained: bool = False) -> nn.Module:
    """ResNet-50 adapted for CIFAR-100 (32x32 input). Deeper than ResNet-18."""
    model = resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class ViTSmall(nn.Module):
    """Small Vision Transformer for CIFAR-100 (32x32 input).

    Patch size 4 → 8x8 = 64 patches.
    4 layers, 4 heads, 256 embed dim.
    ~3M parameters — comparable to ResNet-18's 11M but fundamentally different structure.
    """
    def __init__(self, num_classes=50, img_size=32, patch_size=4,
                 embed_dim=256, depth=4, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def get_vit_small(num_classes: int = 50, pretrained: bool = False) -> nn.Module:
    """Small ViT for CIFAR-100."""
    return ViTSmall(num_classes=num_classes)


def get_model(architecture: str, num_classes: int = 50, **kwargs) -> nn.Module:
    """Factory function for model creation."""
    models = {
        "resnet18": get_resnet18,
        "resnet50": get_resnet50,
        "vit_small": get_vit_small,
    }
    if architecture not in models:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(models.keys())}")
    return models[architecture](num_classes=num_classes, **kwargs)
