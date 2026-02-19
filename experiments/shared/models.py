"""
Standard architectures for experiments.

All models accept num_classes parameter for split-task training.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet50, densenet121, efficientnet_b0,
    vgg16_bn, convnext_tiny, mobilenet_v3_small,
    shufflenet_v2_x1_0, regnet_y_400mf,
)


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


# ─── Wide ResNet-28-10 ───

class WideBasicBlock(nn.Module):
    """Basic block for WideResNet with dropout."""
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.dropout(self.conv1(torch.relu(self.bn1(x))))
        out = self.conv2(torch.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """WideResNet-28-10 for CIFAR-100. ~36.5M parameters.

    Width factor 10 produces very wide feature maps, testing whether
    width (rather than depth) creates topologically deeper basins.
    """
    def __init__(self, depth=28, widen_factor=10, num_classes=50, dropout_rate=0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(WideBasicBlock, nStages[0], nStages[1], n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(WideBasicBlock, nStages[1], nStages[2], n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(WideBasicBlock, nStages[2], nStages[3], n, stride=2, dropout_rate=dropout_rate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.fc = nn.Linear(nStages[3], num_classes)

    def _make_layer(self, block, in_planes, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(in_planes, planes, s, dropout_rate))
            in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.relu(self.bn1(out))
        out = nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def get_wrn2810(num_classes: int = 50, pretrained: bool = False) -> nn.Module:
    """WideResNet-28-10 for CIFAR-100."""
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes)


# ─── MLP-Mixer ───

class MixerBlock(nn.Module):
    """Single Mixer block: token mixing + channel mixing."""
    def __init__(self, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_mix = nn.Sequential(
            nn.Linear(num_patches, tokens_mlp_dim),
            nn.GELU(),
            nn.Linear(tokens_mlp_dim, num_patches),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(hidden_dim, channels_mlp_dim),
            nn.GELU(),
            nn.Linear(channels_mlp_dim, hidden_dim),
        )

    def forward(self, x):
        # Token mixing: transpose to (B, C, S), mix, transpose back
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mix(y)
        y = y.transpose(1, 2)
        x = x + y
        # Channel mixing
        y = self.norm2(x)
        y = self.channel_mix(y)
        return x + y


class MLPMixer(nn.Module):
    """MLP-Mixer for CIFAR-100. No convolutions, no attention — pure MLP.

    Patch size 4 → 64 patches. 8 layers, 256 hidden dim.
    ~5M parameters. Tests whether a purely feedforward architecture
    with no spatial inductive bias creates topological persistence.
    """
    def __init__(self, num_classes=50, img_size=32, patch_size=4,
                 hidden_dim=256, depth=8, tokens_mlp_dim=128, channels_mlp_dim=512):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.Sequential(*[
            MixerBlock(num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, C, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, C)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling over patches
        return self.head(x)


def get_mlp_mixer(num_classes: int = 50, pretrained: bool = False) -> nn.Module:
    """MLP-Mixer for CIFAR-100."""
    return MLPMixer(num_classes=num_classes)


# ─── ResNet-18 Wide (2x channels) ───

class ResNet18Wide(nn.Module):
    """ResNet-18 with 2x channel width. ~44M parameters.

    Same depth as ResNet-18 but double the width at every layer.
    Disentangles the effect of width from depth within the ResNet family.
    """
    def __init__(self, num_classes=50, width_mult=2):
        super().__init__()
        w = [64 * width_mult, 128 * width_mult, 256 * width_mult, 512 * width_mult]

        self.conv1 = nn.Conv2d(3, w[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(w[0], w[0], 2, stride=1)
        self.layer2 = self._make_layer(w[0], w[1], 2, stride=2)
        self.layer3 = self._make_layer(w[1], w[2], 2, stride=2)
        self.layer4 = self._make_layer(w[2], w[3], 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w[3], num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = []
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers.append(WideResBlock(in_ch, out_ch, stride, downsample))
        for _ in range(1, blocks):
            layers.append(WideResBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class WideResBlock(nn.Module):
    """Basic residual block for ResNet18Wide."""
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def get_resnet18_wide(num_classes: int = 50, pretrained: bool = False) -> nn.Module:
    """ResNet-18 with 2x channel width for CIFAR-100."""
    return ResNet18Wide(num_classes=num_classes)


# ─── DenseNet-121 ───

class DenseNet121Wrapper(nn.Module):
    """DenseNet-121 wrapped for CIFAR-100 (32x32 input). ~7M parameters.

    Dense connections: every layer receives input from ALL preceding layers.
    Tests whether dense connectivity (feature reuse) creates topological depth.
    Uses .fc attribute for classifier compatibility with phase3 expansion.
    """
    def __init__(self, num_classes=50):
        super().__init__()
        base = densenet121(weights=None)
        base.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.features.norm0 = nn.BatchNorm2d(64)
        base.features.pool0 = nn.Identity()
        self.features = base.features
        self.fc = nn.Linear(base.classifier.in_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = torch.relu(features)
        out = nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def get_densenet121(num_classes: int = 50, pretrained: bool = False) -> nn.Module:
    """DenseNet-121 for CIFAR-100."""
    return DenseNet121Wrapper(num_classes=num_classes)


# ─── EfficientNet-B0 ───

class EfficientNetB0Wrapper(nn.Module):
    """EfficientNet-B0 wrapped for CIFAR-100 (32x32 input). ~4.1M parameters.

    Compound scaling (balanced depth/width/resolution). Mobile inverted bottlenecks
    with squeeze-and-excitation. Uses .fc for classifier compatibility.
    """
    def __init__(self, num_classes=50):
        super().__init__()
        base = efficientnet_b0(weights=None)
        old_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(3, old_conv.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        base.features[0][1] = nn.BatchNorm2d(old_conv.out_channels)
        self.features = base.features
        self.avgpool = base.avgpool
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(base.classifier[-1].in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def get_efficientnet_b0(num_classes: int = 50, pretrained: bool = False) -> nn.Module:
    """EfficientNet-B0 for CIFAR-100."""
    return EfficientNetB0Wrapper(num_classes=num_classes)


# ─── VGG-16-BN ───

class VGG16BNWrapper(nn.Module):
    """VGG-16 with batch normalization for CIFAR-100 (32x32). ~15M params.

    Classic deep CNN with no skip connections. Tests whether pure depth
    without residual paths creates topological persistence.
    Uses .fc for classifier compatibility.
    """
    def __init__(self, num_classes=50):
        super().__init__()
        base = vgg16_bn(weights=None)
        # Adapt classifier for CIFAR-100 (smaller spatial dims than ImageNet)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_vgg16_bn(num_classes: int = 50, pretrained: bool = False) -> nn.Module:
    """VGG-16-BN for CIFAR-100."""
    return VGG16BNWrapper(num_classes=num_classes)


# ─── ConvNeXt-Tiny ───

class ConvNeXtTinyWrapper(nn.Module):
    """ConvNeXt-Tiny for CIFAR-100 (32x32). ~28M params.

    Modern CNN inspired by Vision Transformers (patchify stem, LayerNorm,
    inverted bottleneck). Uses .fc for classifier compatibility.
    """
    def __init__(self, num_classes=50):
        super().__init__()
        base = convnext_tiny(weights=None)
        # Replace stem's first conv to handle 32x32 (original: 4x4 stride 4)
        base.features[0][0] = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_convnext_tiny(num_classes: int = 50, pretrained: bool = False) -> nn.Module:
    """ConvNeXt-Tiny for CIFAR-100."""
    return ConvNeXtTinyWrapper(num_classes=num_classes)


# ─── MobileNet-V3-Small ───

class MobileNetV3SmallWrapper(nn.Module):
    """MobileNet-V3-Small for CIFAR-100 (32x32). ~1.5M params.

    Lightweight architecture with squeeze-and-excitation, hardswish.
    Very different inductive bias from ResNets. Uses .fc for compatibility.
    """
    def __init__(self, num_classes=50):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        # Replace first conv for CIFAR-100 (no stride-2 downsampling)
        base.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_mobilenet_v3_small(num_classes: int = 50, pretrained: bool = False) -> nn.Module:
    """MobileNet-V3-Small for CIFAR-100."""
    return MobileNetV3SmallWrapper(num_classes=num_classes)


# ─── ViT-Tiny ───

def get_vit_tiny(num_classes: int = 50, pretrained: bool = False) -> nn.Module:
    """Tiny ViT (2 layers, 128 embed dim) for CIFAR-100. ~0.8M params.

    Smaller than ViT-Small to test if transformer topology pattern holds at
    smaller scale.
    """
    return ViTSmall(
        num_classes=num_classes,
        embed_dim=128,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )


# ─── ShuffleNet-V2 ───

class ShuffleNetV2Wrapper(nn.Module):
    """ShuffleNet-V2 x1.0 for CIFAR-100 (32x32). ~1.3M params.

    Channel shuffle operations for efficient feature mixing.
    Uses .fc for classifier compatibility.
    """
    def __init__(self, num_classes=50):
        super().__init__()
        base = shufflenet_v2_x1_0(weights=None)
        # Replace first conv for CIFAR-100
        base.conv1[0] = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        self.features = nn.Sequential(base.conv1, base.maxpool, base.stage2, base.stage3, base.stage4, base.conv5)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_shufflenet_v2(num_classes: int = 50, pretrained: bool = False) -> nn.Module:
    """ShuffleNet-V2 for CIFAR-100."""
    return ShuffleNetV2Wrapper(num_classes=num_classes)


# ─── RegNet-Y-400MF ───

class RegNetY400MFWrapper(nn.Module):
    """RegNet-Y-400MF for CIFAR-100 (32x32). ~4.3M params.

    Systematically designed (NAS-like) architecture with squeeze-and-excitation.
    Uses .fc for classifier compatibility.
    """
    def __init__(self, num_classes=50):
        super().__init__()
        base = regnet_y_400mf(weights=None)
        # Replace stem conv for CIFAR-100
        base.stem[0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem = base.stem
        self.trunk = nn.Sequential(base.trunk_output)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(440, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_regnet_y400mf(num_classes: int = 50, pretrained: bool = False) -> nn.Module:
    """RegNet-Y-400MF for CIFAR-100."""
    return RegNetY400MFWrapper(num_classes=num_classes)


def get_model(architecture: str, num_classes: int = 50, **kwargs) -> nn.Module:
    """Factory function for model creation."""
    models = {
        "resnet18": get_resnet18,
        "resnet50": get_resnet50,
        "vit_small": get_vit_small,
        "wrn2810": get_wrn2810,
        "mlp_mixer": get_mlp_mixer,
        "resnet18_wide": get_resnet18_wide,
        "densenet121": get_densenet121,
        "efficientnet_b0": get_efficientnet_b0,
        "vgg16_bn": get_vgg16_bn,
        "convnext_tiny": get_convnext_tiny,
        "mobilenet_v3_small": get_mobilenet_v3_small,
        "vit_tiny": get_vit_tiny,
        "shufflenet_v2": get_shufflenet_v2,
        "regnet_y400mf": get_regnet_y400mf,
    }
    if architecture not in models:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(models.keys())}")
    return models[architecture](num_classes=num_classes, **kwargs)
