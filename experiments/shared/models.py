"""
Standard architectures for experiments.

All models accept num_classes parameter for split-task training.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_resnet18(num_classes: int = 50, pretrained: bool = False) -> nn.Module:
    """ResNet-18 adapted for CIFAR-100 (32x32 input).

    Modifications from ImageNet ResNet-18:
    - First conv: 3x3 kernel, stride 1, padding 1 (instead of 7x7 stride 2)
    - Remove initial max pool (not needed for 32x32)
    """
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = resnet18(weights=None)

    # Adapt for 32x32 input
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Adjust final FC for num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_model(architecture: str, num_classes: int = 50, **kwargs) -> nn.Module:
    """Factory function for model creation."""
    models = {
        "resnet18": get_resnet18,
    }
    if architecture not in models:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(models.keys())}")
    return models[architecture](num_classes=num_classes, **kwargs)
