"""
Split-CIFAR-100, Split-CIFAR-10, and other continual learning benchmarks.

Provides Task A / Task B splits for sequential training experiments.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar_transforms(train: bool = True, mean=CIFAR100_MEAN, std=CIFAR100_STD):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_cifar100_transforms(train: bool = True):
    return get_cifar_transforms(train, CIFAR100_MEAN, CIFAR100_STD)


def get_cifar10_transforms(train: bool = True):
    return get_cifar_transforms(train, CIFAR10_MEAN, CIFAR10_STD)


class SplitCIFAR100:
    """Split CIFAR-100 into Task A (first N classes) and Task B (remaining)."""

    def __init__(self, data_dir: str, split_at: int = 50):
        self.split_at = split_at
        self.data_dir = data_dir

        self.train_full = datasets.CIFAR100(
            data_dir, train=True, download=True,
            transform=get_cifar100_transforms(train=True),
        )
        self.test_full = datasets.CIFAR100(
            data_dir, train=False, download=True,
            transform=get_cifar100_transforms(train=False),
        )

        # Build index masks
        train_targets = np.array(self.train_full.targets)
        test_targets = np.array(self.test_full.targets)

        self.task_a_train_idx = np.where(train_targets < split_at)[0]
        self.task_a_test_idx = np.where(test_targets < split_at)[0]
        self.task_b_train_idx = np.where(train_targets >= split_at)[0]
        self.task_b_test_idx = np.where(test_targets >= split_at)[0]

    def get_task_a(self, batch_size: int = 128, num_workers: int = 4):
        train_ds = RemappedSubset(self.train_full, self.task_a_train_idx, offset=0)
        test_ds = RemappedSubset(self.test_full, self.task_a_test_idx, offset=0)
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True),
        )

    def get_task_b(self, batch_size: int = 128, num_workers: int = 4):
        train_ds = RemappedSubset(self.train_full, self.task_b_train_idx, offset=self.split_at)
        test_ds = RemappedSubset(self.test_full, self.task_b_test_idx, offset=self.split_at)
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True),
        )


class SplitCIFAR10:
    """Split CIFAR-10 into Task A (first N classes) and Task B (remaining)."""

    def __init__(self, data_dir: str, split_at: int = 5):
        self.split_at = split_at
        self.data_dir = data_dir

        self.train_full = datasets.CIFAR10(
            data_dir, train=True, download=True,
            transform=get_cifar10_transforms(train=True),
        )
        self.test_full = datasets.CIFAR10(
            data_dir, train=False, download=True,
            transform=get_cifar10_transforms(train=False),
        )

        train_targets = np.array(self.train_full.targets)
        test_targets = np.array(self.test_full.targets)

        self.task_a_train_idx = np.where(train_targets < split_at)[0]
        self.task_a_test_idx = np.where(test_targets < split_at)[0]
        self.task_b_train_idx = np.where(train_targets >= split_at)[0]
        self.task_b_test_idx = np.where(test_targets >= split_at)[0]

    def get_task_a(self, batch_size: int = 128, num_workers: int = 4):
        train_ds = RemappedSubset(self.train_full, self.task_a_train_idx, offset=0)
        test_ds = RemappedSubset(self.test_full, self.task_a_test_idx, offset=0)
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True),
        )

    def get_task_b(self, batch_size: int = 128, num_workers: int = 4):
        train_ds = RemappedSubset(self.train_full, self.task_b_train_idx, offset=self.split_at)
        test_ds = RemappedSubset(self.test_full, self.task_b_test_idx, offset=self.split_at)
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True),
        )


def get_split_dataset(cfg):
    """Factory function: returns the right split dataset based on config."""
    dataset_name = cfg.get("dataset", "cifar100")
    data_dir = cfg["data_dir"]
    split_at = cfg["task_a_classes"][1]

    if dataset_name == "cifar100":
        return SplitCIFAR100(data_dir, split_at=split_at)
    elif dataset_name == "cifar10":
        return SplitCIFAR10(data_dir, split_at=split_at)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: cifar100, cifar10")


class RemappedSubset(Dataset):
    """Subset with labels remapped to [0, N) range."""

    def __init__(self, dataset, indices, offset: int = 0):
        self.dataset = dataset
        self.indices = indices
        self.offset = offset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return img, label - self.offset
