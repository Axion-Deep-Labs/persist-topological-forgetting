"""
Split-CIFAR-100 and other continual learning benchmarks.

Provides Task A / Task B splits for sequential training experiments.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar100_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


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
