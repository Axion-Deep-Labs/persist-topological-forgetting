"""
EXP-04: Modular addition dataset for grokking experiments.

Task: (a + b) mod p, where a, b in [0, p-1].
Input: one-hot encoded pair (a, b) -> 2p-dimensional vector.
Output: class label (a + b) mod p.

Standard grokking setup following Power et al. (2022).
"""

import torch
from torch.utils.data import Dataset, DataLoader


class ModularAdditionDataset(Dataset):
    """All pairs (a, b) with label (a + b) mod p."""

    def __init__(self, pairs, modulus):
        self.pairs = pairs
        self.modulus = modulus

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        # One-hot encode: [one_hot(a), one_hot(b)] -> 2p dims
        x = torch.zeros(2 * self.modulus)
        x[a] = 1.0
        x[self.modulus + b] = 1.0
        y = (a + b) % self.modulus
        return x, y


def get_dataloaders(cfg):
    """Create train/test dataloaders from config.

    Generates all p^2 pairs, splits into train_fraction / (1 - train_fraction).
    Split is deterministic (fixed generator seed 0, independent of training seed).

    If batch_size is "full", uses the entire training set as one batch
    (matches Power et al. 2022 canonical grokking setup: 1 step = 1 epoch).
    """
    modulus = cfg["task"]["modulus"]
    train_fraction = cfg["task"]["train_fraction"]
    batch_size_cfg = cfg["training"]["batch_size"]

    # All p^2 pairs
    all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]

    # Deterministic shuffle (fixed seed, independent of training seed)
    generator = torch.Generator()
    generator.manual_seed(0)
    indices = torch.randperm(len(all_pairs), generator=generator).tolist()

    n_train = int(len(all_pairs) * train_fraction)
    train_pairs = [all_pairs[i] for i in indices[:n_train]]
    test_pairs = [all_pairs[i] for i in indices[n_train:]]

    train_ds = ModularAdditionDataset(train_pairs, modulus)
    test_ds = ModularAdditionDataset(test_pairs, modulus)

    # Full-batch: batch_size = entire dataset
    if batch_size_cfg == "full":
        train_batch = len(train_ds)
        test_batch = len(test_ds)
    else:
        train_batch = int(batch_size_cfg)
        test_batch = int(batch_size_cfg)

    train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=test_batch, shuffle=False, drop_last=False)

    return train_loader, test_loader
