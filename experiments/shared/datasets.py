"""
Split-CIFAR-100, Split-CUB-200, Split-RESISC-45 continual learning benchmarks.

Provides Task A / Task B splits for sequential training experiments.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def get_32x32_transforms(train: bool = True, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Transforms for non-CIFAR datasets resized to 32x32."""
    if train:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


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


class CUBImageDataset(Dataset):
    """Helper dataset for CUB-200-2011 that loads images from paths."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.targets = labels  # compatibility with numpy indexing
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class SplitCUB200:
    """Split CUB-200-2011 into Task A (first N classes) and Task B (remaining).

    200 fine-grained bird species. Auto-downloads from Caltech (~1.1GB).
    All images resized to 32x32 for model consistency.
    Default split: 100/100 classes.
    """

    def __init__(self, data_dir: str, split_at: int = 100):
        self.split_at = split_at
        self.data_dir = data_dir
        cub_dir = os.path.join(data_dir, "CUB_200_2011")

        # Download if needed
        if not os.path.exists(cub_dir):
            self._download(data_dir)

        # Parse CUB metadata files
        images_txt = os.path.join(cub_dir, "images.txt")
        labels_txt = os.path.join(cub_dir, "image_class_labels.txt")
        split_txt = os.path.join(cub_dir, "train_test_split.txt")
        img_dir = os.path.join(cub_dir, "images")

        # Read image paths
        id_to_path = {}
        with open(images_txt) as f:
            for line in f:
                img_id, path = line.strip().split()
                id_to_path[int(img_id)] = os.path.join(img_dir, path)

        # Read labels (1-indexed in file, convert to 0-indexed)
        id_to_label = {}
        with open(labels_txt) as f:
            for line in f:
                img_id, label = line.strip().split()
                id_to_label[int(img_id)] = int(label) - 1

        # Read train/test split (1=train, 0=test)
        id_to_is_train = {}
        with open(split_txt) as f:
            for line in f:
                img_id, is_train = line.strip().split()
                id_to_is_train[int(img_id)] = int(is_train) == 1

        # Build train and test sets
        train_paths, train_labels = [], []
        test_paths, test_labels = [], []
        for img_id in sorted(id_to_path.keys()):
            path = id_to_path[img_id]
            label = id_to_label[img_id]
            if id_to_is_train[img_id]:
                train_paths.append(path)
                train_labels.append(label)
            else:
                test_paths.append(path)
                test_labels.append(label)

        train_transform = get_32x32_transforms(train=True)
        test_transform = get_32x32_transforms(train=False)

        self.train_full = CUBImageDataset(train_paths, train_labels, train_transform)
        self.test_full = CUBImageDataset(test_paths, test_labels, test_transform)

        # Build index masks
        train_targets = np.array(train_labels)
        test_targets = np.array(test_labels)

        self.task_a_train_idx = np.where(train_targets < split_at)[0]
        self.task_a_test_idx = np.where(test_targets < split_at)[0]
        self.task_b_train_idx = np.where(train_targets >= split_at)[0]
        self.task_b_test_idx = np.where(test_targets >= split_at)[0]

    def _download(self, data_dir):
        """Download and extract CUB-200-2011."""
        import tarfile
        import urllib.request

        url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
        os.makedirs(data_dir, exist_ok=True)
        tgz_path = os.path.join(data_dir, "CUB_200_2011.tgz")

        print(f"Downloading CUB-200-2011 to {tgz_path}...")
        urllib.request.urlretrieve(url, tgz_path)

        print("Extracting...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=data_dir)

        os.remove(tgz_path)
        print("CUB-200-2011 ready.")

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


class SplitRESISC45:
    """Split NWPU-RESISC45 into Task A and Task B.

    45 satellite scene classes, 700 images/class (31,500 total).
    No official train/test split, so we use stratified 80/20.
    All images resized to 32x32 for model consistency.
    Default split: 23/22 classes.

    Requires: pip install torchgeo (for download only).
    """

    def __init__(self, data_dir: str, split_at: int = 23, seed: int = 42):
        self.split_at = split_at
        self.data_dir = data_dir
        resisc_dir = os.path.join(data_dir, "NWPU-RESISC45")

        # Download if needed
        if not os.path.exists(resisc_dir):
            self._download(data_dir)

        # Load via ImageFolder
        all_dataset = datasets.ImageFolder(resisc_dir)

        # Stratified 80/20 train/test split
        targets = np.array([s[1] for s in all_dataset.samples])
        rng = np.random.RandomState(seed)

        train_indices, test_indices = [], []
        for cls in range(len(all_dataset.classes)):
            cls_idx = np.where(targets == cls)[0]
            rng.shuffle(cls_idx)
            split = int(0.8 * len(cls_idx))
            train_indices.extend(cls_idx[:split].tolist())
            test_indices.extend(cls_idx[split:].tolist())

        train_transform = get_32x32_transforms(train=True)
        test_transform = get_32x32_transforms(train=False)

        # Create transformed subsets
        self.train_full = TransformedSubset(all_dataset, train_indices, train_transform)
        self.test_full = TransformedSubset(all_dataset, test_indices, test_transform)

        # Build index masks within train/test sets
        train_targets = np.array([targets[i] for i in train_indices])
        test_targets = np.array([targets[i] for i in test_indices])

        self.task_a_train_idx = np.where(train_targets < split_at)[0]
        self.task_a_test_idx = np.where(test_targets < split_at)[0]
        self.task_b_train_idx = np.where(train_targets >= split_at)[0]
        self.task_b_test_idx = np.where(test_targets >= split_at)[0]

    def _download(self, data_dir):
        """Download RESISC45 using torchgeo."""
        try:
            from torchgeo.datasets import RESISC45
        except ImportError:
            raise ImportError(
                "torchgeo is required for RESISC45 download. "
                "Install with: pip install torchgeo"
            )
        print(f"Downloading RESISC45 to {data_dir}...")
        RESISC45(root=data_dir, download=True)
        print("RESISC45 ready.")

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


class TransformedSubset(Dataset):
    """Subset of an ImageFolder with a specific transform applied."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.targets = [dataset.targets[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_split_dataset(cfg):
    """Factory function: returns the right split dataset based on config."""
    dataset_name = cfg.get("dataset", "cifar100")
    data_dir = cfg["data_dir"]
    split_at = cfg["task_a_classes"][1]

    if dataset_name == "cifar100":
        return SplitCIFAR100(data_dir, split_at=split_at)
    elif dataset_name == "cub200":
        return SplitCUB200(data_dir, split_at=split_at)
    elif dataset_name == "resisc45":
        return SplitRESISC45(data_dir, split_at=split_at, seed=cfg.get("seed", 42))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: cifar100, cub200, resisc45")
