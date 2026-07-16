"""
Data utilities: DataLoader factory, weighted sampler, and dataset statistics.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.ham10000 import HAM10000Dataset
from datasets.transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    dataset_path: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    use_weighted_sampler: bool = True,
    augmentation_cfg: Optional[Dict] = None,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct train, validation, and test DataLoaders for HAM10000.

    Handles class imbalance via WeightedRandomSampler on the training set
    so that each mini-batch sees a roughly balanced class distribution.

    Args:
        dataset_path:  Root directory of HAM10000 dataset.
        image_size:    Spatial resolution for model input.
        batch_size:    Mini-batch size.
        num_workers:   Parallel data-loading processes.
        pin_memory:    Pin host memory for faster GPU transfers.
        train_ratio:   Fraction of data used for training.
        val_ratio:     Fraction of data used for validation.
        seed:          Random seed.
        use_weighted_sampler:  Balance classes during training via sampling.
        augmentation_cfg:  Optional dict of augmentation hyperparameters.
        mean:          Per-channel normalisation mean.
        std:           Per-channel normalisation std.

    Returns:
        (train_loader, val_loader, test_loader) — PyTorch DataLoader objects.
    """
    train_transform = get_train_transforms(
        image_size=image_size,
        mean=mean,
        std=std,
        augmentation_cfg=augmentation_cfg,
    )
    val_transform = get_val_transforms(image_size=image_size, mean=mean, std=std)

    # Instantiate dataset splits
    train_ds = HAM10000Dataset(
        root_dir=dataset_path,
        split="train",
        transform=train_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    val_ds = HAM10000Dataset(
        root_dir=dataset_path,
        split="val",
        transform=val_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    test_ds = HAM10000Dataset(
        root_dir=dataset_path,
        split="test",
        transform=val_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    # ---------------------------------------------------------------
    # Weighted sampler to combat the severe class imbalance in HAM10000
    # (nv accounts for ~67 % of samples; vasc < 1 %)
    # ---------------------------------------------------------------
    train_sampler = None
    shuffle_train = True

    if use_weighted_sampler:
        labels = np.array(train_ds.labels)
        class_sample_counts = np.bincount(labels, minlength=7).astype(float)
        class_sample_counts = np.where(class_sample_counts == 0, 1.0, class_sample_counts)
        # Weight each sample by the inverse frequency of its class
        sample_weights = (1.0 / class_sample_counts)[labels]
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle_train = False   # Sampler and shuffle are mutually exclusive
        logger.info("Using WeightedRandomSampler to balance training classes.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,   # Keep batch size fixed for stable BatchNorm
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    logger.info(
        "DataLoaders ready — train: %d batches | val: %d batches | test: %d batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Dataset statistics helpers
# ---------------------------------------------------------------------------

def compute_dataset_statistics(
    dataset: HAM10000Dataset,
    sample_size: int = 1000,
    seed: int = 42,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Estimate per-channel mean and std of a dataset from a random subset.

    Uses Welford's online algorithm for numerical stability.

    Args:
        dataset:     HAM10000Dataset instance (un-normalised).
        sample_size: How many images to sample for estimation.
        seed:        Random seed.

    Returns:
        (mean, std) — each a 3-tuple of floats.
    """
    import torchvision.transforms as T

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(sample_size, len(dataset)), replace=False)

    transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for idx in indices:
        row = dataset.samples.iloc[int(idx)]
        img_path = dataset._find_image_path(row["image_id"])
        img = transform(__import__("PIL").Image.open(img_path).convert("RGB"))  # (3, H, W)
        img_np = img.numpy()
        channel_sum += img_np.sum(axis=(1, 2))
        channel_sq_sum += (img_np ** 2).sum(axis=(1, 2))
        n_pixels += img_np.shape[1] * img_np.shape[2]

    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sq_sum / n_pixels - mean ** 2)

    logger.info("Dataset mean: %s | std: %s", mean.round(4), std.round(4))
    return tuple(mean.tolist()), tuple(std.tolist())


# ---------------------------------------------------------------------------
# Dataset info summary
# ---------------------------------------------------------------------------

def print_dataset_summary(
    train_ds: HAM10000Dataset,
    val_ds: HAM10000Dataset,
    test_ds: HAM10000Dataset,
) -> None:
    """Pretty-print a dataset split summary table."""
    from datasets.ham10000 import IDX_TO_CLASS, CLASS_NAMES

    header = f"{'Class':<26} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}"
    print("\n" + "=" * len(header))
    print("HAM10000 Dataset Split Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    train_dist = train_ds.get_class_distribution()
    val_dist = val_ds.get_class_distribution()
    test_dist = test_ds.get_class_distribution()

    for i, abbrev in enumerate(sorted(train_dist.keys())):
        name = CLASS_NAMES[i]
        tr = train_dist.get(abbrev, 0)
        va = val_dist.get(abbrev, 0)
        te = test_dist.get(abbrev, 0)
        total = tr + va + te
        print(f"{name:<26} {tr:>8} {va:>8} {te:>8} {total:>8}")

    print("-" * len(header))
    total_tr = len(train_ds)
    total_va = len(val_ds)
    total_te = len(test_ds)
    total = total_tr + total_va + total_te
    print(f"{'TOTAL':<26} {total_tr:>8} {total_va:>8} {total_te:>8} {total:>8}")
    print("=" * len(header) + "\n")
