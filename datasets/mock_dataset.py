"""
datasets/mock_dataset.py — Synthetic in-memory dataset for unit tests and CI.

Provides ``MockHAM10000Dataset`` — a drop-in replacement for ``HAM10000Dataset``
that generates random RGB tensors instead of loading real images.  This makes
every test that touches a DataLoader completely self-contained:

    * No HAM10000 files needed.
    * Runs on any machine / CI runner in milliseconds.
    * Supports the same API surface as the real dataset class.

Usage
-----
>>> from datasets.mock_dataset import MockHAM10000Dataset, build_mock_dataloaders
>>> train_ds = MockHAM10000Dataset(split="train", num_samples=64)
>>> img, label = train_ds[0]
>>> img.shape
torch.Size([3, 224, 224])

>>> train_loader, val_loader, test_loader = build_mock_dataloaders(batch_size=8)
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.ham10000 import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS

logger = logging.getLogger(__name__)

NUM_CLASSES = len(CLASS_TO_IDX)  # 7


class MockHAM10000Dataset(Dataset):
    """
    Synthetic drop-in for HAM10000Dataset — generates random tensors.

    Args:
        split:        ``'train'``, ``'val'``, or ``'test'``.
        num_samples:  Total number of samples in this split.
        image_size:   Spatial resolution (square) of generated images.
        num_classes:  Number of output classes (default 7 — HAM10000).
        seed:         Seed for the random number generator.
        transform:    Optional transform applied to each PIL/tensor image.
                      When ``None``, raw float tensors are returned directly.
    """

    def __init__(
        self,
        split: str = "train",
        num_samples: int = 128,
        image_size: int = 224,
        num_classes: int = NUM_CLASSES,
        seed: int = 42,
        transform=None,
    ) -> None:
        assert split in {"train", "val", "test"}, f"Unknown split '{split}'"
        self.split = split
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform

        rng = np.random.default_rng(seed)
        # Pre-generate labels with roughly balanced distribution
        self._labels: list[int] = rng.integers(0, num_classes, size=num_samples).tolist()

        logger.info(
            "MockHAM10000Dataset [%s] — %d synthetic samples, %d classes",
            split.upper(),
            num_samples,
            num_classes,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Deterministic per-index image so tests are reproducible
        torch.manual_seed(idx)
        img = torch.rand(3, self.image_size, self.image_size)

        if self.transform is not None:
            # Convert to PIL for torchvision transforms compatibility
            from torchvision.transforms.functional import to_pil_image
            img = self.transform(to_pil_image(img))

        return img, self._labels[idx]

    # ------------------------------------------------------------------
    # HAM10000Dataset-compatible API surface
    # ------------------------------------------------------------------

    @property
    def labels(self) -> list[int]:
        """Integer label list — compatible with WeightedRandomSampler."""
        return self._labels

    @property
    def class_weights(self) -> np.ndarray:
        """Inverse-frequency class weights (float32 array, shape (num_classes,))."""
        counts = np.bincount(self._labels, minlength=self.num_classes).astype(np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        weights = 1.0 / counts
        weights /= weights.sum()
        return weights

    def get_class_distribution(self) -> dict[str, int]:
        """Return ``{class_abbreviation: count}`` dict for this split."""
        counts = np.bincount(self._labels, minlength=self.num_classes)
        abbrevs = list(CLASS_TO_IDX.keys())
        return {abbrevs[i]: int(counts[i]) for i in range(self.num_classes)}


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_mock_dataloaders(
    batch_size: int = 8,
    image_size: int = 224,
    num_train: int = 64,
    num_val: int = 32,
    num_test: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build mock train / val / test DataLoaders for fast unit testing.

    No real images required — all data is randomly generated in memory.

    Args:
        batch_size:   Mini-batch size.
        image_size:   Spatial resolution.
        num_train:    Number of synthetic training samples.
        num_val:      Number of synthetic validation samples.
        num_test:     Number of synthetic test samples.
        num_workers:  DataLoader worker processes (0 = main process).
        seed:         Random seed.

    Returns:
        ``(train_loader, val_loader, test_loader)``
    """
    train_ds = MockHAM10000Dataset("train", num_train, image_size, seed=seed)
    val_ds   = MockHAM10000Dataset("val",   num_val,   image_size, seed=seed + 1)
    test_ds  = MockHAM10000Dataset("test",  num_test,  image_size, seed=seed + 2)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)

    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader
