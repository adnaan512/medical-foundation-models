"""
HAM10000 Dataset — Human Against Machine 10000 Skin Lesion Dataset.

Classes (7):
    akiec — Actinic keratoses and intraepithelial carcinoma
    bcc   — Basal cell carcinoma
    bkl   — Benign keratosis-like lesions
    df    — Dermatofibroma
    mel   — Melanoma
    nv    — Melanocytic nevi
    vasc  — Vascular lesions

Reference:
    Tschandl, P. et al. (2018). The HAM10000 dataset, a large collection of
    multi-source dermatoscopic images of common pigmented skin lesions.
    Scientific Data, 5, 180161. https://doi.org/10.1038/sdata.2018.161
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label mapping — sorted alphabetically for determinism
# ---------------------------------------------------------------------------
CLASS_TO_IDX: Dict[str, int] = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}

IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

CLASS_NAMES: List[str] = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for the HAM10000 skin lesion classification benchmark.

    Supports stratified train/val/test splitting and handles duplicate
    patient IDs by ensuring no patient appears in multiple splits.

    Args:
        root_dir: Path to the dataset root containing images/ and metadata CSV.
        split: One of 'train', 'val', or 'test'.
        transform: Optional torchvision transform applied to each image.
        metadata_file: Name of the CSV metadata file.
        train_ratio: Fraction of data allocated to training.
        val_ratio: Fraction of data allocated to validation.
        seed: Random seed for reproducible splits.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        metadata_file: str = "HAM10000_metadata.csv",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        super().__init__()

        assert split in {"train", "val", "test"}, (
            f"split must be 'train', 'val', or 'test', got '{split}'"
        )
        assert abs(train_ratio + val_ratio + (1 - train_ratio - val_ratio) - 1.0) < 1e-6

        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.seed = seed

        # Load and validate metadata
        metadata_path = self.root_dir / metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                "Please download HAM10000 from: "
                "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000"
            )

        self.metadata = self._load_metadata(metadata_path)
        self.image_dir = self._find_image_dir()

        # Create stratified splits
        self.samples = self._create_split(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        logger.info(
            "HAM10000 [%s] — %d samples, %d classes",
            split.upper(),
            len(self.samples),
            len(CLASS_TO_IDX),
        )
        self._log_class_distribution()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_metadata(self, path: Path) -> pd.DataFrame:
        """Load, validate, and encode metadata CSV."""
        df = pd.read_csv(path)

        required_cols = {"image_id", "dx", "lesion_id"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Metadata CSV missing columns: {missing}")

        # Encode labels
        df["label"] = df["dx"].map(CLASS_TO_IDX)
        unknown_labels = df["label"].isna()
        if unknown_labels.any():
            bad = df.loc[unknown_labels, "dx"].unique().tolist()
            raise ValueError(f"Unknown class labels in metadata: {bad}")

        df["label"] = df["label"].astype(int)
        logger.info("Loaded metadata: %d records", len(df))
        return df

    def _find_image_dir(self) -> Path:
        """Locate the images directory (handles different download layouts)."""
        candidates = [
            self.root_dir / "images",
            self.root_dir / "HAM10000_images_part_1",
            self.root_dir,
        ]
        for candidate in candidates:
            if candidate.exists() and any(candidate.glob("*.jpg")):
                logger.info("Image directory: %s", candidate)
                return candidate
        raise FileNotFoundError(
            f"No image directory with .jpg files found under {self.root_dir}. "
            "Expected 'images/' subfolder."
        )

    def _create_split(
        self,
        train_ratio: float,
        val_ratio: float,
    ) -> pd.DataFrame:
        """
        Build a stratified, patient-aware train/val/test split.

        Patient-awareness: a lesion (lesion_id) is assigned to exactly one
        split so the same patient's images don't leak across splits.
        """
        df = self.metadata.copy()

        # Aggregate by lesion to avoid patient leakage
        lesion_df = (
            df.groupby("lesion_id")
            .agg(label=("label", "first"))
            .reset_index()
        )

        test_ratio = 1.0 - train_ratio - val_ratio

        # First split: train vs (val + test)
        train_lesions, temp_lesions = train_test_split(
            lesion_df,
            test_size=(val_ratio + test_ratio),
            stratify=lesion_df["label"],
            random_state=self.seed,
        )

        # Second split: val vs test (from the temp set)
        relative_val = val_ratio / (val_ratio + test_ratio)
        val_lesions, test_lesions = train_test_split(
            temp_lesions,
            test_size=(1.0 - relative_val),
            stratify=temp_lesions["label"],
            random_state=self.seed,
        )

        split_map = {
            "train": set(train_lesions["lesion_id"]),
            "val": set(val_lesions["lesion_id"]),
            "test": set(test_lesions["lesion_id"]),
        }

        mask = df["lesion_id"].isin(split_map[self.split])
        return df[mask].reset_index(drop=True)

    def _log_class_distribution(self) -> None:
        """Log per-class sample counts to help spot imbalance."""
        counts = self.samples["label"].value_counts().sort_index()
        for idx, count in counts.items():
            cls_name = IDX_TO_CLASS[idx]
            pct = 100.0 * count / len(self.samples)
            logger.debug("  %-8s  %4d  (%.1f%%)", cls_name, count, pct)

    def _find_image_path(self, image_id: str) -> Path:
        """Search for image file across possible sub-directories."""
        # HAM10000 is sometimes spread across part_1 and part_2 directories
        candidates = [
            self.image_dir / f"{image_id}.jpg",
            self.root_dir / "HAM10000_images_part_1" / f"{image_id}.jpg",
            self.root_dir / "HAM10000_images_part_2" / f"{image_id}.jpg",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Image '{image_id}.jpg' not found under {self.root_dir}"
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[any, int]:
        row = self.samples.iloc[idx]
        image_path = self._find_image_path(row["image_id"])

        # Load image as RGB (ensures consistent channel count)
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, int(row["label"])

    # ------------------------------------------------------------------
    # Utility accessors
    # ------------------------------------------------------------------

    @property
    def class_weights(self) -> np.ndarray:
        """
        Compute inverse-frequency class weights for loss weighting.

        Returns:
            float32 array of shape (num_classes,) where rarer classes
            receive higher weights.
        """
        counts = np.zeros(len(CLASS_TO_IDX), dtype=np.float32)
        for label in self.samples["label"]:
            counts[label] += 1.0

        # Avoid division by zero for missing classes
        counts = np.where(counts == 0, 1.0, counts)
        weights = 1.0 / counts
        weights /= weights.sum()   # Normalise so weights sum to 1
        return weights

    @property
    def labels(self) -> List[int]:
        """Return list of integer labels for all samples (useful for samplers)."""
        return self.samples["label"].tolist()

    def get_class_distribution(self) -> Dict[str, int]:
        """Return {class_abbreviation: count} dict for the split."""
        counts = self.samples["label"].value_counts().sort_index()
        return {IDX_TO_CLASS[idx]: int(cnt) for idx, cnt in counts.items()}
