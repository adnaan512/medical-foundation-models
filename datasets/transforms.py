"""
Image transformation pipelines for training, validation, and inference.

Dermatoscopy-specific augmentations are applied during training to improve
generalisation across different imaging devices and patient skin tones.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# ImageNet normalisation statistics (used for all pretrained models)
# ---------------------------------------------------------------------------
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
    augmentation_cfg: Dict | None = None,
) -> T.Compose:
    """
    Build the training augmentation pipeline.

    Augmentation strategy for skin lesion images:
        1. Resize with small random scale variation (simulates zoom).
        2. Random crop to target size.
        3. Horizontal + vertical flips (lesions are rotation-invariant).
        4. Moderate rotation (dermatoscopes capture at arbitrary angles).
        5. Colour jitter (accounts for device colour calibration differences).
        6. Random erasing (improves generalisation; acts as Cutout).
        7. Normalise to ImageNet statistics.

    Args:
        image_size: Target spatial resolution (square).
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation std.
        augmentation_cfg: Optional dict overriding default aug parameters.

    Returns:
        A composed torchvision transform.
    """
    cfg = {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.5,
        "rotation_degrees": 30,
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        },
        "random_erasing": 0.1,
    }
    if augmentation_cfg is not None:
        cfg.update(augmentation_cfg)

    transforms = [
        # Slightly larger crop before random crop improves scale invariance
        T.Resize(int(image_size * 1.15)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=cfg["horizontal_flip"]),
        T.RandomVerticalFlip(p=cfg["vertical_flip"]),
        T.RandomRotation(degrees=cfg["rotation_degrees"]),
        T.ColorJitter(**cfg["color_jitter"]),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]

    # Optional random erasing (applied in tensor space)
    if cfg["random_erasing"] > 0:
        transforms.append(
            T.RandomErasing(
                p=cfg["random_erasing"],
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value="random",
            )
        )

    return T.Compose(transforms)


def get_val_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> T.Compose:
    """
    Build the deterministic validation / test transform pipeline.

    Only resize + centre crop + normalise — no stochastic augmentation.

    Args:
        image_size: Target spatial resolution (square).
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation std.

    Returns:
        A composed torchvision transform.
    """
    return T.Compose(
        [
            T.Resize(int(image_size * 1.14)),   # ~256 for 224 target
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def get_inference_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> T.Compose:
    """
    Alias for val transforms — used during single-image inference.

    Identical to get_val_transforms but semantically named for clarity
    in inference scripts.
    """
    return get_val_transforms(image_size=image_size, mean=mean, std=std)


def get_tta_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
    n_augmentations: int = 5,
) -> List[T.Compose]:
    """
    Test-Time Augmentation (TTA) transform set.

    Returns a list of transforms representing different TTA views of the
    same image. Predictions from each view are averaged at inference time.

    Args:
        image_size: Target spatial resolution.
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation std.
        n_augmentations: Number of TTA views to generate.

    Returns:
        List of transforms; the first is always the canonical centre crop.
    """
    base = get_val_transforms(image_size, mean, std)
    aug = get_train_transforms(image_size, mean, std)

    return [base] + [aug] * (n_augmentations - 1)


def denormalise(
    tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
):
    """
    Reverse ImageNet normalisation for visualisation.

    Args:
        tensor: Normalised image tensor of shape (C, H, W) or (B, C, H, W).
        mean: Per-channel mean used during normalisation.
        std: Per-channel std used during normalisation.

    Returns:
        Denormalised tensor clipped to [0, 1].
    """
    import torch

    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    if tensor.dim() == 4:
        # Batch dimension: (B, C, H, W)
        mean_t = mean_t.view(1, 3, 1, 1)
        std_t = std_t.view(1, 3, 1, 1)
    else:
        # Single image: (C, H, W)
        mean_t = mean_t.view(3, 1, 1)
        std_t = std_t.view(3, 1, 1)

    return (tensor * std_t + mean_t).clamp(0.0, 1.0)
