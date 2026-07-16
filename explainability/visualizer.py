"""
Explainability visualisation utilities.
Overlay Grad-CAM and Attention Rollout heatmaps on original images.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from datasets.transforms import denormalise

logger = logging.getLogger(__name__)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Blend a heatmap onto an RGB image.

    Args:
        image:    Float32 RGB array (H, W, 3) in [0, 1].
        heatmap:  Float32 array (H, W) in [0, 1].
        alpha:    Heatmap opacity.
        colormap: Matplotlib colormap name.

    Returns:
        Blended float32 RGB array (H, W, 3) in [0, 1].
    """
    cmap = cm.get_cmap(colormap)
    heatmap_rgb = cmap(heatmap)[:, :, :3].astype(np.float32)
    blended = (1 - alpha) * image + alpha * heatmap_rgb
    return np.clip(blended, 0, 1)


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised image tensor (C,H,W) to a uint8 numpy array (H,W,3)."""
    img = denormalise(tensor).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0, 1).astype(np.float32)


def save_gradcam_figure(
    images: torch.Tensor,
    heatmaps: List[np.ndarray],
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str,
    title: str = "Grad-CAM Visualisations",
    n_cols: int = 4,
) -> None:
    """
    Save a figure grid of Grad-CAM overlays.

    Args:
        images:      Batch of normalised image tensors (B, C, H, W).
        heatmaps:    List of B heatmap arrays (H, W).
        y_true:      Ground-truth label indices.
        y_pred:      Predicted label indices.
        class_names: Readable class names.
        save_path:   File path for the saved figure.
        title:       Figure suptitle.
        n_cols:      Number of columns in the grid.
    """
    B = len(heatmaps)
    n_rows = (B + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = np.array(axes).flatten()

    for i, (img_t, hmap, gt, pred) in enumerate(zip(images, heatmaps, y_true, y_pred)):
        img_np = tensor_to_numpy_image(img_t)
        overlay = overlay_heatmap(img_np, hmap)
        axes[i].imshow(overlay)
        colour = "green" if gt == pred else "red"
        axes[i].set_title(
            f"True: {class_names[gt]}\nPred: {class_names[pred]}",
            fontsize=8,
            color=colour,
            fontweight="bold",
        )
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def save_attention_rollout_figure(
    images: torch.Tensor,
    heatmaps: List[np.ndarray],
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str,
    title: str = "Attention Rollout Visualisations",
    n_cols: int = 4,
) -> None:
    """Save a figure grid of Attention Rollout overlays (identical signature to Grad-CAM)."""
    save_gradcam_figure(
        images=images,
        heatmaps=heatmaps,
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=save_path,
        title=title,
        n_cols=n_cols,
    )


def save_side_by_side(
    image: torch.Tensor,
    heatmap: np.ndarray,
    true_label: str,
    pred_label: str,
    save_path: str,
    method: str = "Grad-CAM",
) -> None:
    """Save original | heatmap | overlay as a single figure for a single image."""
    img_np = tensor_to_numpy_image(image)
    overlay = overlay_heatmap(img_np, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np); axes[0].set_title("Original", fontsize=11); axes[0].axis("off")

    cmap_img = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"{method} Map", fontsize=11); axes[1].axis("off")
    plt.colorbar(cmap_img, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay); axes[2].axis("off")
    colour = "green" if true_label == pred_label else "red"
    axes[2].set_title(f"Overlay\nTrue: {true_label} | Pred: {pred_label}", fontsize=10, color=colour)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
