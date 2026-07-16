"""
Publication-quality figure generation for training curves, ROC curves,
confusion matrices, and precision-recall curves.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve, auc, precision_recall_curve

logger = logging.getLogger(__name__)

# Publication style defaults
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#795548"]


def save_training_curves(history: Dict, save_dir: str, model_name: str = "") -> None:
    """Plot and save loss + accuracy training curves."""
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color=PALETTE[0], linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color=PALETTE[1], linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Training & Validation Loss\n{model_name}")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color=PALETTE[2], linewidth=2)
    ax2.axhline(max(history["val_acc"]), linestyle=":", color="gray", alpha=0.7,
                label=f"Best: {max(history['val_acc']):.4f}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Validation Accuracy\n{model_name}")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    path = out / f"training_curves_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path); plt.close(fig)
    logger.info("Saved training curves: %s", path)


def save_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_dir: str,
    model_name: str = "",
) -> None:
    """Plot and save per-class ROC curves with macro-average."""
    out = Path(save_dir); out.mkdir(parents=True, exist_ok=True)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(9, 7))
    macro_tpr = np.linspace(0, 1, 200)
    macro_auc_vals = []

    for i, name in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        macro_auc_vals.append(np.interp(macro_tpr, fpr, tpr))
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], linewidth=1.5,
                label=f"{name} (AUC={roc_auc:.3f})", alpha=0.8)

    macro_tpr_mean = np.mean(macro_auc_vals, axis=0)
    macro_auc_score = auc(macro_tpr, macro_tpr_mean)
    ax.plot(macro_tpr, macro_tpr_mean, "k--", linewidth=2.5,
            label=f"Macro-avg (AUC={macro_auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "gray", linestyle=":", linewidth=1, label="Random")

    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {model_name}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    path = out / f"roc_curves_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path); plt.close(fig)
    logger.info("Saved ROC curves: %s", path)


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_dir: str,
    model_name: str = "",
    normalise: bool = True,
) -> None:
    """Plot and save a confusion matrix heatmap."""
    out = Path(save_dir); out.mkdir(parents=True, exist_ok=True)

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm / row_sums, 0.0)
        fmt, vmax = ".2f", 1.0
        cb_label = "Normalised Count"
    else:
        cm_plot = cm.astype(float)
        fmt, vmax = "d", None
        cb_label = "Count"

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", aspect="auto",
                   vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label=cb_label, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")

    thresh = cm_plot.max() / 2.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = f"{cm_plot[i, j]:{fmt}}" if fmt == "d" else f"{cm_plot[i, j]:.2f}"
            ax.text(j, i, val, ha="center", va="center", fontsize=8,
                    color="white" if cm_plot[i, j] > thresh else "black")

    plt.tight_layout()
    path = out / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path); plt.close(fig)
    logger.info("Saved confusion matrix: %s", path)


def save_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_dir: str,
    model_name: str = "",
) -> None:
    """Plot and save per-class Precision-Recall curves."""
    out = Path(save_dir); out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, name in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, y_prob[:, i])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, color=PALETTE[i % len(PALETTE)], linewidth=1.5,
                label=f"{name} (AP={pr_auc:.3f})", alpha=0.85)

    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {model_name}")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    path = out / f"pr_curves_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path); plt.close(fig)
    logger.info("Saved PR curves: %s", path)


def save_efficiency_comparison(
    models_data: List[Dict],
    save_dir: str,
) -> None:
    """
    Bar chart comparing trainable parameters and inference latency between models.

    Args:
        models_data: List of dicts with keys: 'name', 'trainable_params',
                     'inference_latency_ms', 'roc_auc_macro'.
        save_dir:    Output directory.
    """
    out = Path(save_dir); out.mkdir(parents=True, exist_ok=True)
    names = [d["name"] for d in models_data]
    params = [d["trainable_params"] / 1e6 for d in models_data]
    latency = [d["inference_latency_ms"] for d in models_data]
    auc_scores = [d.get("roc_auc_macro", 0) for d in models_data]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].bar(names, params, color=PALETTE[:len(names)], edgecolor="black", linewidth=0.8)
    axes[0].set_ylabel("Trainable Parameters (M)"); axes[0].set_title("Trainable Parameters")
    for i, v in enumerate(params):
        axes[0].text(i, v + 0.1, f"{v:.1f}M", ha="center", fontsize=10, fontweight="bold")

    axes[1].bar(names, latency, color=PALETTE[:len(names)], edgecolor="black", linewidth=0.8)
    axes[1].set_ylabel("Latency (ms / sample)"); axes[1].set_title("Inference Latency")
    for i, v in enumerate(latency):
        axes[1].text(i, v + 0.1, f"{v:.2f}ms", ha="center", fontsize=10, fontweight="bold")

    axes[2].bar(names, auc_scores, color=PALETTE[:len(names)], edgecolor="black", linewidth=0.8)
    axes[2].set_ylabel("Macro ROC-AUC"); axes[2].set_title("ROC-AUC Score")
    axes[2].set_ylim(0, 1.05)
    for i, v in enumerate(auc_scores):
        axes[2].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("Model Efficiency & Performance Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out / "efficiency_comparison.png"
    plt.savefig(path); plt.close(fig)
    logger.info("Saved efficiency comparison: %s", path)
