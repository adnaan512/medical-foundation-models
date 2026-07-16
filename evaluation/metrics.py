"""
Classification and efficiency metrics for model evaluation.
"""
from __future__ import annotations
import time
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, average_precision_score,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    num_classes: int = 7,
) -> Dict:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true:       Ground-truth integer labels (N,).
        y_pred:       Predicted integer labels (N,).
        y_prob:       Softmax probability array (N, num_classes).
        class_names:  Human-readable class names.
        num_classes:  Total number of classes.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc, confusion_matrix,
        per_class metrics, and full classification report.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # One-vs-rest ROC-AUC
    try:
        roc_auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        roc_auc_weighted = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    except ValueError as e:
        logger.warning("ROC-AUC computation failed: %s", e)
        roc_auc_macro = roc_auc_weighted = 0.0

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Average precision (PR-AUC) per class
    per_class_ap = []
    for c in range(num_classes):
        try:
            ap = average_precision_score((y_true == c).astype(int), y_prob[:, c])
        except Exception:
            ap = 0.0
        per_class_ap.append(ap)

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "precision_weighted": float(precision_weighted),
        "recall_macro": float(recall_macro),
        "recall_weighted": float(recall_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "roc_auc_macro": float(roc_auc_macro),
        "roc_auc_weighted": float(roc_auc_weighted),
        "confusion_matrix": cm,
        "classification_report": report,
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "per_class_ap": per_class_ap,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def compute_efficiency_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_warmup_batches: int = 5,
) -> Dict:
    """
    Measure inference speed and GPU memory footprint.

    Args:
        model:               Model in eval mode.
        dataloader:          DataLoader for timing (uses first N batches).
        device:              Target device.
        num_warmup_batches:  Warm-up iterations before timing.

    Returns:
        Dict with parameter counts, inference time, and GPU memory stats.
    """
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # GPU memory before inference
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        mem_before = torch.cuda.memory_allocated(device) / 1e6  # MB

    # Warmup
    batch_iter = iter(dataloader)
    with torch.no_grad():
        for _ in range(min(num_warmup_batches, len(dataloader))):
            try:
                images, _ = next(batch_iter)
                _ = model(images.to(device))
            except StopIteration:
                break

    # Timed inference
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t_start = time.perf_counter()
    n_samples = 0
    n_batches = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)
            n_samples += images.size(0)
            n_batches += 1
            if n_batches >= 50:   # Limit timing to 50 batches
                break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        mem_peak = torch.cuda.max_memory_allocated(device) / 1e6  # MB
    else:
        mem_peak = 0.0

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    throughput = n_samples / max(elapsed, 1e-6)
    latency_ms = (elapsed / max(n_samples, 1)) * 1000

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "trainable_pct": 100.0 * trainable_params / max(total_params, 1),
        "inference_latency_ms": latency_ms,
        "throughput_samples_per_sec": throughput,
        "gpu_peak_memory_mb": mem_peak,
    }
