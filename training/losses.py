"""
Loss functions for imbalanced multi-class classification.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing to reduce overconfidence.

    Args:
        smoothing: Label smoothing factor in [0, 1). 0 = standard CE.
        weight:    Per-class weights tensor for imbalanced datasets.
    """
    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        if self.weight is not None:
            w = self.weight[targets].unsqueeze(1)
            loss = -(smooth_targets * log_probs * w).sum(dim=-1).mean()
        else:
            loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for hard-example mining in imbalanced classification.

    Reference:
        Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV.

    Args:
        gamma:  Focusing parameter (0 = standard CE, 2 recommended).
        weight: Per-class weights.
        reduction: 'mean' | 'sum' | 'none'.
    """
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_loss(cfg: dict, class_weights: Optional[np.ndarray] = None, device: torch.device = None) -> nn.Module:
    """Build the loss function from config."""
    smoothing = cfg.get("training", {}).get("label_smoothing", 0.1)
    weight_tensor = None
    if class_weights is not None and device is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return LabelSmoothingCrossEntropy(smoothing=smoothing, weight=weight_tensor)
