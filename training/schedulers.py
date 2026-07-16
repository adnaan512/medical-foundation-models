"""
Learning rate schedulers with warm-up support.
"""
from __future__ import annotations
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, _LRScheduler


class CosineWithWarmup(_LRScheduler):
    """
    Cosine annealing schedule with a linear warm-up phase.

    During warm-up (epochs 0..warmup_epochs), LR grows linearly from 0 to base_lr.
    Afterwards it follows a cosine decay down to min_lr.

    Args:
        optimizer:      PyTorch optimizer.
        warmup_epochs:  Number of linear warm-up epochs.
        total_epochs:   Total training epochs.
        min_lr_ratio:   min_lr = base_lr * min_lr_ratio.
        last_epoch:     Resume epoch index (-1 = start fresh).
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 3,
        total_epochs: int = 50,
        min_lr_ratio: float = 1e-2,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if epoch < self.warmup_epochs:
                lr = base_lr * (epoch + 1) / max(self.warmup_epochs, 1)
            else:
                progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay)
            lrs.append(lr)
        return lrs


def build_scheduler(optimizer: Optimizer, cfg: dict) -> _LRScheduler:
    """Build LR scheduler from config."""
    sched_cfg = cfg.get("scheduler", {})
    name = sched_cfg.get("name", "cosine_with_warmup")
    total_epochs = cfg.get("training", {}).get("num_epochs", 50)

    if name == "cosine_with_warmup":
        return CosineWithWarmup(
            optimizer,
            warmup_epochs=sched_cfg.get("warmup_epochs", 3),
            total_epochs=total_epochs,
            min_lr_ratio=sched_cfg.get("min_lr", 1e-6) / max(cfg["optimizer"].get("lr", 1e-4), 1e-10),
        )
    elif name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", total_epochs),
            eta_min=sched_cfg.get("min_lr", 1e-6),
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")
