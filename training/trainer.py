"""
Training engine with mixed precision, early stopping, and TensorBoard logging.
"""
from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving."""
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (self.mode == "min" and score < self.best_score - self.min_delta) or \
                   (self.mode == "max" and score > self.best_score + self.min_delta)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.debug("EarlyStopping counter: %d / %d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Full training loop with:
        - Mixed-precision (AMP)
        - Gradient clipping
        - Early stopping
        - Best-model checkpoint saving
        - TensorBoard logging
        - Per-epoch timing

    Args:
        model:           PyTorch model.
        optimizer:       Configured optimizer.
        scheduler:       LR scheduler (step called once per epoch).
        criterion:       Loss function.
        train_loader:    Training DataLoader.
        val_loader:      Validation DataLoader.
        device:          Torch device.
        cfg:             Full config dict.
        experiment_name: Tag for checkpoints and TensorBoard.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        cfg: Dict,
        experiment_name: str = "experiment",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.experiment_name = experiment_name

        # Config shortcuts
        train_cfg = cfg.get("training", {})
        log_cfg = cfg.get("logging", {})

        self.num_epochs = train_cfg.get("num_epochs", 50)
        self.grad_clip = train_cfg.get("gradient_clip_norm", 1.0)
        self.mixed_precision = train_cfg.get("mixed_precision", True) and device.type == "cuda"
        self.log_interval = cfg.get("logging", {}).get("log_interval", 10)

        self.checkpoint_dir = Path(log_cfg.get("checkpoint_dir", "checkpoints")) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        tb_dir = Path(log_cfg.get("tensorboard_dir", "outputs/tensorboard")) / experiment_name
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        self.scaler = GradScaler(enabled=self.mixed_precision)
        self.early_stopping = EarlyStopping(
            patience=train_cfg.get("early_stopping_patience", 10),
            min_delta=train_cfg.get("early_stopping_delta", 0.001),
            mode="max",  # monitor val accuracy
        )

        self.history: Dict = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": [], "epoch_time": []}
        self.best_val_acc = 0.0
        self.best_epoch = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> Dict:
        """Run the full training loop. Returns training history dict."""
        logger.info("Starting training — %d epochs | device: %s | AMP: %s",
                    self.num_epochs, self.device, self.mixed_precision)

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self._val_epoch(epoch)
            epoch_time = time.time() - t0

            # LR scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # History
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)

            # TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("LR", current_lr, epoch)

            logger.info(
                "Epoch [%3d/%3d] | train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f | lr=%.2e | %.1fs",
                epoch, self.num_epochs, train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time,
            )

            # Best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self._save_checkpoint(epoch, val_loss, val_acc, is_best=True)
                logger.info("  ✓ New best val_acc=%.4f — checkpoint saved.", val_acc)

            # Periodic checkpoint every 10 epochs
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_loss, val_acc, is_best=False)

            # Early stopping
            if self.early_stopping(val_acc):
                logger.info("Early stopping triggered at epoch %d. Best epoch: %d (val_acc=%.4f).",
                            epoch, self.best_epoch, self.best_val_acc)
                break

        self.writer.close()
        logger.info("Training complete. Best val_acc=%.4f at epoch %d.", self.best_val_acc, self.best_epoch)
        return self.history

    # ------------------------------------------------------------------
    # Internal epoch loops
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """One training epoch. Returns (avg_loss, avg_accuracy)."""
        from tqdm import tqdm
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            if batch_idx % self.log_interval == 0:
                batch_loss = loss.item()
                global_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar("Loss/train_step", batch_loss, global_step)

        avg_loss = total_loss / max(total, 1)
        avg_acc = correct / max(total, 1)
        return avg_loss, avg_acc

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Tuple[float, float]:
        """One validation epoch. Returns (avg_loss, avg_accuracy)."""
        from tqdm import tqdm
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Val]", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "cfg": self.cfg,
        }
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch:03d}.pth"
        path = self.checkpoint_dir / filename
        torch.save(state, path)

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Return current GPU memory stats in MB."""
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
