"""
tests/test_losses.py — Unit tests for custom loss functions.

Run with:
    pytest tests/test_losses.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.losses import LabelSmoothingCrossEntropy, FocalLoss, build_loss  # noqa: E402

NUM_CLASSES = 7
BATCH_SIZE = 16


@pytest.fixture()
def random_batch():
    """Return (logits, targets) for a random batch."""
    torch.manual_seed(0)
    logits = torch.randn(BATCH_SIZE, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    return logits, targets


# ---------------------------------------------------------------------------
# LabelSmoothingCrossEntropy
# ---------------------------------------------------------------------------

class TestLabelSmoothingCrossEntropy:
    def test_loss_is_scalar(self, random_batch):
        logits, targets = random_batch
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0, "Loss should be a scalar tensor."

    def test_loss_is_positive(self, random_batch):
        logits, targets = random_batch
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_zero_smoothing_matches_cross_entropy(self, random_batch):
        """smoothing=0 should match standard cross-entropy closely."""
        logits, targets = random_batch
        ls_loss = LabelSmoothingCrossEntropy(smoothing=0.0)(logits, targets)
        ce_loss = F.cross_entropy(logits, targets)
        assert abs(ls_loss.item() - ce_loss.item()) < 1e-4

    def test_smoothing_increases_loss(self, random_batch):
        """Higher smoothing → higher loss on low-entropy predictions."""
        logits, targets = random_batch
        loss_low = LabelSmoothingCrossEntropy(smoothing=0.0)(logits, targets)
        loss_high = LabelSmoothingCrossEntropy(smoothing=0.3)(logits, targets)
        # Not guaranteed in all edge cases but holds statistically
        assert loss_high.item() >= loss_low.item() - 1e-2

    def test_weighted_loss_runs(self, random_batch):
        """With per-class weights, loss should still be a positive scalar."""
        logits, targets = random_batch
        weights = torch.ones(NUM_CLASSES)
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1, weight=weights)
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_loss_finite(self, random_batch):
        logits, targets = random_batch
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)

    def test_gradient_flows(self, random_batch):
        logits, targets = random_batch
        logits = logits.requires_grad_(True)
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

class TestFocalLoss:
    def test_loss_is_scalar(self, random_batch):
        logits, targets = random_batch
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_loss_is_positive(self, random_batch):
        logits, targets = random_batch
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_gamma_zero_matches_cross_entropy(self, random_batch):
        """gamma=0 focal loss == standard cross-entropy."""
        logits, targets = random_batch
        focal_loss = FocalLoss(gamma=0.0)(logits, targets)
        ce_loss = F.cross_entropy(logits, targets)
        assert abs(focal_loss.item() - ce_loss.item()) < 1e-4

    def test_gamma_positive_reduces_loss_on_easy(self):
        """Focal loss down-weights well-classified examples (high confidence)."""
        # Create a near-perfect prediction scenario
        logits = torch.zeros(4, NUM_CLASSES)
        targets = torch.zeros(4, dtype=torch.long)
        logits[:, 0] = 10.0  # very high confidence on class 0

        focal_2 = FocalLoss(gamma=2.0)(logits, targets)
        focal_0 = FocalLoss(gamma=0.0)(logits, targets)
        # With high confidence predictions, gamma > 0 gives lower loss
        assert focal_2.item() <= focal_0.item() + 1e-4

    def test_loss_finite(self, random_batch):
        logits, targets = random_batch
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)

    def test_gradient_flows(self, random_batch):
        logits, targets = random_batch
        logits = logits.requires_grad_(True)
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_reduction_sum(self, random_batch):
        logits, targets = random_batch
        loss_mean = FocalLoss(gamma=2.0, reduction="mean")(logits, targets)
        loss_sum = FocalLoss(gamma=2.0, reduction="sum")(logits, targets)
        assert loss_sum.item() == pytest.approx(loss_mean.item() * BATCH_SIZE, rel=1e-5)

    def test_reduction_none_shape(self, random_batch):
        logits, targets = random_batch
        loss_none = FocalLoss(gamma=2.0, reduction="none")(logits, targets)
        assert loss_none.shape == (BATCH_SIZE,)


# ---------------------------------------------------------------------------
# build_loss
# ---------------------------------------------------------------------------

class TestBuildLoss:
    def test_returns_module(self):
        import torch.nn as nn
        cfg = {"training": {"label_smoothing": 0.1}}
        loss_fn = build_loss(cfg)
        assert isinstance(loss_fn, nn.Module)

    def test_default_smoothing(self, random_batch):
        cfg = {"training": {}}
        loss_fn = build_loss(cfg)
        logits, targets = random_batch
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_custom_smoothing(self, random_batch):
        cfg = {"training": {"label_smoothing": 0.2}}
        loss_fn = build_loss(cfg)
        logits, targets = random_batch
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)

    def test_with_class_weights(self, random_batch):
        import numpy as np
        cfg = {"training": {"label_smoothing": 0.1}}
        weights = np.ones(NUM_CLASSES, dtype=np.float32)
        device = torch.device("cpu")
        loss_fn = build_loss(cfg, class_weights=weights, device=device)
        logits, targets = random_batch
        loss = loss_fn(logits, targets)
        assert loss.item() > 0
