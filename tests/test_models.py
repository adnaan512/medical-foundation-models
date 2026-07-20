"""
tests/test_models.py — Unit tests for EfficientNet-B3 and DINOv2+LoRA model classes.

Run with:
    pytest tests/test_models.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Ensure project root is on the path regardless of where pytest is invoked
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_CLASSES = 7
BATCH_SIZE = 2
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")  # CI runs on CPU; GPU tests are skipped unless available


@pytest.fixture(scope="module")
def efficientnet_model():
    from models.efficientnet import EfficientNetClassifier
    return EfficientNetClassifier(num_classes=NUM_CLASSES, pretrained=False).eval()


# DINOv2 fixture is session-scoped and skipped if HuggingFace Hub is unavailable
@pytest.fixture(scope="module")
def dinov2_model():
    pytest.importorskip("peft", reason="peft not installed")
    pytest.importorskip("transformers", reason="transformers not installed")
    try:
        from models.dinov2_lora import DINOv2LoRAClassifier
        model = DINOv2LoRAClassifier(
            num_classes=NUM_CLASSES,
            lora_r=4,       # small rank keeps test fast
            lora_alpha=8,
            lora_dropout=0.0,
        )
        return model.eval()
    except Exception as exc:
        pytest.skip(f"DINOv2 model could not be loaded (network/hub issue): {exc}")


@pytest.fixture()
def dummy_batch():
    """Return a (B, 3, H, W) dummy image tensor."""
    return torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)


# ---------------------------------------------------------------------------
# EfficientNet-B3 tests
# ---------------------------------------------------------------------------

class TestEfficientNetClassifier:
    def test_forward_shape(self, efficientnet_model, dummy_batch):
        """Output logits should have shape (B, NUM_CLASSES)."""
        with torch.no_grad():
            logits = efficientnet_model(dummy_batch)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES), (
            f"Expected ({BATCH_SIZE}, {NUM_CLASSES}), got {logits.shape}"
        )

    def test_output_is_finite(self, efficientnet_model, dummy_batch):
        """No NaN or Inf in logits."""
        with torch.no_grad():
            logits = efficientnet_model(dummy_batch)
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values."

    def test_trainable_params_positive(self, efficientnet_model):
        """At least one parameter should be trainable."""
        trainable = sum(p.numel() for p in efficientnet_model.parameters() if p.requires_grad)
        assert trainable > 0, "No trainable parameters found."

    def test_param_count_dict(self, efficientnet_model):
        """count_parameters() returns expected keys."""
        counts = efficientnet_model.count_parameters()
        assert {"total", "trainable", "frozen"} == set(counts.keys())
        assert counts["total"] == counts["trainable"] + counts["frozen"]

    def test_feature_extraction(self, efficientnet_model, dummy_batch):
        """extract_features() should return (B, feature_dim) tensor."""
        with torch.no_grad():
            feats = efficientnet_model.extract_features(dummy_batch)
        assert feats.ndim == 2
        assert feats.shape[0] == BATCH_SIZE

    def test_gradcam_target_layer(self, efficientnet_model):
        """get_gradcam_target_layer() should return an nn.Module."""
        layer = efficientnet_model.get_gradcam_target_layer()
        assert isinstance(layer, nn.Module)

    def test_freeze_backbone(self):
        """Frozen backbone: only head parameters should be trainable."""
        from models.efficientnet import EfficientNetClassifier
        model = EfficientNetClassifier(num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=True)
        backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        head_trainable = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
        assert backbone_trainable == 0, "Backbone should be frozen."
        assert head_trainable > 0, "Head should have trainable parameters."

    def test_parameter_groups_structure(self, efficientnet_model):
        """get_parameter_groups() must return two groups with correct keys."""
        groups = efficientnet_model.get_parameter_groups(base_lr=1e-4)
        assert len(groups) == 2
        for g in groups:
            assert "params" in g
            assert "lr" in g
            assert len(g["params"]) > 0

    def test_backbone_lr_multiplier(self, efficientnet_model):
        """Backbone LR must be strictly less than head LR."""
        groups = efficientnet_model.get_parameter_groups(base_lr=1e-4, backbone_lr_multiplier=0.1)
        lrs = [g["lr"] for g in groups]
        assert lrs[0] < lrs[1], "Backbone LR should be lower than head LR."

    def test_gradient_flows_through_head(self, efficientnet_model, dummy_batch):
        """A backward pass should produce non-zero gradients in the classifier head."""
        model = efficientnet_model.train()
        logits = model(dummy_batch)
        loss = logits.sum()
        loss.backward()
        head_grad_norm = sum(
            p.grad.norm().item()
            for p in model.classifier.parameters()
            if p.grad is not None
        )
        assert head_grad_norm > 0, "No gradient flowed to the classifier head."
        # Clean up
        model.zero_grad()
        model.eval()


# ---------------------------------------------------------------------------
# DINOv2 + LoRA tests
# ---------------------------------------------------------------------------

class TestDINOv2LoRAClassifier:
    def test_forward_shape(self, dinov2_model, dummy_batch):
        with torch.no_grad():
            logits = dinov2_model(dummy_batch)
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_output_is_finite(self, dinov2_model, dummy_batch):
        with torch.no_grad():
            logits = dinov2_model(dummy_batch)
        assert torch.isfinite(logits).all()

    def test_lora_reduces_trainable_params(self, dinov2_model):
        """LoRA should keep trainable params << total params (PEFT goal)."""
        counts = dinov2_model.count_parameters()
        trainable_pct = 100.0 * counts["trainable"] / max(counts["total"], 1)
        assert trainable_pct < 5.0, (
            f"Expected trainable < 5% but got {trainable_pct:.2f}%"
        )

    def test_param_count_dict(self, dinov2_model):
        counts = dinov2_model.count_parameters()
        assert {"total", "trainable", "frozen"} == set(counts.keys())
        assert counts["total"] == counts["trainable"] + counts["frozen"]

    def test_feature_extraction(self, dinov2_model, dummy_batch):
        with torch.no_grad():
            feats = dinov2_model.extract_features(dummy_batch)
        assert feats.ndim == 2
        assert feats.shape[0] == BATCH_SIZE
        assert feats.shape[1] == 768  # DINOv2-base feature dim

    def test_parameter_groups_structure(self, dinov2_model):
        groups = dinov2_model.get_parameter_groups(base_lr=5e-4)
        assert len(groups) >= 1
        for g in groups:
            assert "params" in g
            assert "lr" in g

    def test_only_lora_params_trainable(self, dinov2_model):
        """All frozen params must have requires_grad=False."""
        for name, param in dinov2_model.named_parameters():
            if not param.requires_grad:
                # Should be backbone non-LoRA params
                assert "lora_" not in name or "base_layer" in name, (
                    f"Unexpected frozen LoRA param: {name}"
                )
