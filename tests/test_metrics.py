"""
tests/test_metrics.py — Unit tests for evaluation metric utilities.

Run with:
    pytest tests/test_metrics.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_classification_metrics, compute_efficiency_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 7
CLASS_NAMES = [
    "Actinic keratoses", "Basal cell carcinoma", "Benign keratosis",
    "Dermatofibroma", "Melanoma", "Melanocytic nevi", "Vascular lesions",
]


def _perfect_predictions(n: int = 100):
    """Generate perfect (y_true == y_pred) arrays for easy assertions."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, NUM_CLASSES, size=n)
    y_pred = y_true.copy()
    y_prob = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    y_prob[np.arange(n), y_true] = 1.0
    return y_true, y_pred, y_prob


def _random_predictions(n: int = 200, seed: int = 0):
    """Generate random (noisy) predictions."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, NUM_CLASSES, size=n)
    y_pred = rng.integers(0, NUM_CLASSES, size=n)
    raw = rng.dirichlet(np.ones(NUM_CLASSES), size=n).astype(np.float32)
    return y_true, y_pred, raw


# ---------------------------------------------------------------------------
# compute_classification_metrics
# ---------------------------------------------------------------------------

class TestComputeClassificationMetrics:
    def test_perfect_accuracy(self):
        y_true, y_pred, y_prob = _perfect_predictions()
        results = compute_classification_metrics(y_true, y_pred, y_prob,
                                                class_names=CLASS_NAMES,
                                                num_classes=NUM_CLASSES)
        assert results["accuracy"] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_f1(self):
        y_true, y_pred, y_prob = _perfect_predictions()
        results = compute_classification_metrics(y_true, y_pred, y_prob,
                                                class_names=CLASS_NAMES,
                                                num_classes=NUM_CLASSES)
        assert results["f1_macro"] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_roc_auc(self):
        y_true, y_pred, y_prob = _perfect_predictions()
        results = compute_classification_metrics(y_true, y_pred, y_prob,
                                                class_names=CLASS_NAMES,
                                                num_classes=NUM_CLASSES)
        assert results["roc_auc_macro"] == pytest.approx(1.0, abs=1e-6)

    def test_metric_keys_present(self):
        y_true, y_pred, y_prob = _random_predictions()
        results = compute_classification_metrics(y_true, y_pred, y_prob,
                                                class_names=CLASS_NAMES,
                                                num_classes=NUM_CLASSES)
        required_keys = {
            "accuracy", "precision_macro", "precision_weighted",
            "recall_macro", "recall_weighted", "f1_macro", "f1_weighted",
            "roc_auc_macro", "roc_auc_weighted", "confusion_matrix",
            "classification_report", "per_class_precision",
            "per_class_recall", "per_class_f1", "per_class_ap",
            "y_true", "y_pred", "y_prob",
        }
        assert required_keys.issubset(set(results.keys()))

    def test_accuracy_in_range(self):
        y_true, y_pred, y_prob = _random_predictions()
        results = compute_classification_metrics(y_true, y_pred, y_prob,
                                                class_names=CLASS_NAMES,
                                                num_classes=NUM_CLASSES)
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_f1_in_range(self):
        y_true, y_pred, y_prob = _random_predictions()
        results = compute_classification_metrics(y_true, y_pred, y_prob,
                                                class_names=CLASS_NAMES,
                                                num_classes=NUM_CLASSES)
        assert 0.0 <= results["f1_macro"] <= 1.0

    def test_roc_auc_in_range(self):
        y_true, y_pred, y_prob = _random_predictions()
        results = compute_classification_metrics(y_true, y_pred, y_prob,
                                                class_names=CLASS_NAMES,
                                                num_classes=NUM_CLASSES)
        assert 0.0 <= results["roc_auc_macro"] <= 1.0

    def test_confusion_matrix_shape(self):
        y_true, y_pred, y_prob = _random_predictions()
        results = compute_classification_metrics(y_true, y_pred, y_prob,
                                                class_names=CLASS_NAMES,
                                                num_classes=NUM_CLASSES)
        cm = results["confusion_matrix"]
        assert cm.shape == (NUM_CLASSES, NUM_CLASSES)

    def test_confusion_matrix_sum_equals_n(self):
        n = 200
        y_true, y_pred, y_prob = _random_predictions(n=n)
        results = compute_classification_metrics(y_true, y_pred, y_prob,
                                                class_names=CLASS_NAMES,
                                                num_classes=NUM_CLASSES)
        assert results["confusion_matrix"].sum() == n

    def test_per_class_lengths(self):
        y_true, y_pred, y_prob = _random_predictions()
        results = compute_classification_metrics(y_true, y_pred, y_prob,
                                                class_names=CLASS_NAMES,
                                                num_classes=NUM_CLASSES)
        assert len(results["per_class_precision"]) == NUM_CLASSES
        assert len(results["per_class_recall"]) == NUM_CLASSES
        assert len(results["per_class_f1"]) == NUM_CLASSES
        assert len(results["per_class_ap"]) == NUM_CLASSES

    def test_y_arrays_preserved(self):
        y_true, y_pred, y_prob = _random_predictions(n=50)
        results = compute_classification_metrics(y_true, y_pred, y_prob,
                                                class_names=CLASS_NAMES,
                                                num_classes=NUM_CLASSES)
        np.testing.assert_array_equal(results["y_true"], y_true)
        np.testing.assert_array_equal(results["y_pred"], y_pred)
        np.testing.assert_array_almost_equal(results["y_prob"], y_prob)

    def test_without_class_names(self):
        """Passing class_names=None should not raise."""
        y_true, y_pred, y_prob = _random_predictions(n=50)
        results = compute_classification_metrics(y_true, y_pred, y_prob,
                                                class_names=None,
                                                num_classes=NUM_CLASSES)
        assert "accuracy" in results

    def test_classification_report_is_string(self):
        y_true, y_pred, y_prob = _random_predictions()
        results = compute_classification_metrics(y_true, y_pred, y_prob,
                                                class_names=CLASS_NAMES,
                                                num_classes=NUM_CLASSES)
        assert isinstance(results["classification_report"], str)
        assert len(results["classification_report"]) > 0

    def test_weighted_metrics_in_range(self):
        y_true, y_pred, y_prob = _random_predictions()
        results = compute_classification_metrics(y_true, y_pred, y_prob,
                                                class_names=CLASS_NAMES,
                                                num_classes=NUM_CLASSES)
        for key in ("precision_weighted", "recall_weighted", "f1_weighted", "roc_auc_weighted"):
            assert 0.0 <= results[key] <= 1.0, f"{key} out of [0,1]: {results[key]}"


# ---------------------------------------------------------------------------
# compute_efficiency_metrics
# ---------------------------------------------------------------------------

class TestComputeEfficiencyMetrics:
    """Use a tiny MLP on CPU so these tests are fast in CI."""

    @pytest.fixture(scope="class")
    def dummy_setup(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from models.efficientnet import EfficientNetClassifier

        model = EfficientNetClassifier(num_classes=NUM_CLASSES, pretrained=False).eval()
        device = torch.device("cpu")

        # Small dataset of (B=4) random images
        imgs = torch.randn(8, 3, 224, 224)
        labels = torch.randint(0, NUM_CLASSES, (8,))
        loader = DataLoader(TensorDataset(imgs, labels), batch_size=4)
        return model, loader, device

    def test_efficiency_keys(self, dummy_setup):
        model, loader, device = dummy_setup
        stats = compute_efficiency_metrics(model, loader, device, num_warmup_batches=1)
        required = {
            "total_params", "trainable_params", "frozen_params",
            "trainable_pct", "inference_latency_ms",
            "throughput_samples_per_sec", "gpu_peak_memory_mb",
        }
        assert required.issubset(set(stats.keys()))

    def test_param_counts_consistent(self, dummy_setup):
        model, loader, device = dummy_setup
        stats = compute_efficiency_metrics(model, loader, device, num_warmup_batches=1)
        assert stats["total_params"] == stats["trainable_params"] + stats["frozen_params"]

    def test_trainable_pct_in_range(self, dummy_setup):
        model, loader, device = dummy_setup
        stats = compute_efficiency_metrics(model, loader, device, num_warmup_batches=1)
        assert 0.0 <= stats["trainable_pct"] <= 100.0

    def test_latency_positive(self, dummy_setup):
        model, loader, device = dummy_setup
        stats = compute_efficiency_metrics(model, loader, device, num_warmup_batches=1)
        assert stats["inference_latency_ms"] > 0

    def test_throughput_positive(self, dummy_setup):
        model, loader, device = dummy_setup
        stats = compute_efficiency_metrics(model, loader, device, num_warmup_batches=1)
        assert stats["throughput_samples_per_sec"] > 0

    def test_gpu_memory_zero_on_cpu(self, dummy_setup):
        """GPU memory should be 0 when running on CPU."""
        model, loader, device = dummy_setup
        stats = compute_efficiency_metrics(model, loader, device, num_warmup_batches=1)
        assert stats["gpu_peak_memory_mb"] == 0.0
