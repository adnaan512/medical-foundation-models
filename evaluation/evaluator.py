"""
Model evaluator: runs inference on test set and computes all metrics.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from evaluation.metrics import compute_classification_metrics, compute_efficiency_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluate a trained model on the test split.

    Args:
        model:       Trained model in eval mode.
        test_loader: Test DataLoader.
        device:      Target device.
        class_names: Human-readable class names for reporting.
        cfg:         Config dict (used for mixed-precision flag).
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        class_names: Optional[List[str]] = None,
        cfg: Optional[Dict] = None,
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names or [str(i) for i in range(7)]
        self.cfg = cfg or {}
        self.mixed_precision = (
            self.cfg.get("training", {}).get("mixed_precision", True)
            and device.type == "cuda"
        )

    def run(self, output_dir: Optional[str] = None) -> Dict:
        """
        Run full evaluation and return metrics dict.

        Args:
            output_dir: If provided, save metrics JSON and classification report.

        Returns:
            Dict with all classification and efficiency metrics.
        """
        self.model.eval()
        y_true, y_pred, y_prob = self._collect_predictions()

        cls_metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=self.class_names,
            num_classes=len(self.class_names),
        )

        eff_metrics = compute_efficiency_metrics(
            model=self.model,
            dataloader=self.test_loader,
            device=self.device,
        )

        results = {**cls_metrics, **eff_metrics}
        self._log_results(cls_metrics, eff_metrics)

        if output_dir:
            self._save_results(results, output_dir)

        return results

    @torch.no_grad()
    def _collect_predictions(self):
        """Run inference over the full test set and collect predictions."""
        all_probs = []
        all_preds = []
        all_labels = []

        for images, labels in self.test_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.mixed_precision):
                logits = self.model(images)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)
        return y_true, y_pred, y_prob

    def _log_results(self, cls_metrics: Dict, eff_metrics: Dict) -> None:
        logger.info("=" * 60)
        logger.info("TEST SET EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info("Accuracy         : %.4f", cls_metrics["accuracy"])
        logger.info("Precision (macro): %.4f", cls_metrics["precision_macro"])
        logger.info("Recall (macro)   : %.4f", cls_metrics["recall_macro"])
        logger.info("F1 (macro)       : %.4f", cls_metrics["f1_macro"])
        logger.info("ROC-AUC (macro)  : %.4f", cls_metrics["roc_auc_macro"])
        logger.info("-" * 60)
        logger.info("Total params     : %s", f"{eff_metrics['total_params']:,}")
        logger.info("Trainable params : %s (%.2f%%)", f"{eff_metrics['trainable_params']:,}", eff_metrics['trainable_pct'])
        logger.info("Inference latency: %.2f ms/sample", eff_metrics["inference_latency_ms"])
        logger.info("Throughput       : %.1f samples/sec", eff_metrics["throughput_samples_per_sec"])
        logger.info("GPU peak memory  : %.1f MB", eff_metrics["gpu_peak_memory_mb"])
        logger.info("=" * 60)
        logger.info("\n%s", cls_metrics["classification_report"])

    def _save_results(self, results: Dict, output_dir: str) -> None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Serialisable subset (exclude numpy arrays)
        serialisable = {
            k: v for k, v in results.items()
            if not isinstance(v, np.ndarray) and k not in ("y_true", "y_pred", "y_prob")
        }
        serialisable["confusion_matrix"] = results["confusion_matrix"].tolist()

        with open(out_path / "metrics.json", "w") as f:
            json.dump(serialisable, f, indent=2)

        with open(out_path / "classification_report.txt", "w") as f:
            f.write(results["classification_report"])

        # Save raw predictions for later analysis
        np.save(out_path / "y_true.npy", results["y_true"])
        np.save(out_path / "y_pred.npy", results["y_pred"])
        np.save(out_path / "y_prob.npy", results["y_prob"])

        logger.info("Results saved to %s", out_path)
