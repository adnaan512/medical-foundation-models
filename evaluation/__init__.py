"""Evaluation package: metrics computation and model evaluator."""
from evaluation.metrics import compute_classification_metrics, compute_efficiency_metrics
from evaluation.evaluator import Evaluator
__all__ = ["compute_classification_metrics", "compute_efficiency_metrics", "Evaluator"]
