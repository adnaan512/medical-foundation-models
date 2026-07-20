#!/usr/bin/env python3
"""
examples/run_demo.py — Quick end-to-end demo of the medical-foundation-models pipeline.

This script demonstrates:
    1. Building EfficientNet-B3 and DINOv2+LoRA models.
    2. Running a forward pass on randomly generated "dummy" images.
    3. Printing trainable-parameter counts and basic efficiency stats.

No HAM10000 dataset is required — it runs entirely with synthetic data so you
can verify your environment is set up correctly before downloading the dataset.

Usage:
    python examples/run_demo.py
    python examples/run_demo.py --model efficientnet_b3
    python examples/run_demo.py --model dinov2_vitb14
    python examples/run_demo.py --batch_size 4 --image_size 224
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path when executed from examples/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import set_seed, get_device, setup_logger  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: build models and run a synthetic forward pass.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["efficientnet_b3", "dinov2_vitb14", "both"],
        help="Which model(s) to demo.",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Demo batch size.")
    parser.add_argument("--image_size", type=int, default=224, help="Spatial resolution.")
    parser.add_argument("--num_classes", type=int, default=7, help="Output classes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g. 'cpu', 'cuda'). Auto-detected if omitted.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper: profile a single forward pass
# ---------------------------------------------------------------------------

def _profile_forward(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    device: torch.device,
    n_runs: int = 5,
) -> dict:
    """Warm-up then time N forward passes; return latency stats."""
    model.eval()
    dummy = dummy.to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(2):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            logits = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    batch_size = dummy.size(0)
    avg_batch_ms = (sum(times) / len(times)) * 1_000
    avg_sample_ms = avg_batch_ms / batch_size

    return {
        "output_shape": tuple(logits.shape),
        "avg_batch_ms": avg_batch_ms,
        "avg_sample_ms": avg_sample_ms,
    }


# ---------------------------------------------------------------------------
# Demo runners
# ---------------------------------------------------------------------------

def demo_efficientnet(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> None:
    logger.info("=" * 60)
    logger.info("Demo: EfficientNet-B3 Baseline")
    logger.info("=" * 60)

    from models.efficientnet import EfficientNetClassifier

    model = EfficientNetClassifier(
        num_classes=args.num_classes,
        pretrained=True,
        dropout_rate=0.3,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    stats = _profile_forward(model, dummy, device)

    logger.info("  Output shape     : %s", stats["output_shape"])
    logger.info("  Total params     : %s", f"{total:,}")
    logger.info("  Trainable params : %s (%.2f%%)", f"{trainable:,}", 100.0 * trainable / max(total, 1))
    logger.info("  Avg latency/batch: %.2f ms  (%.2f ms/sample)", stats["avg_batch_ms"], stats["avg_sample_ms"])
    logger.info("  Grad-CAM target  : %s", model.get_gradcam_target_layer().__class__.__name__)
    logger.info("")


def demo_dinov2(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> None:
    logger.info("=" * 60)
    logger.info("Demo: DINOv2 ViT-B/14 + LoRA")
    logger.info("=" * 60)

    from models.dinov2_lora import DINOv2LoRAClassifier

    model = DINOv2LoRAClassifier(
        num_classes=args.num_classes,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"],
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    stats = _profile_forward(model, dummy, device)

    logger.info("  Output shape     : %s", stats["output_shape"])
    logger.info("  Total params     : %s", f"{total:,}")
    logger.info("  Trainable params : %s (%.2f%%)", f"{trainable:,}", 100.0 * trainable / max(total, 1))
    logger.info("  Avg latency/batch: %.2f ms  (%.2f ms/sample)", stats["avg_batch_ms"], stats["avg_sample_ms"])
    logger.info("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logger = setup_logger("demo", level=logging.INFO)
    set_seed(args.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device(device_str)
    logger.info("Device : %s", device)
    logger.info("Batch  : %d  |  Image size: %dx%d  |  Classes: %d",
                args.batch_size, args.image_size, args.image_size, args.num_classes)
    logger.info("")

    if args.model in ("efficientnet_b3", "both"):
        demo_efficientnet(args, device, logger)

    if args.model in ("dinov2_vitb14", "both"):
        demo_dinov2(args, device, logger)

    logger.info("Demo complete. ✓")


if __name__ == "__main__":
    main()
