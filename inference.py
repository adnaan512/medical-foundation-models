#!/usr/bin/env python3
"""
inference.py — Run inference on a single image or a folder of images.

Usage:
    # Single image
    python inference.py --model efficientnet_b3 \\
                        --checkpoint checkpoints/efficientnet/best_model.pth \\
                        --image path/to/lesion.jpg

    # Folder of images
    python inference.py --model dinov2_vitb14 \\
                        --checkpoint checkpoints/dinov2_lora/best_model.pth \\
                        --image_dir path/to/images/ \\
                        --output_csv results.csv

    # With Test-Time Augmentation
    python inference.py --model efficientnet_b3 \\
                        --checkpoint checkpoints/efficientnet/best_model.pth \\
                        --image path/to/lesion.jpg --tta
"""
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from utils import get_device, setup_logger, load_and_merge
from models import build_model, load_checkpoint
from datasets import CLASS_NAMES, IDX_TO_CLASS
from datasets.transforms import get_inference_transforms, get_tta_transforms, denormalise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained skin lesion classifier.")
    parser.add_argument("--model", type=str, required=True, choices=["efficientnet_b3", "dinov2_vitb14"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None, help="Path to a single image.")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory of images.")
    parser.add_argument("--output_csv", type=str, default=None, help="Save predictions to CSV.")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation.")
    parser.add_argument("--tta_n", type=int, default=5, help="Number of TTA augmentations.")
    parser.add_argument("--top_k", type=int, default=3, help="Show top-k class predictions.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def predict_single(
    model: torch.nn.Module,
    image_path: str,
    transform,
    device: torch.device,
    top_k: int = 3,
) -> Dict:
    """Run inference on a single image and return top-k predictions."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)   # (1, 3, H, W)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    latency_ms = (time.perf_counter() - t0) * 1000

    probs = F.softmax(logits, dim=1).squeeze(0).cpu()
    topk_probs, topk_idx = probs.topk(min(top_k, len(CLASS_NAMES)))

    return {
        "image": str(image_path),
        "predicted_class": IDX_TO_CLASS[topk_idx[0].item()],
        "predicted_class_name": CLASS_NAMES[topk_idx[0].item()],
        "confidence": float(topk_probs[0]),
        "top_k": [
            {
                "rank": i + 1,
                "class_abbrev": IDX_TO_CLASS[idx.item()],
                "class_name": CLASS_NAMES[idx.item()],
                "probability": float(prob),
            }
            for i, (idx, prob) in enumerate(zip(topk_idx, topk_probs))
        ],
        "all_probabilities": {IDX_TO_CLASS[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "latency_ms": latency_ms,
    }


def predict_with_tta(
    model: torch.nn.Module,
    image_path: str,
    tta_transforms: List,
    device: torch.device,
    top_k: int = 3,
) -> Dict:
    """Run TTA inference: average softmax probabilities over N augmented views."""
    img = Image.open(image_path).convert("RGB")
    all_probs = []

    with torch.no_grad():
        for transform in tta_transforms:
            tensor = transform(img).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
            all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    topk_probs, topk_idx = avg_probs.topk(min(top_k, len(CLASS_NAMES)))

    return {
        "image": str(image_path),
        "predicted_class": IDX_TO_CLASS[topk_idx[0].item()],
        "predicted_class_name": CLASS_NAMES[topk_idx[0].item()],
        "confidence": float(topk_probs[0]),
        "tta_n": len(tta_transforms),
        "top_k": [
            {
                "rank": i + 1,
                "class_abbrev": IDX_TO_CLASS[idx.item()],
                "class_name": CLASS_NAMES[idx.item()],
                "probability": float(prob),
            }
            for i, (idx, prob) in enumerate(zip(topk_idx, topk_probs))
        ],
        "all_probabilities": {IDX_TO_CLASS[i]: float(avg_probs[i]) for i in range(len(CLASS_NAMES))},
    }


def main() -> None:
    args = parse_args()
    setup_logger("medical_foundation")
    logger = logging.getLogger("medical_foundation")

    model_cfg_map = {
        "efficientnet_b3": "configs/efficientnet_config.yaml",
        "dinov2_vitb14": "configs/dinov2_lora_config.yaml",
    }
    cfg = load_and_merge("configs/base_config.yaml", model_cfg_map[args.model])

    device = get_device(args.device)
    model = build_model(cfg, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    mean = tuple(cfg["data"]["mean"])
    std = tuple(cfg["data"]["std"])

    if args.tta:
        transforms = get_tta_transforms(args.image_size, mean, std, args.tta_n)
        predict_fn = lambda p: predict_with_tta(model, p, transforms, device, args.top_k)
    else:
        transform = get_inference_transforms(args.image_size, mean, std)
        predict_fn = lambda p: predict_single(model, p, transform, device, args.top_k)

    # Collect image paths
    if args.image:
        image_paths = [Path(args.image)]
    elif args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_paths = [p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in exts]
        logger.info("Found %d images in %s", len(image_paths), args.image_dir)
    else:
        logger.error("Provide --image or --image_dir.")
        sys.exit(1)

    results = []
    for path in image_paths:
        try:
            result = predict_fn(str(path))
            results.append(result)
            logger.info(
                "%s → %s (%.1f%%) [top: %s]",
                path.name,
                result["predicted_class_name"],
                result["confidence"] * 100,
                ", ".join(f"{d['class_abbrev']}:{d['probability']:.3f}" for d in result["top_k"]),
            )
        except Exception as e:
            logger.error("Failed on %s: %s", path, e)

    # Print result for single image
    if len(results) == 1:
        print("\n" + "=" * 50)
        print(json.dumps(results[0], indent=2))
        print("=" * 50)

    # Save CSV
    if args.output_csv and results:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "predicted_class", "predicted_class_name", "confidence"])
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in ["image", "predicted_class", "predicted_class_name", "confidence"]})
        logger.info("Predictions saved to %s", args.output_csv)


if __name__ == "__main__":
    main()
