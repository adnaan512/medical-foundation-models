#!/usr/bin/env python3
"""
compare_models.py — Side-by-side efficiency and performance comparison.

Usage:
    python compare_models.py \\
        --efficientnet_ckpt checkpoints/efficientnet/best_model.pth \\
        --dinov2_ckpt checkpoints/dinov2_lora/best_model.pth
"""
import argparse
import json
import logging
from pathlib import Path

import torch

from utils import set_seed, get_device, setup_logger, load_and_merge
from datasets import build_dataloaders, CLASS_NAMES
from models import build_model, load_checkpoint
from evaluation import Evaluator
from utils.visualization import save_efficiency_comparison


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--efficientnet_ckpt", type=str, required=True)
    p.add_argument("--dinov2_ckpt", type=str, required=True)
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def evaluate_model(cfg, model_name, ckpt_path, test_loader, device):
    model = build_model(cfg, device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()
    evaluator = Evaluator(model=model, test_loader=test_loader, device=device,
                          class_names=CLASS_NAMES, cfg=cfg)
    results = evaluator.run()
    results["model_name"] = model_name
    return results


def main():
    args = parse_args()
    setup_logger("medical_foundation")
    logger = logging.getLogger("medical_foundation")
    set_seed(args.seed)
    device = get_device("cuda")

    eff_cfg = load_and_merge("configs/base_config.yaml", "configs/efficientnet_config.yaml")
    dino_cfg = load_and_merge("configs/base_config.yaml", "configs/dinov2_lora_config.yaml")
    if args.data_path:
        eff_cfg["data"]["dataset_path"] = args.data_path
        dino_cfg["data"]["dataset_path"] = args.data_path

    _, _, test_loader = build_dataloaders(
        dataset_path=eff_cfg["data"]["dataset_path"],
        image_size=224, batch_size=args.batch_size,
        num_workers=eff_cfg["data"]["num_workers"], seed=args.seed,
    )

    logger.info("Evaluating EfficientNet-B3 …")
    eff_results = evaluate_model(eff_cfg, "EfficientNet-B3", args.efficientnet_ckpt, test_loader, device)

    logger.info("Evaluating DINOv2+LoRA …")
    dino_results = evaluate_model(dino_cfg, "DINOv2+LoRA", args.dinov2_ckpt, test_loader, device)

    # Comparison table
    print("\n" + "=" * 75)
    print(f"{'Metric':<30} {'EfficientNet-B3':>20} {'DINOv2+LoRA':>20}")
    print("=" * 75)
    metrics = [
        ("Accuracy", "accuracy"), ("F1 Macro", "f1_macro"),
        ("ROC-AUC Macro", "roc_auc_macro"), ("Trainable Params (M)", None),
        ("Total Params (M)", None), ("Latency ms/sample", "inference_latency_ms"),
        ("GPU Memory MB", "gpu_peak_memory_mb"),
    ]
    for label, key in metrics:
        if key:
            ev = eff_results.get(key, 0)
            dv = dino_results.get(key, 0)
            print(f"{label:<30} {ev:>20.4f} {dv:>20.4f}")
        else:
            if "Trainable" in label:
                ev = eff_results["trainable_params"] / 1e6
                dv = dino_results["trainable_params"] / 1e6
            else:
                ev = eff_results["total_params"] / 1e6
                dv = dino_results["total_params"] / 1e6
            print(f"{label:<30} {ev:>20.2f}M {dv:>20.2f}M")
    print("=" * 75)

    # Save comparison figure
    models_data = [
        {**eff_results, "name": "EfficientNet-B3"},
        {**dino_results, "name": "DINOv2+LoRA"},
    ]
    save_efficiency_comparison(models_data, save_dir="figures")

    # Save JSON
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/comparison_results.json", "w") as f:
        out = {}
        for m in models_data:
            out[m["name"]] = {k: v for k, v in m.items()
                              if not hasattr(v, "__len__") or isinstance(v, (str, list)) and len(v) < 20}
        json.dump(out, f, indent=2, default=str)
    logger.info("Comparison results saved to outputs/comparison_results.json")


if __name__ == "__main__":
    main()
