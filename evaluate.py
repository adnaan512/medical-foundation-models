#!/usr/bin/env python3
"""
evaluate.py — Evaluate a trained model and generate all figures.

Usage:
    python evaluate.py --model efficientnet_b3 \\
                       --checkpoint checkpoints/efficientnet/best_model.pth

    python evaluate.py --model dinov2_vitb14 \\
                       --checkpoint checkpoints/dinov2_lora/best_model.pth \\
                       --save_explainability
"""
import argparse
import logging
from pathlib import Path

import torch

from utils import set_seed, get_device, setup_logger, load_and_merge
from datasets import build_dataloaders, CLASS_NAMES, CLASS_TO_IDX
from models import build_model, load_checkpoint
from evaluation import Evaluator
from utils import save_roc_curves, save_confusion_matrix, save_pr_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on HAM10000 test set.")
    parser.add_argument("--model", type=str, required=True, choices=["efficientnet_b3", "dinov2_vitb14"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint.")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_explainability", action="store_true",
                        help="Generate and save Grad-CAM / Attention Rollout figures.")
    parser.add_argument("--n_explain_samples", type=int, default=16,
                        help="Number of test samples to visualise.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def generate_explainability(model, test_loader, device, args, cfg, figures_dir):
    """Generate and save Grad-CAM or Attention Rollout figures."""
    import numpy as np
    from datasets.transforms import denormalise

    model.eval()
    # Collect a small batch for visualisation
    images_list, labels_list, preds_list = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs_dev = imgs.to(device)
            logits = model(imgs_dev)
            preds = logits.argmax(dim=1)
            images_list.append(imgs)
            labels_list.extend(labels.tolist())
            preds_list.extend(preds.cpu().tolist())
            if len(labels_list) >= args.n_explain_samples:
                break

    images = torch.cat(images_list, dim=0)[:args.n_explain_samples]
    y_true = labels_list[:args.n_explain_samples]
    y_pred = preds_list[:args.n_explain_samples]

    if args.model == "efficientnet_b3":
        from explainability import GradCAM, save_gradcam_figure
        gradcam = GradCAM(model, target_layer=model.get_gradcam_target_layer())
        heatmaps = gradcam.generate_batch(images.to(device))
        gradcam.remove_hooks()
        save_gradcam_figure(
            images=images, heatmaps=heatmaps,
            y_true=y_true, y_pred=y_pred,
            class_names=CLASS_NAMES,
            save_path=str(Path(figures_dir) / "gradcam_samples.png"),
            title="Grad-CAM — EfficientNet-B3",
        )
    else:
        from explainability import AttentionRollout, save_attention_rollout_figure
        rollout = AttentionRollout(model, discard_ratio=0.9, head_fusion="mean")
        heatmaps = rollout.generate_batch(images.to(device), patch_size=14)
        save_attention_rollout_figure(
            images=images, heatmaps=heatmaps,
            y_true=y_true, y_pred=y_pred,
            class_names=CLASS_NAMES,
            save_path=str(Path(figures_dir) / "attention_rollout_samples.png"),
            title="Attention Rollout — DINOv2+LoRA",
        )

    logging.getLogger(__name__).info("Explainability figures saved to %s", figures_dir)


def main() -> None:
    args = parse_args()
    setup_logger("medical_foundation")
    logger = logging.getLogger("medical_foundation")

    model_cfg_map = {
        "efficientnet_b3": "configs/efficientnet_config.yaml",
        "dinov2_vitb14": "configs/dinov2_lora_config.yaml",
    }
    cfg = load_and_merge("configs/base_config.yaml", model_cfg_map[args.model])
    if args.data_path:
        cfg["data"]["dataset_path"] = args.data_path
    cfg["training"]["batch_size"] = args.batch_size

    set_seed(args.seed)
    device = get_device(cfg["project"].get("device", "cuda"))

    _, _, test_loader = build_dataloaders(
        dataset_path=cfg["data"]["dataset_path"],
        image_size=cfg["data"]["image_size"],
        batch_size=args.batch_size,
        num_workers=cfg["data"]["num_workers"],
        seed=args.seed,
    )

    model = build_model(cfg, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    experiment_name = cfg["logging"].get("experiment_name", args.model)
    figures_dir = str(Path("figures") / experiment_name)
    output_dir = str(Path("outputs") / experiment_name)

    evaluator = Evaluator(model=model, test_loader=test_loader, device=device,
                          class_names=CLASS_NAMES, cfg=cfg)
    results = evaluator.run(output_dir=output_dir)

    save_roc_curves(results["y_true"], results["y_prob"], CLASS_NAMES, figures_dir, experiment_name)
    save_confusion_matrix(results["confusion_matrix"], CLASS_NAMES, figures_dir, experiment_name)
    save_pr_curves(results["y_true"], results["y_prob"], CLASS_NAMES, figures_dir, experiment_name)

    if args.save_explainability:
        generate_explainability(model, test_loader, device, args, cfg, figures_dir)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
