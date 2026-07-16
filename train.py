#!/usr/bin/env python3
"""
train.py — Main training entry point.

Usage:
    # Train EfficientNet-B3 baseline
    python train.py --model efficientnet_b3

    # Train DINOv2 + LoRA
    python train.py --model dinov2_vitb14

    # Override config values via CLI
    python train.py --model efficientnet_b3 --batch_size 16 --num_epochs 30

    # Resume from checkpoint
    python train.py --model efficientnet_b3 --resume checkpoints/efficientnet/best_model.pth
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import torch

from utils import set_seed, get_device, setup_logger, load_and_merge, print_config
from datasets import build_dataloaders, print_dataset_summary, CLASS_NAMES
from models import build_model, load_checkpoint
from training import Trainer, build_loss, build_scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B3 or DINOv2+LoRA on HAM10000",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["efficientnet_b3", "dinov2_vitb14"],
        help="Model architecture to train.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--data_path", type=str, default=None, help="Path to HAM10000 dataset root.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--no_weighted_sampler", action="store_true", help="Disable WeightedRandomSampler.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging.")
    return parser.parse_args()


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    """Build AdamW with discriminative learning rates."""
    opt_cfg = cfg.get("optimizer", {})
    base_lr = opt_cfg.get("lr", 1e-4)
    weight_decay = opt_cfg.get("weight_decay", 1e-4)
    backbone_mult = opt_cfg.get("backbone_lr_multiplier", 0.1)

    param_groups = model.get_parameter_groups(
        base_lr=base_lr,
        backbone_lr_multiplier=backbone_mult,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
    return optimizer


def main() -> None:
    args = parse_args()
    t_start = time.time()

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = f"outputs/logs/train_{args.model}_{int(time.time())}.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("medical_foundation", log_file=log_file, level=log_level)
    logger.info("=" * 65)
    logger.info("Medical Foundation Models — Training")
    logger.info("Model: %s", args.model)
    logger.info("=" * 65)

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    model_cfg_map = {
        "efficientnet_b3": "configs/efficientnet_config.yaml",
        "dinov2_vitb14": "configs/dinov2_lora_config.yaml",
    }
    cfg = load_and_merge("configs/base_config.yaml", model_cfg_map[args.model])

    # CLI overrides
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_epochs:
        cfg["training"]["num_epochs"] = args.num_epochs
    if args.lr:
        cfg["optimizer"]["lr"] = args.lr
    if args.data_path:
        cfg["data"]["dataset_path"] = args.data_path

    logger.info("Active configuration:")
    print_config(cfg)

    # -----------------------------------------------------------------------
    # Reproducibility & device
    # -----------------------------------------------------------------------
    seed = args.seed or cfg["project"].get("seed", 42)
    set_seed(seed)
    device = get_device(cfg["project"].get("device", "cuda"))

    # -----------------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------------
    logger.info("Building DataLoaders …")
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_path=cfg["data"]["dataset_path"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        train_ratio=cfg["data"]["train_split"],
        val_ratio=cfg["data"]["val_split"],
        seed=seed,
        use_weighted_sampler=not args.no_weighted_sampler,
        mean=tuple(cfg["data"]["mean"]),
        std=tuple(cfg["data"]["std"]),
    )

    from datasets import HAM10000Dataset
    _tr = train_loader.dataset
    _va = val_loader.dataset
    _te = test_loader.dataset
    print_dataset_summary(_tr, _va, _te)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    logger.info("Building model: %s", args.model)
    model = build_model(cfg, device)

    # Resume from checkpoint
    if args.resume:
        ckpt = load_checkpoint(model, args.resume, device)
        logger.info("Resumed from epoch %d.", ckpt.get("epoch", 0))

    # -----------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # -----------------------------------------------------------------------
    class_weights = _tr.class_weights if not args.no_weighted_sampler else None
    criterion = build_loss(cfg, class_weights=class_weights, device=device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    experiment_name = cfg["logging"].get("experiment_name", args.model)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
        experiment_name=experiment_name,
    )

    history = trainer.fit()
    elapsed = (time.time() - t_start) / 60.0
    logger.info("Total training time: %.1f minutes.", elapsed)

    # -----------------------------------------------------------------------
    # Save training curves
    # -----------------------------------------------------------------------
    from utils import save_training_curves
    figures_dir = cfg["logging"].get("figures_dir", "figures")
    save_training_curves(history, save_dir=figures_dir, model_name=experiment_name)

    # -----------------------------------------------------------------------
    # Quick evaluation on test set
    # -----------------------------------------------------------------------
    logger.info("Loading best model for final test evaluation …")
    best_ckpt_path = Path(cfg["logging"]["checkpoint_dir"]) / experiment_name / "best_model.pth"
    if best_ckpt_path.exists():
        load_checkpoint(model, str(best_ckpt_path), device)
    else:
        logger.warning("Best checkpoint not found at %s — using last weights.", best_ckpt_path)

    from evaluation import Evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=CLASS_NAMES,
        cfg=cfg,
    )
    output_dir = Path("outputs") / experiment_name
    results = evaluator.run(output_dir=str(output_dir))

    from utils import save_roc_curves, save_confusion_matrix, save_pr_curves
    save_roc_curves(results["y_true"], results["y_prob"], CLASS_NAMES, figures_dir, experiment_name)
    save_confusion_matrix(results["confusion_matrix"], CLASS_NAMES, figures_dir, experiment_name)
    save_pr_curves(results["y_true"], results["y_prob"], CLASS_NAMES, figures_dir, experiment_name)

    logger.info("Training and evaluation complete.")
    logger.info("Test Accuracy : %.4f", results["accuracy"])
    logger.info("Test F1 Macro : %.4f", results["f1_macro"])
    logger.info("Test ROC-AUC  : %.4f", results["roc_auc_macro"])


if __name__ == "__main__":
    main()
