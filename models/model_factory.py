"""
Model factory — instantiate EfficientNet or DINOv2+LoRA from a config dict.
"""
from __future__ import annotations
import logging
from typing import Any, Dict
import torch
import torch.nn as nn
from models.efficientnet import EfficientNetClassifier
from models.dinov2_lora import DINOv2LoRAClassifier

logger = logging.getLogger(__name__)


def build_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Build and return a model from a config dictionary.

    Args:
        cfg:    Full config dict (loaded from YAML).
        device: Target device.

    Returns:
        Initialised model moved to device.
    """
    model_name = cfg["model"]["name"]
    num_classes = cfg["project"]["num_classes"]

    if model_name == "efficientnet_b3":
        model = EfficientNetClassifier(
            num_classes=num_classes,
            pretrained=cfg["model"].get("pretrained", True),
            dropout_rate=cfg["model"].get("dropout_rate", 0.3),
            freeze_backbone=cfg["model"].get("freeze_backbone", False),
        )
    elif model_name == "dinov2_vitb14":
        lora_cfg = cfg.get("lora", {})
        model = DINOv2LoRAClassifier(
            num_classes=num_classes,
            lora_r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.1),
            target_modules=lora_cfg.get("target_modules", ["query", "value"]),
            dropout_rate=cfg["model"].get("dropout_rate", 0.1),
            pretrained=cfg["model"].get("pretrained", True),
        )
    else:
        raise ValueError(f"Unknown model name: '{model_name}'. Choose 'efficientnet_b3' or 'dinov2_vitb14'.")

    model = model.to(device)
    logger.info("Model '%s' built and moved to %s.", model_name, device)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> Dict:
    """Load model weights from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Checkpoint loaded from '%s' (epoch %d).", checkpoint_path, checkpoint.get("epoch", -1))
    return checkpoint
