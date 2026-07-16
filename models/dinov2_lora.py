"""
DINOv2 Vision Transformer with LoRA (Low-Rank Adaptation).

Architecture:
    - DINOv2 ViT-B/14 backbone (frozen)
    - LoRA adapters injected into Q and V projection matrices
    - [CLS] token pooling
    - Dropout + Linear classification head (7 classes)

References:
    Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision.
    Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel

logger = logging.getLogger(__name__)
DINOV2_MODEL_ID = "facebook/dinov2-base"


class DINOv2LoRAClassifier(nn.Module):
    """DINOv2 ViT-B/14 with LoRA adapters for PEFT skin lesion classification."""

    def __init__(
        self,
        num_classes: int = 7,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        dropout_rate: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if target_modules is None:
            target_modules = ["query", "value"]

        self.num_classes = num_classes
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.feature_dim = 768

        logger.info("Loading DINOv2 backbone: %s", DINOV2_MODEL_ID)
        backbone = AutoModel.from_pretrained(DINOV2_MODEL_ID)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.backbone = get_peft_model(backbone, lora_config)
        logger.info("LoRA adapters injected into: %s", target_modules)
        self.backbone.print_trainable_parameters()

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, num_classes),
        )
        self._log_param_counts()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        return outputs.last_hidden_state[:, 0, :]

    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = self.backbone(pixel_values=x, output_attentions=True)
        return list(outputs.attentions)

    def get_parameter_groups(self, base_lr: float = 5e-4, weight_decay: float = 1e-4, **kwargs) -> List[Dict]:
        trainable = [p for p in self.parameters() if p.requires_grad]
        return [{"params": trainable, "lr": base_lr, "weight_decay": weight_decay, "name": "lora_and_head"}]

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def _log_param_counts(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "DINOv2+LoRA — Total: %s | Trainable: %s (%.2f%%) | Frozen: %s",
            f"{total:,}", f"{trainable:,}", 100.0 * trainable / max(total, 1), f"{total - trainable:,}",
        )

    def save_lora_weights(self, save_path: str) -> None:
        self.backbone.save_pretrained(save_path)
        logger.info("LoRA weights saved to: %s", save_path)
