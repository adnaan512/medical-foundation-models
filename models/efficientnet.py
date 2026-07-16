"""
EfficientNet-B3 Transfer Learning Baseline.

Architecture:
    - EfficientNet-B3 backbone (pretrained on ImageNet)
    - Global average pooling (built-in)
    - Dropout regularisation
    - Linear classification head (7 classes)

Training strategy:
    - Backbone: fine-tuned with a lower learning rate (discriminative LR)
    - Head: trained with the base learning rate
    - Mixed-precision training support

Reference:
    Tan, M. & Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling for
    CNNs. ICML. https://arxiv.org/abs/1905.11946
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B3 fine-tuned for 7-class skin lesion classification.

    Args:
        num_classes:    Number of output classes.
        pretrained:     Load ImageNet-pretrained weights.
        dropout_rate:   Dropout probability before the classifier head.
        freeze_backbone: If True, freeze all backbone parameters (only
                        trains the head — useful for quick baselines).
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------------
        # Load EfficientNet-B3 from timm (SOTA pretrained weights)
        # num_classes=0 removes timm's built-in head so we can add ours
        # ---------------------------------------------------------------
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,          # Returns pooled feature vector
            global_pool="avg",
        )

        # Retrieve the feature dimension from timm model config
        self.feature_dim: int = self.backbone.num_features   # 1536 for B3

        # ---------------------------------------------------------------
        # Custom classification head
        # ---------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.feature_dim, num_classes),
        )

        # ---------------------------------------------------------------
        # Optionally freeze backbone for head-only baseline
        # ---------------------------------------------------------------
        if freeze_backbone:
            self._freeze_backbone()
            logger.info("EfficientNet-B3: backbone frozen (head-only training).")
        else:
            logger.info(
                "EfficientNet-B3: full fine-tuning enabled (discriminative LR)."
            )

        self._log_param_counts()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        features = self.backbone(x)      # (B, feature_dim)
        logits = self.classifier(features)  # (B, num_classes)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled features without the classification head."""
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Parameter groups (discriminative learning rates)
    # ------------------------------------------------------------------

    def get_parameter_groups(
        self,
        base_lr: float = 1e-4,
        backbone_lr_multiplier: float = 0.1,
        weight_decay: float = 1e-4,
    ) -> List[Dict]:
        """
        Return parameter groups with discriminative learning rates.

        The pretrained backbone uses a lower LR (backbone_lr_multiplier × base_lr)
        to preserve learned representations, while the new head uses the full LR.

        Args:
            base_lr:               Learning rate for the classification head.
            backbone_lr_multiplier: LR scale factor for backbone layers.
            weight_decay:          L2 weight decay (not applied to biases/norms).

        Returns:
            List of dicts suitable for torch.optim constructors.
        """
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.classifier.parameters())

        return [
            {
                "params": backbone_params,
                "lr": base_lr * backbone_lr_multiplier,
                "weight_decay": weight_decay,
                "name": "backbone",
            },
            {
                "params": head_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "classifier_head",
            },
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (e.g., for progressive unfreezing)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("EfficientNet-B3: backbone unfrozen.")

    def _log_param_counts(self) -> None:
        """Log total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "EfficientNet-B3 — Total params: %s | Trainable: %s | Frozen: %s",
            f"{total:,}",
            f"{trainable:,}",
            f"{total - trainable:,}",
        )

    def count_parameters(self) -> Dict[str, int]:
        """Return a dict of parameter counts for reporting."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    # ------------------------------------------------------------------
    # Hooks for Grad-CAM
    # ------------------------------------------------------------------

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Return the target convolutional layer for Grad-CAM visualisation.

        For EfficientNet-B3, the last convolutional block ('blocks[-1]')
        produces the highest-level spatial feature maps.
        """
        return self.backbone.blocks[-1]
