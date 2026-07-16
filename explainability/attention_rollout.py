"""
Attention Rollout for DINOv2 Vision Transformer.

Recursively multiplies attention matrices across all transformer layers
to account for residual connections and produce a single spatial
attention map from the [CLS] token to each image patch.

Reference:
    Abnar & Zuidema (2020). Quantifying Attention Flow in Transformers.
    https://arxiv.org/abs/2005.00928
"""
from __future__ import annotations
import logging
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AttentionRollout:
    """
    Attention Rollout visualisation for DINOv2 ViT.

    Args:
        model:       DINOv2LoRAClassifier instance.
        discard_ratio: Fraction of lowest-attention heads to zero out
                       before rollout (reduces noise).
        head_fusion:  How to aggregate multi-head attention maps.
                      'mean' | 'max' | 'min'.
    """

    def __init__(
        self,
        model,
        discard_ratio: float = 0.9,
        head_fusion: str = "mean",
    ):
        self.model = model
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion

    @torch.no_grad()
    def __call__(
        self,
        image: torch.Tensor,
        patch_size: int = 14,
    ) -> np.ndarray:
        """
        Generate an Attention Rollout heatmap for a single image.

        Args:
            image:      Input tensor of shape (1, 3, H, W).
            patch_size: DINOv2 patch size in pixels (14 for ViT-B/14).

        Returns:
            Heatmap np.ndarray of shape (H, W) normalised to [0, 1].
        """
        self.model.eval()

        # Extract per-layer attention maps: list of (1, heads, seq, seq)
        attention_maps = self.model.get_attention_maps(image)

        rollout = self._compute_rollout(attention_maps)

        # rollout[0, 1:] → patch attentions from CLS token (skip CLS itself)
        h = w = image.shape[2] // patch_size
        patch_attn = rollout[0, 1:].reshape(h, w).cpu().numpy()

        # Upsample to original image resolution
        patch_attn_t = torch.tensor(patch_attn).unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(
            patch_attn_t,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()

        # Normalise to [0, 1]
        vmin, vmax = upsampled.min(), upsampled.max()
        if vmax - vmin > 1e-8:
            upsampled = (upsampled - vmin) / (vmax - vmin)

        return upsampled.astype(np.float32)

    def _compute_rollout(self, attention_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply Attention Rollout across all transformer layers.

        Steps for each layer:
          1. Fuse multi-head attention via mean/max/min.
          2. Discard lowest-attention heads (noise reduction).
          3. Add residual identity (accounts for skip connections).
          4. Normalise rows to sum to 1.
          5. Matrix-multiply with the accumulated rollout.

        Args:
            attention_maps: List of (B, heads, seq_len, seq_len) tensors.

        Returns:
            Rollout matrix of shape (B, seq_len, seq_len).
        """
        batch_size = attention_maps[0].shape[0]
        seq_len = attention_maps[0].shape[-1]
        rollout = torch.eye(seq_len, device=attention_maps[0].device).unsqueeze(0)
        rollout = rollout.expand(batch_size, -1, -1).clone()

        for attn in attention_maps:
            # attn: (B, heads, seq, seq)
            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=1)
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=1).values
            else:
                attn_fused = attn.min(dim=1).values

            # Discard low-attention tokens
            flat = attn_fused.view(batch_size, -1)
            threshold_idx = int(flat.shape[1] * self.discard_ratio)
            if threshold_idx < flat.shape[1]:
                threshold_val = flat.kthvalue(threshold_idx + 1, dim=1).values
                mask = attn_fused >= threshold_val.view(batch_size, 1, 1)
                attn_fused = attn_fused * mask.float()

            # Add identity (residual connection) and row-normalise
            attn_fused = attn_fused + torch.eye(seq_len, device=attn_fused.device).unsqueeze(0)
            row_sums = attn_fused.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            attn_fused = attn_fused / row_sums

            rollout = torch.bmm(attn_fused, rollout)

        return rollout

    def generate_batch(
        self,
        images: torch.Tensor,
        patch_size: int = 14,
    ) -> List[np.ndarray]:
        """Generate Attention Rollout heatmaps for a batch."""
        return [self(images[i:i+1], patch_size=patch_size) for i in range(images.size(0))]
