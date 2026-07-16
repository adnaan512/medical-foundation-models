"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for EfficientNet.

Produces spatial heatmaps highlighting which image regions most influenced
the model's classification decision.

Reference:
    Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization. ICCV. https://arxiv.org/abs/1610.02391
"""
from __future__ import annotations
import logging
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM implementation compatible with EfficientNet-B3 (timm).

    Usage:
        gradcam = GradCAM(model, target_layer=model.get_gradcam_target_layer())
        heatmap = gradcam(image_tensor, class_idx=None)  # None → predicted class
        gradcam.remove_hooks()

    Args:
        model:        EfficientNetClassifier instance.
        target_layer: Conv layer to hook (typically last conv block).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: List = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attach forward and backward hooks to the target layer."""
        def save_activation(module, input, output):
            self._activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(save_activation)
        h2 = self.target_layer.register_full_backward_hook(save_gradient)
        self._hooks = [h1, h2]

    def remove_hooks(self) -> None:
        """Remove registered hooks (call after done to free memory)."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __call__(
        self,
        image: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given image.

        Args:
            image:     Input tensor (1, 3, H, W) on same device as model.
            class_idx: Target class. If None, uses the predicted class.

        Returns:
            Normalised heatmap as float32 np.ndarray of shape (H, W) in [0, 1].
        """
        self.model.eval()
        image = image.requires_grad_(True)

        # Forward pass
        logits = self.model(image)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for the target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        # Compute Grad-CAM: global-average-pool gradients → weight feature maps
        gradients = self._gradients   # (1, C, H', W')
        activations = self._activations  # (1, C, H', W')

        weights = gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Upsample to original image size
        h, w = image.shape[2], image.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.astype(np.float32)

    def generate_batch(
        self,
        images: torch.Tensor,
        class_indices: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """
        Generate Grad-CAM heatmaps for a batch of images.

        Args:
            images:        Batch tensor (B, 3, H, W).
            class_indices: Per-image target classes. None → use predicted.

        Returns:
            List of B heatmap arrays, each (H, W) float32 in [0, 1].
        """
        heatmaps = []
        for i in range(images.size(0)):
            img = images[i:i + 1]
            cls = class_indices[i] if class_indices is not None else None
            heatmaps.append(self(img, class_idx=cls))
        return heatmaps
