"""Reproducibility utilities — set all random seeds deterministically."""
from __future__ import annotations
import logging
import os
import random
import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Covers: Python random, NumPy, PyTorch (CPU + CUDA), and CUDA cuDNN.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # cuDNN determinism — may slightly reduce performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Global seed set to %d.", seed)


def get_device(preferred: str = "cuda") -> torch.device:
    """
    Return the best available torch device.

    Args:
        preferred: 'cuda' | 'mps' | 'cpu'.

    Returns:
        torch.device instance.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using device: CUDA (%s)", torch.cuda.get_device_name(0))
    elif preferred == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using device: Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
    return device
