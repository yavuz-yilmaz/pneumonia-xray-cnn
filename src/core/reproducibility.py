"""Utilities for deterministic and reproducible experiment behavior."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Set random seeds for Python, NumPy, and PyTorch when available.

    Args:
        seed: Non-negative integer seed value.
        deterministic_torch: Whether to request deterministic PyTorch algorithms where possible.

    Raises:
        ValueError: If `seed` is negative.
    """
    if seed < 0:
        message = f"Seed must be a non-negative integer, received {seed}."
        raise ValueError(message)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
