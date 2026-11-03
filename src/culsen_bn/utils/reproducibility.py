"""
Reproducibility utilities for CulSen-BN.
Fixes seeds across Python, NumPy, PyTorch, and CUDA.
"""

import os
import random
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Set fixed random seed across all libraries to ensure reproducible results.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(requested: Optional[str] = None) -> str:
    """
    Determine the computation device (cuda / mps / cpu).
    """
    if requested and requested.lower() == "cpu":
        return "cpu"

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"
