"""
Model checkpointing and artifact saving utilities.
"""

from pathlib import Path
from typing import Any, Dict, Optional


def save_checkpoint(
    state_dict: Dict[str, Any],
    metadata: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "best_model.pt",
) -> Path:
    """
    Save model weights and training metadata.
    """
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    try:
        import torch
        payload = {
            "state_dict": state_dict,
            "metadata": metadata,
        }
        torch.save(payload, str(out_path))
    except ImportError:
        import pickle
        payload = {
            "state_dict": state_dict,
            "metadata": metadata,
        }
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)

    return out_path


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load model weights and metadata.
    """
    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    try:
        import torch
        return torch.load(str(p), map_location="cpu")
    except ImportError:
        import pickle
        with open(p, "rb") as f:
            return pickle.load(f)
