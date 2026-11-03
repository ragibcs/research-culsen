"""
Uncertainty-aware learning and calibration modules for CulSen-BN.
Mitigates noisy pseudo-label supervision through sample-level reliability weighting
and temperature scaling for post-hoc probability calibration.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyWeightingModule(nn.Module):
    """
    Computes sample-level reliability weights and confidence-weighted classification loss.
    Prevents low-confidence or potentially mislabeled pseudo-labels from dominating gradient updates.
    """

    def __init__(self, clipping_floor: float = 0.10, label_smoothing: float = 0.0):
        super().__init__()
        self.clipping_floor = clipping_floor
        self.label_smoothing = label_smoothing

    def compute_weights(self, confidences: torch.Tensor) -> torch.Tensor:
        """
        Clip sample confidence scores to [clipping_floor, 1.0] to prevent zero gradients.
        """
        return torch.clamp(confidences, min=self.clipping_floor, max=1.0)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.
        """
        # Standard cross-entropy loss per sample
        loss_fn = nn.CrossEntropyLoss(reduction="none", label_smoothing=self.label_smoothing)
        per_sample_loss = loss_fn(logits, labels)

        if confidences is not None:
            weights = self.compute_weights(confidences)
            # Weighted average
            return (per_sample_loss * weights).sum() / (weights.sum() + 1e-8)
        else:
            return per_sample_loss.mean()


class TemperatureScaler(nn.Module):
    """
    Temperature scaling for post-hoc confidence calibration.
    Calibrates model probabilities without altering classification accuracy.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scale logits by temperature parameter.
        """
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50) -> float:
        """
        Optimize temperature parameter on validation set to minimize NLL.
        """
        import torch.optim as optim

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return float(self.temperature.item())
