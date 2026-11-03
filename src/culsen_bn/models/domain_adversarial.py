"""
Domain-Adversarial Neural Network (DANN) module for season-invariant sentiment representations.
Uses a Gradient Reversal Layer (GRL) and season discriminator network
to prevent sentiment representations from overfitting to season-specific vocabulary or artists.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer: forward pass is an identity operation,
    backward pass scales gradients by -alpha.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    """
    Module wrapper for GradientReversalFunction with adjustable adversarial weight alpha.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class SeasonDiscriminator(nn.Module):
    """
    Auxiliary domain classifier predicting season identity from sentence representations.
    Trained adversarially via Gradient Reversal to encourage season-invariant sentiment embeddings.
    """

    def __init__(self, in_dim: int = 768, hidden_dim: int = 256, num_seasons: int = 4, alpha: float = 1.0):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=alpha)
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_seasons),
        )

    def set_alpha(self, alpha: float) -> None:
        self.grl.set_alpha(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reversed_x = self.grl(x)
        return self.network(reversed_x)
