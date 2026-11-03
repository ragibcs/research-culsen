"""
Supervised Contrastive Learning (SupCon) module for CulSen-BN.
Includes a normalized projection head and confidence-filtered contrastive loss
that prevents noisy pseudo-labels from forming untrustworthy positive pairs.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveHead(nn.Module):
    """
    Projection network projecting sentence embeddings to lower-dimensional normalized sentiment space.
    """

    def __init__(self, in_dim: int = 768, projection_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project and L2-normalize embeddings.
        """
        proj = self.projector(x)
        return F.normalize(proj, p=2, dim=-1)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss with confidence filtering.
    Based on Khosla et al. (NeurIPS 2020), adapted for noisy pseudo-label learning.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        confidence_threshold: float = 0.65,
        filter_uncertain_pairs: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.filter_uncertain_pairs = filter_uncertain_pairs

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [batch_size, projection_dim] normalized embeddings.
            labels: [batch_size] integer class labels.
            confidences: Optional [batch_size] float confidence scores.
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)
        # Binary mask of same-class pairs: mask[i, j] = 1 if label[i] == label[j]
        label_mask = torch.eq(labels, labels.T).float().to(device)

        # Mask out self-contrast (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(label_mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0,
        )
        mask = label_mask * logits_mask

        # Confidence filtering: uncertain pseudo-labels cannot act as valid positive pairs
        if self.filter_uncertain_pairs and confidences is not None:
            conf_mask = (confidences >= self.confidence_threshold).float().view(-1, 1)
            pair_conf = torch.matmul(conf_mask, conf_mask.T)
            mask = mask * pair_conf

        # Compute cosine similarity / temperature
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature,
        )

        # Numerical stability: subtract max per row
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Denominator: sum over all contrast pairs (excluding self)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Mean of log-likelihood over positive pairs
        positive_counts = mask.sum(1)
        # Avoid division by zero for samples with no valid positive pairs
        valid_anchors = positive_counts > 0

        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_anchors] / positive_counts[valid_anchors]
        loss = -mean_log_prob_pos.mean()

        return loss
