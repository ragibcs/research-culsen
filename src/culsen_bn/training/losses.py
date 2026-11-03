"""
Multi-task loss formulations for CulSen-BN.
Integrates uncertainty-weighted classification, supervised contrastive learning,
domain invariance, and consistency regularization.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from ..models.uncertainty import UncertaintyWeightingModule
from ..models.contrastive import SupConLoss


class IntegratedCulSenLoss(nn.Module):
    """
    Combined loss function:
    L = L_cls + alpha * L_con + beta * L_domain + gamma * L_consistency
    """

    def __init__(
        self,
        alpha: float = 0.20,
        beta: float = 0.10,
        gamma: float = 0.0,
        clipping_floor: float = 0.10,
        label_smoothing: float = 0.0,
        contrastive_temp: float = 0.07,
        contrastive_threshold: float = 0.65,
        filter_uncertain_pairs: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.cls_criterion = UncertaintyWeightingModule(
            clipping_floor=clipping_floor,
            label_smoothing=label_smoothing,
        )
        self.con_criterion = SupConLoss(
            temperature=contrastive_temp,
            confidence_threshold=contrastive_threshold,
            filter_uncertain_pairs=filter_uncertain_pairs,
        )
        self.domain_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        sentiment_logits: torch.Tensor,
        labels: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
        projected_embeddings: Optional[torch.Tensor] = None,
        season_logits: Optional[torch.Tensor] = None,
        season_labels: Optional[torch.Tensor] = None,
        perturbed_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and component losses for logging and backpropagation.
        """
        # 1. Classification Loss (Uncertainty-Weighted)
        l_cls = self.cls_criterion(sentiment_logits, labels, confidences)
        total_loss = l_cls

        # 2. Contrastive Loss
        l_con = torch.tensor(0.0, device=sentiment_logits.device)
        if self.alpha > 0 and projected_embeddings is not None:
            l_con = self.con_criterion(projected_embeddings, labels, confidences)
            total_loss = total_loss + self.alpha * l_con

        # 3. Domain Invariance Loss
        l_domain = torch.tensor(0.0, device=sentiment_logits.device)
        if self.beta > 0 and season_logits is not None and season_labels is not None:
            l_domain = self.domain_criterion(season_logits, season_labels)
            total_loss = total_loss + self.beta * l_domain

        # 4. Consistency Regularization Loss (symmetric KL divergence)
        l_cons = torch.tensor(0.0, device=sentiment_logits.device)
        if self.gamma > 0 and perturbed_logits is not None:
            p = torch.softmax(sentiment_logits, dim=-1)
            q = torch.softmax(perturbed_logits, dim=-1)
            kl_div = torch.sum(p * torch.log((p + 1e-9) / (q + 1e-9)), dim=-1)
            l_cons = kl_div.mean()
            total_loss = total_loss + self.gamma * l_cons

        return {
            "loss": total_loss,
            "loss_cls": l_cls,
            "loss_con": l_con,
            "loss_domain": l_domain,
            "loss_cons": l_cons,
        }
