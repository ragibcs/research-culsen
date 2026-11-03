"""
Unit tests for CulSen-BN loss functions.
Tests numerical stability, single-class batches, finite gradients, and confidence weighting.
"""

import pytest
import torch
from culsen_bn.models.uncertainty import UncertaintyWeightingModule
from culsen_bn.models.contrastive import SupConLoss
from culsen_bn.training.losses import IntegratedCulSenLoss


def test_uncertainty_weighted_loss():
    loss_module = UncertaintyWeightingModule(clipping_floor=0.10)
    logits = torch.randn(4, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    confidences = torch.tensor([0.9, 0.4, 0.95, 0.2], dtype=torch.float)

    loss = loss_module(logits, labels, confidences)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Test gradient flow
    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_supcon_loss_numerics_and_gradients():
    supcon = SupConLoss(temperature=0.07, confidence_threshold=0.65)
    raw_features = torch.randn(6, 128, requires_grad=True)
    features = torch.nn.functional.normalize(raw_features, p=2, dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    confidences = torch.tensor([0.9, 0.8, 0.85, 0.95, 0.7, 0.8], dtype=torch.float)

    loss = supcon(features, labels, confidences)
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)

    loss.backward()
    assert raw_features.grad is not None
    assert torch.all(torch.isfinite(raw_features.grad))


def test_supcon_single_sample_or_missing_positives():
    supcon = SupConLoss(temperature=0.07)
    # Only 1 sample in batch
    features = torch.nn.functional.normalize(torch.randn(1, 128, requires_grad=True), p=2, dim=-1)
    labels = torch.tensor([0], dtype=torch.long)
    loss = supcon(features, labels)
    assert loss.item() == 0.0


def test_integrated_loss():
    criterion = IntegratedCulSenLoss(alpha=0.2, beta=0.1, gamma=0.0)
    logits = torch.randn(4, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    confidences = torch.tensor([0.8, 0.9, 0.7, 0.85])
    projected = torch.nn.functional.normalize(torch.randn(4, 64, requires_grad=True), p=2, dim=-1)

    out = criterion(logits, labels, confidences=confidences, projected_embeddings=projected)
    assert "loss" in out
    assert "loss_cls" in out
    assert "loss_con" in out
    assert torch.isfinite(out["loss"])
