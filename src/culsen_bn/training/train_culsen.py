"""
Integrated CulSen-BN Model Training Pipeline.
Combines Contextual Transformer Representation, Uncertainty-Aware Supervision,
Confidence-Filtered Supervised Contrastive Learning, and Domain-Adversarial Season Invariance.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ..models.transformer import CulSenTransformerEncoder
from ..models.contrastive import SupervisedContrastiveHead
from ..models.domain_adversarial import SeasonDiscriminator
from .losses import IntegratedCulSenLoss
from .train_transformer import SentimentDataset, evaluate_transformer
from ..evaluation.metrics import compute_classification_metrics
from ..evaluation.calibration import compute_calibration_metrics
from ..evaluation.robustness import evaluate_subgroup_robustness
from ..utils.logging import get_logger, save_json
from ..utils.reproducibility import set_seed, get_device
from ..utils.checkpointing import save_checkpoint


class CulSenModel(nn.Module):
    """
    Unified CulSen-BN Architecture combining:
    1. Base Encoder
    2. Sentiment Classifier Head
    3. Contrastive Projection Head
    4. Adversarial Season Discriminator Head
    """

    def __init__(
        self,
        pretrained_model_name: str = "csebuetnlp/banglabert",
        projection_dim: int = 128,
        num_seasons: int = 4,
        enable_contrastive: bool = True,
        enable_domain_inv: bool = True,
    ):
        super().__init__()
        self.encoder_module = CulSenTransformerEncoder(pretrained_model_name=pretrained_model_name)
        hidden_dim = self.encoder_module.hidden_size

        self.enable_contrastive = enable_contrastive
        self.enable_domain_inv = enable_domain_inv

        if enable_contrastive:
            self.contrastive_head = SupervisedContrastiveHead(
                in_dim=hidden_dim,
                projection_dim=projection_dim,
            )

        if enable_domain_inv:
            self.domain_discriminator = SeasonDiscriminator(
                in_dim=hidden_dim,
                num_seasons=num_seasons,
            )

    def set_grl_alpha(self, alpha: float):
        if self.enable_domain_inv:
            self.domain_discriminator.set_alpha(alpha)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc_out = self.encoder_module(input_ids, attention_mask, token_type_ids)
        pooled = enc_out["pooled"]
        logits = enc_out["logits"]

        out = {"logits": logits, "pooled": pooled}

        if self.enable_contrastive:
            out["projected"] = self.contrastive_head(pooled)

        if self.enable_domain_inv:
            out["season_logits"] = self.domain_discriminator(pooled)

        return out


def train_culsen_model(
    experiment_config: str = "configs/experiment.yaml",
    variant_name: str = "B5_full",
    pretrained_model_name: str = "csebuetnlp/banglabert",
) -> Dict[str, Any]:
    """
    Train and evaluate the proposed CulSen-BN framework or one of its ablation variants.
    """
    logger = get_logger("culsen_training")
    with open(experiment_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    variant_cfg = cfg["experiments"]["E8_ablation"]["variants"].get(variant_name, {})
    use_uncertainty = variant_cfg.get("uncertainty", True)
    use_contrastive = variant_cfg.get("contrastive", True)
    use_domain_inv = variant_cfg.get("domain_inv", True)
    use_consistency = variant_cfg.get("consistency", False)

    logger.info(
        f"Initializing CulSen-BN variant '{variant_name}': Uncertainty={use_uncertainty}, "
        f"Contrastive={use_contrastive}, DomainInv={use_domain_inv}, Consistency={use_consistency}"
    )

    seed = cfg["experiment"].get("random_seeds", [42])[0]
    set_seed(seed)
    device = torch.device(get_device())

    # Load splits
    import pandas as pd
    split_file = "data/processed/splits_protocol_a.parquet"
    splits_df = pd.read_parquet(split_file)

    train_df = splits_df[splits_df["split"] == "train"].copy()
    val_df = splits_df[splits_df["split"] == "val"].copy()
    test_df = splits_df[splits_df["split"] == "test"].copy()

    # Filter binary
    train_df = train_df[train_df["sentiment"].isin(["negative", "positive"])]
    val_df = val_df[val_df["sentiment"].isin(["negative", "positive"])]
    test_df = test_df[test_df["sentiment"].isin(["negative", "positive"])]

    text_col = "text_clean"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    # Convert seasons (1-indexed to 0-indexed: Season 1 -> 0, Season 2 -> 1, ...)
    train_seasons = (train_df["season"] - 1).clip(0, 3).tolist()

    train_dataset = SentimentDataset(
        train_df[text_col],
        (train_df["sentiment"] == "positive").astype(int),
        tokenizer,
        max_length=128,
        confidences=train_df["confidence"].fillna(0.7).tolist() if use_uncertainty else None,
    )
    val_dataset = SentimentDataset(
        val_df[text_col],
        (val_df["sentiment"] == "positive").astype(int),
        tokenizer,
        max_length=128,
    )
    test_dataset = SentimentDataset(
        test_df[text_col],
        (test_df["sentiment"] == "positive").astype(int),
        tokenizer,
        max_length=128,
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = CulSenModel(
        pretrained_model_name=pretrained_model_name,
        projection_dim=128,
        num_seasons=4,
        enable_contrastive=use_contrastive,
        enable_domain_inv=use_domain_inv,
    )
    model.to(device)

    loss_fn = IntegratedCulSenLoss(
        alpha=0.20 if use_contrastive else 0.0,
        beta=0.10 if use_domain_inv else 0.0,
        gamma=0.05 if use_consistency else 0.0,
        clipping_floor=0.10,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 4
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_val_f1 = -1.0
    best_weights = None
    best_epoch = 0

    out_dir = Path(f"artifacts/culsen_{variant_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # Dynamic GRL alpha schedule: p in [0, 1]
        p = float(epoch - 1) / float(epochs)
        grl_alpha = 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0
        model.set_grl_alpha(float(grl_alpha))

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            confidences = batch.get("confidence", None)
            if confidences is not None:
                confidences = confidences.to(device)

            outputs = model(input_ids, attention_mask)

            loss_dict = loss_fn(
                sentiment_logits=outputs["logits"],
                labels=labels,
                confidences=confidences,
                projected_embeddings=outputs.get("projected", None),
            )

            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        val_metrics, _ = evaluate_transformer(model.encoder_module, val_loader, device)
        val_f1 = val_metrics["macro_f1"]
        logger.info(f"Variant {variant_name} Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

    total_training_sec = time.time() - start_time
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Test Evaluation
    test_metrics, test_probs = evaluate_transformer(model.encoder_module, test_loader, device)
    y_test = (test_df["sentiment"] == "positive").astype(int).tolist()
    y_pred = [1 if p >= 0.5 else 0 for p in test_probs]
    calib = compute_calibration_metrics(y_test, test_probs)
    subgroups = evaluate_subgroup_robustness(test_df, y_pred, test_probs, text_col=text_col)

    results = {
        "variant": variant_name,
        "encoder": pretrained_model_name,
        "training_time_seconds": total_training_sec,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "calibration": calib,
        "subgroups": subgroups,
    }

    save_json(results, str(out_dir / "metrics.json"))
    save_checkpoint(
        state_dict=best_weights,
        metadata={"variant": variant_name, "best_epoch": best_epoch, "val_f1": best_val_f1},
        checkpoint_dir=str(out_dir),
        filename="best_checkpoint.pt",
    )
    return results
