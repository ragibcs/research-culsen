"""
Pretrained Transformer baseline training script for CulSen-BN.
Fine-tunes BanglaBERT / XLM-RoBERTa using standard Cross-Entropy with PyTorch & Hugging Face.
Optimizes validation Macro-F1 with early stopping and learning rate warmup.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import time
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ..models.transformer import CulSenTransformerEncoder
from ..evaluation.metrics import compute_classification_metrics
from ..evaluation.calibration import compute_calibration_metrics
from ..evaluation.robustness import evaluate_subgroup_robustness
from ..utils.logging import get_logger, save_json
from ..utils.reproducibility import set_seed, get_device
from ..utils.checkpointing import save_checkpoint


class SentimentDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenized comments and binary sentiment labels.
    """

    def __init__(self, texts, labels, tokenizer, max_length=128, confidences=None):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.confidences = list(confidences) if confidences is not None else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        if self.confidences is not None:
            item["confidence"] = torch.tensor(self.confidences[idx], dtype=torch.float)
        return item


def train_transformer_baseline(
    config_path: str = "configs/baseline_transformer.yaml",
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train and evaluate a standard pretrained transformer baseline.
    """
    logger = get_logger("transformer_training")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)
    device = torch.device(get_device())
    logger.info(f"Using compute device: {device}")

    # Determine model candidate
    model_name = model_override or cfg.get("model", {}).get("default_model", "banglabert")
    candidates = {c["name"]: c for c in cfg.get("model", {}).get("candidates", [])}
    if model_name not in candidates:
        hf_model_path = model_name  # direct huggingface path
    else:
        hf_model_path = candidates[model_name]["pretrained_model_name"]

    logger.info(f"Selected model: {model_name} ({hf_model_path})")

    # Load data splits
    import pandas as pd
    split_file = cfg["data"]["split_file"]
    splits_df = pd.read_parquet(split_file)

    train_df = splits_df[splits_df["split"] == "train"].copy()
    val_df = splits_df[splits_df["split"] == "val"].copy()
    test_df = splits_df[splits_df["split"] == "test"].copy()

    # Filter binary classes
    train_df = train_df[train_df["sentiment"].isin(["negative", "positive"])]
    val_df = val_df[val_df["sentiment"].isin(["negative", "positive"])]
    test_df = test_df[test_df["sentiment"].isin(["negative", "positive"])]

    text_col = cfg["data"].get("text_column", "text_clean")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    max_len = cfg.get("model", {}).get("max_length", 128)

    train_dataset = SentimentDataset(
        train_df[text_col],
        (train_df["sentiment"] == "positive").astype(int),
        tokenizer,
        max_length=max_len,
    )
    val_dataset = SentimentDataset(
        val_df[text_col],
        (val_df["sentiment"] == "positive").astype(int),
        tokenizer,
        max_length=max_len,
    )
    test_dataset = SentimentDataset(
        test_df[text_col],
        (test_df["sentiment"] == "positive").astype(int),
        tokenizer,
        max_length=max_len,
    )

    batch_size = cfg.get("training", {}).get("batch_size", 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = CulSenTransformerEncoder(pretrained_model_name=hf_model_path, num_labels=2)
    model.to(device)

    # Optimizer & Scheduler
    lr = float(cfg.get("training", {}).get("learning_rate", 2e-5))
    weight_decay = float(cfg.get("training", {}).get("weight_decay", 0.01))
    epochs = int(cfg.get("training", {}).get("num_epochs", 4))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * float(cfg.get("training", {}).get("warmup_ratio", 0.10)))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_weights = None
    best_epoch = 0

    out_dir = Path(cfg["output"].get("artifacts_dir", "artifacts/baseline_transformer")) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    start_train_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs["logits"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        val_metrics, _ = evaluate_transformer(model, val_loader, device)
        val_f1 = val_metrics["macro_f1"]
        logger.info(
            f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Macro-F1: {val_f1:.4f} - Val Acc: {val_metrics['accuracy']:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

    total_training_sec = time.time() - start_train_time
    logger.info(f"Training completed in {total_training_sec:.1f}s. Best Epoch: {best_epoch} with Val Macro-F1: {best_val_f1:.4f}")

    # Load best model for testing
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save best checkpoint
    save_checkpoint(
        state_dict=best_weights,
        metadata={"model_name": model_name, "best_epoch": best_epoch, "val_f1": best_val_f1},
        checkpoint_dir=str(out_dir),
        filename="best_checkpoint.pt",
    )

    # Test evaluation
    test_metrics, test_probs = evaluate_transformer(model, test_loader, device)
    y_test = (test_df["sentiment"] == "positive").astype(int).tolist()
    y_pred = [1 if p >= 0.5 else 0 for p in test_probs]
    calib_metrics = compute_calibration_metrics(y_test, test_probs)
    subgroups = evaluate_subgroup_robustness(test_df, y_pred, test_probs, text_col=text_col)

    results = {
        "model_name": model_name,
        "pretrained_path": hf_model_path,
        "training_time_seconds": total_training_sec,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "calibration": calib_metrics,
        "subgroups": subgroups,
    }

    save_json(results, str(out_dir / "metrics.json"))
    return results


def evaluate_transformer(model, dataloader, device):
    """
    Compute predictions and probabilities over a DataLoader.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
            preds = torch.argmax(outputs["logits"], dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    metrics = compute_classification_metrics(all_labels, all_preds, all_probs)
    return metrics, all_probs
