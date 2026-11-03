"""
Pretrained Transformer Encoder for CulSen-BN.
Supports BanglaBERT (Bengali encoder) and XLM-RoBERTa (multilingual encoder).
Outputs pooled sentence representations and sentiment classification logits.
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class CulSenTransformerEncoder(nn.Module):
    """
    Contextual Bengali / Multilingual transformer encoder with pooling and classification head.
    """

    def __init__(
        self,
        pretrained_model_name: str = "csebuetnlp/banglabert",
        num_labels: int = 2,
        dropout_rate: float = 0.2,
        pooling_strategy: str = "mean",  # 'mean' or 'cls'
    ):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        self.pretrained_model_name = pretrained_model_name
        self.num_labels = num_labels
        self.pooling_strategy = pooling_strategy

        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=self.config)

        self.hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through transformer encoder and classification head.
        """
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = token_type_ids

        encoder_outputs = self.encoder(**kwargs)
        sequence_output = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Pooling
        if self.pooling_strategy == "cls":
            pooled = sequence_output[:, 0, :]
        else:
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled = sum_embeddings / sum_mask

        features = self.dropout(pooled)
        logits = self.classifier(features)

        return {
            "logits": logits,
            "pooled": pooled,
            "features": features,
        }
