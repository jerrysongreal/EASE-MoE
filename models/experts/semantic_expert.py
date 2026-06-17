"""
Semantic Expert for EASE-MoE.
Encodes news text via RoBERTa online encoding with self-attention pooling.
Outputs h_sem (batch, hidden_dim).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class SemanticExpert(nn.Module):
    def __init__(self, roberta_path: str = "roberta-base", hidden_dim: int = 512,
                 num_heads: int = 8, dropout: float = 0.1, freeze_backbone: bool = True,
                 freeze_last_n: int = 3, shared_roberta=None, shared_tokenizer=None):
        super().__init__()
        from transformers import RobertaModel, RobertaTokenizer

        if shared_roberta is not None:
            self.roberta = shared_roberta
            self.tokenizer = shared_tokenizer
        else:
            self.roberta = RobertaModel.from_pretrained(roberta_path, local_files_only=True)
            self.tokenizer = RobertaTokenizer.from_pretrained(roberta_path, local_files_only=True)
        roberta_dim = self.roberta.config.hidden_size

        self.text_proj = nn.Linear(roberta_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        if freeze_backbone:
            self._freeze_layers(freeze_last_n)

    def _freeze_layers(self, unfreeze_last_n: int):
        for name, param in self.roberta.named_parameters():
            param.requires_grad = False
            for i in range(unfreeze_last_n):
                if f"encoder.layer.{11 - i}" in name:
                    param.requires_grad = True
                    break

    def encode_text(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.roberta(**text_inputs)
        features = outputs.last_hidden_state  # (B, L, 768)
        features = self.text_proj(features)    # (B, L, hidden_dim)
        features = self.dropout(features)
        attended, _ = self.self_attn(features, features, features)
        features = self.norm(features + attended)  # residual
        pooled = features.mean(dim=1)  # (B, hidden_dim)
        return pooled

    def forward(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode_text(text_inputs)
