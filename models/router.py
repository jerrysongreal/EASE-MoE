"""
Gating Router for EASE-MoE.
Trainable 2-layer MLP that maps [expert distances + expert features] → routing weights.
No hand-crafted normalization — the network learns to weigh experts end-to-end.

Input:  [d_sem, d_vis, d_prop, d_emp, h_sem, h_vis, h_prop, h_emp]
        4 + 4*256 = 1028 dims
Output: softmax weights over 4 experts.
"""
import torch
import torch.nn as nn
from typing import List


class GatingRouter(nn.Module):
    def __init__(self, expert_dim: int = 256, hidden_dim: int = 128,
                 n_experts: int = 4):
        super().__init__()
        self.n_experts = n_experts
        input_dim = n_experts + n_experts * expert_dim  # 4 + 1024 = 1028

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts)
        )

    def forward(self, distances: List[torch.Tensor],
                expert_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            distances: list of 4 tensors, each (B,)
            expert_features: list of 4 tensors, each (B, expert_dim)
        Returns:
            routing_weights: (B, n_experts) softmax weights
        """
        d_stack = torch.stack(distances, dim=1)  # (B, 4)
        h_stack = torch.cat(expert_features, dim=1)  # (B, 4*256)

        x = torch.cat([d_stack, h_stack], dim=1)  # (B, 1028)
        logits = self.mlp(x)  # (B, 4)
        weights = torch.softmax(logits, dim=-1)
        return weights

    def compute_final_score(self, distances: List[torch.Tensor],
                            expert_features: List[torch.Tensor]) -> torch.Tensor:
        """Weighted sum of OCC distances = final anomaly score."""
        weights = self.forward(distances, expert_features)
        d_stack = torch.stack(distances, dim=1)
        return (weights * d_stack).sum(dim=1)
