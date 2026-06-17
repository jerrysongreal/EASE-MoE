"""
Propagation Expert for EASE-MoE v2.
All samples go through 3-layer GCN + JumpingKnowledge — no MLP fallback.
Single-node trees are treated as valid propagation structures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge, global_mean_pool


class PropagationExpert(nn.Module):
    def __init__(self, dim_features: int = 768, hidden_dim: int = 256,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # All GCN layers operate at full RoBERTa dimension (768) as per paper III.C.3
        # JK cat = 768 × 3 = 2304, then projected to 256
        gcn_dim = dim_features  # 768 — paper keeps full dimension across all layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(gcn_dim, gcn_dim))
            self.norms.append(nn.BatchNorm1d(gcn_dim))

        self.dropout = nn.Dropout(dropout)
        self.jk = JumpingKnowledge(mode='cat', channels=gcn_dim, num_layers=num_layers)
        # JK cat = gcn_dim × 3 = 2304 → 256 (paper: W_out^prop ∈ R^{256×2304})
        self.output_proj = nn.Sequential(
            nn.Linear(gcn_dim * num_layers, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        All samples (including single-node trees) go through GCN.
        """
        layer_outputs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
            layer_outputs.append(global_mean_pool(x, batch))

        jk_out = self.jk(layer_outputs)        # (B, hidden_dim * 3)
        return self.output_proj(jk_out)         # (B, hidden_dim)
