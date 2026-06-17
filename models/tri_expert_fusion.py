import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Tuple


class ParametricMaskNet(nn.Module):
    def __init__(self, num_segments: int, mask_dim: int, mode: str = 'softmax', dropout: float = 0.1):
        super(ParametricMaskNet, self).__init__()
        self.num_segments = num_segments
        self.mask_dim = mask_dim
        self.mode = mode
        self.raw_mask = Parameter(torch.randn(num_segments, mask_dim))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self) -> torch.Tensor:
        raw = self.dropout(self.raw_mask)
        if self.mode == 'softmax':
            mask = F.softmax(raw, dim=-1)
        elif self.mode == 'sigmoid':
            mask = torch.sigmoid(raw)
        elif self.mode == 'relu':
            mask = F.relu(raw)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return mask


class TriExpertDisentangle(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_chunks: int = 4,
        dropout: float = 0.1
    ):
        super(TriExpertDisentangle, self).__init__()
        
        self.num_chunks = num_chunks
        self.hidden_dim = hidden_dim
        self.chunk_dim = hidden_dim // num_chunks
        
        self.mask_net_gnn_llm = ParametricMaskNet(
            num_segments=num_chunks,
            mask_dim=self.chunk_dim,
            dropout=dropout
        )
        
        self.mask_net_gnn_vis = ParametricMaskNet(
            num_segments=num_chunks,
            mask_dim=self.chunk_dim,
            dropout=dropout
        )
        
        self.mask_net_llm_vis = ParametricMaskNet(
            num_segments=num_chunks,
            mask_dim=self.chunk_dim,
            dropout=dropout
        )

    def split_chunks(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(1) % self.num_chunks == 0, \
            f"Hidden dim {x.size(1)} must be divisible by num_chunks {self.num_chunks}"
        return x.view(x.size(0), self.num_chunks, -1)

    def cross_attention_transform(
        self,
        query_chunks: torch.Tensor,
        key_chunks: torch.Tensor,
        value_chunks: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        query_chunks = query_chunks * mask
        key_chunks = key_chunks * mask
        value_chunks = value_chunks * mask
        
        attn_output = F.scaled_dot_product_attention(
            query=query_chunks,
            key=key_chunks,
            value=value_chunks
        )
        
        return attn_output.reshape(attn_output.size(0), -1)

    def forward(
        self,
        gnn_emb: torch.Tensor,
        llm_emb: torch.Tensor,
        vis_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g_chunks = self.split_chunks(gnn_emb)
        l_chunks = self.split_chunks(llm_emb)
        v_chunks = self.split_chunks(vis_emb)
        
        batch_size = g_chunks.size(0)
        
        mask_gl = self.mask_net_gnn_llm().unsqueeze(0).expand(batch_size, -1, -1).to(gnn_emb.device)
        mask_gv = self.mask_net_gnn_vis().unsqueeze(0).expand(batch_size, -1, -1).to(gnn_emb.device)
        mask_lv = self.mask_net_llm_vis().unsqueeze(0).expand(batch_size, -1, -1).to(gnn_emb.device)
        
        g_from_l = self.cross_attention_transform(g_chunks, l_chunks, l_chunks, mask_gl)
        g_from_v = self.cross_attention_transform(g_chunks, v_chunks, v_chunks, mask_gv)
        
        l_from_g = self.cross_attention_transform(l_chunks, g_chunks, g_chunks, mask_gl)
        l_from_v = self.cross_attention_transform(l_chunks, v_chunks, v_chunks, mask_lv)
        
        v_from_g = self.cross_attention_transform(v_chunks, g_chunks, g_chunks, mask_gv)
        v_from_l = self.cross_attention_transform(v_chunks, l_chunks, l_chunks, mask_lv)
        
        g_new = gnn_emb + 0.1 * (g_from_l + g_from_v)
        l_new = llm_emb + 0.1 * (l_from_g + l_from_v)
        v_new = vis_emb + 0.1 * (v_from_g + v_from_l)
        
        return g_new, l_new, v_new


class DualAttentionTransform(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_chunks: int = 4,
        dropout: float = 0.1
    ):
        super(DualAttentionTransform, self).__init__()
        
        self.num_chunks = num_chunks
        self.hidden_dim = hidden_dim
        self.chunk_dim = hidden_dim // num_chunks
        
        self.mask_net = ParametricMaskNet(
            num_segments=num_chunks,
            mask_dim=self.chunk_dim,
            dropout=dropout
        )

    def split_chunks(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(1) % self.num_chunks == 0
        return x.view(x.size(0), self.num_chunks, -1)

    def forward(
        self,
        graph_emb: torch.Tensor,
        text_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g_chunks = self.split_chunks(graph_emb)
        t_chunks = self.split_chunks(text_emb)
        
        batch_size = g_chunks.size(0)
        
        mask_matrix = self.mask_net()
        mask_matrix = mask_matrix.unsqueeze(0).expand(batch_size, -1, -1).to(graph_emb.device)
        
        g_chunks = g_chunks * mask_matrix
        t_chunks = t_chunks * mask_matrix
        
        attn1 = F.scaled_dot_product_attention(query=g_chunks, key=t_chunks, value=t_chunks)
        attn2 = F.scaled_dot_product_attention(query=t_chunks, key=g_chunks, value=g_chunks)
        
        g_new = attn1.reshape(attn1.size(0), -1)
        t_new = attn2.reshape(attn2.size(0), -1)
        
        return g_new, t_new


class ExpertFusionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 3,
        dropout: float = 0.1
    ):
        super(ExpertFusionLayer, self).__init__()
        
        self.num_experts = num_experts
        
        self.expert_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout
        )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        expert_features: list,
        routing_weights: torch.Tensor
    ) -> torch.Tensor:
        projected_features = []
        for i, (feat, proj) in enumerate(zip(expert_features, self.expert_projections)):
            proj_feat = proj(feat)
            weighted_feat = proj_feat * routing_weights[:, i:i+1]
            projected_features.append(weighted_feat)
        
        stacked_features = torch.stack(projected_features, dim=0)
        
        fused, _ = self.fusion_attention(
            stacked_features,
            stacked_features,
            stacked_features
        )
        
        fused = fused.mean(dim=0)
        output = self.output_proj(fused)
        
        return output


if __name__ == "__main__":
    batch_size = 16
    hidden_dim = 512
    
    disentangle = TriExpertDisentangle(hidden_dim=hidden_dim)
    
    gnn_emb = torch.randn(batch_size, hidden_dim)
    llm_emb = torch.randn(batch_size, hidden_dim)
    vis_emb = torch.randn(batch_size, hidden_dim)
    
    g_new, l_new, v_new = disentangle(gnn_emb, llm_emb, vis_emb)
    
    print(f"GNN output shape: {g_new.shape}")
    print(f"LLM output shape: {l_new.shape}")
    print(f"Vision output shape: {v_new.shape}")
    
    fusion_layer = ExpertFusionLayer(hidden_dim=hidden_dim)
    routing_weights = torch.softmax(torch.randn(batch_size, 3), dim=-1)
    
    fused = fusion_layer([g_new, l_new, v_new], routing_weights)
    print(f"Fused output shape: {fused.shape}")
