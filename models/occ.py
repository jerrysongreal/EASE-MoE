"""
Multi-Prototype One-Class Classification for EASE-MoE v2.
Each expert learns M=5 prototypes representing 5 real-news archetypes:
  1. Political news     (policy, elections, diplomacy)
  2. Health news        (disease, public health, medicine)
  3. Entertainment news (films, celebrities, culture)
  4. Breaking events    (disasters, accidents, emergencies)
  5. Short social text  (tweets, snippets, fragmented info)

Prototypes are learnable parameters in R^256, initialized via K-means on real-news features.
Anomaly distance: d_i^e = min_m ||f_i^e - theta_m^e||^2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MultiPrototypeOCC(nn.Module):
    def __init__(self, n_experts: int = 4, hidden_dim: int = 256,
                 n_prototypes: int = 5, margin: float = 3.0):
        super().__init__()
        self.n_experts = n_experts
        self.n_prototypes = n_prototypes
        self.margin = margin

        # Prototype-space projection: one per expert (Linear(256,256))
        self.proto_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(n_experts)
        ])

        # Theta^e = {theta_1^e, ..., theta_M^e} — learnable prototype vectors
        self.prototypes = nn.Parameter(
            torch.randn(n_experts, n_prototypes, hidden_dim) * 0.1
        )

    def get_distances(self, expert_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        d_i^e = min_m ||proj_e(f_i^e) - theta_m^e||^2
        Features are first projected to prototype space, then L2-normalized.
        """
        import torch.nn.functional as F
        distances = []
        for e, feat in enumerate(expert_features):
            protos = self.prototypes[e].clamp(-10.0, 10.0)
            # Project to prototype space, then L2 normalize
            feat_proj = self.proto_proj[e](feat)
            feat_n = F.normalize(feat_proj, dim=-1)
            protos_n = F.normalize(protos, dim=-1)
            dists = torch.cdist(feat_n, protos_n, p=2).pow(2).clamp(max=4.0)
            d_min, _ = dists.min(dim=1)
            distances.append(d_min)
        return distances

    def forward(self, expert_features: List[torch.Tensor],
                labels: torch.Tensor,
                routing_weights: torch.Tensor = None,
                sample_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Weighted multi-prototype OCC loss.
        Real news (y=1): minimize distance to nearest prototype.
        Fake news (y=0): push beyond margin, penalize if too close.
        """
        B = labels.size(0)
        if routing_weights is None:
            routing_weights = torch.ones(B, self.n_experts, device=labels.device) / self.n_experts
        if sample_weights is None:
            sample_weights = torch.ones(B, device=labels.device)

        total_loss = 0.0
        real_mask = (labels == 1)
        fake_mask = (labels == 0)

        for e, feat in enumerate(expert_features):
            protos = self.prototypes[e].clamp(-10.0, 10.0)
            feat_proj = self.proto_proj[e](feat)
            feat_n = F.normalize(feat_proj, dim=-1)
            protos_n = F.normalize(protos, dim=-1)
            dists = torch.cdist(feat_n, protos_n, p=2).pow(2).clamp(max=4.0)
            d_min, _ = dists.min(dim=1)

            rw = routing_weights[:, e]
            sw = sample_weights

            expert_loss = 0.0
            if real_mask.any():
                expert_loss += (sw[real_mask] * rw[real_mask] * d_min[real_mask]).mean()
            if fake_mask.any():
                margin_violation = F.relu(self.margin - d_min[fake_mask])
                expert_loss += (sw[fake_mask] * rw[fake_mask] * margin_violation.pow(2)).mean()
            total_loss += expert_loss

        return total_loss / self.n_experts

    @torch.no_grad()
    def init_prototypes(self, expert_features: List[torch.Tensor], labels: torch.Tensor):
        """K-means initialization of prototypes on projected real-news features (10 iterations)."""
        for e, feat in enumerate(expert_features):
            feat = feat.to(self.prototypes.device)
            feat_proj = self.proto_proj[e](feat)
            feat = F.normalize(feat_proj, dim=-1)  # L2 normalize before K-means
            real_feats = feat[labels == 1]
            if real_feats.size(0) < self.n_prototypes:
                continue

            centroids = real_feats[:self.n_prototypes].clone()
            for _ in range(10):
                dists = torch.cdist(real_feats, centroids, p=2).pow(2)
                assignments = dists.argmin(dim=1)
                for k in range(self.n_prototypes):
                    members = real_feats[assignments == k]
                    if members.size(0) > 0:
                        centroids[k] = members.mean(dim=0)
                if assignments.diff().abs().sum() == 0:
                    break
            self.prototypes.data[e] = centroids
