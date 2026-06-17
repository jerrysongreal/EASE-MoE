"""
Expert-Consistency Pseudo-Labeling for EASE-MoE v2.
Independent module — assigns pseudo-labels only when experts agree.

Paper formulas:
  v_i = Var({d̂_i^e})                              — expert disagreement
  pseudo-real:  s_i ≤ τ_real AND v_i ≤ β
  pseudo-fake:  s_i ≥ τ_fake AND v_i ≤ β
  conf_i = (1 − ṽ_i) · σ(|s_i − τ_s| / τ_c)      — confidence with sigmoid
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ExpertConsistencyPL:
    """
    Pseudo-label strategy based on expert agreement (paper III.E.2).
    """
    def __init__(self, alpha: float = 1.5, delta: float = 4.0,
                 beta: float = 1.0, warmup_epochs: int = 5,
                 tau_c: float = 0.5):
        self.alpha = alpha          # d_mean < alpha → pseudo-real
        self.delta = delta          # d_mean > delta → pseudo-fake
        self.beta = beta            # variance < beta → experts agree
        self.warmup_epochs = warmup_epochs
        self.tau_s = (alpha + delta) / 2  # decision boundary midpoint
        self.tau_c = tau_c          # temperature for confidence sigmoid

    def assign(self, distances, routing_weights, epoch: int = 0):
        """
        Args:
            distances: list of 4 tensors, each (B,) — expert anomaly distances
            routing_weights: (B, 4) — router weights
            epoch: current epoch (for ramp-up)
        Returns:
            pseudo_labels: (B,) with values +1 (real), 0 (fake), -1 (uncertain)
            confidences: (B,) confidence scores in [0, 1]
            sample_weights: (B,) final weights with ramp-up
        """
        d_stack = torch.stack(distances, dim=1)  # (B, 4)

        # Weighted anomaly score (paper Eq. 16)
        d_mean = (routing_weights * d_stack).sum(dim=1)  # (B,)

        # Expert variance (disagreement) — paper Eq. 17
        # Normalize d to compute variance on comparable scale
        d_norm = (d_stack - d_stack.mean(dim=1, keepdim=True)) / (
            d_stack.std(dim=1, keepdim=True) + 1e-8)
        d_var = d_norm.var(dim=1)  # (B,) — on normalized distances

        # Assign pseudo-labels (paper III.E.2)
        pseudo_labels = torch.full_like(d_mean, -1, dtype=torch.long)
        pseudo_real = (d_mean < self.alpha) & (d_var < self.beta)
        pseudo_fake = (d_mean > self.delta) & (d_var < self.beta)
        pseudo_labels[pseudo_real] = 1
        pseudo_labels[pseudo_fake] = 0

        # Confidence: (1 − ṽ) · σ(|s − τ_s| / τ_c)  — paper Eq. 19
        # d_var is already on normalized distances, upper-bounded approximately
        d_var_clipped = torch.clamp(d_var, 0.0, 1.0)  # ṽ ∈ [0, 1]
        conf = (1.0 - d_var_clipped) * torch.sigmoid(
            torch.abs(d_mean - self.tau_s) / self.tau_c
        )
        # conf ∈ [0, ~0.5] roughly — per-sample, no batch dependency

        # Ramp-up weight (paper: curriculum scheduling)
        ramp = min(epoch / max(self.warmup_epochs, 1), 1.0)
        sample_weights = torch.where(
            pseudo_labels >= 0, ramp * conf, torch.zeros_like(conf))

        return pseudo_labels, conf, sample_weights
