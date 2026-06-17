import torch
import torch.nn as nn
import torch.nn.functional as F


class OCCLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(OCCLoss, self).__init__()
        self.margin = margin

    def forward(self, features, labels, center):
        center = center.to(features.device)
        dist = torch.norm(features - center, dim=1)
        normal_mask = (labels == 1).float()
        anomalous_mask = (labels == 0).float()
        normal_loss = normal_mask * dist.pow(2)
        anomalous_loss = anomalous_mask * F.relu(self.margin - dist).pow(2)
        return (normal_loss + anomalous_loss).mean()


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features1, features2):
        B = features1.size(0)
        if B < 2:
            return torch.tensor(0.0, device=features1.device)

        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)

        sim = torch.mm(features1, features2.T) / self.temperature

        labels = torch.arange(B, device=sim.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss


class TriExpertDisentangleLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat_gnn, feat_llm, feat_vis):
        feats = [feat_gnn, feat_llm, feat_vis]
        loss = 0.0

        for feat in feats:
            f = feat - feat.mean(dim=0, keepdim=True)
            n, d = f.shape
            std = f.std(dim=0, keepdim=True) + 1e-8
            f_normed = f / std
            corr = (f_normed.T @ f_normed) / (n - 1 + 1e-8)
            loss += (corr.pow(2) - torch.eye(d, device=corr.device)).mean()

        for i in range(3):
            for j in range(i + 1, 3):
                f1 = feats[i] - feats[i].mean(dim=0, keepdim=True)
                f2 = feats[j] - feats[j].mean(dim=0, keepdim=True)
                n = f1.size(0)
                f1 = f1 / (f1.std(dim=0, keepdim=True) + 1e-8)
                f2 = f2 / (f2.std(dim=0, keepdim=True) + 1e-8)
                cross = (f1.T @ f2) / (n - 1 + 1e-8)
                loss += cross.pow(2).mean()

        return loss / 6


class EmpathyOrthogonalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cognitive_emb, emotional_emb):
        cognitive_emb = F.normalize(cognitive_emb, dim=-1)
        emotional_emb = F.normalize(emotional_emb, dim=-1)
        sim = torch.mm(cognitive_emb, emotional_emb.T)
        eye = torch.eye(sim.size(0), device=sim.device)
        return F.mse_loss(sim, eye)


class CrossModalConsistencyLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, text_features, image_features, labels=None):
        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)

        sim = torch.mm(text_features, image_features.T) / self.temperature
        eye = torch.eye(sim.size(0), device=sim.device)
        loss = F.mse_loss(sim, eye)

        if labels is not None:
            return loss
        return loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.5,
        gamma: float = 0.1,
        delta: float = 0.1,
        margin: float = 1.0,
        temperature: float = 0.5,
        hidden_dim: int = 512,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.occ_loss = OCCLoss(margin)
        self.info_nce_loss = InfoNCELoss(temperature)
        self.disentangle_loss = TriExpertDisentangleLoss()
        self.empathy_ortho_loss = EmpathyOrthogonalLoss()

        self.proj_gnn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, proj_dim),
        )
        self.proj_llm = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, proj_dim),
        )
        self.proj_vis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, proj_dim),
        )

    def forward(
        self,
        feat_gnn: torch.Tensor,
        feat_llm: torch.Tensor,
        feat_vis: torch.Tensor,
        labels: torch.Tensor,
        center_gnn: torch.Tensor,
        center_llm: torch.Tensor,
        center_vis: torch.Tensor,
        cognitive_emb: torch.Tensor = None,
        emotional_emb: torch.Tensor = None
    ) -> dict:
        feat_gnn_p = self.proj_gnn(feat_gnn)
        feat_llm_p = self.proj_llm(feat_llm)
        feat_vis_p = self.proj_vis(feat_vis)

        info_nce = self.info_nce_loss(feat_gnn_p, feat_vis_p) + self.info_nce_loss(feat_llm_p, feat_vis_p)
        info_nce = info_nce / 2

        occ = self.occ_loss(feat_gnn, labels, center_gnn) + \
              self.occ_loss(feat_llm, labels, center_llm) + \
              self.occ_loss(feat_vis, labels, center_vis)
        occ = occ / 3

        dis = self.disentangle_loss(feat_gnn, feat_llm, feat_vis)

        total_loss = self.alpha * info_nce + self.beta * occ + self.gamma * dis

        losses = {
            'total': total_loss,
            'info_nce': info_nce,
            'occ': occ,
            'covariance': dis
        }

        if cognitive_emb is not None and emotional_emb is not None:
            emp = self.empathy_ortho_loss(cognitive_emb, emotional_emb)
            total_loss = total_loss + self.delta * emp
            losses['empathy_ortho'] = emp
            losses['total'] = total_loss

        return losses
