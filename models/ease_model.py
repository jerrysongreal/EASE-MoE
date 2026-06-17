"""
EASE-MoE v2: Empathy-Aware Semi-Supervised Mixture of Experts.
Architecture: 4 experts → Multi-Prototype OCC → Gating Router (MLP 1028→128→4).
No projection layer (dimensions already unified at 256).
No classifier head (OCC anomaly score is the detection signal).
No image_mask hard blocking (router learns modality reliability autonomously).
Loss = L_mp-occ + lambda * L_con  (L_con MUST be active, lambda = 0.1).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from models.experts import (SemanticExpert, VisualMultimodalExpert,
                              PropagationExpert, EmpathyResponseExpert)
from models.occ import MultiPrototypeOCC
from models.router import GatingRouter


class CrossExpertContrastive(nn.Module):
    """InfoNCE contrastive loss on selected stable expert pairs."""
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        B = f1.size(0)
        if B < 2:
            return torch.tensor(0.0, device=f1.device)
        f1 = F.normalize(f1, dim=-1)
        f2 = F.normalize(f2, dim=-1)
        sim = torch.mm(f1, f2.T) / self.temperature
        labels = torch.arange(B, device=sim.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


CONTRASTIVE_PAIRS = [(0, 2), (0, 3), (1, 3)]  # (sem,str), (sem,emp), (vis,emp)


class EASEMoE(nn.Module):
    def __init__(self, dim_features: int = 768, hidden_dim: int = 256,
                 n_experts: int = 4, n_prototypes: int = 5,
                 roberta_path: str = "roberta-base",
                 swin_path: str = "microsoft/swin-base-patch4-window7-224",
                 device: str = "cpu", dropout: float = 0.3,
                 margin: float = 3.0, lambda_con: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.hidden_dim = hidden_dim
        self.device = device
        self.margin = margin
        self.lambda_con = lambda_con

        # ── Shared RoBERTa ─────────────────────────────────────────
        from transformers import RobertaModel, RobertaTokenizer
        self.shared_roberta = RobertaModel.from_pretrained(roberta_path, local_files_only=True)
        self.shared_tokenizer = RobertaTokenizer.from_pretrained(roberta_path, local_files_only=True)

        # ── 4 Experts ──────────────────────────────────────────────
        self.semantic_expert = SemanticExpert(
            roberta_path=roberta_path, hidden_dim=hidden_dim, dropout=dropout,
            shared_roberta=self.shared_roberta, shared_tokenizer=self.shared_tokenizer)
        self.visual_expert = VisualMultimodalExpert(
            swin_path=swin_path, hidden_dim=hidden_dim, dropout=dropout,
            freeze_backbone=True, freeze_last_n=0)

        # ── CLIP: frozen vision-language alignment for visual expert ──
        try:
            import open_clip
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai')
            clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
            clip_model.to(device)
            clip_model.eval()
            self.visual_expert.set_clip_model(clip_model, clip_preprocess, clip_tokenizer)
            print(f"  CLIP (ViT-B-32/openai) loaded via open_clip")
        except Exception as e:
            print(f"  CLIP not available via open_clip ({e})")
            # Fallback: try transformers CLIPModel
            try:
                from transformers import CLIPModel, CLIPProcessor
                clip_model2 = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32", local_files_only=True)
                clip_proc2 = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32", local_files_only=True)
                clip_model2.to(device)
                clip_model2.eval()
                self.visual_expert.set_clip_model(clip_model2, clip_proc2)
                print(f"  CLIP (ViT-B/32) loaded via transformers")
            except Exception as e2:
                print(f"  CLIP not available ({e2}) — visual expert uses Swin-only channel")
        self.propagation_expert = PropagationExpert(
            dim_features=dim_features, hidden_dim=hidden_dim, dropout=dropout)
        self.empathy_expert = EmpathyResponseExpert(
            hidden_dim=hidden_dim, dropout=dropout, device=device)

        # ── Prototype init projection (fixed, structure-preserving) ──
        self.proto_init_proj = nn.Linear(dim_features, hidden_dim, bias=False)
        for p in self.proto_init_proj.parameters():
            p.requires_grad = False

        # ── Semantic adapter (precomputed embedding → hidden_dim, 2-layer) ──
        self.semantic_proj = nn.Sequential(
            nn.Linear(dim_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim))

        # ── Multi-Prototype OCC ────────────────────────────────────
        self.occ = MultiPrototypeOCC(
            n_experts=n_experts, hidden_dim=hidden_dim,
            n_prototypes=n_prototypes, margin=margin)

        # ── Gating Router ──────────────────────────────────────────
        self.router = GatingRouter(expert_dim=hidden_dim, hidden_dim=128,
                                    n_experts=n_experts)

        # ── Contrastive Loss ──────────────────────────────────────
        self.contrastive = CrossExpertContrastive()

    # ═══════════════════════════════════════════════════════════════
    #  Encoding
    # ═══════════════════════════════════════════════════════════════

    def _encode_all(self, text_inputs, image_inputs, graph_data, llm_emb,
                    comment_emb=None, texts=None, images_pil=None,
                    comment_texts_list=None):
        """Encode all 4 expert features. No image_mask intervention."""
        B = llm_emb.size(0)

        # E0: Semantic — adapter on precomputed embeddings
        h_sem = self.semantic_proj(llm_emb.float().to(self.device))

        # E1: Visual — Swin + CLIP, no hard blocking
        h_mul = self.visual_expert(image_inputs, h_sem, texts=texts, images_pil=images_pil)

        # E2: Propagation — all samples through GCN
        h_str = self.propagation_expert(
            graph_data.x.float().to(self.device),
            graph_data.edge_index.to(self.device),
            graph_data.batch.to(self.device))

        # E3: Empathy — Creator (llm_emb + emotion_cls) + Reader (comment_emb + emo_cls)
        if comment_emb is not None:
            h_emp = self.empathy_expert(
                llm_emb.float().to(self.device), h_mul,
                comment_emb.float().to(self.device),
                texts=texts, comment_texts_list=comment_texts_list)
        else:
            # Fallback: use zeros (should not happen with proper data)
            h_emp = self.empathy_expert(
                llm_emb.float().to(self.device), h_mul,
                torch.zeros(B, 768, device=self.device))

        return [h_sem, h_mul, h_str, h_emp]

    # ═══════════════════════════════════════════════════════════════
    #  Forward
    # ═══════════════════════════════════════════════════════════════

    def forward(self, text_inputs: Dict, image_inputs: Dict, graph_data,
                llm_emb: torch.Tensor, comment_emb: torch.Tensor = None,
                labels: torch.Tensor = None,
                sample_weights: torch.Tensor = None,
                texts: list = None, images_pil: list = None,
                comment_texts_list: list = None):
        """
        Returns: (total_loss, extra_dict)
        No logits — OCC anomaly score is the detection signal.
        """
        B = llm_emb.size(0)
        extra = {}

        # Step 1: Encode 4 experts (no projection needed, dims already 256)
        features = self._encode_all(
            text_inputs, image_inputs, graph_data, llm_emb,
            comment_emb=comment_emb, texts=texts, images_pil=images_pil,
            comment_texts_list=comment_texts_list)

        # Step 2: OCC distances
        occ_distances = self.occ.get_distances(features)  # list of (B,) x 4

        # Step 3: Router weights via gating MLP (distances + features)
        routing_weights = self.router(occ_distances, features)  # (B, 4)
        extra["routing_weights"] = {
            f"e{i}": routing_weights[:, i].mean().item() for i in range(self.n_experts)}

        # Step 4: Losses
        total_loss = torch.tensor(0.0, device=self.device)

        if labels is not None:
            # L_mp-occ: weighted multi-prototype OCC
            occ_loss = self.occ.forward(
                features, labels,
                routing_weights=routing_weights,
                sample_weights=sample_weights)

            # L_con: cross-expert contrastive on PROTOTYPE-SPACE features (paper: h̃)
            con_loss = torch.tensor(0.0, device=self.device)
            # Project each expert feature to prototype space: h̃^e = proj_e(h^e)
            proj_features = []
            for e, feat in enumerate(features):
                proj = self.occ.proto_proj[e](feat)
                proj = F.normalize(proj, dim=-1)
                proj_features.append(proj)
            for (i, j) in CONTRASTIVE_PAIRS:
                con_loss += self.contrastive(proj_features[i], proj_features[j])

            total_loss = occ_loss + self.lambda_con * con_loss
            extra["loss_components"] = {
                "occ": occ_loss.item(),
                "contrastive": con_loss.item()
            }

        extra["occ_distances"] = [d.mean().item() for d in occ_distances]
        extra["final_score"] = self.router.compute_final_score(occ_distances, features)

        return total_loss, extra

    # ═══════════════════════════════════════════════════════════════
    #  Prototype initialization
    # ═══════════════════════════════════════════════════════════════

    @torch.no_grad()
    def init_occ_prototypes(self, dataloader, device: str):
        """K-means initialize prototypes on real-news features.
        Uses structure-preserving random projection of pretrained embeddings,
        NOT expert features (which have random weights at initialization).
        Loads directly from llm_embeddings.pt when available to skip slow dataloader."""
        self.eval()
        all_feats = []
        all_lbls = []
        proj = self.proto_init_proj.to(device)

        # Fast path: load pre-computed embeddings directly from .pt file
        # (avoids slow image loading + tokenization in single-thread dataloader)
        ds = dataloader.dataset
        if hasattr(ds, 'dataset'):
            ds = ds.dataset  # unwrap Subset
        emb_path = getattr(ds, 'llm_embedding_path', None)
        if emb_path and os.path.exists(emb_path):
            emb_data = torch.load(emb_path, weights_only=True)
            raw_embs = emb_data["embeddings"]
            raw_lbls = emb_data["labels"]
            # Handle multiple .pt formats: list, dict, or pre-stacked tensor
            if isinstance(raw_embs, list):
                embs = torch.stack([e.float() for e in raw_embs])
                lbls = torch.tensor(raw_lbls, dtype=torch.long)
            elif isinstance(raw_embs, dict):
                embs = torch.stack([e.float() for _, e in raw_embs.items()])
                lbls = torch.tensor([int(l) for _, l in raw_lbls.items()], dtype=torch.long)
            else:
                embs = raw_embs.float()
                lbls = torch.tensor(raw_lbls, dtype=torch.long) if not isinstance(raw_lbls, torch.Tensor) else raw_lbls.long()
            # If dataloader is a Subset, filter to subset indices
            if hasattr(dataloader.dataset, 'indices'):
                idx = dataloader.dataset.indices
                embs, lbls = embs[idx], lbls[idx]
            all_feats = proj(embs.to(device)).cpu()
            all_lbls = lbls
            print(f"    Fast-loaded {all_feats.size(0)} embeddings from {emb_path}")
        else:
            # Slow fallback: iterate dataloader
            for batch in dataloader:
                llm_emb = batch["llm_embeddings"].to(device).float()
                feats = proj(llm_emb)
                all_feats.append(feats.cpu())
                all_lbls.append(batch["labels"])
            all_feats = torch.cat(all_feats, dim=0)
            all_lbls = torch.cat(all_lbls, dim=0)

        # Use same projected features for all experts as starting point
        features_per_expert = [all_feats.clone() for _ in range(self.n_experts)]
        self.occ.init_prototypes(features_per_expert, all_lbls)
