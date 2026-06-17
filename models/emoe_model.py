import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from transformers import RobertaTokenizer, RobertaModel

from models.vision_expert import SwinExpert
from models.tri_expert_fusion import TriExpertDisentangle, ExpertFusionLayer
from models.losses import CombinedLoss


class StructuralEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class LLMExpert(nn.Module):
    def __init__(
        self,
        roberta_path: str = "roberta-base",
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super(LLMExpert, self).__init__()

        self.roberta = RobertaModel.from_pretrained(roberta_path, local_files_only=True)
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_path, local_files_only=True)

        roberta_hidden = self.roberta.config.hidden_size

        self.text_proj = nn.Linear(roberta_hidden, hidden_dim)
        self.comments_proj = nn.Linear(roberta_hidden, hidden_dim)

        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout)

        self.classifier = nn.Linear(hidden_dim, 2)

        self.dropout = nn.Dropout(dropout)

        self.mean_pooling = nn.AdaptiveAvgPool1d(1)
        self.max_pooling = nn.AdaptiveMaxPool1d(1)

    def forward(
        self,
        text_input: Dict[str, torch.Tensor],
        comments_input: Dict[str, torch.Tensor] = None,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        text_output = self.roberta(**text_input)
        text_features = text_output.last_hidden_state
        text_features = self.text_proj(text_features)
        text_features = self.dropout(text_features)

        text_features = text_features.transpose(0, 1)
        text_features, _ = self.self_attention(text_features, text_features, text_features)
        text_features = text_features.transpose(0, 1)

        text_pooled = self.mean_pooling(text_features.transpose(1, 2)).squeeze(-1)

        if comments_input is not None:
            B, N, L = comments_input["input_ids"].shape
            flattened_input_ids = comments_input["input_ids"].view(B * N, L)
            flattened_attention_mask = comments_input["attention_mask"].view(B * N, L)

            comments_output = self.roberta(
                input_ids=flattened_input_ids,
                attention_mask=flattened_attention_mask
            )

            cls_tokens = comments_output.last_hidden_state[:, 0, :]
            cls_tokens = self.comments_proj(cls_tokens)

            comments_features = cls_tokens.view(B, N, -1)

            comments_features = comments_features.transpose(0, 1)
            comments_features, _ = self.self_attention(comments_features, comments_features, comments_features)
            comments_features = comments_features.transpose(0, 1)

            comments_pooled = self.max_pooling(comments_features.transpose(1, 2)).squeeze(-1)

            combined = (text_pooled + comments_pooled) / 2
        else:
            combined = text_pooled

        logits = self.classifier(combined)

        if return_features:
            return logits, combined
        return logits, text_features

    def encode_text(self, text: str, device: str = 'cpu') -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = self.roberta(**inputs)

        return outputs.last_hidden_state.mean(dim=1)

    def encode_comments(self, comments: list, device: str = 'cpu', max_length: int = 128) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            comments,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=max_length
        ).to(device)

        return encoded


class EMoEF(nn.Module):
    def __init__(
        self,
        dim_features: int,
        device: str,
        roberta_path: str = "roberta-base",
        swin_path: str = "microsoft/swin-base-patch4-window7-224",
        hidden_dim: int = 512,
        num_chunks: int = 4,
        alpha: float = 0.2,
        beta: float = 0.5,
        gamma: float = 0.3,
        delta: float = 0.1,
        eta: float = 0.05,
        dropout: float = 0.1
    ):
        super(EMoEF, self).__init__()

        self.device = device
        self.num_chunks = num_chunks
        self.hidden_dim = hidden_dim

        self.structural_encoder = StructuralEncoder(
            input_dim=dim_features,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(device)

        self.llm_expert = LLMExpert(
            roberta_path=roberta_path,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(device)

        self.vision_expert = SwinExpert(
            model_path=swin_path,
            hidden_dim=hidden_dim,
            dropout_rate=dropout
        ).to(device)

        self.llm_proj = nn.Linear(dim_features, hidden_dim).to(device)

        self.tri_disentangle = TriExpertDisentangle(
            hidden_dim=hidden_dim,
            num_chunks=num_chunks,
            dropout=dropout
        ).to(device)

        self.expert_fusion = ExpertFusionLayer(
            hidden_dim=hidden_dim,
            num_experts=3,
            dropout=dropout
        ).to(device)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        ).to(device)

        self.center_gnn = nn.Parameter(torch.empty(1, hidden_dim), requires_grad=True)
        nn.init.normal_(self.center_gnn, mean=0, std=0.1)
        self.center_llm = nn.Parameter(torch.empty(1, hidden_dim), requires_grad=True)
        nn.init.normal_(self.center_llm, mean=0, std=0.1)
        self.center_vis = nn.Parameter(torch.empty(1, hidden_dim), requires_grad=True)
        nn.init.normal_(self.center_vis, mean=0, std=0.1)
        self.center_gnn.data = self.center_gnn.data.to(device)
        self.center_llm.data = self.center_llm.data.to(device)
        self.center_vis.data = self.center_vis.data.to(device)

        self.combined_loss = CombinedLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            margin=3.0,
            hidden_dim=hidden_dim,
            temperature=0.1,
        )

        self.eta = eta

        self.pseudo_label_threshold = 0.95

        self.routing_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        ).to(device)
        nn.init.zeros_(self.routing_head[-1].bias)
        nn.init.xavier_uniform_(self.routing_head[-1].weight, gain=0.01)

    def _compute_routing(
        self,
        g_new: torch.Tensor,
        l_new: torch.Tensor,
        v_new: torch.Tensor
    ) -> torch.Tensor:
        concat = torch.cat([g_new, l_new, v_new], dim=-1)
        logits = self.routing_head(concat)
        weights = F.softmax(logits, dim=-1)
        return weights

    def forward(
        self,
        graph_data,
        llama_emb: torch.Tensor,
        text_input: Dict[str, torch.Tensor],
        comments_input: Dict[str, torch.Tensor],
        image_input: Dict[str, torch.Tensor],
        cognitive_comments_input: Dict[str, torch.Tensor] = None,
        emotional_scores: torch.Tensor = None,
        labels: torch.Tensor = None,
        return_feat: bool = False,
        has_real_image: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        struct_emb = self.structural_encoder(llama_emb.float().to(self.device))
        struct_emb = F.normalize(struct_emb, dim=1)

        llm_emb = self.llm_proj(llama_emb.float().to(self.device))
        llm_emb = F.normalize(llm_emb, dim=1)

        _, image_features = self.vision_expert(image_input['pixel_values'], return_features=True)
        if image_features.dim() == 3:
            image_pooled = image_features.mean(dim=1)
        else:
            image_pooled = image_features
        image_pooled = F.normalize(image_pooled, dim=1)

        g_new, l_new, v_new = self.tri_disentangle(struct_emb, llm_emb, image_pooled)

        routing_weights = self._compute_routing(g_new, l_new, v_new)

        if has_real_image is not None:
            no_img_mask = (~has_real_image.to(self.device)).float().unsqueeze(1)
            routing_weights = routing_weights.clone()
            routing_weights[:, 2] = routing_weights[:, 2] * (1 - no_img_mask.squeeze(1))
            routing_weights = routing_weights / (routing_weights.sum(dim=1, keepdim=True) + 1e-8)

        fused = self.expert_fusion([g_new, l_new, v_new], routing_weights)

        logits = self.classifier(fused)

        total_loss = torch.tensor(0.0, device=self.device)
        extra_info = {'loss_components': {}, 'routing_weights': {}}

        if routing_weights is not None:
            rw = routing_weights.detach().cpu()
            extra_info['routing_weights'] = {
                'gnn': rw[:, 0].mean().item(),
                'llm': rw[:, 1].mean().item(),
                'vis': rw[:, 2].mean().item()
            }

        if labels is not None:
            losses = self.combined_loss(
                feat_gnn=g_new,
                feat_llm=l_new,
                feat_vis=v_new,
                labels=labels,
                center_gnn=self.center_gnn,
                center_llm=self.center_llm,
                center_vis=self.center_vis
            )
            total_loss = losses['total']

            for key in ['info_nce', 'occ', 'covariance', 'empathy_ortho']:
                if key in losses:
                    extra_info['loss_components'][key] = losses[key].item()

            dist_gnn = torch.norm(g_new - self.center_gnn, p=2, dim=1)
            dist_llm = torch.norm(l_new - self.center_llm, p=2, dim=1)
            dist_vis = torch.norm(v_new - self.center_vis, p=2, dim=1)
            dists = torch.stack([dist_gnn, dist_llm, dist_vis], dim=1)
            target_weights = F.softmax(-dists / 0.5, dim=1)
            rl_policy = F.mse_loss(routing_weights, target_weights)
            total_loss = total_loss + self.eta * rl_policy
            extra_info['loss_components']['rl_policy'] = rl_policy.item()

        if return_feat:
            return logits, total_loss, fused, extra_info
        else:
            return logits, total_loss

    def generate_pseudo_labels(
        self,
        graph_data,
        llama_emb: torch.Tensor,
        text_input: Dict[str, torch.Tensor],
        image_input: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()

        with torch.no_grad():
            struct_emb = self.structural_encoder(llama_emb.float().to(self.device))
            struct_emb = F.normalize(struct_emb, dim=1)

            llm_emb = self.llm_proj(llama_emb.float().to(self.device))
            llm_emb = F.normalize(llm_emb, dim=1)

            _, image_features = self.vision_expert(image_input['pixel_values'], return_features=True)
            if image_features.dim() == 3:
                image_pooled = image_features.mean(dim=1)
            else:
                image_pooled = image_features
            image_pooled = F.normalize(image_pooled, dim=1)

            g_new, l_new, v_new = self.tri_disentangle(struct_emb, llm_emb, image_pooled)

            confidence = self._compute_confidence(g_new, l_new, v_new)

            pseudo_labels = (confidence > self.pseudo_label_threshold).long()

        return pseudo_labels, confidence

    def _compute_confidence(
        self,
        g_new: torch.Tensor,
        l_new: torch.Tensor,
        v_new: torch.Tensor
    ) -> torch.Tensor:
        dist_g = torch.norm(g_new - self.center_gnn, p=2, dim=1)
        dist_l = torch.norm(l_new - self.center_llm, p=2, dim=1)
        dist_v = torch.norm(v_new - self.center_vis, p=2, dim=1)
        avg_dist = (dist_g + dist_l + dist_v) / 3
        confidence = 1 / (1 + avg_dist)
        return confidence

    def compute_center(self, dataloader, device: str):
        self.eval()
        all_g, all_l, all_v = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                labels = batch['labels'].to(device)
                real_mask = (labels == 1)
                if not real_mask.any():
                    continue

                llama_emb = batch['llm_embeddings'].to(device).float()
                image_input = {k: v.to(device) for k, v in batch['images_processed'].items()}

                struct_emb = self.structural_encoder(llama_emb)
                struct_emb = F.normalize(struct_emb, dim=1)

                llm_emb = self.llm_proj(llama_emb)
                llm_emb = F.normalize(llm_emb, dim=1)

                _, image_features = self.vision_expert(image_input['pixel_values'], return_features=True)
                if image_features.dim() == 3:
                    image_pooled = image_features.mean(dim=1)
                else:
                    image_pooled = image_features
                image_pooled = F.normalize(image_pooled, dim=1)

                g_new, l_new, v_new = self.tri_disentangle(struct_emb, llm_emb, image_pooled)

                all_g.append(g_new[real_mask])
                all_l.append(l_new[real_mask])
                all_v.append(v_new[real_mask])

        if all_g:
            all_g = torch.cat(all_g, dim=0)
            all_l = torch.cat(all_l, dim=0)
            all_v = torch.cat(all_v, dim=0)
            self.center_gnn.data = all_g.mean(dim=0, keepdim=True)
            self.center_llm.data = all_l.mean(dim=0, keepdim=True)
            self.center_vis.data = all_v.mean(dim=0, keepdim=True)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EMoEF(
        dim_features=768,
        device=device,
        hidden_dim=512
    )

    print("EASE-MoE model initialized successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
