"""
Visual-Multimodal Expert for EASE-MoE v2.
Two parallel channels:
  Channel 1 — Pure visual: Swin-Base → Self-Attention → h_vis_pooled
  Channel 2 — CLIP alignment: CLIP text+image → cosine_sim → h_clip
  Fusion: [h_vis_pooled; h_clip] → Linear → h_mul

CLIP is frozen (like RoBERTa). No image_mask hard blocking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class VisualMultimodalExpert(nn.Module):
    def __init__(self, swin_path: str = "microsoft/swin-base-patch4-window7-224",
                 hidden_dim: int = 256, clip_dim: int = 512,
                 num_heads: int = 8, dropout: float = 0.1,
                 freeze_backbone: bool = True, freeze_last_n: int = 1):
        super().__init__()
        # ── Channel 1: Swin visual encoder ────────────────────────
        from transformers import SwinModel
        self.swin = SwinModel.from_pretrained(swin_path, local_files_only=True)
        swin_dim = self.swin.config.hidden_size  # 1024 for swin-base

        self.vis_proj = nn.Linear(swin_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads,
                                                dropout=dropout, batch_first=True)
        self.norm_vis = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # ── Channel 2: CLIP text-image alignment ──────────────────
        # CLIP model loaded via open_clip (or transformers fallback).
        # Set externally via set_clip_model() before training.
        self.clip_model = None          # CLIP model with encode_image / encode_text
        self.clip_preprocess = None     # torchvision transform for CLIP images
        self.clip_tokenizer = None      # tokenizer callable: list[str] → tensor (B, 77)
        self.clip_dim = clip_dim
        self.clip_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ── Fusion ────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if freeze_backbone:
            self._freeze_layers(freeze_last_n)

    def _freeze_layers(self, unfreeze_last_n: int):
        for name, param in self.swin.named_parameters():
            param.requires_grad = False
            for i in range(unfreeze_last_n):
                if f"encoder.layers.{3 - i}" in name:
                    param.requires_grad = True
                    break

    def set_clip_model(self, clip_model, clip_preprocess, clip_tokenizer=None):
        """Set CLIP model (frozen). Call before training.

        Args:
            clip_model: model with encode_image(tensor) and encode_text(tensor) methods
                        (open_clip.CLIP or transformers.CLIPModel)
            clip_preprocess: torchvision transform for image preprocessing, or
                             transformers.CLIPProcessor
            clip_tokenizer: callable list[str] → tensor (B, 77), or None if
                            clip_preprocess is a CLIPProcessor
        """
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        for p in self.clip_model.parameters():
            p.requires_grad = False

    # ── Channel 1: Swin encoding ──────────────────────────────────

    def encode_visual(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Swin → projection → self-attention → pooled + sequence features."""
        outputs = self.swin(pixel_values=pixel_values)
        features = outputs.last_hidden_state              # (B, 49, 1024)
        features = self.vis_proj(features)                # (B, 49, 256)
        features = self.dropout(features)
        attended, _ = self.self_attn(features, features, features)
        features = self.norm_vis(features + attended)     # residual
        h_vis_pooled = features.mean(dim=1)               # (B, 256)
        return h_vis_pooled, features

    # ── Channel 2: CLIP alignment ─────────────────────────────────

    def compute_clip_alignment(self, texts, images, device) -> Optional[torch.Tensor]:
        """
        CLIP image-text cosine similarity → Linear(1→256).
        Supports both open_clip (primary) and transformers.CLIPModel (fallback).
        Returns None if CLIP is not configured or images/texts are missing.
        """
        if self.clip_model is None or self.clip_preprocess is None:
            return None
        if not texts or not images:
            return None

        # ── Detect backend ──────────────────────────────────────────
        from transformers import CLIPProcessor
        is_transformers_processor = isinstance(self.clip_preprocess, CLIPProcessor)

        with torch.no_grad():
            if is_transformers_processor:
                # ── transformers.CLIPModel path ──────────────────
                inputs = self.clip_preprocessor(
                    text=texts, images=images, return_tensors="pt",
                    padding=True, truncation=True, max_length=77
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                text_features = self.clip_model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"))
                image_features = self.clip_model.get_image_features(
                    pixel_values=inputs["pixel_values"])
            else:
                # ── open_clip path ──────────────────────────────
                # Preprocess images (torchvision transform)
                clip_images = torch.stack(
                    [self.clip_preprocess(img) for img in images]
                ).to(device)
                # Tokenize texts
                if self.clip_tokenizer is not None:
                    text_tokens = self.clip_tokenizer(texts).to(device)
                else:
                    text_tokens = self.clip_preprocess.tokenize(texts).to(device)
                text_features = self.clip_model.encode_text(text_tokens)
                image_features = self.clip_model.encode_image(clip_images)

            # ── Cosine similarity (common to both backends) ────────
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cos_sim = (text_features * image_features).sum(dim=-1, keepdim=True)  # (B, 1)

        return self.clip_proj(cos_sim)  # (B, 256)

    # ── Forward ───────────────────────────────────────────────────

    def forward(self, image_inputs: Dict[str, torch.Tensor],
                h_sem: Optional[torch.Tensor] = None,
                texts: list = None,
                images_pil: list = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            image_inputs: {'pixel_values': (B,3,224,224)} from Swin processor
            h_sem: (B,256) semantic features (unused in v2, kept for compatibility)
            texts: list of raw text strings (for CLIP)
            images_pil: list of PIL Images (for CLIP preprocessing)
        Returns:
            h_mul: (B, 256) fused visual-multimodal feature
        """
        device = image_inputs["pixel_values"].device

        # Channel 1: Swin
        h_vis_pooled, _ = self.encode_visual(image_inputs["pixel_values"])

        # Channel 2: CLIP
        h_clip = None
        if texts is not None and images_pil is not None:
            h_clip = self.compute_clip_alignment(texts, images_pil, device)

        # Fusion
        if h_clip is not None:
            h_mul = self.fusion(torch.cat([h_vis_pooled, h_clip], dim=-1))
        else:
            # No CLIP available → pad with zeros
            h_mul = self.fusion(torch.cat([h_vis_pooled, torch.zeros_like(h_vis_pooled)], dim=-1))

        return h_mul
