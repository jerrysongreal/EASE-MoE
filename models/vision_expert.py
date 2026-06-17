import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, SwinModel
from typing import Optional, Tuple


class SwinExpert(nn.Module):
    def __init__(
        self,
        model_path: str = "microsoft/swin-base-patch4-window7-224",
        hidden_dim: int = 512,
        dropout_rate: float = 0.1,
        freeze_layers: bool = True,
        num_classes: int = 2
    ):
        super(SwinExpert, self).__init__()
        
        self.swin_model = SwinModel.from_pretrained(model_path, local_files_only=True)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        
        swin_output_size = self.swin_model.config.hidden_size
        
        self.proj = nn.Linear(swin_output_size, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self.max_pooling = nn.AdaptiveMaxPool1d(1)
        self.mean_pooling = nn.AdaptiveAvgPool1d(1)
        
        if freeze_layers:
            self._freeze_swin_layers()
    
    def _freeze_swin_layers(self):
        swin_layers = []
        for module in self.swin_model.modules():
            if module.__class__.__name__ == 'SwinLayer':
                swin_layers.append(module)
        
        if swin_layers:
            for layer in swin_layers[:-1]:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        outputs = self.swin_model(pixel_values=pixel_values)
        features = outputs.last_hidden_state
        
        features = self.proj(features)
        features = self.dropout(features)
        
        features = features.transpose(0, 1)
        attn_output, _ = self.self_attention(features, features, features)
        features = attn_output.transpose(0, 1)
        
        pooled = self.mean_pooling(features.transpose(1, 2)).squeeze(-1)
        
        logits = self.classifier(pooled)
        
        if return_features:
            return logits, pooled
        return logits, features
    
    def get_visual_anomaly_score(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)
        
        image_pooled = self.mean_pooling(image_features.transpose(1, 2)).squeeze(-1)
        text_pooled = text_features.mean(dim=1) if text_features.dim() == 3 else text_features
        
        text_pooled = F.normalize(text_pooled, dim=-1)
        image_pooled = F.normalize(image_pooled, dim=-1)
        
        similarity = F.cosine_similarity(text_pooled, image_pooled, dim=-1)
        anomaly_score = 1 - similarity
        
        return anomaly_score


class VisionFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_path: str = "microsoft/swin-base-patch4-window7-224",
        hidden_dim: int = 512
    ):
        super(VisionFeatureExtractor, self).__init__()
        
        self.swin_model = SwinModel.from_pretrained(model_path, local_files_only=True)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        
        swin_output_size = self.swin_model.config.hidden_size
        self.proj = nn.Linear(swin_output_size, hidden_dim)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.swin_model(pixel_values=pixel_values)
        features = outputs.last_hidden_state
        features = self.proj(features)
        return features
    
    def process_images(self, images: list) -> torch.Tensor:
        processed = self.image_processor(images=images, return_tensors='pt')
        return processed['pixel_values']


if __name__ == "__main__":
    expert = SwinExpert(hidden_dim=512, freeze_layers=True)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits, features = expert(dummy_input, return_features=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
