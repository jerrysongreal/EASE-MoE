"""
Empathy Response Expert for EASE-MoE v2.
Creator-Reader dual-tower with empathy divergence signal.

Creator side: encodes creator's cognitive intent + emotional manipulation.
  - Cognitive: semantic features → h_cre_cog (256)
  - Emotional: emotion_classifier(news_text) → distribution → h_cre_emo (256)

Reader side: encodes reader reactions from filtered comments.
  - Cognitive: mean-pooled RoBERTa embedding of top-5 comments → h_read_cog (256)
  - Emotional: mean(emotion_classifier(comment_j)) → h_read_emo (256)

Empathy divergence: g_cog = |h_cre_cog - h_read_cog|, g_emo = |h_cre_emo - h_read_emo|
Core hypothesis: fake news creators manipulate emotions → larger creator-reader gap.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class EmotionClassifier(nn.Module):
    """Frozen pre-trained emotion classifier (DistilRoBERTa, 7 Ekman emotions).
    Labels: anger, disgust, fear, joy, neutral, sadness, surprise.
    """
    EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

    def __init__(self, device: str = "cpu"):
        super().__init__()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.EMOTION_MODEL, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.EMOTION_MODEL, local_files_only=True)
        self.n_emotions = self.model.config.num_labels  # 7
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def to(self, device):
        self.model.to(device)
        return super().to(device)

    @torch.no_grad()
    def get_distribution(self, texts: List[str], device) -> torch.Tensor:
        """texts: list of N strings → (N, n_emotions) softmax distribution.
        Returns zero vector for empty list.
        """
        if not texts:
            return torch.zeros(1, self.n_emotions, device=device)
        # Filter out empty strings
        valid = [t for t in texts if t and len(t.strip()) > 0]
        if not valid:
            return torch.zeros(len(texts), self.n_emotions, device=device)
        tokens = self.tokenizer(valid, padding=True, truncation=True,
                                max_length=512, return_tensors='pt')
        tokens = {k: v.to(device) for k, v in tokens.items()}
        logits = self.model(**tokens).logits
        return F.softmax(logits, dim=-1)  # (N_valid, 7)


class CreatorEncoder(nn.Module):
    """Encode creator-side empathy signals from news RoBERTa [CLS] + text."""
    def __init__(self, roberta_dim: int = 768, hidden_dim: int = 256,
                 n_emotions: int = 7, dropout: float = 0.1):
        super().__init__()
        # Cognitive: h_cre_cog = MLP_cre-cog(RoBERTa_cre(t_i)[CLS]), input = 768-dim
        self.cognitive_encoder = nn.Sequential(
            nn.Linear(roberta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Emotional: h_cre_emo = MLP_cre-emo(EmotionCls(t_i)), input = n_emotions
        self.emotion_proj = nn.Sequential(
            nn.Linear(n_emotions, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, llm_emb: torch.Tensor,
                emotion_dist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            llm_emb: (B, 768) RoBERTa [CLS] embedding of news text
            emotion_dist: (B, n_emotions) EmotionCls(news_text) distribution
        Returns:
            h_cre_cog: (B, 256) creator cognitive intent
            h_cre_emo: (B, 256) creator emotional manipulation signal
        """
        h_cre_cog = self.cognitive_encoder(llm_emb)
        h_cre_emo = self.emotion_proj(emotion_dist)
        return h_cre_cog, h_cre_emo


class ReaderEncoder(nn.Module):
    """
    Encode reader-side reactions from comment embeddings and emotion signals.
    Cognitive: pre-computed RoBERTa mean-pool of top-5 comments (768d).
    Emotional: average emotion distribution of individual comments.
    """
    def __init__(self, roberta_dim: int = 768, hidden_dim: int = 256,
                 n_emotions: int = 7, dropout: float = 0.1):
        super().__init__()
        # Cognitive: from comment_emb (unchanged)
        self.cognitive_encoder = nn.Sequential(
            nn.Linear(roberta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Emotional: emotion distribution → projection
        self.emotion_proj = nn.Sequential(
            nn.Linear(n_emotions, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, comment_emb: torch.Tensor,
                emotion_dist_avg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            comment_emb: (B, 768) mean-pooled RoBERTa embedding of top-5 comments
            emotion_dist_avg: (B, n_emotions) average emotion distribution across comments
        Returns:
            h_read_cog: (B, 256) reader cognitive reaction
            h_read_emo: (B, 256) reader emotional reaction
        """
        h_read_cog = self.cognitive_encoder(comment_emb)
        h_read_emo = self.emotion_proj(emotion_dist_avg)
        return h_read_cog, h_read_emo


class EmpathyResponseExpert(nn.Module):
    """Full empathy expert: Creator + Reader + emotion classifier + 6-way fusion."""
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1,
                 emotion_model_path: str = None, device: str = "cpu"):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Emotion classifier (shared, frozen, applied to both creator & reader text)
        self.emotion_classifier = None
        self._n_emotions = 7
        if emotion_model_path is None:
            emotion_model_path = EmotionClassifier.EMOTION_MODEL
        try:
            self.emotion_classifier = EmotionClassifier(device=device)
            self._n_emotions = self.emotion_classifier.n_emotions
            print(f"  Emotion classifier loaded: {emotion_model_path} "
                  f"({self._n_emotions} emotions)")
        except Exception as e:
            print(f"  Emotion classifier unavailable ({e}) — using zero fallback")

        self.creator_encoder = CreatorEncoder(
            roberta_dim=768, hidden_dim=hidden_dim,
            n_emotions=self._n_emotions, dropout=dropout)
        self.reader_encoder = ReaderEncoder(
            roberta_dim=768, hidden_dim=hidden_dim,
            n_emotions=self._n_emotions, dropout=dropout)

        # 6-way fusion: [cre_cog, read_cog, g_cog, cre_emo, read_emo, g_emo]
        fusion_input_dim = hidden_dim * 6  # 1536
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def to(self, device):
        if self.emotion_classifier is not None:
            self.emotion_classifier.to(device)
        return super().to(device)

    def forward(self, llm_emb: torch.Tensor,
                h_vis: torch.Tensor,
                comment_emb: torch.Tensor,
                texts: list = None,
                comment_texts_list: list = None) -> torch.Tensor:
        """
        Args:
            llm_emb: (B, 768) RoBERTa [CLS] embedding of news text (for creator cognitive)
            h_vis: (B, 256) visual features (for fusion, not used in creator directly)
            comment_emb: (B, 768) offline pre-computed RoBERTa mean-pool of top-5 comments
            texts: list of B raw news text strings (for creator emotion classifier)
            comment_texts_list: list of B lists of raw comment strings (for reader emotion)
        Returns:
            h_emp: (B, 256)
        """
        B = llm_emb.size(0)
        device = llm_emb.device

        # ── Creator emotional: EmotionCls(news_text) → p_cre → h_cre_emo ──
        if self.emotion_classifier is not None and texts:
            creator_emo_dist = self.emotion_classifier.get_distribution(texts, device)
            if creator_emo_dist.size(0) == 1 and B > 1:
                creator_emo_dist = creator_emo_dist.expand(B, -1)
            elif creator_emo_dist.size(0) != B:
                creator_emo_dist = creator_emo_dist[:B]
                if creator_emo_dist.size(0) < B:
                    pad = torch.zeros(B - creator_emo_dist.size(0),
                                      self._n_emotions, device=device)
                    creator_emo_dist = torch.cat([creator_emo_dist, pad], dim=0)
        else:
            creator_emo_dist = torch.zeros(B, self._n_emotions, device=device)

        # h_cre_cog = MLP_cre-cog(RoBERTa(t_i)[CLS]),  h_cre_emo = MLP_cre-emo(p_cre)
        h_cre_cog, h_cre_emo = self.creator_encoder(llm_emb, creator_emo_dist)

        # ── Reader emotional: mean_j(EmotionCls(c_i^j)) → p_read → h_read_emo ──
        if self.emotion_classifier is not None and comment_texts_list:
            reader_emo_dists = []
            for comments in comment_texts_list:
                if comments and len(comments) > 0:
                    dists = self.emotion_classifier.get_distribution(comments, device)
                    reader_emo_dists.append(dists.mean(dim=0))  # (n_emotions,)
                else:
                    reader_emo_dists.append(
                        torch.zeros(self._n_emotions, device=device))
            reader_emo_dist = torch.stack(reader_emo_dists, dim=0)  # (B, n_emotions)
        else:
            reader_emo_dist = torch.zeros(B, self._n_emotions, device=device)

        # h_read_cog = MLP_read-cog(1/Ks Σ RoBERTa(c_i^j)), h_read_emo = MLP_read-emo(p_read)
        h_read_cog, h_read_emo = self.reader_encoder(comment_emb, reader_emo_dist)

        # ── Empathy divergence ──────────────────────────────────
        g_cog = torch.abs(h_cre_cog - h_read_cog)
        g_emo = torch.abs(h_cre_emo - h_read_emo)

        # ── 6-way fusion: [cre_cog; read_cog; cre_emo; read_emo; g_cog; g_emo] ──
        h_emp = self.fusion(torch.cat([
            h_cre_cog, h_read_cog, h_cre_emo, h_read_emo, g_cog, g_emo
        ], dim=-1))

        return h_emp
