"""
Multimodal data loader for EASE-MoE v2.
Changes from v1:
  - No has_real_image / image_mask (router learns modality reliability autonomously)
  - Comments trimmed to top-5 (after quality filtering)
  - New: comment_embeddings — offline pre-computed RoBERTa mean-pool of top-5 comments
  - Propagation graphs loaded from pre-built .pt files
"""
import os, json
import torch
import numpy as np
import pandas as pd
from functools import partial
from PIL import Image
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, coalesce
from transformers import AutoTokenizer, AutoImageProcessor
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

MIN_IMAGE_SIZE = 50 * 1024
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')


class MultimodalFakeNewsDataset(Dataset):
    def __init__(
        self,
        json_file: str,
        image_folder: str,
        empathy_comments_csv: str = None,
        tokenizer_path: str = "roberta-base",
        image_processor_path: str = "microsoft/swin-base-patch4-window7-224",
        llm_embedding_path: str = None,
        comment_embedding_path: str = None,
        ablation_comment_embedding_path: str = None,
        llm_emb_dim: int = 768,
        max_text_length: int = 512,
        max_comment_length: int = 128,
        max_comments: int = 5,
        device: str = 'cpu'
    ):
        self.image_folder = image_folder
        self.max_text_length = max_text_length
        self.max_comment_length = max_comment_length
        self.max_comments = max_comments
        self.device = device
        self.llm_emb_dim = llm_emb_dim

        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        print(f"Loaded {len(raw_data)} samples from {json_file}")

        self.data = []
        no_image_count = 0
        small_image_count = 0
        image_paths = {}

        for item in raw_data:
            data_id = str(item.get('id', ''))
            img_path = self._find_image_path(data_id)

            if img_path is None:
                no_image_count += 1
                image_paths[data_id] = None
            elif os.path.getsize(img_path) < MIN_IMAGE_SIZE:
                small_image_count += 1
                image_paths[data_id] = None
            else:
                image_paths[data_id] = img_path

            self.data.append(item)

        print(f"  Real images={len(self.data) - no_image_count - small_image_count}, "
              f"missing={no_image_count}, too_small={small_image_count} (using black placeholder)")
        print(f"  Total: {len(self.data)} samples")

        self._image_paths = image_paths

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_path, local_files_only=True)

        self.empathy_comments = {}
        if empathy_comments_csv and os.path.exists(empathy_comments_csv):
            self._load_empathy_comments(empathy_comments_csv)

        self.llm_embedding_path = llm_embedding_path
        self.llm_embeddings = {}
        if llm_embedding_path and os.path.exists(llm_embedding_path):
            self._load_llm_embeddings(llm_embedding_path)

        # Optional: offline pre-computed comment embeddings (mean-pooled top-5)
        self.comment_embeddings = {}
        if comment_embedding_path and os.path.exists(comment_embedding_path):
            self._load_comment_embeddings(comment_embedding_path)

        # Optional: offline pre-computed ablation comment embeddings (top 6-10)
        self.ablation_comment_embeddings = {}
        if ablation_comment_embedding_path and os.path.exists(ablation_comment_embedding_path):
            self._load_ablation_comment_embeddings(ablation_comment_embedding_path)

        self.user_ids = [f"user_{i}" for i in range(len(self.data))]
        self.post_ids = [item.get('id', f"post_{i}") for i, item in enumerate(self.data)]

        # Load pre-built propagation graphs
        self.propagation_graphs = {}
        graph_candidates = [
            os.path.join(os.path.dirname(json_file), "..", "graphs", "propagation_graphs_train.pt"),
            os.path.join(os.path.dirname(json_file), "..", "propagation_graphs_train.pt"),
        ]
        for gp in graph_candidates:
            if os.path.exists(gp):
                self.propagation_graphs = torch.load(gp, weights_only=False)
                print(f"Loaded {len(self.propagation_graphs)} pre-built propagation graphs from {gp}")
                break

    def _find_image_path(self, data_id: str) -> Optional[str]:
        for ext in IMAGE_EXTENSIONS:
            candidate = os.path.join(self.image_folder, data_id + ext)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _load_empathy_comments(self, csv_path: str):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            data_id = str(row['id'])
            cognitive = str(row.get('cognitive_comments', '')).split('\n')
            emotional = str(row.get('emotional_comments', '')).split('\n')
            sentiment_scores = str(row.get('sentiment_scores', '0.5')).split(',')
            self.empathy_comments[data_id] = {
                'cognitive': [c.strip() for c in cognitive if c.strip()][:self.max_comments],
                'emotional': [e.strip() for e in emotional if e.strip()][:self.max_comments],
                'sentiment_scores': [float(s.strip()) for s in sentiment_scores if s.strip()][:self.max_comments]
            }
        print(f"Loaded empathy comments for {len(self.empathy_comments)} samples")

    def _load_llm_embeddings(self, pt_path: str):
        data = torch.load(pt_path, weights_only=True)
        for emb, label, fname in zip(data["embeddings"], data["labels"], data["filenames"]):
            self.llm_embeddings[fname] = (emb.float(), int(label))
        print(f"Loaded LLM embeddings for {len(self.llm_embeddings)} samples")

    def _load_comment_embeddings(self, pt_path: str):
        """Load offline pre-computed mean-pooled RoBERTa embeddings of top-5 comments."""
        data = torch.load(pt_path, weights_only=True)
        for emb, fname in zip(data["embeddings"], data["filenames"]):
            self.comment_embeddings[fname] = emb.float()
        print(f"Loaded comment embeddings for {len(self.comment_embeddings)} samples")

    def _load_ablation_comment_embeddings(self, pt_path: str):
        """Load offline pre-computed mean-pooled RoBERTa embeddings of top 6-10 comments."""
        data = torch.load(pt_path, weights_only=True)
        for emb, fname in zip(data["embeddings"], data["filenames"]):
            self.ablation_comment_embeddings[fname] = emb.float()
        print(f"Loaded ablation comment embeddings for {len(self.ablation_comment_embeddings)} samples")

    def __len__(self):
        return len(self.data)

    def _create_graph_data(self, text_feature: torch.Tensor, label: int,
                           user_id: str, post_id: str, data_id: str = None) -> Data:
        # Use pre-built propagation graph if available
        if data_id and data_id in self.propagation_graphs:
            pg = self.propagation_graphs[data_id]
            if isinstance(pg, dict) and "edge_index" in pg:
                num_nodes = pg.get("num_nodes", 1)
                x = pg.get("x", torch.zeros(num_nodes, text_feature.size(-1)))
                if x.size(0) != num_nodes:
                    x = torch.zeros(num_nodes, text_feature.size(-1))
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                graph_data = Data(
                    x=x.float(), edge_index=pg["edge_index"].long(),
                    y=torch.tensor([label], dtype=torch.long),
                    filename=f"{post_id}.json")
                graph_data.post_id = post_id
                graph_data.user_id = user_id
                return graph_data

        # Fallback: simple post-user graph (v2 treats this as a valid tree for GCN)
        node_ids = [post_id, user_id]
        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        num_nodes = len(node_ids)
        edge_index = torch.tensor([
            [node_id_to_idx[post_id], node_id_to_idx[user_id]],
            [node_id_to_idx[user_id], node_id_to_idx[post_id]]
        ], dtype=torch.long).t().contiguous()
        if edge_index.dim() == 1:
            edge_index = edge_index.view(2, -1)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_index, _ = coalesce(edge_index, num_nodes=num_nodes)
        if edge_index.dim() == 1:
            edge_index = edge_index.view(2, -1)
        node_features = torch.cat([text_feature, text_feature], dim=0)
        graph_data = Data(
            x=node_features, edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
            filename=f"{post_id}.json")
        graph_data.post_id = post_id
        graph_data.user_id = user_id
        return graph_data

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        data_id = str(item.get('id', f"post_{idx}"))
        text = item.get('text', '')
        label = item.get('label', 0)

        comments = item.get('comments', [])
        if isinstance(comments, str):
            comments = [c.strip() for c in comments.split("\n") if c.strip()]
        comments = [c.replace('@user', '').strip() for c in comments[:self.max_comments]]

        empathy_data = self.empathy_comments.get(data_id, {})
        cognitive_comments = empathy_data.get('cognitive', [])
        emotional_comments = empathy_data.get('emotional', [])
        sentiment_scores = empathy_data.get('sentiment_scores', [0.5] * len(emotional_comments))

        if data_id in self.llm_embeddings:
            llm_emb, _ = self.llm_embeddings[data_id]
        else:
            llm_emb = torch.zeros(self.llm_emb_dim)

        if data_id in self.comment_embeddings:
            comment_emb = self.comment_embeddings[data_id]
        else:
            comment_emb = torch.zeros(self.llm_emb_dim)

        if data_id in self.ablation_comment_embeddings:
            ablation_comment_emb = self.ablation_comment_embeddings[data_id]
        else:
            ablation_comment_emb = torch.zeros(self.llm_emb_dim)

        user_id = self.user_ids[idx] if idx < len(self.user_ids) else f"user_{idx}"
        post_id = self.post_ids[idx] if idx < len(self.post_ids) else f"post_{idx}"

        graph_data = self._create_graph_data(llm_emb.unsqueeze(0), label, user_id, post_id, data_id)

        image_path = self._image_paths.get(data_id)
        black_img = Image.new('RGB', (224, 224), color=(0, 0, 0))
        try:
            image = Image.open(image_path).convert('RGB') if image_path else black_img
        except Exception:
            image = black_img

        return {
            'id': data_id,
            'text': text,
            'comments': comments,
            'cognitive_comments': cognitive_comments,
            'emotional_comments': emotional_comments,
            'sentiment_scores': sentiment_scores,
            'image': image,
            'label': label,
            'llm_embedding': llm_emb,
            'comment_embedding': comment_emb,
            'ablation_comment_embedding': ablation_comment_emb,
            'graph_data': graph_data,
        }


def collate_fn(batch: List[Dict], tokenizer, image_processor, device: str = 'cpu') -> Dict:
    data_ids = [item['id'] for item in batch]
    texts = [item['text'] for item in batch]
    comments_list = [item['comments'] for item in batch]
    cognitive_comments_list = [item['cognitive_comments'] for item in batch]
    emotional_comments_list = [item['emotional_comments'] for item in batch]
    sentiment_scores_list = [item['sentiment_scores'] for item in batch]
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    llm_embeddings = [item['llm_embedding'] for item in batch]
    comment_embeddings = [item['comment_embedding'] for item in batch]
    ablation_comment_embeddings = [item['ablation_comment_embedding'] for item in batch]
    graph_data_list = [item['graph_data'] for item in batch]

    text_inputs = tokenizer(texts, padding=True, truncation=True,
                            return_tensors='pt', max_length=512)

    # Tokenize comments
    max_comments = max((len(c) for c in comments_list), default=0)
    max_comments = max(max_comments, 1)
    max_comments = min(max_comments, 5)
    batch_input_ids, batch_attention_mask = [], []
    for comments in comments_list:
        sample_ids, sample_mask = [], []
        for comment in comments[:max_comments]:
            tokens = tokenizer(comment, padding='max_length', truncation=True,
                              max_length=128, return_tensors='pt')
            sample_ids.append(tokens['input_ids'].squeeze(0))
            sample_mask.append(tokens['attention_mask'].squeeze(0))
        while len(sample_ids) < max_comments:
            sample_ids.append(torch.zeros(128, dtype=torch.long))
            sample_mask.append(torch.zeros(128, dtype=torch.long))
        batch_input_ids.append(torch.stack(sample_ids, dim=0))
        batch_attention_mask.append(torch.stack(sample_mask, dim=0))
    comments_inputs = {
        'input_ids': torch.stack(batch_input_ids, dim=0),
        'attention_mask': torch.stack(batch_attention_mask, dim=0)
    }

    # Tokenize cognitive comments
    max_cog = max((len(c) for c in cognitive_comments_list), default=0)
    max_cog = max(max_cog, 1)
    cog_ids, cog_mask = [], []
    for comments in cognitive_comments_list:
        sample_ids, sample_mask = [], []
        for comment in comments[:max_cog]:
            tokens = tokenizer(comment, padding='max_length', truncation=True,
                              max_length=128, return_tensors='pt')
            sample_ids.append(tokens['input_ids'].squeeze(0))
            sample_mask.append(tokens['attention_mask'].squeeze(0))
        while len(sample_ids) < max_cog:
            sample_ids.append(torch.zeros(128, dtype=torch.long))
            sample_mask.append(torch.zeros(128, dtype=torch.long))
        cog_ids.append(torch.stack(sample_ids, dim=0))
        cog_mask.append(torch.stack(sample_mask, dim=0))
    cognitive_inputs = {
        'input_ids': torch.stack(cog_ids, dim=0),
        'attention_mask': torch.stack(cog_mask, dim=0)
    }

    # Emotional scores → mean per sample
    emotional_scores = torch.tensor([
        sum(s) / len(s) if s else 0.5 for s in sentiment_scores_list
    ], dtype=torch.float)

    images_processed = image_processor(images=images, return_tensors='pt')
    labels = torch.tensor(labels, dtype=torch.long)
    llm_embeddings = torch.stack(llm_embeddings, dim=0)
    comment_embeddings_stacked = torch.stack(comment_embeddings, dim=0)
    graph_batch = Batch.from_data_list(graph_data_list)

    return {
        'data_ids': data_ids,
        'text_inputs': text_inputs,
        'comments_inputs': comments_inputs,
        'comments_raw': comments_list,
        'cognitive_comments_raw': cognitive_comments_list,
        'emotional_comments_raw': emotional_comments_list,
        'cognitive_inputs': cognitive_inputs,
        'emotional_scores': emotional_scores,
        'images_processed': images_processed,
        'labels': labels,
        'llm_embeddings': llm_embeddings,
        'comment_embeddings': comment_embeddings_stacked,
        'ablation_comment_embeddings': torch.stack(ablation_comment_embeddings, dim=0),
        'graph_data': graph_batch,
        'images_pil': images,
        'texts_raw': texts,
    }


def mixup_embeddings(batch, alpha=0.2):
    """Mixup at embedding level."""
    B = batch["llm_embeddings"].size(0)
    if B < 2:
        return batch
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(B)
    batch["llm_embeddings"] = lam * batch["llm_embeddings"] + (1 - lam) * batch["llm_embeddings"][idx]
    batch["comment_embeddings"] = lam * batch["comment_embeddings"] + (1 - lam) * batch["comment_embeddings"][idx]
    batch["images_processed"]["pixel_values"] = (
        lam * batch["images_processed"]["pixel_values"] +
        (1 - lam) * batch["images_processed"]["pixel_values"][idx])
    batch["emotional_scores"] = lam * batch["emotional_scores"] + (1 - lam) * batch["emotional_scores"][idx]
    batch["labels"] = batch["labels"].float()
    batch["labels"] = lam * batch["labels"] + (1 - lam) * batch["labels"][idx]
    batch["mixup_lambda"] = lam
    return batch


# ── Split utilities ──────────────────────────────────────────────────

def save_split_indices(save_path, train_indices, val_indices, test_indices,
                       test_size=0.2, val_size=0.25, random_state=42, split_hash=None):
    split_info = {
        'train_indices': [int(i) for i in train_indices],
        'val_indices': [int(i) for i in val_indices],
        'test_indices': [int(i) for i in test_indices],
        'config': {'test_size': test_size, 'val_size': val_size, 'random_state': random_state},
        'split_hash': split_hash
    }
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    print(f"Split indices saved to {save_path}")


def load_split_indices(split_path):
    with open(split_path, 'r', encoding='utf-8') as f:
        split_info = json.load(f)
    print(f"Split indices loaded from {split_path}")
    return split_info


def create_split_loaders(dataset, batch_size=32, val_size=0.2, random_state=42,
                         device='cpu', num_workers=4):
    """Split a dataset into train/val and return DataLoaders."""
    from torch.utils.data import Subset
    tokenizer = dataset.tokenizer
    image_processor = dataset.image_processor
    total_size = len(dataset)
    indices = list(range(total_size))
    all_labels = [item.get('label', 0) for item in dataset.data]
    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, random_state=random_state, stratify=all_labels)
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    collate = partial(collate_fn, tokenizer=tokenizer, image_processor=image_processor, device=device)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, collate_fn=collate)
    return train_loader, val_loader
