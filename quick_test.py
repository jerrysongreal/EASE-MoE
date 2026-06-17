import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("EASE-MoE Quick Test Script")
print("=" * 60)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'gossipcop')
ROBERTA_PATH = 'roberta-base'
SWIN_PATH = 'microsoft/swin-base-patch4-window7-224'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Device: {DEVICE}")


def test_data_loading():
    print("\n[1/4] Testing data loading...")
    from database.multimodal_loader import MultimodalFakeNewsDataset

    json_path = os.path.join(DATA_DIR, 'gossipcop_clean.json')
    image_folder = os.path.join(DATA_DIR, 'images')
    csv_path = os.path.join(DATA_DIR, 'empathy_clean.csv')
    emb_path = os.path.join(DATA_DIR, 'llm_embeddings_clean.pt')

    try:
        dataset = MultimodalFakeNewsDataset(
            json_file=json_path,
            image_folder=image_folder,
            empathy_comments_csv=csv_path,
            llm_embedding_path=emb_path,
            tokenizer_path=ROBERTA_PATH,
            image_processor_path=SWIN_PATH,
            device=str(DEVICE)
        )
        print(f"  Dataset loaded: {len(dataset)} samples")

        sample = dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Sample ID: {sample['id']}")
        print(f"  Label: {sample['label']}")
        print(f"  Text length: {len(sample['text'])}")
        print(f"  Comments count: {len(sample['comments'])}")
        print(f"  Cognitive comments: {len(sample['cognitive_comments'])}")
        print(f"  Emotional comments: {len(sample['emotional_comments'])}")
        print(f"  Sentiment scores: {sample['sentiment_scores']}")
        print(f"  LLM embedding shape: {sample['llm_embedding'].shape}")
        print(f"  Image size: {sample['image'].size}")
        print(f"  Graph data: {sample['graph_data']}")
        print(f"  Graph x shape: {sample['graph_data'].x.shape}")
        print(f"  Graph edge_index shape: {sample['graph_data'].edge_index.shape}")
        print("  [PASS] Data loading successful!")
        return dataset
    except Exception as e:
        print(f"  [FAIL] Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_dataloader(dataset):
    print("\n[2/4] Testing DataLoader with collate_fn...")
    from database.multimodal_loader import collate_fn

    if dataset is None:
        print("  [SKIP] No dataset available")
        return None

    try:
        subset_indices = list(range(8))
        from torch.utils.data import Subset
        subset = Subset(dataset, subset_indices)

        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, dataset.tokenizer, dataset.image_processor, str(DEVICE))
        )

        batch = next(iter(loader))
        print(f"  Batch keys: {list(batch.keys())}")
        print(f"  Data IDs: {batch['data_ids']}")
        print(f"  Text input_ids shape: {batch['text_inputs']['input_ids'].shape}")
        print(f"  Text attention_mask shape: {batch['text_inputs']['attention_mask'].shape}")
        print(f"  Comments input_ids shape: {batch['comments_inputs']['input_ids'].shape}")
        print(f"  Cognitive input_ids shape: {batch['cognitive_inputs']['input_ids'].shape}")
        print(f"  Emotional scores shape: {batch['emotional_scores'].shape}")
        print(f"  Image pixel_values shape: {batch['images_processed']['pixel_values'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  Labels: {batch['labels'].tolist()}")
        print(f"  LLM embeddings shape: {batch['llm_embeddings'].shape}")
        print(f"  Graph batch: {batch['graph_data']}")
        print(f"  Graph batch x shape: {batch['graph_data'].x.shape}")
        print(f"  Graph batch edge_index shape: {batch['graph_data'].edge_index.shape}")
        print(f"  Graph batch batch shape: {batch['graph_data'].batch.shape}")
        print("  [PASS] DataLoader successful!")
        return batch
    except Exception as e:
        print(f"  [FAIL] DataLoader failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_init():
    print("\n[3/4] Testing model initialization...")
    try:
        print(f"  Using device: {DEVICE}")

        from models.emoe_model import EMoEF
        model = EMoEF(
            dim_features=768,
            device=str(DEVICE),
            roberta_path=ROBERTA_PATH,
            swin_path=SWIN_PATH,
            hidden_dim=512,
            alpha=0.2,
            beta=0.5,
            gamma=0.3,
            delta=0.1
        ).to(DEVICE)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print("  [PASS] Model initialization successful!")
        return model, DEVICE
    except Exception as e:
        print(f"  [FAIL] Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_forward_pass(model, device, batch):
    print("\n[4/4] Testing forward pass...")
    if model is None or batch is None:
        print("  [SKIP] No model or batch available")
        return

    try:
        model.eval()
        with torch.no_grad():
            graph_data = batch['graph_data'].to(device)
            llm_emb = batch['llm_embeddings'].to(device).float()
            text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
            comments_inputs = {k: v.to(device) for k, v in batch['comments_inputs'].items()}
            cognitive_inputs = {k: v.to(device) for k, v in batch['cognitive_inputs'].items()}
            emotional_scores = batch['emotional_scores'].to(device)
            images_processed = {k: v.to(device) for k, v in batch['images_processed'].items()}
            labels = batch['labels'].to(device)

            logits, total_loss, features, _ = model(
                graph_data=graph_data,
                llama_emb=llm_emb,
                text_input=text_inputs,
                comments_input=comments_inputs,
                image_input=images_processed,
                cognitive_comments_input=cognitive_inputs,
                emotional_scores=emotional_scores,
                labels=labels,
                return_feat=True
            )

            print(f"  Logits shape: {logits.shape}")
            print(f"  Logits: {logits}")
            print(f"  Total loss: {total_loss.item():.4f}")
            print(f"  Features shape: {features.shape}")

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            print(f"  Predictions: {preds.tolist()}")
            print(f"  Ground truth: {labels.tolist()}")
            print(f"  Probabilities (Fake/Real):")
            for j, (p, l) in enumerate(zip(probs, labels)):
                print(f"    Sample {j}: Fake={p[0]:.3f}, Real={p[1]:.3f} | GT={'Real' if l==1 else 'Fake'}")

        print("  [PASS] Forward pass successful!")
    except Exception as e:
        print(f"  [FAIL] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    dataset = test_data_loading()
    batch = test_dataloader(dataset)
    model, device = test_model_init()
    test_forward_pass(model, device, batch)

    print("\n" + "=" * 60)
    print("Quick test complete!")
    print("=" * 60)
