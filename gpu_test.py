import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

from models.emoe_model import EMoEF
from database.multimodal_loader import MultimodalFakeNewsDataset, collate_fn
from functions.visualize import TrainingVisualizer
from torch.utils.data import Subset, DataLoader

ROBERTA_PATH = 'roberta-base'
SWIN_PATH = 'microsoft/swin-base-patch4-window7-224'
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'gossipcop')


def main():
    random.seed(42)
    np.random.seed(seed=42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Device: {device}")

    print("\nLoading dataset...")
    dataset = MultimodalFakeNewsDataset(
        json_file=os.path.join(DATA_DIR, 'gossipcop_clean.json'),
        image_folder=os.path.join(DATA_DIR, 'images'),
        empathy_comments_csv=os.path.join(DATA_DIR, 'empathy_clean.csv'),
        llm_embedding_path=os.path.join(DATA_DIR, 'llm_embeddings_clean.pt'),
        tokenizer_path=ROBERTA_PATH,
        image_processor_path=SWIN_PATH,
        device=str(device)
    )

    train_dataset = Subset(dataset, list(range(0, 16)))
    test_dataset = Subset(dataset, list(range(16, 24)))

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, dataset.tokenizer, dataset.image_processor, str(device))
    )
    test_loader = DataLoader(
        test_dataset, batch_size=2, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, dataset.tokenizer, dataset.image_processor, str(device))
    )

    print(f"Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples, Batch size: 2")

    print("Initializing model on GPU...")
    model = EMoEF(
        dim_features=768,
        device=str(device),
        roberta_path=ROBERTA_PATH,
        swin_path=SWIN_PATH,
        hidden_dim=512,
        alpha=0.2, beta=0.5, gamma=0.3, delta=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1024**2
        print(f"GPU memory used: {alloc:.0f} MB")

    visualizer = TrainingVisualizer(log_dir="logs", experiment_name="gpu-test")
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=5e-5)
    ce_loss_fn = nn.CrossEntropyLoss()

    num_epochs = 3
    print(f"\n{'='*60}")
    print(f"GPU Training Test - {num_epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            graph_data = batch['graph_data'].to(device)
            llm_emb = batch['llm_embeddings'].to(device).float()
            text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
            comments_inputs = {k: v.to(device) for k, v in batch['comments_inputs'].items()}
            cognitive_inputs = {k: v.to(device) for k, v in batch['cognitive_inputs'].items()}
            emotional_scores = batch['emotional_scores'].to(device)
            images_processed = {k: v.to(device) for k, v in batch['images_processed'].items()}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits, total_loss, _, _ = model(
                graph_data=graph_data, llama_emb=llm_emb,
                text_input=text_inputs, comments_input=comments_inputs,
                image_input=images_processed,
                cognitive_comments_input=cognitive_inputs,
                emotional_scores=emotional_scores,
                labels=labels, return_feat=True
            )
            cls_loss = ce_loss_fn(logits, labels)
            combined_loss = total_loss + 0.1 * cls_loss
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += combined_loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        avg_loss = running_loss / len(train_loader)

        model.eval()
        test_preds, test_labels, test_probs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                graph_data = batch['graph_data'].to(device)
                llm_emb = batch['llm_embeddings'].to(device).float()
                text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
                comments_inputs = {k: v.to(device) for k, v in batch['comments_inputs'].items()}
                cognitive_inputs = {k: v.to(device) for k, v in batch['cognitive_inputs'].items()}
                emotional_scores = batch['emotional_scores'].to(device)
                images_processed = {k: v.to(device) for k, v in batch['images_processed'].items()}
                labels = batch['labels'].to(device)

                logits, _, _, _ = model(
                    graph_data=graph_data, llama_emb=llm_emb,
                    text_input=text_inputs, comments_input=comments_inputs,
                    image_input=images_processed,
                    cognitive_comments_input=cognitive_inputs,
                    emotional_scores=emotional_scores,
                    return_feat=True
                )
                probs = torch.softmax(logits, dim=1)
                test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                test_probs.extend(probs[:, 1].cpu().numpy())

        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)

        visualizer.log_epoch(
            epoch=epoch + 1, train_loss=avg_loss,
            train_acc=train_acc, val_acc=test_acc, val_f1=test_f1,
            lr=optimizer.param_groups[0]['lr'], stage=2
        )

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1024**2
            vram_info = f" | VRAM: {alloc:.0f} MB"
        else:
            vram_info = ""

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
              f"Test F1: {test_f1:.4f}{vram_info}")

    plot_files = visualizer.generate_all_plots()
    visualizer.save_log()

    print(f"\n{'='*60}")
    print("GPU Training Test PASSED!")
    print(f"{'='*60}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Final Test F1: {test_f1:.4f}")
    print(f"\nVisualization plots saved:")
    for name, path in plot_files.items():
        print(f"  {name}: {path}")
    print(f"\nTraining log: logs/gpu-test_log.json")


if __name__ == "__main__":
    main()
