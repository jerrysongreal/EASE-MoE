import os
import argparse
import numpy as np
import torch
from functools import partial
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from models.emoe_model import EMoEF
from database.multimodal_loader import (
    MultimodalFakeNewsDataset, create_dataloaders, load_split_indices, collate_fn
)
from torch.utils.data import DataLoader


def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            graph_data = batch['graph_data'].to(device)
            llm_emb = batch['llm_embeddings'].to(device).float()
            text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
            comments_inputs = {k: v.to(device) for k, v in batch['comments_inputs'].items()}
            cognitive_inputs = {k: v.to(device) for k, v in batch['cognitive_inputs'].items()}
            emotional_scores = batch['emotional_scores'].to(device)
            images_processed = {k: v.to(device) for k, v in batch['images_processed'].items()}
            labels = batch['labels'].to(device)

            logits, _, _, _ = model(
                graph_data=graph_data,
                llama_emb=llm_emb,
                text_input=text_inputs,
                comments_input=comments_inputs,
                image_input=images_processed,
                cognitive_comments_input=cognitive_inputs,
                emotional_scores=emotional_scores,
                return_feat=True,
                has_real_image=batch.get('has_real_image')
            )

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }


def print_results(name, metrics):
    print(f"\n  {'='*50}")
    print(f"  {name}")
    print(f"  {'='*50}")
    print(f"  Samples:    {len(metrics['labels'])}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"  AUC:        {metrics['auc']:.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"              Pred Fake  Pred Real")
    print(f"  True Fake   {metrics['confusion_matrix'][0,0]:9d}  {metrics['confusion_matrix'][0,1]:9d}")
    print(f"  True Real   {metrics['confusion_matrix'][1,0]:9d}  {metrics['confusion_matrix'][1,1]:9d}")

    print(f"\n  Classification Report:")
    print(classification_report(
        metrics['labels'], metrics['predictions'],
        target_names=['Fake', 'Real'],
        zero_division=0
    ))


def main():
    parser = argparse.ArgumentParser(description='EASE-MoE Independent Testing')

    parser.add_argument('--test_json', type=str, required=True, help='Test JSON file')
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--empathy_csv', type=str, default=None)
    parser.add_argument('--llm_emb_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument('--cross_test_json', type=str, default=None, help='Cross-domain test JSON')
    parser.add_argument('--cross_image_folder', type=str, default=None)
    parser.add_argument('--cross_empathy_csv', type=str, default=None)
    parser.add_argument('--cross_llm_emb_path', type=str, default=None)
    parser.add_argument('--cross_name', type=str, default='Cross-Domain', help='Name for cross-domain results')

    parser.add_argument('--roberta_path', type=str, default='roberta-base')
    parser.add_argument('--swin_path', type=str, default='microsoft/swin-base-patch4-window7-224')

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--llm_dim', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=0.1)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("  EASE-MoE Independent Testing")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    print(f"\n[1] Loading model...")
    model = EMoEF(
        dim_features=args.llm_dim,
        device=str(device),
        roberta_path=args.roberta_path,
        swin_path=args.swin_path,
        hidden_dim=args.hidden_dim,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        dropout=args.dropout
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    print(f"  Model: {args.model_path}")

    def load_and_evaluate(data_json, image_folder, empathy_csv, llm_emb_path, name):
        print(f"\n[{name}] Loading dataset...")
        print(f"  Dataset: {data_json}")
        dataset = MultimodalFakeNewsDataset(
            json_file=data_json,
            image_folder=image_folder,
            empathy_comments_csv=empathy_csv,
            llm_embedding_path=llm_emb_path,
            tokenizer_path=args.roberta_path,
            image_processor_path=args.swin_path,
            device=str(device)
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=partial(
                collate_fn, tokenizer=dataset.tokenizer,
                image_processor=dataset.image_processor, device=str(device)
            ),
            num_workers=4, pin_memory=True
        )
        print(f"  Samples: {len(loader.dataset)}")
        metrics = evaluate(model, loader, device)
        print_results(name, metrics)
        return metrics

    # In-domain test
    in_metrics = load_and_evaluate(
        args.test_json, args.image_folder, args.empathy_csv, args.llm_emb_path,
        "In-Domain (GossipCop)"
    )

    # Cross-domain test
    if args.cross_test_json and args.cross_image_folder:
        cross_metrics = load_and_evaluate(
            args.cross_test_json, args.cross_image_folder,
            args.cross_empathy_csv, args.cross_llm_emb_path,
            args.cross_name
        )

    print(f"\n{'='*60}")
    print("  Testing complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()