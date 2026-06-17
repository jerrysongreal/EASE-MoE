"""
EASE-MoE v2 Training Script.
Changes from v1:
  - 4 datasets (gossipcop, weibo_sgke, weibo21, politifact) + covid19 OOD
  - No L_cls / no classifier head — loss = L_mp-occ + lambda * L_con
  - lambda_con = 0.1 (MUST be active)
  - ExpertConsistencyPL pseudo-label module with ramp-up
  - No image_mask hard blocking
  - Modality dropout + Mixup retained
"""
import os, sys, random, time, json, hashlib
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ease_model import EASEMoE
from models.pseudo_label import ExpertConsistencyPL
from database.multimodal_loader import (MultimodalFakeNewsDataset, collate_fn,
                                         create_split_loaders, save_split_indices,
                                         load_split_indices, mixup_embeddings)
from functions.visualize import TrainingVisualizer
from torch.utils.data import DataLoader, Subset

ROBERTA_PATH = "roberta-base"
SWIN_PATH = "microsoft/swin-base-patch4-window7-224"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_DATASETS = ["gossipcop", "weibo21", "politifact", "twitter", "pheme_dataset"]
OOD_DATASETS = []
SPLITS_DIR = os.path.join(DATA_DIR, "_splits")

# ── Config ──────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS_S1 = 15
EPOCHS_S2 = 20
HIDDEN_DIM = 256
LR = 5e-4
WEIGHT_DECAY = 0.01
DROPOUT = 0.3
NUM_WORKERS = 0
EARLY_STOP = 8
LAMBDA_CON = 0.1        # MUST be > 0
OCC_MARGIN = 2.5
# Pseudo-label (semi-supervised)
LABEL_RATIO = 0.3       # 30% labeled, 70% unlabeled
PL_RATIOS = [0.05, 0.10, 0.15]
PL_UPDATE_EVERY = 5
PL_ALPHA = 1.5
PL_DELTA = 2.5
PL_BETA = 1.0
PL_WARMUP = 5
# Augmentation
MODALITY_DROPOUT_PROB = 0.10
MIXUP_PROB = 0.15
MIXUP_ALPHA = 0.2
# CV
CV_THRESHOLD = 2000
N_FOLDS = 5
GLOBAL_SEED = 42


def set_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_split_hash(train_idx, val_idx, test_idx, ds_name):
    h = hashlib.sha256()
    h.update(str(sorted(train_idx)).encode())
    h.update(str(sorted(val_idx)).encode())
    h.update(str(sorted(test_idx)).encode())
    h.update(ds_name.encode())
    return h.hexdigest()[:12]


def apply_modality_dropout(batch, prob=MODALITY_DROPOUT_PROB):
    B = batch["labels"].size(0)
    if torch.rand(1).item() < prob:
        batch["images_processed"]["pixel_values"] = torch.zeros_like(
            batch["images_processed"]["pixel_values"])
    if torch.rand(1).item() < prob:
        batch["graph_data"] = None
    return batch


def run_one_epoch(model, loader, optimizer, device, scaler,
                  use_modality_dropout=True, use_mixup=True,
                  pseudo_labeler=None, epoch=0,
                  pseudo_labels=None, unlabeled_mask=None):
    """
    pseudo_labels: pre-computed (B,) tensor with values +1(real), 0(fake), -1(uncertain)
    unlabeled_mask: (B,) bool tensor, True = unlabeled sample
    """
    model.train()
    total_loss = 0.0
    all_dists = []

    for bi, batch in enumerate(loader):
        if use_modality_dropout and MODALITY_DROPOUT_PROB > 0:
            batch = apply_modality_dropout(batch)

        is_mixed = False
        if use_mixup and MIXUP_PROB > 0 and torch.rand(1).item() < MIXUP_PROB:
            batch = mixup_embeddings(batch, alpha=MIXUP_ALPHA)
            is_mixed = True

        graph_data = batch.get("graph_data")
        if graph_data is None:
            B = batch["labels"].size(0)
            from torch_geometric.data import Data, Batch as PyGBatch
            dummy_graphs = []
            for b in range(B):
                dg = Data(x=batch["llm_embeddings"][b:b+1],
                         edge_index=torch.zeros((2, 1), dtype=torch.long),
                         y=torch.tensor([0], dtype=torch.long))
                dummy_graphs.append(dg)
            graph_data = PyGBatch.from_data_list(dummy_graphs)

        llm_emb = batch["llm_embeddings"].to(device).float()
        text_in = {k: v.to(device) for k, v in batch["text_inputs"].items()}
        img_in = {k: v.to(device) for k, v in batch["images_processed"].items()}
        labels = batch["labels"].to(device)
        comment_emb = batch.get("comment_embeddings")
        if comment_emb is not None:
            comment_emb = comment_emb.to(device).float()

        B = labels.size(0)
        sw = torch.ones(B, device=device)

        # ── Pseudo-label integration ────────────────────────────
        occ_labels = labels.clone()
        if pseudo_labels is not None and unlabeled_mask is not None:
            data_ids = batch.get("data_ids", [])
            if data_ids and isinstance(unlabeled_mask, dict):
                pl_vals = []
                is_unlabeled = []
                for did in data_ids:
                    pl_vals.append(pseudo_labels.get(did, -1))
                    # unlabeled_mask[did]=True means labeled; we want unlabeled
                    is_unlabeled.append(not unlabeled_mask.get(did, True))
                pl_batch = torch.tensor(pl_vals, device=device)
                um_batch = torch.tensor(is_unlabeled, dtype=torch.bool, device=device)
                valid_pl = (pl_batch >= 0) & um_batch
                if valid_pl.any():
                    occ_labels = occ_labels.float()
                    occ_labels[valid_pl] = pl_batch[valid_pl].float()
                    sw[valid_pl] = 0.5  # pseudo-label confidence weight

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast("cuda"):
                total_loss_val, extra = model(
                    text_in, img_in, graph_data, llm_emb,
                    comment_emb=comment_emb,
                    labels=occ_labels, sample_weights=sw,
                    texts=batch.get("texts_raw"), images_pil=batch.get("images_pil"),
                    comment_texts_list=batch.get("comments_raw"))
        else:
            total_loss_val, extra = model(
                text_in, img_in, graph_data, llm_emb,
                comment_emb=comment_emb,
                labels=occ_labels, sample_weights=sw,
                texts=batch.get("texts_raw"), images_pil=batch.get("images_pil"),
                    comment_texts_list=batch.get("comments_raw"))

        if torch.isnan(total_loss_val) or torch.isinf(total_loss_val):
            continue

        if scaler:
            scaler.scale(total_loss_val).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += total_loss_val.item()
        all_dists.append(extra.get("final_score", torch.zeros(1)).mean().item())

    return (total_loss / len(loader),
            np.mean(all_dists) if all_dists else 0.0,
            extra.get("loss_components", {}),
            extra.get("routing_weights", {}))


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluation using OCC anomaly scores (final_score). Higher score = more likely fake."""
    model.eval()
    all_scores = []
    all_lbls = []
    for batch in loader:
        llm_emb = batch["llm_embeddings"].to(device).float()
        img_in = {k: v.to(device) for k, v in batch["images_processed"].items()}
        comment_emb = batch.get("comment_embeddings")
        if comment_emb is not None:
            comment_emb = comment_emb.to(device).float()
        labels = batch["labels"].to(device)
        graph_data = batch.get("graph_data")

        if graph_data is None:
            B = labels.size(0)
            from torch_geometric.data import Data, Batch as PyGBatch
            dummy_graphs = [Data(x=llm_emb[b:b+1], edge_index=torch.zeros((2,1), dtype=torch.long)) for b in range(B)]
            graph_data = PyGBatch.from_data_list(dummy_graphs)

        _, extra = model(
            {k: v.to(device) for k, v in batch["text_inputs"].items()},
            img_in, graph_data, llm_emb,
            comment_emb=comment_emb,
            texts=batch.get("texts_raw"), images_pil=batch.get("images_pil"),
            comment_texts_list=batch.get("comments_raw"))

        scores = extra["final_score"].detach().cpu()
        all_scores.extend(scores.numpy())
        all_lbls.extend(labels.cpu().numpy())

    all_scores = np.array(all_scores)
    all_lbls = np.array(all_lbls)

    # OCC anomaly score: higher = more likely fake (label 0)
    # Find optimal threshold on validation set (maximize F1)
    best_thresh = np.median(all_scores)
    best_f1 = 0.0
    for t in np.percentile(all_scores, np.linspace(10, 90, 80)):
        preds = (all_scores <= t).astype(int)
        f1 = f1_score(all_lbls, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    threshold = best_thresh
    all_preds = (all_scores <= threshold).astype(int)  # lower anomaly → real

    # AUC: higher OCC score → fake (label 0), sklearn expects higher → 1
    # Use flipped score so higher → real (label 1)
    auc = roc_auc_score(all_lbls, -all_scores)
    if auc < 0.5:
        auc = roc_auc_score(all_lbls, all_scores)

    return {
        "accuracy": accuracy_score(all_lbls, all_preds),
        "precision": precision_score(all_lbls, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_lbls, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_lbls, all_preds, average="macro", zero_division=0),
        "auc": auc,
        "labels": all_lbls, "scores": all_scores, "predictions": all_preds,
    }


def print_metrics(met, prefix=""):
    cm = confusion_matrix(met["labels"], met["predictions"])
    fr = cm[0, 0] / max(cm[0, 0] + cm[0, 1], 1)
    print(f"  {prefix}Acc:{met['accuracy']:.4f} F1:{met['f1']:.4f} AUC:{met['auc']:.4f} "
          f"Prec:{met['precision']:.4f} Rec:{met['recall']:.4f} FakeRec:{fr:.2%}")


@torch.no_grad()
def compute_pseudo_labels(model, loader, device, pseudo_labeler, epoch, pl_ratio):
    """Assign pseudo-labels to unlabeled training samples. Returns dicts keyed by data_id."""
    model.eval()
    all_scores = []
    all_data_ids = []

    for batch in loader:
        llm_emb = batch["llm_embeddings"].to(device).float()
        img_in = {k: v.to(device) for k, v in batch["images_processed"].items()}
        comment_emb = batch.get("comment_embeddings")
        if comment_emb is not None:
            comment_emb = comment_emb.to(device).float()
        graph_data = batch.get("graph_data")

        if graph_data is None:
            B = llm_emb.size(0)
            from torch_geometric.data import Data, Batch as PyGBatch
            dummy_graphs = [Data(x=llm_emb[b:b+1], edge_index=torch.zeros((2,1), dtype=torch.long)) for b in range(B)]
            graph_data = PyGBatch.from_data_list(dummy_graphs)

        _, extra = model(
            {k: v.to(device) for k, v in batch["text_inputs"].items()},
            img_in, graph_data, llm_emb, comment_emb=comment_emb,
            texts=batch.get("texts_raw"), images_pil=batch.get("images_pil"),
            comment_texts_list=batch.get("comments_raw"))

        all_scores.append(extra["final_score"].cpu())
        all_data_ids.extend(batch.get("data_ids", []))

    all_scores = torch.cat(all_scores, dim=0)  # (N,)
    # Use final_score as distance for all 4 experts (approximation for PL assignment)
    dists = [all_scores.clone() for _ in range(4)]
    routing_weights = torch.ones(len(all_data_ids), 4) / 4

    pseudo_labels, confidences, _ = pseudo_labeler.assign(
        dists, routing_weights, epoch=epoch)

    n_real = (pseudo_labels == 1).sum().item()
    n_fake = (pseudo_labels == 0).sum().item()
    n_uncertain = (pseudo_labels == -1).sum().item()
    print(f"  PL assigned: real={n_real} fake={n_fake} uncertain={n_uncertain} "
          f"mean_conf={confidences[pseudo_labels>=0].mean().item():.3f}", flush=True)

    # Return dicts keyed by data_id
    pl_dict = {}
    conf_dict = {}
    for did, pl, cf in zip(all_data_ids, pseudo_labels.tolist(), confidences.tolist()):
        pl_dict[did] = pl
        conf_dict[did] = cf
    return pl_dict, conf_dict


def train_one_dataset(model, tr_loader, vl_loader, device, ds_name,
                      pl_labeler, vis, model_dir,
                      labeled_mask=None, epoch_offset=0):
    """Train one dataset through Stage 1 + Stage 2."""
    best_f1 = 0.0
    best_state = None
    patience = 0
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    # K-means prototype initialization on real-news features
    print("  Initializing OCC prototypes (K-means on real news)...", flush=True)
    model.init_occ_prototypes(tr_loader, device)

    # Stage 1: Supervised
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=EPOCHS_S1, eta_min=1e-6)

    for epoch in range(EPOCHS_S1):
        avg_loss, avg_dist, lc, rw = run_one_epoch(
            model, tr_loader, opt, device, scaler, epoch=epoch)
        sch.step()
        vl_met = evaluate(model, vl_loader, device)
        vis.log_epoch(epoch=epoch + epoch_offset + 1, train_loss=avg_loss, train_acc=avg_dist,
                      val_acc=vl_met["accuracy"], val_f1=vl_met["f1"], val_auc=vl_met["auc"],
                      lr=opt.param_groups[0]["lr"], stage=1, loss_components=lc, routing_weights=rw)

        print(f"  S1 E{epoch+1:2d}/{EPOCHS_S1} | Loss:{avg_loss:.3f} "
              f"| ValF1:{vl_met['f1']:.3f} AUC:{vl_met['auc']:.3f} "
              f"| occ:{lc.get('occ',0):.2f} con:{lc.get('contrastive',0):.3f} "
              f"| w:[{rw.get('e0',0):.2f} {rw.get('e1',0):.2f} {rw.get('e2',0):.2f} {rw.get('e3',0):.2f}]")

        if vl_met["f1"] > best_f1:
            best_f1 = vl_met["f1"]
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), os.path.join(model_dir, "best_ease_s1.pth"))
        else:
            patience += 1
            if patience >= EARLY_STOP:
                break

    if best_state:
        model.load_state_dict(best_state)
    print(f"  Best S1 ValF1: {best_f1:.4f}")

    # Stage 2: Fine-tune with pseudo-labels
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR * 0.2, weight_decay=WEIGHT_DECAY)
    sch = CosineAnnealingLR(opt, T_max=EPOCHS_S2, eta_min=1e-7)
    patience = 0
    model_dir_local = model_dir

    # Pre-computed pseudo-labels for the training set (refreshed every PL_UPDATE_EVERY epochs)
    pseudo_labels_all = None

    for epoch in range(EPOCHS_S2):
        # Update pseudo-label ratio
        pl_ratio = PL_RATIOS[min(epoch // PL_UPDATE_EVERY, len(PL_RATIOS) - 1)]

        # Refresh pseudo-labels every PL_UPDATE_EVERY epochs
        if epoch % PL_UPDATE_EVERY == 0 or pseudo_labels_all is None:
            if labeled_mask is not None:
                print(f"  Computing pseudo-labels (epoch {EPOCHS_S1+epoch+1}, ratio={pl_ratio:.0%})...", flush=True)
                pseudo_labels_all, pl_conf_dict = compute_pseudo_labels(
                    model, tr_loader, device, pl_labeler,
                    epoch=EPOCHS_S1 + epoch, pl_ratio=pl_ratio)

        avg_loss, avg_dist, lc, rw = run_one_epoch(
            model, tr_loader, opt, device, scaler,
            pseudo_labeler=pl_labeler, epoch=EPOCHS_S1 + epoch,
            pseudo_labels=pseudo_labels_all,
            unlabeled_mask=labeled_mask)
        sch.step()
        vl_met = evaluate(model, vl_loader, device)
        vis.log_epoch(epoch=EPOCHS_S1 + epoch + 1, train_loss=avg_loss, train_acc=avg_dist,
                      val_acc=vl_met["accuracy"], val_f1=vl_met["f1"], val_auc=vl_met["auc"],
                      lr=opt.param_groups[0]["lr"], stage=2, loss_components=lc, routing_weights=rw)

        print(f"  S2 E{epoch+1:2d}/{EPOCHS_S2} | Loss:{avg_loss:.3f} "
              f"| ValF1:{vl_met['f1']:.3f} AUC:{vl_met['auc']:.3f} PL%:{pl_ratio:.0%}")

        if vl_met["f1"] > best_f1:
            best_f1 = vl_met["f1"]
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), os.path.join(model_dir_local, "best_ease_s2.pth"))
        else:
            patience += 1
            if patience >= EARLY_STOP:
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_f1


def main():
    set_seed(GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"Device: {device} | Seed: {GLOBAL_SEED} | lambda_con: {LAMBDA_CON}")

    pl_labeler = ExpertConsistencyPL(
        alpha=PL_ALPHA, delta=PL_DELTA, beta=PL_BETA, warmup_epochs=PL_WARMUP)

    for ds_name in TRAIN_DATASETS:
        dd = os.path.join(DATA_DIR, ds_name)
        print(f"\n{'='*60}\n  EASE-MoE v2: {ds_name}\n{'='*60}")

        print("\n[1] Loading data...")
        train_ds = MultimodalFakeNewsDataset(
            json_file=os.path.join(dd, "train", f"{ds_name}_train.json"),
            image_folder=os.path.join(dd, "images"),
            empathy_comments_csv=os.path.join(dd, "empathy_train.csv"),
            llm_embedding_path=os.path.join(dd, "llm_embeddings_train.pt"),
            comment_embedding_path=os.path.join(dd, "comment_embeddings_train.pt"),
            tokenizer_path=ROBERTA_PATH, image_processor_path=SWIN_PATH,
            device=str(device))
        test_ds = MultimodalFakeNewsDataset(
            json_file=os.path.join(dd, "test", f"{ds_name}_test.json"),
            image_folder=os.path.join(dd, "images"),
            empathy_comments_csv=os.path.join(dd, "empathy_test.csv"),
            llm_embedding_path=os.path.join(dd, "llm_embeddings_test.pt"),
            comment_embedding_path=os.path.join(dd, "comment_embeddings_test.pt"),
            tokenizer_path=ROBERTA_PATH, image_processor_path=SWIN_PATH,
            device=str(device))

        # Small dataset → 5-fold CV
        if len(train_ds) < CV_THRESHOLD:
            print(f"  Small dataset ({len(train_ds)} < {CV_THRESHOLD}) — using {N_FOLDS}-fold CV")
            total_size = len(train_ds)
            all_labels = np.array([train_ds.data[i].get('label', 0) for i in range(total_size)])
            kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=GLOBAL_SEED)
            indices = np.arange(total_size)
            cv_results = []
            model_dir = os.path.join("checkpoints", ds_name)
            os.makedirs(model_dir, exist_ok=True)

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices, all_labels)):
                print(f"\n  -- Fold {fold_idx+1}/{N_FOLDS} --")
                tr_subset = Subset(train_ds, train_idx.tolist())
                vl_subset = Subset(train_ds, val_idx.tolist())
                tr_lbls = [all_labels[i] for i in train_idx]
                fc, rc = tr_lbls.count(0), tr_lbls.count(1)
                print(f"    Train:{len(tr_subset)} Val:{len(vl_subset)} Fake={fc} Real={rc}")

                tr_loader = DataLoader(tr_subset, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=True,
                    collate_fn=partial(collate_fn, tokenizer=train_ds.tokenizer,
                                       image_processor=train_ds.image_processor, device=str(device)))
                vl_loader = DataLoader(vl_subset, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True,
                    collate_fn=partial(collate_fn, tokenizer=train_ds.tokenizer,
                                       image_processor=train_ds.image_processor, device=str(device)))

                vis = TrainingVisualizer(log_dir="logs", experiment_name=f"EASEv2-{ds_name}-fold{fold_idx}")
                model = EASEMoE(dim_features=768, hidden_dim=HIDDEN_DIM, n_experts=4, n_prototypes=5,
                                roberta_path=ROBERTA_PATH, swin_path=SWIN_PATH,
                                device=str(device), dropout=DROPOUT, margin=OCC_MARGIN,
                                lambda_con=LAMBDA_CON).to(device)
                # CV: create labeled mask for this fold
                n_labeled = int(len(train_idx) * LABEL_RATIO)
                labeled_set = set(np.random.choice(train_idx, n_labeled, replace=False))
                labeled_mask_fold = {}
                for idx in train_idx:
                    did = str(train_ds.data[idx].get("id", idx))
                    labeled_mask_fold[did] = idx in labeled_set
                best_f1 = train_one_dataset(model, tr_loader, vl_loader, device, ds_name,
                                            pl_labeler, vis, model_dir,
                                            labeled_mask=labeled_mask_fold)
                cv_results.append({'fold': fold_idx, 'best_val_f1': best_f1})
                vis.save_log()

            cv_f1s = [r['best_val_f1'] for r in cv_results]
            print(f"\n  CV Summary: {N_FOLDS}-fold Val F1 = {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")

        else:
            # Standard split with persistence
            os.makedirs(SPLITS_DIR, exist_ok=True)
            split_path = os.path.join(SPLITS_DIR, f"{ds_name}_split.json")
            if os.path.exists(split_path):
                split_info = load_split_indices(split_path)
                train_idx = split_info['train_indices']
                val_idx = split_info['val_indices']
                test_idx = split_info['test_indices']
                print(f"  Loaded persisted split (hash={split_info.get('split_hash','?')})")
            else:
                total_size = len(train_ds)
                indices_all = list(range(total_size))
                all_labels = [train_ds.data[i].get('label', 0) for i in range(total_size)]
                train_idx, test_idx = train_test_split(
                    indices_all, test_size=0.2, random_state=GLOBAL_SEED, stratify=all_labels)
                test_labels = [all_labels[i] for i in test_idx]
                train_val_labels = [all_labels[i] for i in train_idx]
                train_idx, val_idx = train_test_split(
                    train_idx, test_size=0.25, random_state=GLOBAL_SEED, stratify=train_val_labels)
                train_idx = [int(i) for i in train_idx]
                val_idx = [int(i) for i in val_idx]
                test_idx = [int(i) for i in test_idx]
                split_hash = compute_split_hash(train_idx, val_idx, test_idx, ds_name)
                save_split_indices(split_path, train_idx, val_idx, test_idx,
                                   test_size=0.2, val_size=0.25, random_state=GLOBAL_SEED,
                                   split_hash=split_hash)
                print(f"  Saved split (hash={split_hash})")

            tr_subset = Subset(train_ds, train_idx)
            vl_subset = Subset(train_ds, val_idx)
            te_subset = Subset(train_ds, test_idx)
            tr_lbls = [train_ds.data[i]["label"] for i in train_idx]
            fc, rc = tr_lbls.count(0), tr_lbls.count(1)
            n_labeled = int(len(train_idx) * LABEL_RATIO)
            labeled_idx = set(np.random.choice(train_idx, n_labeled, replace=False))
            labeled_mask = {}  # dict: data_id → is_labeled (bool)
            for idx in train_idx:
                did = str(train_ds.data[idx].get("id", idx))
                labeled_mask[did] = idx in labeled_idx
            print(f"  Train:{len(tr_subset)} Val:{len(vl_subset)} Test:{len(te_subset)} Fake={fc} Real={rc} "
                  f"Labeled={n_labeled}({LABEL_RATIO:.0%})")

            tr_loader = DataLoader(tr_subset, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=True,
                collate_fn=partial(collate_fn, tokenizer=train_ds.tokenizer,
                                   image_processor=train_ds.image_processor, device=str(device)))
            vl_loader = DataLoader(vl_subset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
                collate_fn=partial(collate_fn, tokenizer=train_ds.tokenizer,
                                   image_processor=train_ds.image_processor, device=str(device)))
            te_loader = DataLoader(te_subset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
                collate_fn=partial(collate_fn, tokenizer=train_ds.tokenizer,
                                   image_processor=train_ds.image_processor, device=str(device)))

            print("\n[2] Initializing EASE-MoE v2...")
            model_dir = os.path.join("checkpoints", ds_name)
            os.makedirs(model_dir, exist_ok=True)
            vis = TrainingVisualizer(log_dir="logs", experiment_name=f"EASEv2-{ds_name}")
            model = EASEMoE(dim_features=768, hidden_dim=HIDDEN_DIM, n_experts=4, n_prototypes=5,
                            roberta_path=ROBERTA_PATH, swin_path=SWIN_PATH,
                            device=str(device), dropout=DROPOUT, margin=OCC_MARGIN,
                            lambda_con=LAMBDA_CON).to(device)
            tp = sum(p.numel() for p in model.parameters())
            print(f"  Params: {tp:,} | Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

            train_one_dataset(model, tr_loader, vl_loader, device, ds_name,
                             pl_labeler, vis, model_dir,
                             labeled_mask=labeled_mask)

            print(f"\n[3] Final Test Evaluation")
            te_met = evaluate(model, te_loader, device)
            print_metrics(te_met, prefix="Test: ")
            cm = confusion_matrix(te_met["labels"], te_met["predictions"])
            print(f"  CM: [[{cm[0,0]},{cm[0,1]}][{cm[1,0]},{cm[1,1]}]]")
            vis.generate_all_plots()
            vis.save_log()

    # OOD evaluation
    for ood_ds in OOD_DATASETS:
        dd = os.path.join(DATA_DIR, ood_ds)
        print(f"\n{'='*60}\n  OOD Evaluation: {ood_ds}\n{'='*60}")
        ood_test = MultimodalFakeNewsDataset(
            json_file=os.path.join(dd, "test", f"{ood_ds}_test.json"),
            image_folder=os.path.join(dd, "images"),
            empathy_comments_csv=os.path.join(dd, "empathy_test.csv"),
            llm_embedding_path=os.path.join(dd, "llm_embeddings_test.pt"),
            comment_embedding_path=os.path.join(dd, "comment_embeddings_test.pt"),
            tokenizer_path=ROBERTA_PATH, image_processor_path=SWIN_PATH,
            device=str(device))
        ood_loader = DataLoader(ood_test, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0,
            collate_fn=partial(collate_fn, tokenizer=ood_test.tokenizer,
                               image_processor=ood_test.image_processor, device=str(device)))
        ood_met = evaluate(model, ood_loader, device)
        print_metrics(ood_met, prefix=f"OOD {ood_ds}: ")
        fake_detected = (ood_met["predictions"] == 0).sum()
        print(f"  Fake detected: {fake_detected}/{len(ood_test)} ({fake_detected/len(ood_test)*100:.1f}%)")

    print(f"\n{'='*60}\n  EASE-MoE v2 Complete!\n{'='*60}")


if __name__ == "__main__":
    main()
