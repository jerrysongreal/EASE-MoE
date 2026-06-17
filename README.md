# EASE-MoE: Empathy-Aware Semi-Supervised Mixture of Experts for Multimodal Fake News Detection

Official implementation. Multi-expert Mixture-of-Experts with multi-prototype one-class classification (OCC) for multimodal fake news detection.

## Key Innovations

- **Multi-Prototype OCC**: K=5 prototypes per expert capture diverse real news patterns across domains
- **4 Experts**: SemanticExpert (text), VisualMultimodalExpert (image), PropagationExpert (graph), EmpathyResponseExpert (comments)
- **Expert-Consistency Pseudo-Labeling**: Samples pseudo-labeled only when 4 experts agree
- **Cross-Expert Contrastive Learning**: Prototype-space contrastive loss across expert pairs

## Datasets

| Dataset | Samples | Modalities |
|---------|:------:|------------|
| GossipCop | ~22,000 | Text, Image, Graph, Comments |
| Weibo21 | ~9,200 | Text, Graph, Comments |
| PolitiFact | ~1,060 | Text, Image, Graph, Comments |
| COVID-19 | ~2,200 | Text, Image, Graph, Comments |
| Twitter | ~5,000 | Text, Image, Graph, Comments |

Data format: `data/{dataset}/train/{dataset}_train.json` with pre-computed LLM embeddings, comment embeddings, and propagation graphs.

> **Note**: Datasets are not included in this repository. Please refer to the paper appendix or contact the authors for data access.

## Baselines

| Baseline | Modalities | Description |
|----------|-----------|-------------|
| RoBERTa | Text | Text-only transformer baseline |
| BiGCN | Text + Graph | Graph-enhanced fake news detection |
| EANN | Text + Image | Event-adversarial neural network |
| DAE | Text + Image + Comments | Dual-attention explainable framework |
| S2MOE-F | Text + Graph | Sparse MoE for fake news detection |

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch 2.12+, CUDA 12.8, torch-geometric, transformers, open-clip, scikit-learn.

## Training

```bash
# EASE-MoE main training
python train_ease.py

# Quick integration test (3 epochs)
python quick_test.py
```

## Testing

```bash
# GPU availability check
python gpu_test.py

# Standalone model evaluation
python test.py \
  --test_json data/gossipcop/test/gossipcop_test.json \
  --image_folder data/gossipcop/images \
  --llm_emb_path data/gossipcop/llm_embeddings_test.pt \
  --model_path checkpoints/best_model.pt
```

## Project Structure

```
EASE-MoE/
├── train_ease.py           # Main training script (EASE-MoE)
├── test.py                 # Independent model evaluation
├── quick_test.py           # Fast integration test (3 epochs)
├── gpu_test.py             # GPU availability check
├── config.py               # Global configuration
├── requirements.txt        # Python dependencies
│
├── models/                 # Model implementations
│   ├── ease_model.py       #   EASE-MoE (latest)
│   ├── emoe_model.py       #   E-MoE base model
│   ├── occ.py              #   Multi-prototype one-class classification
│   ├── pseudo_label.py     #   Expert-consistency pseudo-labeling
│   ├── router.py           #   Reliability router
│   ├── vision_expert.py    #   Swin visual expert
│   ├── tri_expert_fusion.py#   Tri-expert fusion
│   ├── losses.py           #   Loss functions
│   └── experts/            #   Expert modules
│       ├── semantic_expert.py
│       ├── visual_expert.py
│       ├── propagation_expert.py
│       └── empathy_expert.py
│
├── database/               # Data loading
│   └── multimodal_loader.py
│
└── scripts/functions/      # Utility functions
    └── visualize.py
```

## Acknowledgments

We sincerely thank the following researchers for releasing their baseline code:

- **Adrien Benamira** (CentraleSupélec, University of Paris-Saclay, France) — *Semi-Supervised Learning and Graph Neural Networks for Fake News Detection* (S2MOE-F)
- **Benjamin Devillers** (CentraleSupélec, University of Paris-Saclay, France) — *Semi-Supervised Learning and Graph Neural Networks for Fake News Detection* (S2MOE-F)
- **Fragkiskos D. Malliaros** (CentraleSupélec, University of Paris-Saclay, France) — *Semi-Supervised Learning and Graph Neural Networks for Fake News Detection* (S2MOE-F)
- **Lu Yuan** (Communication University of China, Beijing, China) — *Bridging Cognition and Emotion: Empathy-Driven Multimodal Misinformation Detection* (DAE)
- **Zihan Wang** (Communication University of China, Beijing, China) — *Bridging Cognition and Emotion: Empathy-Driven Multimodal Misinformation Detection* (DAE)
- **Lei Shi** (Communication University of China, Beijing, China) — *Bridging Cognition and Emotion: Empathy-Driven Multimodal Misinformation Detection* (DAE)
