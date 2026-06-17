# EASE-MoE 项目结构说明

## 仓库概述

EASE-MoE（Empathy-Aware Semi-Supervised Mixture of Experts for Multimodal Fake News Detection）的官方实现。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# GPU 检测
python gpu_test.py

# 快速集成测试（3 epoch）
python quick_test.py

# 主训练
python train_ease.py

# 独立评估
python test.py \
  --test_json <test.json> \
  --image_folder <image_dir> \
  --model_path <checkpoint.pt>
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `train_ease.py` | EASE-MoE 主训练脚本 |
| `test.py` | 独立模型评估（支持跨域测试） |
| `quick_test.py` | 3-epoch 快速集成测试 |
| `gpu_test.py` | GPU 可用性检测 |
| `config.py` | 全局配置文件 |
| `requirements.txt` | Python 依赖列表 |

## 模型架构

```
models/
├── ease_model.py           # EASE-MoE 主模型（最新版）
├── emoe_model.py           # E-MoE 基类模型
├── occ.py                  # 多原型单类分类（K=5 prototypes）
├── pseudo_label.py         # 专家一致性伪标签生成
├── router.py               # 可靠性路由器
├── losses.py               # 损失函数（OCC + InfoNCE + 对比学习）
│
├── experts/                # 四个专家模块
│   ├── semantic_expert.py  #   E₀ 语义专家（文本）
│   ├── visual_expert.py    #   E₁ 视觉多模态专家
│   ├── propagation_expert.py # E₂ 传播专家（图）
│   └── empathy_expert.py   #   E₃ 共情响应专家（评论）
│
├── vision_expert.py        # Swin Transformer 视觉特征提取
├── tri_expert_fusion.py    # 三专家特征解耦与融合
│
├── database/
│   └── multimodal_loader.py # 多模态数据加载器
│
└── scripts/functions/
    └── visualize.py         # 训练可视化工具
```

## 数据集

| 数据集 | 样本数 | 模态 |
|--------|:-----:|------|
| GossipCop | ~22,000 | 文本、图像、传播图、评论 |
| Weibo21 | ~9,200 | 文本、传播图、评论 |
| PolitiFact | ~1,060 | 文本、图像、传播图、评论 |
| COVID-19 | ~2,200 | 文本、图像、传播图、评论 |
| Twitter | ~5,000 | 文本、图像、传播图、评论 |

> 数据集不在本仓库中，请联系作者或参考论文附录获取数据。

## 致谢

衷心感谢以下研究者公开发布 baseline 代码：

- **Adrien Benamira** (CentraleSupélec, University of Paris-Saclay, France) — *Semi-Supervised Learning and Graph Neural Networks for Fake News Detection* (S2MOE-F)
- **Benjamin Devillers** (CentraleSupélec, University of Paris-Saclay, France) — *Semi-Supervised Learning and Graph Neural Networks for Fake News Detection* (S2MOE-F)
- **Fragkiskos D. Malliaros** (CentraleSupélec, University of Paris-Saclay, France) — *Semi-Supervised Learning and Graph Neural Networks for Fake News Detection* (S2MOE-F)
- **Lu Yuan** (Communication University of China, Beijing, China) — *Bridging Cognition and Emotion: Empathy-Driven Multimodal Misinformation Detection* (DAE)
- **Zihan Wang** (Communication University of China, Beijing, China) — *Bridging Cognition and Emotion: Empathy-Driven Multimodal Misinformation Detection* (DAE)
- **Lei Shi** (Communication University of China, Beijing, China) — *Bridging Cognition and Emotion: Empathy-Driven Multimodal Misinformation Detection* (DAE)
