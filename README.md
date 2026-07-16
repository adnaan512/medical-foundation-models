<div align="center">

# 🔬 Parameter-Efficient Fine-Tuning of Vision Foundation Models for Medical Image Analysis

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://black.readthedocs.io)

**Investigating whether LoRA-adapted DINOv2 can match or surpass EfficientNet-B3  
on skin lesion classification with a fraction of the trainable parameters.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Motivation](#-motivation)
- [Dataset](#-dataset)
- [Model Architectures](#-model-architectures)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference](#-inference)
- [Explainability](#-explainability)
- [Results](#-results)
- [Configuration](#-configuration)

---

## 🔭 Overview

This project provides a complete, reproducible research implementation comparing:

| Model | Strategy | Trainable Params |
|:------|:---------|:---------------:|
| **EfficientNet-B3** | Full fine-tuning (discriminative LR) | ~12 M |
| **DINOv2 ViT-B/14 + LoRA** | PEFT — backbone frozen, LoRA rank-16 | ~1.5 M |

Both models are trained on the [HAM10000](https://www.nature.com/articles/sdata2018161) 
skin lesion dataset (7 classes, 10,015 images) and evaluated on identical test splits.

---

## 💡 Motivation

Large vision foundation models (DINOv2, SAM, CLIP) have demonstrated remarkable transfer 
capabilities, but full fine-tuning is prohibitively expensive in data-scarce medical settings.  
**LoRA (Low-Rank Adaptation)** injects small trainable rank-decomposition matrices into attention 
layers, achieving competitive performance while updating <2% of total parameters — critical for:

- 🏥 **Clinical deployment** on resource-constrained hospital hardware
- 🔒 **Regulatory auditability** — fewer moving parts, simpler validation
- ⚡ **Rapid iteration** — 5–10× faster training than full fine-tuning
- 🌍 **Multi-site adaptation** — quickly adapt to new imaging devices/populations

---

## 📊 Dataset

**HAM10000 — Human Against Machine 10000**

| Class | Abbreviation | Count | % |
|:------|:------------|------:|--:|
| Melanocytic nevi | `nv` | 6,705 | 66.9% |
| Melanoma | `mel` | 1,113 | 11.1% |
| Benign keratosis | `bkl` | 1,099 | 11.0% |
| Basal cell carcinoma | `bcc` | 514 | 5.1% |
| Actinic keratoses | `akiec` | 327 | 3.3% |
| Vascular lesions | `vasc` | 142 | 1.4% |
| Dermatofibroma | `df` | 115 | 1.1% |

**Splits:** 70% train / 15% val / 15% test (patient-stratified — no lesion appears in multiple splits).

📥 See [`data/download_instructions.md`](data/download_instructions.md) for setup.

---

## 🏗️ Model Architectures

### Model 1 — EfficientNet-B3 (Baseline)

```
Input (224×224×3)
    └── EfficientNet-B3 Backbone (pretrained, ImageNet)
        ├── MBConv Blocks (× 26)
        └── Global Average Pool → 1536-d feature
    └── Dropout(0.3)
    └── Linear(1536 → 7)   ← classification head
Output: logits (7,)
```

- **Total params:** ~12.2 M | **Trainable:** ~12.2 M
- Discriminative learning rates: backbone LR = 0.1 × head LR
- Mixed-precision training (AMP)

### Model 2 — DINOv2 ViT-B/14 + LoRA

```
Input (224×224×3)
    └── Patch Embedding (14×14 patches → 256 tokens)
    └── Transformer Encoder (12 layers)
        └── Each layer: Multi-Head Attention
            ├── Q_proj → [Q_frozen + LoRA_A × LoRA_B]  ← rank-16
            ├── K_proj → K_frozen
            └── V_proj → [V_frozen + LoRA_A × LoRA_B]  ← rank-16
    └── [CLS] token → LayerNorm → 768-d feature
    └── Dropout(0.1)
    └── Linear(768 → 7)
Output: logits (7,)
```

- **Total params:** ~86.6 M | **Trainable:** ~1.5 M (**1.7%**)
- Backbone completely frozen; only LoRA adapters + head updated

---

## 📁 Project Structure

```
medical-foundation-models/
│
├── configs/
│   ├── base_config.yaml           # Shared hyperparameters
│   ├── efficientnet_config.yaml   # EfficientNet-B3 settings
│   └── dinov2_lora_config.yaml    # DINOv2 + LoRA settings
│
├── data/
│   └── download_instructions.md   # HAM10000 setup guide
│
├── datasets/
│   ├── ham10000.py                # Dataset class + stratified splits
│   ├── transforms.py              # Train/val/TTA augmentation pipelines
│   └── data_utils.py              # DataLoader factory + class balancing
│
├── models/
│   ├── efficientnet.py            # EfficientNet-B3 classifier
│   ├── dinov2_lora.py             # DINOv2 + LoRA (PEFT)
│   └── model_factory.py           # Model instantiation from config
│
├── training/
│   ├── trainer.py                 # Training engine (AMP, early stopping, TensorBoard)
│   ├── losses.py                  # Label-smoothing CE + Focal Loss
│   └── schedulers.py              # Cosine schedule with linear warmup
│
├── evaluation/
│   ├── metrics.py                 # Classification + efficiency metrics
│   └── evaluator.py               # Full test-set evaluation pipeline
│
├── explainability/
│   ├── gradcam.py                 # Grad-CAM for EfficientNet
│   ├── attention_rollout.py       # Attention Rollout for DINOv2
│   └── visualizer.py              # Heatmap overlay + figure saving
│
├── utils/
│   ├── seed.py                    # Deterministic seeding + device selection
│   ├── logger.py                  # Console + file logging setup
│   ├── config.py                  # YAML config loading + deep merge
│   └── visualization.py           # ROC, confusion matrix, PR, training curves
│
├── notebooks/
│   └── exploration.ipynb          # EDA + dataset statistics
│
├── figures/                       # Auto-generated publication figures
├── checkpoints/                   # Model checkpoints
├── outputs/                       # Metrics JSON, predictions, TensorBoard
│
├── train.py                       # Main training entry point
├── evaluate.py                    # Standalone evaluation + explainability
├── inference.py                   # Single-image / batch inference
├── compare_models.py              # Side-by-side model comparison
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/medical-foundation-models.git
cd medical-foundation-models
```

### 2. Create environment

```bash
conda create -n medvision python=3.10 -y
conda activate medvision
```

### 3. Install PyTorch (CUDA 11.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install project dependencies

```bash
pip install -r requirements.txt
```

### 5. Download dataset

```bash
# See data/download_instructions.md for full options
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d data/HAM10000
```

---

## 🚀 Quick Start

```bash
# Verify dataset is set up correctly
python -c "from datasets import HAM10000Dataset; d = HAM10000Dataset('data/HAM10000', 'train'); print(f'{len(d)} samples loaded.')"

# Train EfficientNet-B3
python train.py --model efficientnet_b3

# Train DINOv2 + LoRA
python train.py --model dinov2_vitb14

# Compare both models
python compare_models.py \
    --efficientnet_ckpt checkpoints/efficientnet/best_model.pth \
    --dinov2_ckpt checkpoints/dinov2_lora/best_model.pth
```

---

## 🏋️ Training

### EfficientNet-B3

```bash
python train.py \
    --model efficientnet_b3 \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 1e-4 \
    --data_path data/HAM10000
```

### DINOv2 + LoRA

```bash
python train.py \
    --model dinov2_vitb14 \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 5e-4 \
    --data_path data/HAM10000
```

### Key training features

| Feature | Details |
|:--------|:--------|
| Mixed precision | AMP with GradScaler |
| Class imbalance | WeightedRandomSampler + label-smoothing CE |
| LR schedule | Cosine annealing with 3-epoch linear warm-up |
| Early stopping | Patience = 10 epochs on val accuracy |
| Checkpointing | Best model + every 10 epochs |
| Logging | TensorBoard + structured file logs |

Monitor training:
```bash
tensorboard --logdir outputs/tensorboard
```

---

## 📈 Evaluation

```bash
# Full evaluation with all figures
python evaluate.py \
    --model efficientnet_b3 \
    --checkpoint checkpoints/efficientnet/best_model.pth \
    --save_explainability

python evaluate.py \
    --model dinov2_vitb14 \
    --checkpoint checkpoints/dinov2_lora/best_model.pth \
    --save_explainability
```

**Generated outputs:**

| Output | Location |
|:-------|:---------|
| Metrics JSON | `outputs/<experiment>/metrics.json` |
| Classification report | `outputs/<experiment>/classification_report.txt` |
| ROC curves | `figures/<experiment>/roc_curves_*.png` |
| Confusion matrix | `figures/<experiment>/confusion_matrix_*.png` |
| PR curves | `figures/<experiment>/pr_curves_*.png` |
| Training curves | `figures/<experiment>/training_curves_*.png` |
| Grad-CAM grids | `figures/<experiment>/gradcam_samples.png` |
| Attention Rollout | `figures/<experiment>/attention_rollout_samples.png` |
| Comparison chart | `figures/efficiency_comparison.png` |

---

## 🔍 Inference

```bash
# Single image
python inference.py \
    --model efficientnet_b3 \
    --checkpoint checkpoints/efficientnet/best_model.pth \
    --image path/to/lesion.jpg \
    --top_k 3

# Batch inference on folder
python inference.py \
    --model dinov2_vitb14 \
    --checkpoint checkpoints/dinov2_lora/best_model.pth \
    --image_dir path/to/images/ \
    --output_csv predictions.csv

# With Test-Time Augmentation (5 views)
python inference.py \
    --model efficientnet_b3 \
    --checkpoint checkpoints/efficientnet/best_model.pth \
    --image path/to/lesion.jpg \
    --tta --tta_n 5
```

---

## 🔎 Explainability

### Grad-CAM (EfficientNet-B3)

Highlights convolutional feature regions driving the classification decision.

```python
from explainability import GradCAM
gradcam = GradCAM(model, target_layer=model.get_gradcam_target_layer())
heatmap = gradcam(image_tensor, class_idx=None)   # None = predicted class
```

### Attention Rollout (DINOv2)

Propagates attention through all 12 transformer layers via matrix multiplication,
producing a single [CLS]→patch attention map.

```python
from explainability import AttentionRollout
rollout = AttentionRollout(model, discard_ratio=0.9, head_fusion="mean")
heatmap = rollout(image_tensor, patch_size=14)
```

---

## 📊 Results

> Results below are indicative targets. Run training to obtain your actual results.

| Metric | EfficientNet-B3 | DINOv2 + LoRA |
|:-------|:--------------:|:-------------:|
| Accuracy | ~0.84 | ~0.86 |
| F1 (macro) | ~0.72 | ~0.75 |
| ROC-AUC (macro) | ~0.95 | ~0.96 |
| **Trainable params** | **~12.2 M** | **~1.5 M** |
| Trainable % | 100% | ~1.7% |
| Inference latency | ~3.2 ms | ~4.8 ms |

---

## ⚙️ Configuration

All hyperparameters live in `configs/`. Override any value via CLI:

```bash
python train.py --model efficientnet_b3 --batch_size 16 --num_epochs 30 --lr 5e-5
```

Key config sections:

```yaml
# configs/dinov2_lora_config.yaml
lora:
  r: 16           # LoRA rank (try 8, 16, 32)
  lora_alpha: 32  # Scaling (alpha/r = 2 is standard)
  target_modules: ["query", "value"]

training:
  batch_size: 32
  num_epochs: 50
  early_stopping_patience: 10
  mixed_precision: true
  label_smoothing: 0.1
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [HAM10000](https://www.nature.com/articles/sdata2018161) — Tschandl et al., 2018
- [DINOv2](https://arxiv.org/abs/2304.07193) — Oquab et al., Meta AI, 2023
- [LoRA](https://arxiv.org/abs/2106.09685) — Hu et al., Microsoft Research, 2021
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, Google Brain, 2019
- [PEFT](https://github.com/huggingface/peft) — Hugging Face
- [timm](https://github.com/huggingface/pytorch-image-models) — Ross Wightman
