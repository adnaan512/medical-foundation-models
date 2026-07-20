# Research Notes — Parameter-Efficient Fine-Tuning of Vision Foundation Models

> **Paper:** Hassnain, A. (2026). *Parameter-Efficient Fine-Tuning of Vision Foundation Models for Medical Image Analysis.*
> GitHub: [adnaan512/medical-foundation-models](https://github.com/adnaan512/medical-foundation-models)

---

## 1. Motivation & Problem Statement

Medical image datasets are inherently small due to privacy constraints, annotation cost, and rarity of certain conditions. The **HAM10000** skin lesion dataset (~10 K images, 7 classes) is representative of this regime.

Large Vision Foundation Models (VFMs) — particularly **DINOv2 ViT-B/14** (86 M parameters) — achieve excellent transfer performance but are computationally prohibitive to fine-tune in full. Standard full fine-tuning also risks **catastrophic overfitting** when the target dataset is orders of magnitude smaller than the pre-training corpus.

**Central research question:**

> *Can parameter-efficient fine-tuning (LoRA) applied to Vision Foundation Models achieve competitive skin lesion classification performance while requiring significantly fewer trainable parameters than traditional CNN-based transfer learning?*

---

## 2. Background

### 2.1 Vision Foundation Models

| Model | Architecture | Pre-training | Params |
|---|---|---|---|
| DINOv2 ViT-B/14 | Vision Transformer | Self-supervised (LVD-142M) | 86 M |
| EfficientNet-B3 | CNN | Supervised (ImageNet-1K) | 12 M |

DINOv2 is trained with a combination of self-distillation (DINO) and masked image modelling (iBOT) objectives on a curated 142 M image dataset, producing features that transfer remarkably well to downstream tasks without any labels.

### 2.2 Low-Rank Adaptation (LoRA)

LoRA (Hu et al., 2021) approximates weight updates with low-rank matrices:

```
W' = W₀ + ΔW = W₀ + BA
```

where `B ∈ ℝ^(d×r)` and `A ∈ ℝ^(r×k)` with rank `r ≪ min(d, k)`.

Only `A` and `B` are trained; the original weight matrix `W₀` is frozen.

**Key properties:**
- Trainable parameters scale as `O(r(d + k))` instead of `O(dk)`.
- At inference, `W' = W₀ + BA` is merged — **zero additional latency**.
- Applied to Query (`Q`) and Value (`V`) projection matrices in every attention layer.

**Our configuration:** `r = 16`, `α = 32` → scaling factor `α/r = 2`.

---

## 3. Experimental Setup

### 3.1 Dataset — HAM10000

| Property | Value |
|---|---|
| Total images | 10,015 |
| Classes | 7 (akiec, bcc, bkl, df, mel, nv, vasc) |
| Dominant class | Melanocytic nevi (nv) — 66.9% |
| Rarest class | Dermatofibroma (df) — 1.1% |
| Split | 70% train / 15% val / 15% test (patient-stratified) |
| Source | [Kaggle — kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |

**Class-imbalance handling:**
- `WeightedRandomSampler` during training — each batch sees a roughly balanced class distribution.
- `LabelSmoothingCrossEntropy` (ε = 0.1) — reduces overconfidence on the majority class.

### 3.2 Data Augmentation

Training augmentations are chosen to respect the physical properties of dermatoscopic imaging:

| Augmentation | Rationale |
|---|---|
| Random horizontal & vertical flip (p = 0.5) | Lesions are rotation-invariant |
| Random rotation ±30° | Camera orientation varies |
| Colour jitter (brightness/contrast/saturation/hue) | Device calibration differences |
| Random erasing (p = 0.1) | Simulates hair / artefacts; acts as Cutout |

### 3.3 Training Configuration

| Hyperparameter | EfficientNet-B3 | DINOv2 + LoRA |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-4 (head), 1e-5 (backbone) | 5e-4 |
| Scheduler | Cosine Annealing + Warmup | Cosine Annealing + Warmup |
| Warmup epochs | 3 | 3 |
| Batch size | 32 | 32 |
| Max epochs | 50 | 50 |
| Early stopping patience | 10 | 10 |
| Mixed precision (AMP) | ✅ | ✅ |
| Gradient clip norm | 1.0 | 1.0 |
| Label smoothing | 0.1 | 0.1 |

---

## 4. Results

### 4.1 Classification Performance

| Model | Accuracy ↑ | F1 Macro ↑ | ROC-AUC Macro ↑ |
|:---|:---:|:---:|:---:|
| **EfficientNet-B3** (baseline) | 0.3714 | 0.3528 | 0.8821 |
| **DINOv2 + LoRA** (ours) | **0.7535** | **0.6938** | **0.9526** |

DINOv2+LoRA achieves **+38.2 pp accuracy** and **+7.0 pp ROC-AUC** over the full fine-tuning CNN baseline.

### 4.2 Efficiency Comparison

| Model | Trainable Params | % of Total | GPU Memory | Latency (ms/sample) |
|:---|:---:|:---:|:---:|:---:|
| **EfficientNet-B3** | ~10.71 M | 100% | ~637 MB | ~4.9 ms |
| **DINOv2 + LoRA** | **~0.60 M** | **0.68%** | ~733 MB | ~21.3 ms |

LoRA reduces trainable parameters by **~18×** vs. full EfficientNet-B3 fine-tuning, despite using a model that is 7× larger in total.

### 4.3 Analysis

**Why does EfficientNet-B3 underperform so severely?**
- HAM10000 (7 K train samples) is too small for full fine-tuning of 10 M parameters — severe overfitting occurs despite dropout and augmentation.
- The model memorises the majority class (`nv`) and fails to learn rare-class boundaries.

**Why does DINOv2+LoRA succeed?**
- The frozen DINOv2 backbone provides rich, general visual representations that require only minimal task-specific adaptation (0.68% parameters).
- LoRA acts as a strong implicit regulariser — the low-rank constraint limits the degrees of freedom of the update.
- DINOv2's self-supervised pre-training on 142 M diverse images produces features far more transferable than ImageNet-supervised CNNs.

---

## 5. Explainability Analysis

### 5.1 EfficientNet-B3 — Grad-CAM

Grad-CAM (Selvaraju et al., 2017) computes the gradient of the class score with respect to the final convolutional feature map, then uses the gradient magnitudes as channel-wise weights.

**Target layer:** `backbone.blocks[-1]` (last MBConv block).

**Observation:** EfficientNet Grad-CAM maps are often diffuse and noisy, sometimes highlighting skin texture outside the lesion boundary. This correlates with the model's poor generalisation — it is attending to spurious features.

### 5.2 DINOv2 — Attention Rollout

Attention Rollout (Abnar & Zuidema, 2020) propagates attention weights recursively through all 12 transformer layers:

```
Ā_eff = Ā_1 · Ā_2 · ... · Ā_L
```

where `Ā_l = 0.5 · A_l + 0.5 · I` (residual connection correction).

The `[CLS]` token row of `Ā_eff` gives the attention weight each patch receives from the classification token — a proxy for "how much did each patch contribute to the prediction."

**Observation:** DINOv2 attention maps produce tight, semantically meaningful localisation of lesion boundaries, comparable to and often sharper than CNN Grad-CAM. This validates that the model has learned clinically relevant features.

---

## 6. Limitations & Future Work

| Limitation | Proposed mitigation |
|---|---|
| Single dataset (HAM10000) | Extend to ISIC-2019, PH2, Derm7pt |
| Only skin lesions evaluated | Apply PEFT to X-ray (CheXpert) and MRI (BraTS) |
| No domain-specific pre-training | Pre-train DINOv2 on large unlabelled dermoscopy corpora |
| No segmentation | Adapt SAM (Segment Anything Model) with LoRA |
| Centralised training | Federated LoRA across clinical institutions |
| No multimodality | Fuse clinical metadata with image features |

---

## 7. References

1. Oquab, M. et al. (2023). **DINOv2: Learning Robust Visual Features without Supervision.** *TMLR.* [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
2. Hu, E. J. et al. (2021). **LoRA: Low-Rank Adaptation of Large Language Models.** *ICLR 2022.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
3. Tschandl, P. et al. (2018). **The HAM10000 dataset.** *Scientific Data, 5, 180161.* [DOI:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)
4. Tan, M. & Le, Q.V. (2019). **EfficientNet: Rethinking Model Scaling for CNNs.** *ICML.* [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
5. Selvaraju, R.R. et al. (2017). **Grad-CAM: Visual Explanations from Deep Networks.** *ICCV.* [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
6. Abnar, S. & Zuidema, W. (2020). **Quantifying Attention Flow in Transformers.** *ACL.* [arXiv:2005.00928](https://arxiv.org/abs/2005.00928)
7. Lin, T.Y. et al. (2017). **Focal Loss for Dense Object Detection.** *ICCV.* [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
