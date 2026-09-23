# High-Sensitivity Glaucoma Screening via Deep Learning Segmentation of the Optic Nerve Head

> A comparative analysis of U-Net architectures and optimization strategies for robust clinical application.  
> **CS 5100 – Foundations of AI · Northeastern University · Fall 2025**

**Team:** Jiahua (Liz) Wu · [Teammate 2] · [Teammate 3]

---

## Overview

Glaucoma is a leading cause of irreversible blindness. Early detection relies on measuring the **Cup-to-Disc Ratio (CDR)** from retinal fundus images — a task that is tedious, subjective, and specialist-dependent.

This project builds an **automated, high-sensitivity screening pipeline** that:
1. Segments the **Optic Disc and Optic Cup** from fundus images using a custom U-Net
2. Calculates the **CDR** directly from predicted masks
3. Applies a clinically-tuned threshold to flag patients for follow-up review

**Core design principle:** _Missing a glaucoma diagnosis is far worse than a false alarm._ The system is optimized for **Recall (Sensitivity) ≥ 85%**, not overall accuracy.

---

## Pipeline

```
Retinal Fundus Image
        │
        ▼
   [U-Net Model]          ← Custom Deep U-Net (31M params, trained from scratch)
        │
        ▼
Segmented Cup & Disc Mask
        │
        ▼
  CDR Calculation         ← CDR = d_cup / d_disc
        │
        ▼
 Clinical Decision        ← CDR > 0.3414 → Flagged for Review
```

---

## Model Comparison

Three architectures were trained and evaluated on the REFUGE dataset:

| Model | Architecture | Parameters | Init | Dice Score | Cup IoU |
|-------|-------------|------------|------|------------|---------|
| Model 1 – Custom Baseline | Basic U-Net | 7.8M | Scratch | — | — |
| **Model 2 – Custom Deep U-Net** ✓ | Deep U-Net | **31M** | Scratch | **0.778** | **0.647** |
| Model 3 – Expert Outsider | SMP U-Net (ResNet34) | 24M | ImageNet Pre-trained | 0.000 | 0.000 |

**Key finding:** The ImageNet pre-trained model (Model 3) suffered **catastrophic negative transfer** — features learned from natural images (cat fur, car wheels) are actively detrimental to identifying the subtle, low-contrast structures in medical fundus imagery. Training from scratch proved decisively superior.

---

## Training Strategy (Champion Model 2)

**Tactic 1 — Aggressive Loss Weighting** to handle severe class imbalance:
```
Weighted Cross-Entropy Loss: [Background: 1.0, Disc: 2.0, Cup: 20.0]
```
The Optic Cup occupies a tiny fraction of the image, so its loss weight is amplified 20× to force the model to learn it.

**Tactic 2 — Stochastic Optimization** with a batch size of 2 to escape local minima where the model simply ignores the cup entirely.

---

## Clinical Results

**Optimal CDR Threshold: > 0.3414**

| Metric | Score | Status |
|--------|-------|--------|
| **Recall (Sensitivity)** | **88.89%** | ✅ Exceeds 85% clinical mandate |
| Specificity | 18.18% | ⚠ High false-positive rate (area for future work) |

The system correctly identifies nearly **9 out of 10 glaucoma patients**, meeting its primary patient safety objective. The low specificity (82% of healthy patients flagged) reflects the intentional precision–recall tradeoff and highlights the need for a hybrid Dice + Cross-Entropy loss in future iterations.

---

## Datasets

| Dataset | Images | Purpose |
|---------|--------|---------|
| [REFUGE](https://refuge.grand-challenge.org/) | 1,200 | Primary training & evaluation |
| [ORIGA](https://pubmed.ncbi.nlm.nih.gov/21095735/) | 650 | Supplementary validation |
| [G1020](https://arxiv.org/abs/2006.09581) | 1,020 | Additional benchmarking |

Sample images from each dataset are in [`samples/`](samples/).

> **Note:** Full datasets are not included due to size (~8 GB total). Download from the links above and place under `data/REFUGE/`, `data/ORIGA/`, `data/G1020/`.

---

## Repository Structure

```
glaucoma-screening-unet/
├── models/
│   └── refuge_clf.pkl          # Trained glaucoma classifier (CDR threshold)
│   └── refuge_segmentation.pth # U-Net segmentation weights (1 GB — not tracked by git)
├── samples/
│   ├── REFUGE/                 # Sample fundus images
│   ├── ORIGA/
│   └── G1020/
└── docs/
    └── presentation.pdf        # Final project presentation slides
```

---

## Key Takeaways

1. **Architecture & strategy win:** A custom deep U-Net with tailored loss weighting and stochastic training outperforms both simpler and pre-trained architectures for this task.
2. **Negative transfer is a real risk:** Pre-trained ImageNet weights can be actively harmful in specialized medical imaging domains.
3. **Clinical mandate met:** The system achieves 88.9% sensitivity, functioning as a reliable first-pass screening tool.

---

## Tech Stack

`Python` · `PyTorch` · `scikit-learn` · `NumPy` · `OpenCV`
