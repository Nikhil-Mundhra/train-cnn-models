# Architecture & Design Guide

This document describes the neural network architectures, multi-scale feature aggregation, strict hierarchical conditioning, and loss function formulations implemented in the repository.

---

## 1. Architectural Philosophy: "Best of Both Worlds" Unified Network

The repository addresses the constraint of disjoint medical datasets (healthy layer segmentation masks in OCT5K vs 15-class pathology labels in Classified without pixel masks) through a unified architecture.

```text
                     ┌────────────────────────┐
                     │    Input OCT Image     │
                     └───────────┬────────────┘
                                 │
                     ┌───────────▼────────────┐
                     │     Shared Encoder     │
                     │  (ConvNeXt / U-Net)    │
                     └─┬─────────┬───────────┬┘
                       │         │           │
           Stage 2 (x3)│ Stage 3 │ Stage 4   │
                       │   (x4)  │   (x5)    │
                       ▼         ▼           ▼
                     ┌────────────────────────┐
                     │ Multi-Scale CBAM Block │
                     └───────────┬────────────┘
                                 │
                     ┌───────────▼────────────┐
                     │  Multi-Scale Pooling   │
                     └───────────┬────────────┘
                                 │
          ┌──────────────────────┴──────────────────────┐
          │                                             │
┌─────────▼───────────┐                       ┌─────────▼───────────┐
│ Multi-Head Classifier│                       │ Independent Decoder │
│   (Strict Cascade)  │                       │(Layer Segmentation) │
└─────────────────────┘                       └─────────────────────┘
```

---

## 2. Multi-Head ConvNeXt Classification Architecture

### Multi-Scale Encoder Aggregation
To prevent loss of fine spatial details (such as subretinal fluid pockets or small drusen) at the deep bottleneck, features are extracted across three backbone stages:
- **Stage 2 (x3):** Fine spatial resolution (e.g. 28x28 for 224x224 input)
- **Stage 3 (x4):** Intermediate feature representation (e.g. 14x14)
- **Stage 4 (x5):** High-level semantic bottleneck (e.g. 7x7)

Each stage passes through a Convolutional Block Attention Module (CBAM) before adaptive global pooling, concatenating into a multi-scale representation vector.

### CBAM Attention Blocks
CBAM blocks apply sequential Channel Attention followed by Spatial Attention:
$$\text{Channel Attention: } \mathbf{M}_c(\mathbf{F}) = \sigma\left(\text{MLP}(\text{AvgPool}(\mathbf{F})) + \text{MLP}(\text{MaxPool}(\mathbf{F}))\right)$$
$$\text{Spatial Attention: } \mathbf{M}_s(\mathbf{F}) = \sigma\left(f^{7\times 7}\left([\text{AvgPool}(\mathbf{F}); \text{MaxPool}(\mathbf{F})]\right)\right)$$

### Strict Hierarchical Conditioning
The classification pass enforces explicit hierarchical dependencies:
1. **Head 1 (Normal vs Abnormal):** Evaluates Level 1 binary status $\hat{y}_{L1} = \sigma(W_{H1} \cdot \mathbf{f}_{S4})$.
2. **Head 2 (Pathology Category):** Conditioned on Level 1 probability vector:
   $$\mathbf{f}_{L2} = [\mathbf{f}_{\text{multi\_scale}} \,\|\, \hat{y}_{L1}]$$
   $$\hat{y}_{L2} = \text{softmax}(W_{H2} \cdot \mathbf{f}_{L2})$$
3. **Head 3 (Granular Specialists):** Conditioned on both Level 1 and Level 2 probability vectors:
   $$\mathbf{f}_{L3} = [\mathbf{f}_{\text{multi\_scale}} \,\|\, \hat{y}_{L1} \,\|\, \hat{y}_{L2}]$$

---

## 3. 15-Layer Hierarchical U-Net Segmentation

The segmentation architecture (`HierarchicalUNet`) isolates 15 distinct retinal layers and lesions.

### Attention-Gated Skip Connections
Every decoder skip connection incorporates an Attention Gate ($W_g, W_x$) that filters background noise and highlights tissue boundaries:
$$\alpha = \sigma\left(\text{BN}\left(W_\psi \cdot \text{ReLU}(\text{BN}(W_g \cdot \mathbf{g}) + \text{BN}(W_x \cdot \mathbf{x}))\right)\right)$$
$$\mathbf{x}_{\text{attributed}} = \mathbf{x} \odot \alpha$$

### Two-Stage Coarse-to-Granular Heads
1. **Coarse Head:** Predicts 3 broad anatomical zones (Vitreous, Retinal Tissue, Choroid/Sclera).
2. **Granular Head:** Concatenates shared decoder features with coarse softmax probabilities to refine predictions into 15 specific retinal layers.

---

## 4. Loss Formulations

### Bounded Focal Loss (Multi-Class Head 2)
To handle severe class imbalance across pathology types without destabilizing gradient steps:
$$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
where $\alpha_t$ is normalized and bounded to $[0.2, 3.0]$ based on inverse class frequencies.

### Total Multi-Task Loss
$$\mathcal{L}_{\text{Total}} = w_{H1} \cdot \mathcal{L}_{\text{BCE}}(\hat{y}_{L1}, y_{L1}) + w_{H2} \cdot \mathcal{L}_{\text{Focal}}(\hat{y}_{L2}, y_{L2})$$
Default loss weights: $w_{H1} = 1.0$, $w_{H2} = 1.0$.
