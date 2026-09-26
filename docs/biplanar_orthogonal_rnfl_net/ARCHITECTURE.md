# Architecture Specification: Bi-Planar Orthogonal RNFL Network

---

## 1. Physical Input Dimensions & Coordinate Mapping

| Axis | Physiological Direction | Solix Physical Dimension | Voxel Count | Physical Spacing |
| :--- | :--- | :--- | :--- | :--- |
| **$X$ (Fast)** | Temporal $\longleftrightarrow$ Nasal | $6.00\,\text{mm}$ | $320$ pixels | $\Delta x = 18.750\,\mu\text{m}$ |
| **$Y$ (Slow)** | Superior $\longleftrightarrow$ Inferior | $6.02\,\text{mm}$ | $320$ B-scans | $\Delta y = 18.809\,\mu\text{m}$ |
| **$Z$ (Axial)** | Anterior $\longleftrightarrow$ Posterior (Depth) | $2.40\,\text{mm}$ | $768$ rows | $\Delta z = 3.124\,\mu\text{m}$ |

Because $\Delta x \approx \Delta y$, the lateral sampling grid is **essentially isotropic** ($\approx 18.8\,\mu\text{m}$), permitting lossless transpose of the 3D volume:
$$\mathbf{V}_{\text{horiz}} \in \mathbb{R}^{320 \times 768 \times 320} \quad \xleftrightarrow{\text{transpose}(2, 1, 0)} \quad \mathbf{V}_{\text{vert}} \in \mathbb{R}^{320 \times 768 \times 320}$$

---

## 2. Multi-Planar 2.5D Stacking

Rather than single-slice 2D processing, the network ingests a 5-slice adjacent context window centered at index $i$:
$$\mathbf{X}_{i} = \left[ \mathbf{S}_{i-2},\; \mathbf{S}_{i-1},\; \mathbf{S}_{i},\; \mathbf{S}_{i+1},\; \mathbf{S}_{i+2} \right] \in \mathbb{R}^{5 \times 768 \times 320}$$

* **Horizontal Plane Mode**:
  $$\mathbf{S}_{i} = \mathbf{V}[i, :, :] \quad (i \in [0, 319] \text{ along slow axis } Y)$$
* **Vertical Plane Mode**:
  $$\mathbf{S}_{j} = \mathbf{V}[:, :, j]^T \quad (j \in [0, 319] \text{ along fast axis } X)$$

---

## 3. Network Architecture Backbone

The model utilizes a deep residual U-Net backbone with multi-scale skip connections:

```
Input: (B, 5, 768, 320)
 │
 ├── Stage 1:  16 channels, stride 1  (ResUnits: 2) -> (B, 16, 768, 320)
 │     ↓ [Downsample 2x]
 ├── Stage 2:  32 channels, stride 2  (ResUnits: 2) -> (B, 32, 384, 160)
 │     ↓ [Downsample 2x]
 ├── Stage 3:  64 channels, stride 2  (ResUnits: 2) -> (B, 64, 192, 80)
 │     ↓ [Downsample 2x]
 ├── Stage 4: 128 channels, stride 2  (ResUnits: 2) -> (B, 128, 96, 40)
 │     ↓ [Downsample 2x]
 ├── Bottleneck: 256 channels, stride 2 (ResUnits: 2) -> (B, 256, 48, 20)
 │     ↑ [Upsample & Skip Concat]
 ├── Decoder Stages 4 → 1 with Residual Blocks
 │
 └── Multi-Task Feature Representation: (B, 16, 768, 320)
       ├── Head 1 (Dense Voxel Mask): Conv2d(16 -> 1, 1x1) -> (B, 1, 768, 320)
       └── Head 2 (Boundary & Cup):
             ├── AdaptiveAvgPool2d((1, 320)) -> (B, 16, 320)
             ├── 1D Conv Stack (ILM Surface Depth): -> (B, 320) in [0, 768]
             ├── 1D Conv Stack (NFL Surface Depth): -> (B, 320) in [0, 768]
             └── 1D Conv Stack (Optic Cup Absence Logits): -> (B, 320)
```

---

## 4. Orthogonal Bi-Planar Fusion Formulation

During inference, the 3D volume is evaluated through two orthogonal sweeps:

1. **Horizontal Pass**:
   $$\mathcal{P}_{\text{horiz}} \in [0, 1]^{320 \times 768 \times 320}, \quad \text{ILM}_{\text{horiz}} \in \mathbb{R}^{320 \times 320}, \quad \text{NFL}_{\text{horiz}} \in \mathbb{R}^{320 \times 320}$$
2. **Vertical Pass**:
   $$\mathcal{P}_{\text{vert}} \in [0, 1]^{320 \times 768 \times 320}, \quad \text{ILM}_{\text{vert}} \in \mathbb{R}^{320 \times 320}, \quad \text{NFL}_{\text{vert}} \in \mathbb{R}^{320 \times 320}$$
3. **Multi-Planar Fusion**:
   $$\mathcal{P}_{\text{fused}} = \frac{1}{2} \left( \mathcal{P}_{\text{horiz}} + \mathcal{P}_{\text{vert}} \right)$$
   $$\text{ILM}_{\text{fused}} = \frac{1}{2} \left( \text{ILM}_{\text{horiz}} + \text{ILM}_{\text{vert}}^T \right), \quad \text{NFL}_{\text{fused}} = \frac{1}{2} \left( \text{NFL}_{\text{horiz}} + \text{NFL}_{\text{vert}}^T \right)$$
   $$\text{Cup}_{\text{fused}} = \frac{1}{2} \left( \text{Cup}_{\text{horiz}} + \text{Cup}_{\text{vert}}^T \right)$$

---

## 5. Eye Orientation (OD vs OS) Standardization

To maintain uniform anatomical priors regardless of which eye is imaged:
* For **Left Eyes (OS)**:
  * Horizontal B-scans: Flipped horizontally (`axis=-1`) so that the Nasal quadrant is always oriented on the same side as in Right Eyes (OD).
  * Vertical B-scans: Slices are sequenced in reverse ($a \to 319 - a$) so that Temporal-Nasal geometry aligns with the OD training manifold.
  * At inference termination, all predictions are mapped back into native patient coordinates.
