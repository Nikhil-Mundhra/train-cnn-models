# Bi-Planar Orthogonal Volumetric RNFL Network
## Multi-Planar 2.5D Residual U-Net with Directional Invariance & Surface Coupling

---

### Executive Overview

The **Bi-Planar Orthogonal Volumetric RNFL Network** (`BiplanarVolumetricRNFLNet`) is an advanced deep learning architecture designed specifically for 3D Spectral Domain Optical Coherence Tomography (SD-OCT; 840 nm) volumes acquired on the Optovue Solix platform.

It directly resolves the fundamental limitation of traditional 2.5D networks—directional blind spots along the slow acquisition axis that cause thin peripapillary and peripheral nerve fiber layer (RNFL) masks to cut off or vanish when moving away from the optic disc.

```
       Horizontal Pane (Y-Axis Slicing)               Vertical Pane (X-Axis Slicing)
   ┌─────────────────────────────────────┐        ┌─────────────────────────────────────┐
   │ Temporal-to-Nasal In-Plane View     │        │ Superior-to-Inferior In-Plane View  │
   │ Transverse cross-section of arcuate │   +    │ Continuous trajectory of arcuate    │
   │ bundles entering the disc poles     │        │ fibers streaming into disc margins  │
   └─────────────────────────────────────┘        └─────────────────────────────────────┘
                                      ╲          ╱
                                       ╲        ╱
                        ┌──────────────────────────────────────┐
                        │   Bi-Planar Orthogonal Fusion Layer  │
                        │   - Dense Probabilities: P_fused     │
                        │   - Continuous Surfaces: ILM, NFL    │
                        │   - Cup Absence Logits: Cup_fused    │
                        └──────────────────────────────────────┘
                                           │
                        ┌──────────────────────────────────────┐
                        │   Hybrid Surface-Guided Recovery     │
                        │   - Guarantees thin peripheral mask  │
                        │     retention across all 320 slices  │
                        └──────────────────────────────────────┘
```

---

### Key Architectural Pillars

1. **Orthogonal Bi-Planar 2.5D Sampling**:
   * Leverages the **near-perfect lateral isotropy** of the Optovue Solix platform ($\Delta x = 18.75\,\mu\text{m} \approx \Delta y = 18.81\,\mu\text{m}$).
   * Evaluates orthogonal 2.5D context stacks ($5 \times 768 \times 320$) along both horizontal and vertical axes without resampling distortion.
   * Eliminates the $1.56\%$ out-of-plane field-of-view bottleneck of single-plane 2.5D models.

2. **Multi-Task Volumetric Heads**:
   * **Dense Voxel Mask Head**: $(B, 1, 768, 320)$ predicting volumetric RNFL probability maps.
   * **Continuous 1D Boundary Heads**: $(B, 320)$ directly regressing sub-pixel physical depths in microns for both the **Inner Limiting Membrane (ILM)** and **Nerve Fiber Layer (NFL)**.
   * **1D Optic Cup Absence Classifier**: $(B, 320)$ detecting anatomical void across the Bruch's Membrane Opening (BMO) and cup excavation cavity.

3. **Physics-Informed Composite Loss**:
   * **Focal Tversky Overlap Loss ($\alpha=0.3, \beta=0.7$)**: Heavily penalizes false negatives, preventing thin peripheral sheets from vanishing.
   * **Boundary-Weighted Cross-Entropy ($\omega_1=4.0, \omega_2=2.0$)**: Boosts gradients along thin layer transitions to counteract $> 99.4\%$ background voxel dominance.
   * **Differentiable Column-Thickness Consistency ($\mathcal{L}_{\text{thick}}$)**: Forces the vertical integral of dense probabilities $\sum_y \sigma(\text{logit}_{x,y})$ to match the continuous 1D physical thickness $(\text{NFL}_x - \text{ILM}_x)$.
   * **Optical Reflectance Edge Alignment ($\mathcal{L}_{\text{edge}}$)**: Snaps the NFL boundary to the physical bright-to-dark drop-off in raw backscatter intensity.
   * **Topological Ordering Guarantee ($\mathcal{L}_{\text{topo}}$)**: Enforces $\text{ILM}_x \le \text{NFL}_x$.

4. **Hybrid Surface-Guided Recovery**:
   * If dense voxel probabilities dip in peripheral slices due to optical roll-off or tissue obliquity, the continuous 1D surface heads seamlessly recover the thin mask between predicted ILM and NFL boundaries.
