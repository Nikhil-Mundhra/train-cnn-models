# Dense Anisotropic 3D RNFL Architecture & Multi-Task Contract

## 1. Overview
This module implements the **Dense Anisotropic 3D U-Net** (`AnisotropicRNFLUNet3D`) for volumetric retinal nerve fiber layer (RNFL) segmentation from Spectral-Domain Optical Coherence Tomography (SD-OCT, Optovue Solix) volumes.

To ensure strict parity with the 2.5D models ([`VolumetricRNFLNet`](../train_rnfl_volumetric/model.py) and [`TransUNetRNFLNet`](../train_rnfl_volumetric/transunet.py)), the 3D model is equipped with standard continuous boundary regression and tissue-absence heads.

---

## 2. Architecture & Head Contracts

- **Backbone**: `AnisotropicRNFLUNet3D`
  - Input patch dimensions: `(B, 1, slow_z=64, axial_y=768, fast_x=64)`
  - Strides: Anisotropic downsampling `((1, 2, 1), (1, 2, 1), (2, 2, 2), (2, 2, 2))` preserving fine axial layer boundaries.
  - Normalization: `GroupNorm(8, C)` with `SiLU` activations.

- **Output Heads**:
  1. `mask_logits`: `(B, 1, slow_z, axial_y, fast_x)` dense voxel segmentation logits.
  2. `ilm_pred`: `(B, slow_z, fast_x)` continuous 1D/2D ILM surface depth in pixel rows (scaled to `[0, 768.0]`).
  3. `nfl_pred`: `(B, slow_z, fast_x)` continuous 1D/2D NFL surface depth in pixel rows (scaled to `[0, 768.0]`).
  4. `cup_logits`: `(B, slow_z, fast_x)` optic cup / tissue absence logits.
  5. `column_thickness`: `(B, slow_z, fast_x)` differentiable axial integral $\sum_y P_{\text{voxel}}(z, y, x)$ from sigmoid mask probabilities.

---

## 3. Standardized Multi-Task Loss Formulation

$$\mathcal{L}_{\text{total}} = w_{\text{dice}}\mathcal{L}_{\text{dice}} + w_{\text{bce}}\mathcal{L}_{\text{bce}} + w_{\text{boundary}}(\mathcal{L}_{\text{ilm}} + \mathcal{L}_{\text{nfl}}) + w_{\text{cup}}\mathcal{L}_{\text{cup}} + w_{\text{topo}}\mathcal{L}_{\text{topo}} + w_{\text{thick}}\mathcal{L}_{\text{thick}}$$

- **Dense Voxel Segmentation**: Combined Soft-Dice + BCE on the 3D voxel volume.
- **Continuous Surface Regression**: Smooth L1 (Huber) loss in physical micrometres ($\times \text{AXIAL\_UM} = 3.12367\,\mu\text{m}$) against ground truth curves.
- **Optic Cup Absence Detection**: BCE against sentinel cup absence columns (`nfl_absent`).
- **Topological Ordering Constraint**: $\text{ReLU}(\text{ILM} - \text{NFL})$ penalizing inverted crossing.
- **Column Thickness Consistency**: Enforces consistency between the vertical integral of voxel probabilities and physical 2D boundary thickness.

---

## 4. Evaluation Contract & Metrics

Evaluations adhere to [`evaluation_metrics.py`](evaluation_metrics.py) wrapping `seg-metrics==1.2.8` with native Solix voxel spacing:
- **Spacing**: `(18.81, 3.12367, 18.75) µm` for `(slow_z, axial_y, fast_x)`.
- **Metrics Computed**:
  - `dice`: Volumetric Dice coefficient.
  - `hd95_um`: 95th-percentile Hausdorff Distance in micrometres.
  - `hd_um`: Maximum Hausdorff Distance in micrometres.
  - `asd_um` / `msd`: Mean Surface Distance in micrometres.
  - `volume_similarity`: Bounded relative volume similarity $1 - \frac{|V_p - V_g|}{V_p + V_g}$.

---

## 5. Completed Benchmark Runs & Empirical Findings

Two full 20-epoch runs exist on NYUAD Jubail HPC:

| Run Job ID | Model Description | Heads | Train Epochs | Held-Out Dice | Held-Out MABE | Peripapillary Gain | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`18574378`** | Dense 3D U-Net Baseline | 1 Head (Voxel mask) | 20 | **0.9471** | **5.50 µm** | **+55.5%** | Pure Voxel Upper Bound |
| **`18770720`** | Standardized Multi-Task 3D | 5 Heads (Boundary/Cup) | 20 | **0.8722** | **7.21 µm** | **+54.3%** | Standardized API Contract |

**Key Finding**:
Attaching continuous 1D physical-coordinate regression heads onto shared 3D convolutional decoder features created cross-objective gradient tension, driving boundary errors down to sub-pixel accuracy (ILM: 6.9 µm, NFL: 11.6 µm) while reducing voxel Dice from 0.9471 to 0.8722. In contrast, 2.5D Bi-Planar architectures harmonize both objectives without gradient interference (Dice: 0.9518, MABE: 4.71 µm).

For the full architectural analysis and mathematical breakdown, see the technical specification:  
[`OCT-Analyser-Capstone/docs/technical_specs/dense_3d_multitask_standardization_finding.md`](../../../OCT-Analyser-Capstone/docs/technical_specs/dense_3d_multitask_standardization_finding.md).

