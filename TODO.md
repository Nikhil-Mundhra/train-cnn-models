# Train CNN Models - Project To-Do List

---

## 🔴 High Priority (CNN Training & Validation)

- [ ] **Finalize Multi-Head ConvNeXt V2 Model**
  - [x] Complete dataset mapping, loss functions (BCE + CrossEntropy), and training loop.
  - [x] Generate dataset manifest from local dataset.
  - [x] Execute MPS-optimized Python training script.
  - [ ] Validate multi-head performance across Level 1 (Normal/Abnormal), Level 2 (Pathology Category), Level 3 (Specific Sub-Type) on Kaggle GPU.
  - [ ] Export final trained checkpoint (`multi_head.pth`) into `models_suite/`.

- [ ] **15-Layer U-Net Hierarchical Segmentation Optimization**
  - [x] Train 15-layer Hierarchical U-Net checkpoint (`unet_hierarchical_best_cls.pth`).
  - [ ] Fine-tune boundary loss for thin retinal layers (RPE, Bruch's membrane).

---

## 🟡 Medium Priority (Model Architecture & Diagnostics)

- [ ] **Segmentation-Driven Cropping & Smart Padding**
  - Dynamically calculate tight bounding boxes around retinal tissue from U-Net outputs and letterbox-pad to eliminate background UI artifacts without spatial information loss.

- [ ] **Grad-CAM & Explainability Auditing**
  - [x] Implement MultiHeadGradCAM for ConvNeXt classification heads.
  - [x] Implement HierarchicalUNetGradCAM for U-Net encoder features.
  - [ ] Run automated Grad-CAM diagnostic battery to verify attention maps overlap with biological markers.

---

## 🟢 Low Priority (Deep Learning Enhancements)

- [ ] **Anatomical Topological Constraints**
  - Implement soft topological graph constraints (Viterbi/Dijkstra) on U-Net boundary outputs to enforce retinal layer ordering while accommodating macular holes and fluid elevations.

- [ ] **Uncertainty Estimation**
  - Add Monte Carlo Dropout heads for pixel-level confidence maps in segmentation.
