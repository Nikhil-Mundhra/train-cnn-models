# Data Pipelines & Medical Augmentation Rules

This document outlines the dataset configuration contract (`hierarchy.yaml`), MONAI augmentation pipelines, medical spatial constraints, and PyTorch DataLoader optimization strategies.

---

## 1. Dataset Hierarchy Mapping (`hierarchy.yaml`)

The dataset structure is governed by `image-classification-model-training/config/hierarchy.yaml`.

```yaml
data_root: "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Classified-preprocessed"

l1_labels:
  Normal: 0
  Abnormal: 1

l2_labels:
  AMD: 0
  DR: 1
  Vascular: 2
  Fluid: 3
  Structural: 4

granular_classes:
  DRUSEN: 0
  CNV: 1
  GA: 2
  NPDR: 3
  PDR: 4
  RVO: 5
  RAO: 6
  ERM: 7
  DME: 8
  SRF: 9
  PED: 10
  MH: 11
```

`MultiHeadOCTDataset` parses this configuration, recursively discovers images across subdirectories, extracts namespaced patient IDs (`dataset_key::pathology_class::patient_id`), and prepares `StratifiedGroupKFold` patient-aware cross-validation splits.

---

## 2. Medical Data Augmentation Constraints

Medical OCT scans differ fundamentally from general natural images. Indiscriminate application of standard image augmentations can corrupt diagnostic signals.

### Mandated Rules

1. **NO Destructive Spatial Cropping (`RandomResizedCrop`):**
   - **Rationale:** Retinal pathologies (e.g. peripheral cysts, drusen) often reside near scan boundaries. Chopping off image borders acts as random spatial dropout, erasing true positive biological markers.
   - **Mandated Solution:** **Segmentation-Driven / Morphological Tissue Cropping** (`TissueMaskCrop`). Isolates retinal tissue using morphological contour extraction and zero-pads surrounding scanner UI overlays (compasses, headers, logos) while preserving 100% of the tissue.

2. **Horizontal Flipping Only (`RandFlip(spatial_axis=1)`):**
   - **Rationale:** OCT scans possess strict vertical anatomical orientation (Vitreous $\rightarrow$ Retinal Layers $\rightarrow$ RPE $\rightarrow$ Choroid). Vertical flips invert gravity and destroy anatomical layer ordering.
   - **Mandated Solution:** Only apply horizontal flips (`prob=0.5, spatial_axis=1`).

3. **Restricted Anatomical Rotation (`RandRotate(range_x=0.09)`):**
   - **Rationale:** Large rotations alter layer slope and simulate artificial patient head tilts.
   - **Mandated Solution:** Cap tilt rotations to $\approx 5^\circ$ ($\pm 0.09$ radians).

4. **Luminance Normalization & Contrast Enhancement (`CLAHETransform`):**
   - Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) with `clip_limit=2.0` and `tile_grid=(8, 8)` to enhance fine retinal layer boundaries under varying signal strengths.

---

## 3. MONAI Transform Pipelines

### Training Augmentation Pipeline (`get_train_transforms(img_size)`)
```python
Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    CLAHETransform(clip_limit=2.0, tile_grid=(8, 8)),
    Ensure3Channels(),
    ScaleIntensity(),  # Scale [0, 255] -> [0, 1]
    Resize((img_size, img_size)),
    RandFlip(prob=0.5, spatial_axis=1),
    RandRotate(range_x=0.09, prob=0.5, keep_size=True),
    RandGaussianNoise(prob=0.3, std=0.05),
    NormalizeIntensity(subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
    RandCoarseDropout(holes=1, spatial_size=(32, 32), dropout_holes=True, fill_value=0, prob=0.2)
])
```

### Validation Pipeline (`get_val_transforms(img_size)`)
```python
Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    CLAHETransform(clip_limit=2.0, tile_grid=(8, 8)),
    Ensure3Channels(),
    ScaleIntensity(),
    Resize((img_size, img_size)),
    NormalizeIntensity(subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True)
])
```

---

## 4. PyTorch DataLoader Optimization & Deadlock Prevention

To prevent multiprocessing deadlocks and shared memory (`/dev/shm`) Out-Of-Memory (OOM) crashes during multi-worker training on Kaggle or macOS:
- Explicitly disable OpenCV multi-threading: `cv2.setNumThreads(0)`
- Set OpenMP threads: `os.environ['OMP_NUM_THREADS'] = '1'`
- Enable PyTorch MPS fallback on Apple Silicon: `os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'`
- Configure DataLoader workers: Set `--num-workers 2` on cloud GPUs (`0` on restricted shared memory nodes).
- Enable `pin_memory=True` when CUDA or MPS acceleration is active.
