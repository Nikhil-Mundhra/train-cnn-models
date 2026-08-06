# Train CNN Models Suite

A unified PyTorch deep learning framework for training, evaluating, and validating Convolutional Neural Networks (CNNs) on Optical Coherence Tomography (OCT) retinal images.

This framework unifies **Hierarchical Retinal Layer & Lesion Segmentation** (U-Net architectures) and **Multi-Level Disease Classification** (ConvNeXt, ResNet, EfficientNet, DenseNet, Swin, ViT) into a single CLI training engine (`train.py`).

---

## Architectural Principles

1. **Shared Encoder Architecture:** Extracts universal representation features across medical imaging tasks.
2. **Multi-Scale Encoder Aggregation:** Feature maps from multiple encoder depths (e.g. stages 2, 3, 4) are pooled directly into classification heads to retain fine spatial details lost at the bottleneck.
3. **Strict Hierarchical Conditioning:** Cascaded feature propagation enforces Level 2 (Pathology Category) conditioned on Level 1 (Normal/Abnormal) probabilities, preventing contradictory predictions.
4. **Decoupled Decoder:** The segmentation decoder operates independently to predict retinal tissue boundaries without feeding back into classification, preventing catastrophic failure on unseen pathologies.
5. **Medical Augmentation Constraints:** Enforces **Segmentation-Driven Cropping** (isolating retinal tissue while zeroing out background scanner artifacts) over destructive spatial crops (`RandomResizedCrop`).

---

## Codebase Navigation

```text
train-cnn-models/
├── train.py                             # Unified Master Training CLI Application
├── Makefile                             # Build, training, and diagnostic test commands
├── README.md                            # Primary documentation guide
├── pytest.ini                           # Pytest configuration
├── docs/                                # Technical documentation suite
│   ├── CLI_REFERENCE.md                 # Complete train.py argument reference
│   ├── ARCHITECTURE.md                  # Unified network design & loss formulations
│   ├── MODEL_ZOO.md                     # Supported backbones & resolution contracts
│   ├── DATA_AND_AUGMENTATIONS.md        # Medical augmentations & dataset mapping
│   └── KAGGLE_COLAB_GUIDE.md            # Cloud GPU training & HuggingFace Hub backup guide
├── core_ml/                             # Core PyTorch architectures & analyzers
│   ├── classification/                  # Multi-head ConvNeXt, Grad-CAM & device managers
│   └── segmentation/                    # 15-Layer Hierarchical U-Net & analyzers
├── image-classification-model-training/ # Disease classification pipeline (configs & data loaders)
├── image-segmentation-model-training/   # Layer & lesion segmentation pipeline
├── model_training/                      # Detailed model training experiment setups
├── models_suite/                        # Trained model suite architecture definitions
├── scripts/                             # Notebooks for Kaggle and Colab training
├── tests/                               # Battery of diagnostic tests for model health
├── train_cls_frozen.py                  # Frozen-encoder training script
├── generate_final_validation_csv.py     # Suite validation metrics calculation script
└── diagnostics_report.md                # Automated model health diagnostic report
```

---

## Quickstart

### 1. Environment Setup

Requirements: Python 3.10+, PyTorch 2.0+ (CUDA or Apple Silicon MPS).

```bash
make venv
make install
```

### 2. Run Diagnostic Battery Tests

Verify model tensor scaling, bias, and Grad-CAM explainability outputs:
```bash
make test
```

### 3. Run Pipeline Smoke Test

Execute a fast 1-epoch smoke test on the unified training application:
```bash
make smoke-test
```

---

## CLI Training Application (`train.py`)

All training tasks are driven through `train.py`.

### Basic Multi-Head Classification
```bash
python3 train.py --task multi_head --arch convnextv2_base --batch-size 32
```

### Alternative Backbone (ResNet-50)
```bash
python3 train.py --task multi_head --arch resnet50 --img-size 224
```

### Hierarchical U-Net Segmentation
```bash
python3 train.py --task segmentation --arch hierarchical_unet --img-size 512
```

### Fine-Tuning & Multi-GPU Training
```bash
python3 train.py \
  --arch convnext_small \
  --epochs-warmup 5 \
  --epochs-finetune 15 \
  --lr-head 1e-4 \
  --lr-backbone 1e-6 \
  --use-weighted-sampler \
  --hf-repo username/oct-convnext-checkpoint
```

For full CLI parameter documentation, see [CLI Reference](file:///Users/nikhilmundhra/Documents/Github/train-cnn-models/docs/CLI_REFERENCE.md).

---

## Documentation Index

- [CLI Reference](file:///Users/nikhilmundhra/Documents/Github/train-cnn-models/docs/CLI_REFERENCE.md): Complete parameter reference for `train.py`.
- [Architecture Guide](file:///Users/nikhilmundhra/Documents/Github/train-cnn-models/docs/ARCHITECTURE.md): Network design, loss functions, and hierarchical features.
- [Model Zoo](file:///Users/nikhilmundhra/Documents/Github/train-cnn-models/docs/MODEL_ZOO.md): Supported classification backbones and U-Net segmenters.
- [Augmentation & Data Guide](file:///Users/nikhilmundhra/Documents/Github/train-cnn-models/docs/DATA_AND_AUGMENTATIONS.md): Medical augmentation rules, MONAI pipelines, and dataset configuration.
- [Kaggle & Colab Guide](file:///Users/nikhilmundhra/Documents/Github/train-cnn-models/docs/KAGGLE_COLAB_GUIDE.md): Remote GPU execution, HuggingFace Hub sync, and multi-GPU DDP setup.

---

## Environment Variables

- `OCT_LOCAL_DEVICE`: Override compute device selection (`cpu`, `mps`, `cuda`, or `auto`). Default is `auto`.
- `OCT_DATA_ROOT`: Root path to the preprocessed dataset directory.
