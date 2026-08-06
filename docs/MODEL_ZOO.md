# Model Zoo & Backbone Reference

This document lists all supported classification backbones and segmentation models available in `train.py`, along with their input spatial contracts, feature dimensions, and recommended hyperparameter profiles.

---

## Supported Model Architectures

| Model Identifier (`--arch`) | Category | Default Input Size | Feature Channels (S2, S3, S4) | Target Use Case |
| :--- | :--- | :--- | :--- | :--- |
| `convnextv2_base` | Multi-Head ConvNeXt | 224x224 / 384x384 | (256, 512, 1024) | Default multi-head disease classifier |
| `convnext_small` | Multi-Head ConvNeXt | 224x224 | (192, 384, 768) | Medium-capacity classification |
| `convnext_tiny` | Multi-Head ConvNeXt | 224x224 | (96, 192, 384) | Lightweight classification |
| `convnext_large` | Multi-Head ConvNeXt | 224x224 | (384, 768, 1536) | High-capacity classification |
| `resnet18` | Residual Network | 224x224 | (64, 128, 256) | Ultra-fast baseline classification |
| `resnet34` | Residual Network | 224x224 | (128, 256, 512) | Fast baseline classification |
| `resnet50` | Residual Network | 224x224 | (256, 512, 1024) | Standard ResNet classification |
| `resnet101` | Residual Network | 224x224 | (256, 512, 2048) | Deep ResNet classification |
| `efficientnet_b0` | EfficientNet | 224x224 | (40, 112, 320) | Mobile/efficient classification |
| `efficientnet_b2` | EfficientNet | 260x260 | (48, 120, 352) | Balanced EfficientNet classification |
| `efficientnet_b4` | EfficientNet | 380x380 | (56, 160, 448) | High-resolution EfficientNet |
| `densenet121` | DenseNet | 224x224 | (256, 512, 1024) | Dense connection feature reuse |
| `densenet201` | DenseNet | 224x224 | (256, 512, 1792) | Deep DenseNet classification |
| `vit_base_patch16_224` | Vision Transformer | 224x224 | (768, 768, 768) | Transformer attention backbone |
| `swin_tiny_patch4_window7_224` | Swin Transformer | 224x224 | (192, 384, 768) | Shifted-window transformer |
| `hierarchical_unet` | Attention U-Net | 512x512 | (64, 128, 256) | 15-Layer Retinal Layer & Lesion Segmentation |
| `unet` | Standard U-Net | 512x512 | (64, 128, 256) | Standard tissue boundary segmentation |

---

## Parameter Group Management & Differential Learning Rates

All multi-head classification models automatically partition model parameters into 4 distinct optimizer parameter groups:

1. **Backbone Decay Group:** Deep stage weight matrices. Learning rate: `--lr-backbone`. Weight decay: `1e-2`.
2. **Backbone No-Decay Group:** Biases, LayerNorm, and BatchNorm parameters. Learning rate: `--lr-backbone`. Weight decay: `0.0`.
3. **Head Decay Group:** Linear classification heads and CBAM conv weights. Learning rate: `--lr-head`. Weight decay: `1e-2`.
4. **Head No-Decay Group:** Head biases and norm parameters. Learning rate: `--lr-head`. Weight decay: `0.0`.

### Recommended Differential Learning Rates
- **Warmup Phase (Epochs 1..10):** Backbone parameters are strictly frozen (`requires_grad = False`). Only head & CBAM parameters train at `--lr-head 1e-4`.
- **Fine-tuning Phase (Epochs 11..20):** Deep backbone stages unfreeze at `--lr-backbone 1e-6`, while heads continue fine-tuning at `--lr-head 1e-4`. Early backbone stem stages receive 0.1x `--lr-backbone` to preserve low-level edge detectors.

---

## Dynamic Image Size Resolution Contract

When running `train.py`:
- Omitting `--img-size` (or setting `--img-size 0`) automatically selects the optimal input resolution from the matrix above.
- Passing an explicit `--img-size N` (e.g. `--img-size 384`) overrides the default and updates the MONAI image resizing transforms accordingly.
