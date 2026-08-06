# Command Line Interface (CLI) Reference: `train.py`

`train.py` is the master CLI application for training, fine-tuning, and evaluating all CNN models in the repository.

---

## Command Syntax

```bash
python3 train.py [SELECTION_FLAGS] [HYPERPARAMETERS] [HARDWARE_FLAGS]
```

---

## 1. Selection & Model Flags

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--task` | `str` | `multi_head` | Training task target. Choices: `multi_head`, `classification`, `segmentation`, `unified`. |
| `--arch`, `--model`, `--backbone` | `str` | `convnextv2_base` | Target architecture or timm backbone name. See [Model Zoo](file:///Users/nikhilmundhra/Documents/Github/train-cnn-models/docs/MODEL_ZOO.md) for full list. |
| `--img-size`, `--image-size` | `int` | `0` | Input spatial resolution (Height/Width). `0` enables automatic resolution assignment per architecture. |
| `--n-coarse-classes` | `int` | `3` | Number of coarse segmentation classes for U-Net (default: 3). |
| `--n-granular-classes` | `int` | `15` | Number of granular layer/lesion segmentation classes for U-Net (default: 15). |

---

## 2. Training Hyperparameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--config` | `str` | `image-classification-model-training/config/hierarchy.yaml` | Path to dataset hierarchy mapping YAML file. |
| `--batch-size` | `int` | `32` | Batch size per GPU/device during training and evaluation. |
| `--epochs-warmup` | `int` | `10` | Number of initial warmup epochs with a frozen backbone to train classification heads and CBAM blocks. |
| `--epochs-finetune` | `int` | `10` | Number of gradual fine-tuning epochs for deep backbone stages. |
| `--patience` | `int` | `5` | Early stopping patience (epochs without macro-F1 improvement). |
| `--lr-head` | `float` | `1e-4` | Learning rate for classification heads, CBAM blocks, and decoder layers. |
| `--lr-backbone` | `float` | `1e-6` | Learning rate for deep backbone stages (early stages receive 0.1x this rate). |
| `--w-h1` | `float` | `1.0` | Loss weighting multiplier for Head 1 (Normal/Abnormal binary classification). |
| `--w-h2` | `float` | `1.0` | Loss weighting multiplier for Head 2 (12-class pathology classification). |
| `--use-weighted-sampler` | flag | `False` | Enables `WeightedRandomSampler` for inverse class frequency oversampling. |

---

## 3. Checkpointing & Remote Backup Flags

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--checkpoint-dir` | `str` | `checkpoints` | Directory to save local model checkpoints (`/kaggle/working` automatically selected on Kaggle). |
| `--resume` | `str` | `None` | Path to an existing `.pth` checkpoint file to resume training. |
| `--hf-repo` | `str` | `None` | Hugging Face Hub repository ID (e.g. `username/oct-checkpoint`) for real-time cloud backup. |
| `--save-steps` | `int` | `2250` | Mid-epoch checkpoint saving frequency (batches). `0` disables mid-epoch saves. |
| `--accum-steps` | `int` | `1` | Gradient accumulation steps (effective batch size = `--batch-size` * `--accum-steps`). |

---

## 4. Hardware & Execution Flags

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--num-workers` | `int` | `2` | Number of DataLoader worker subprocesses (`0` bypasses shared memory OOM traps). |
| `--smoke-test` | flag | `False` | Executes a 1-epoch dry-run to verify pipeline integrity before full training. |
| `--use-data-parallel` | flag | `False` | Enables PyTorch `DataParallel` across multiple local GPUs. |
| `--use-ddp` | flag | `False` | Enables PyTorch `DistributedDataParallel` across multi-GPU environments via `torchrun`. |

---

## Usage Examples

### Fast Smoke Test
```bash
python3 train.py --smoke-test --batch-size 8
```

### Training ResNet-50 Classifier
```bash
python3 train.py \
  --task multi_head \
  --arch resnet50 \
  --img-size 224 \
  --batch-size 16 \
  --epochs-warmup 5 \
  --epochs-finetune 15
```

### Training 15-Layer Hierarchical U-Net
```bash
python3 train.py \
  --task segmentation \
  --arch hierarchical_unet \
  --img-size 512 \
  --batch-size 8 \
  --lr-head 2e-4
```

### Kaggle Multi-GPU Training with Hugging Face Backup
```bash
python3 train.py \
  --arch convnextv2_base \
  --use-weighted-sampler \
  --hf-repo user/oct-convnextv2-weights \
  --batch-size 32 \
  --num-workers 2
```
