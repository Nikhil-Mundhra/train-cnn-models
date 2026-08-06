# Cloud GPU Training Guide: Kaggle & Google Colab

This guide provides step-by-step instructions for deploying, training, and backing up models on Kaggle GPUs (T4 / P100 / L4) and Google Colab instances using real-time Hugging Face Hub checkpoint backups.

---

## 1. Kaggle Execution Setup

Kaggle provides free P100 / T4 x2 GPUs.

### Pre-Flight Smoke Test on Kaggle
Before launching a multi-hour training run, execute a 1-epoch smoke test to verify DataLoader paths, memory allocations, and GPU CUDA initialization:

```bash
python3 train.py \
  --arch convnextv2_base \
  --config image-classification-model-training/config/hierarchy.yaml \
  --smoke-test \
  --epochs-warmup 1 \
  --epochs-finetune 1 \
  --batch-size 8 \
  --num-workers 2
```

### Full Multi-GPU Training Run with Hugging Face Backup

On Kaggle, set `--checkpoint-dir /kaggle/working` and pass your Hugging Face Hub token to enable automatic checkpoint uploads after every fold or mid-epoch interval:

```bash
export HF_TOKEN="hf_your_write_token_here"

python3 train.py \
  --task multi_head \
  --arch convnextv2_base \
  --batch-size 32 \
  --epochs-warmup 10 \
  --epochs-finetune 10 \
  --lr-head 1e-4 \
  --lr-backbone 1e-6 \
  --use-weighted-sampler \
  --hf-repo "username/oct-convnextv2-model" \
  --checkpoint-dir "/kaggle/working" \
  --num-workers 2
```

---

## 2. Google Colab Setup

### Mount Drive & Clone Repository
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!git clone https://github.com/Nikhil-Mundhra/train-cnn-models.git
%cd train-cnn-models
!pip install -r image-classification-model-training/requirements.txt
```

### Launch Colab Training
```bash
!python3 train.py \
  --arch resnet50 \
  --img-size 224 \
  --batch-size 16 \
  --epochs-warmup 5 \
  --epochs-finetune 10 \
  --checkpoint-dir "/content/drive/MyDrive/OCT_Checkpoints"
```

---

## 3. Multi-GPU Distributed Training (`torchrun` DDP)

For multi-GPU Kaggle instances (2x T4):

```bash
torchrun --nproc_per_node=2 train.py \
  --arch convnextv2_base \
  --use-ddp \
  --batch-size 16 \
  --epochs-warmup 10 \
  --epochs-finetune 10 \
  --hf-repo "username/oct-convnextv2-ddp"
```

---

## 4. Troubleshooting Cloud Environments

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `DataLoader worker process killed (SIGBUS / OOM)` | Shared memory (`/dev/shm`) limit exceeded on Kaggle | Set `--num-workers 0` or `--num-workers 2` |
| `KMP_DUPLICATE_LIB_OK / OpenMP deadlock` | Duplicate libomp runtimes loaded by OpenCV and PyTorch | Handled automatically by `train.py` (`os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'`) |
| `Loss NaN during fp16 training` | Unbounded loss values or extreme class imbalance | Bounded Focal Loss in `train.py` constrains alpha weights to $[0.2, 3.0]$ |
| `HuggingFace Upload Timeout` | Network instability during checkpoint save | Set `--save-steps 0` to upload only end-of-epoch best checkpoints |
