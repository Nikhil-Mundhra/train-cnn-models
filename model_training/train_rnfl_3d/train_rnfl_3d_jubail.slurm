#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: Dense Anisotropic 3D RNFL Network (Jubail HPC)
# Features:
#   - Dense 3D Convolutional U-Net Architecture (AnisotropicRNFLUNet3D)
#   - Full-axial transverse patch sampling: 64x768x64
#   - Micro-batch size 1, Gradient accumulation steps 8
#   - Mixed Precision: BF16 (bfloat16) on NVIDIA A100 GPU
#   - Zero-Leakage Option B2 Held-Out Validation Cohort (20 subjects, 40 scans)
# Partition: nvidia (NVIDIA A100 GPU, 40GB/80GB VRAM)
# ==============================================================================

#SBATCH -p nvidia                      # Request GPU partition (nvidia)
#SBATCH --gres=gpu:a100:1              # Request 1 NVIDIA A100 GPU
#SBATCH -n 1                           # 1 task
#SBATCH -c 8                           # 8 CPU cores for data loading
#SBATCH --mem=48G                      # 48 GB system memory
#SBATCH -t 08:00:00                    # Wall clock time limit (8 hours)
#SBATCH -J rnfl_3d                     # Job name
#SBATCH -o slurm_rnfl_3d_%j.out        # Standard output log (%j expands to jobID)
#SBATCH -e slurm_rnfl_3d_%j.err        # Standard error log

set -e

echo "=================================================================="
echo "=== NYUAD HPC Job Started on $(hostname) at $(date) ==="
echo "=== Job ID: ${SLURM_JOB_ID} | Allocated GPU: ${CUDA_VISIBLE_DEVICES} ==="
echo "=== Dense Anisotropic 3D Volumetric RNFL Architecture ==="
echo "=================================================================="

# ----------------------------------------------------------------------
# 1. Environment & Paths Configuration
# ----------------------------------------------------------------------
module purge

NETID="${USER}"
SCRATCH_DIR="/scratch/${NETID}"
PYTHON_BIN="${SCRATCH_DIR}/envs/oct-env/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[ERROR] Python interpreter not found at: ${PYTHON_BIN}"
    exit 1
fi

echo "Python Executable: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import torch; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

if [ -d "${SCRATCH_DIR}/deidentified-new" ]; then
    DATASET_ROOT="${SCRATCH_DIR}/deidentified-new"
    echo "[Dataset] Using expanded cohort at: ${DATASET_ROOT}"
else
    echo "[ERROR] Canonical dataset not found at ${SCRATCH_DIR}/deidentified-new!"
    exit 1
fi

REPO_DIR="${HOME}/train-cnn-models"
TRAIN_SCRIPT="${REPO_DIR}/model_training/train_rnfl_3d/train_3d.py"
CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_3d_${SLURM_JOB_ID}"
mkdir -p "${CHECKPOINT_DIR}"

TRAIN_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/phase1_train_manifest.json"
VAL_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/held_out_validation_scans.json"

if [ ! -f "${VAL_MANIFEST}" ]; then
    # Fallback to stratified held out v2 if scans manifest is not present
    VAL_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/stratified_held_out_v2.json"
fi

echo "Training Manifest:   ${TRAIN_MANIFEST}"
echo "Validation Manifest: ${VAL_MANIFEST}"
echo "Checkpoint Output:   ${CHECKPOINT_DIR}"

echo ""
echo "##################################################################"
echo "### STEP 1: TRAINING DENSE ANISOTROPIC 3D U-NET                ###"
echo "##################################################################"

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --dataset_root "${DATASET_ROOT}" \
    --train_manifest "${TRAIN_MANIFEST}" \
    --val_manifest "${VAL_MANIFEST}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --patch_shape 64 768 64 \
    --patches_per_volume 32 \
    --base_channels 32 \
    --batch_size 1 \
    --grad_accum_steps 8 \
    --epochs 20 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --num_workers 6 \
    --device "cuda" \
    --amp \
    --val_stride 64 \
    --vol_eval_interval 5 \
    --log_interval 25

echo ""
echo "=================================================================="
echo "=== 3D RNFL Training Completed Successfully at $(date) ==="
echo "=== Checkpoints saved to: ${CHECKPOINT_DIR} ==="
echo "=================================================================="
