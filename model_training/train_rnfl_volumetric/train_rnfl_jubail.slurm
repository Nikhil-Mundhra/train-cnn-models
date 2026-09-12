#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: Multi-Epoch Volumetric RNFL Training on NYUAD HPC (Jubail)
# Target Partition: nvidia (NVIDIA A100 / V100 GPU)
# Reference: https://crc-docs.abudhabi.nyu.edu/hpc/jobs/quick_start.html
# ==============================================================================

#SBATCH -p nvidia                      # Request GPU partition (nvidia)
#SBATCH --gres=gpu:1                   # Request 1 GPU: uses --gres=gpu:1 (any free GPU) or --gres=gpu:a100:1 or --gres=gpu:v100:1
#SBATCH -n 1                           # 1 task
#SBATCH -c 8                           # 8 CPU cores for fast data loading
#SBATCH --mem=32G                      # 32 GB RAM
#SBATCH -t 02:00:00                    # Wall clock time limit (2 hours is plenty; ~20-30 min actual run)
#SBATCH -J rnfl_train                  # Job name
#SBATCH -o slurm_rnfl_%j.out           # Standard output log (%j expands to jobID)
#SBATCH -e slurm_rnfl_%j.err           # Standard error log

set -e

echo "=================================================================="
echo "=== NYUAD HPC Job Started on $(hostname) at $(date) ==="
echo "=== Job ID: ${SLURM_JOB_ID} | Allocated GPU: ${CUDA_VISIBLE_DEVICES} ==="
echo "=================================================================="

# ----------------------------------------------------------------------
# 1. Environment & Module Setup (NYUAD Pre-Installed MONAI & Singularity)
# ----------------------------------------------------------------------
# Reuses Jubail's pre-built MONAI & PyTorch container in /share/apps/NYUAD5
# Zero pip install, zero download, instant launch.
module purge
module load monai/0.8.0

# ----------------------------------------------------------------------
# 2. Paths Configuration (Data & Outputs on Scratch, Code in Home)
# ----------------------------------------------------------------------
# Soumen's Rule: Data MUST live on fast parallel /scratch filesystem,
# code and repositories live in /home.
NETID="${USER}"
SCRATCH_DIR="/scratch/${NETID}"
DATASET_ROOT="${SCRATCH_DIR}/deidentified"
CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_volumetric_${SLURM_JOB_ID}"
OUTPUT_DIR="${SCRATCH_DIR}/predictions_${SLURM_JOB_ID}"

REPO_DIR="${HOME}/train-cnn-models"
SCRIPT_PATH="${REPO_DIR}/model_training/train_rnfl_volumetric/train.py"

mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "User:           ${NETID}"
echo "Dataset Root:   ${DATASET_ROOT}"
echo "Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "Code Location:  ${REPO_DIR}"

# Check dataset presence
if [ ! -d "${DATASET_ROOT}" ]; then
    echo "[ERROR] Dataset not found at ${DATASET_ROOT}!"
    echo "Please rsync your 'deidentified' dataset to /scratch/${NETID}/deidentified before running."
    exit 1
fi

# ----------------------------------------------------------------------
# 3. Launch Multi-Epoch Volumetric Training on GPU
# ----------------------------------------------------------------------
echo "=================================================================="
echo "=== Launching Volumetric RNFL Training on GPU via Jubail Module ==="
echo "=================================================================="

monai python "${SCRIPT_PATH}" \
    --dataset_root "${DATASET_ROOT}" \
    --val_subjects "BEH0335,BEH0314" \
    --epochs 15 \
    --batch_size 8 \
    --accum_steps 2 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --context_slices 5 \
    --num_workers 4 \
    --device "cuda" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --log_interval 50

echo "=================================================================="
echo "=== Training Completed Successfully at $(date) ==="
echo "=== Checkpoints saved to: ${CHECKPOINT_DIR} ==="
echo "=================================================================="

