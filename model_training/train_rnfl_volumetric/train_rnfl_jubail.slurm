#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: Multi-Epoch Volumetric RNFL Training on NYUAD HPC (Jubail)
# Target Partition: nvidia (NVIDIA A100 / V100 GPU)
# Reference: https://crc-docs.abudhabi.nyu.edu/hpc/jobs/quick_start.html
# ==============================================================================

#SBATCH -p nvidia                      # Request GPU partition (nvidia)
#SBATCH --gres=gpu:a100:1              # Request 1 NVIDIA A100 GPU (or use --gres=gpu:1 for any available GPU)
#SBATCH -n 1                           # 1 task
#SBATCH -c 8                           # 8 CPU cores for fast parallel data loading
#SBATCH --mem=32G                      # 32 GB RAM
#SBATCH -t 04:00:00                    # Wall clock time limit (4 hours)
#SBATCH -J rnfl_train                  # Job name
#SBATCH -o slurm_rnfl_%j.out           # Standard output log (%j expands to jobID)
#SBATCH -e slurm_rnfl_%j.err           # Standard error log

set -e

echo "=================================================================="
echo "=== NYUAD HPC Job Started on $(hostname) at $(date) ==="
echo "=== Job ID: ${SLURM_JOB_ID} | Allocated GPU: ${CUDA_VISIBLE_DEVICES} ==="
echo "=================================================================="

# ----------------------------------------------------------------------
# 1. Environment Setup (Using Verified Conda Env on Scratch)
# ----------------------------------------------------------------------
module purge

NETID="${USER}"
SCRATCH_DIR="/scratch/${NETID}"
PYTHON_BIN="${SCRATCH_DIR}/envs/oct-env/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[ERROR] Python interpreter not found or not executable at: ${PYTHON_BIN}"
    exit 1
fi

echo "Python Executable: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import torch; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# ----------------------------------------------------------------------
# 2. Paths Configuration (Data & Outputs on Scratch, Code in Home)
# ----------------------------------------------------------------------
# Soumen's Rule: Data MUST live on fast parallel /scratch filesystem,
# code and repositories live in /home.
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
echo "=== Launching Volumetric RNFL Training on GPU                  ==="
echo "=================================================================="

"${PYTHON_BIN}" "${SCRIPT_PATH}" \
    --dataset_root "${DATASET_ROOT}" \
    --val_subjects "BEH0335,BEH0314,BEH0086" \
    --epochs 20 \
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

