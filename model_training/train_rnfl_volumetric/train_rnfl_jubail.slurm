#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: Multi-Epoch Volumetric RNFL Training on NYUAD HPC (Jubail)
# Target Partition: nvidia (NVIDIA A100 / V100 GPU)
# Reference: https://crc-docs.abudhabi.nyu.edu/hpc/jobs/quick_start.html
# ==============================================================================

#SBATCH -p nvidia                      # Request GPU partition (nvidia)
#SBATCH --gres=gpu:1                   # Request 1 GPU (or specify --gres=gpu:a100:1 for A100)
#SBATCH -n 1                           # 1 task
#SBATCH -c 8                           # 8 CPU cores for fast data loading
#SBATCH --mem=32G                      # 32 GB RAM
#SBATCH -t 06:00:00                    # Wall clock time limit (6 hours)
#SBATCH -J rnfl_train                  # Job name
#SBATCH -o slurm_rnfl_%j.out           # Standard output log (%j expands to jobID)
#SBATCH -e slurm_rnfl_%j.err           # Standard error log

set -e

echo "=================================================================="
echo "=== NYUAD HPC Job Started on $(hostname) at $(date) ==="
echo "=== Job ID: ${SLURM_JOB_ID} | Allocated GPU: ${CUDA_VISIBLE_DEVICES} ==="
echo "=================================================================="

# ----------------------------------------------------------------------
# 1. Environment & Module Setup (NYUAD Lmod)
# ----------------------------------------------------------------------
# Load Python / Conda on Jubail
module purge
module load miniconda || module load python/3.10 || true

# Initialize conda for bash subshell
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/share/apps/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/share/apps/miniconda3/etc/profile.d/conda.sh"
fi

# Activate Conda environment
if conda info --envs 2>/dev/null | grep -q "oct"; then
    conda activate oct
elif [ -d "${HOME}/.conda/envs/oct" ]; then
    conda activate "${HOME}/.conda/envs/oct"
elif [ -d "${HOME}/miniconda3/envs/oct" ]; then
    conda activate "${HOME}/miniconda3/envs/oct"
elif [ -d "${HOME}/env_rnfl" ]; then
    source "${HOME}/env_rnfl/bin/activate"
fi


# ----------------------------------------------------------------------
# 2. Paths Configuration (Data on Scratch, Code in Home)
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
    echo "Please rsync or copy your 'deidentified' dataset to /scratch/${NETID}/deidentified before submitting."
    exit 1
fi

# Check CUDA availability
python -c "import torch; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# ----------------------------------------------------------------------
# 3. Launch Multi-Epoch Training on GPU
# ----------------------------------------------------------------------
echo "=================================================================="
echo "=== Launching Multi-Epoch Volumetric RNFL Training             ==="
echo "=================================================================="

python "${SCRIPT_PATH}" \
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
