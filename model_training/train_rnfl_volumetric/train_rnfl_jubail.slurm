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
# 1. Environment & Module Setup (NYUAD Lmod Singularity)
# ----------------------------------------------------------------------
module purge
module load singularity

# ----------------------------------------------------------------------
# 2. Paths Configuration (Data & Container on Scratch, Code in Home)
# ----------------------------------------------------------------------
# Soumen's Rule: Data and containers MUST live on fast parallel /scratch
# filesystem. Code and repositories live in /home.
NETID="${USER}"
SCRATCH_DIR="/scratch/${NETID}"
IMAGE_DIR="${SCRATCH_DIR}/singularityimages"
SINGULARITY_IMG="${SINGULARITY_IMG:-${IMAGE_DIR}/monai.sif}"
DATASET_ROOT="${SCRATCH_DIR}/deidentified"
CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_volumetric_${SLURM_JOB_ID}"
OUTPUT_DIR="${SCRATCH_DIR}/predictions_${SLURM_JOB_ID}"

REPO_DIR="${HOME}/train-cnn-models"
SCRIPT_PATH="${REPO_DIR}/model_training/train_rnfl_volumetric/train.py"

mkdir -p "${IMAGE_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# ----------------------------------------------------------------------
# 3. Ensure Container Image Exists (Zero pip install required)
# ----------------------------------------------------------------------
# Official MONAI GPU container bundles PyTorch (CUDA), MONAI, SciPy,
# NumPy, Pandas and pydicom directly into a single immutable SIF image.
if [ ! -f "${SINGULARITY_IMG}" ]; then
    echo "=================================================================="
    echo "[INFO] Singularity image not found at ${SINGULARITY_IMG}"
    echo "[INFO] Pulling official MONAI/PyTorch CUDA container to scratch..."
    echo "=================================================================="
    export SINGULARITY_CACHEDIR="${SCRATCH_DIR}/.singularity_cache"
    export SINGULARITY_TMPDIR="${SCRATCH_DIR}/.singularity_tmp"
    mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"
    singularity pull "${SINGULARITY_IMG}" docker://projectmonai/monai:latest
fi

echo "User:            ${NETID}"
echo "Dataset Root:    ${DATASET_ROOT}"
echo "Singularity Img: ${SINGULARITY_IMG}"
echo "Checkpoint Dir:  ${CHECKPOINT_DIR}"
echo "Code Location:   ${REPO_DIR}"

# Check dataset presence
if [ ! -d "${DATASET_ROOT}" ]; then
    echo "[ERROR] Dataset not found at ${DATASET_ROOT}!"
    echo "Please rsync your 'deidentified' dataset to /scratch/${NETID}/deidentified before running."
    exit 1
fi

# Verify CUDA within Singularity container
singularity exec --nv \
    --bind /scratch:/scratch \
    "${SINGULARITY_IMG}" \
    python3 -c "import torch; print('Container PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# ----------------------------------------------------------------------
# 4. Launch Multi-Epoch Training on GPU inside Container
# ----------------------------------------------------------------------
echo "=================================================================="
echo "=== Launching Volumetric RNFL Training in Singularity (--nv)   ==="
echo "=================================================================="

singularity exec --nv \
    --bind /scratch:/scratch \
    "${SINGULARITY_IMG}" \
    python3 "${SCRIPT_PATH}" \
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

