#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: End-to-End Volumetric RNFL Pipeline on NYUAD HPC (Jubail)
# Stages Executed Sequentially in the Same Job:
#   1. Training: Multi-Task Volumetric U-Net (Dense ResNet Backbone, 20 Epochs, GPU)
#   2. Batch Evaluation: Cohort-Wide Inference across all 23 Subjects (46 Volumes, GPU)
#   3. Automated Reporting: Quantitative Metrics Tables & Executive PDF Report
# Target Partition: nvidia (NVIDIA A100 / V100 GPU)
# Reference: https://crc-docs.abudhabi.nyu.edu/hpc/jobs/quick_start.html
# ==============================================================================

#SBATCH -p nvidia                      # Request GPU partition (nvidia)
#SBATCH --gres=gpu:a100:1              # Request 1 NVIDIA A100 GPU
#SBATCH -n 1                           # 1 task
#SBATCH -c 8                           # 8 CPU cores for fast parallel data loading
#SBATCH --mem=32G                      # 32 GB RAM
#SBATCH -t 04:00:00                    # Wall clock time limit (4 hours)
#SBATCH -J rnfl_pipeline               # Job name
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
DATASET_ROOT="${SCRATCH_DIR}/deidentified"
CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_volumetric_${SLURM_JOB_ID}"
REPORT_DIR="${SCRATCH_DIR}/reports/report_${SLURM_JOB_ID}"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"

REPO_DIR="${HOME}/train-cnn-models"
TRAIN_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/train.py"
EVAL_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/batch_cohort_evaluator.py"
REPORT_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/build_cohort_report.py"

mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${ASSETS_DIR}"

echo "User:           ${NETID}"
echo "Dataset Root:   ${DATASET_ROOT}"
echo "Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "Report Dir:     ${REPORT_DIR}"
echo "Code Location:  ${REPO_DIR}"

# Check dataset presence
if [ ! -d "${DATASET_ROOT}" ]; then
    echo "[ERROR] Dataset not found at ${DATASET_ROOT}!"
    echo "Please rsync your 'deidentified' dataset to /scratch/${NETID}/deidentified before running."
    exit 1
fi

# ----------------------------------------------------------------------
# 3. Stage 1: Volumetric RNFL Training on GPU (Dense Model, A100 Optimized)
# ----------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "=== STAGE 1: Volumetric RNFL Training on GPU                  ==="
echo "=================================================================="

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --dataset_root "${DATASET_ROOT}" \
    --val_subjects "BEH0335,BEH0314,BEH0086" \
    --epochs 20 \
    --base_channels 32 \
    --batch_size 32 \
    --accum_steps 2 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --context_slices 5 \
    --num_workers 6 \
    --device "cuda" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --log_interval 25

BEST_MODEL="${CHECKPOINT_DIR}/best_volumetric_rnfl_net.pt"
if [ ! -f "${BEST_MODEL}" ]; then
    echo "[WARNING] ${BEST_MODEL} not found. Falling back to latest_volumetric_rnfl_net.pt"
    BEST_MODEL="${CHECKPOINT_DIR}/latest_volumetric_rnfl_net.pt"
fi

echo "[STAGE 1 SUCCESS] Best model checkpoint saved to: ${BEST_MODEL}"

# ----------------------------------------------------------------------
# 4. Stage 2: Automated Cohort Batch Evaluation on GPU (Same Job)
# ----------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "=== STAGE 2: Cohort Batch Evaluation on GPU (46 Volumes)       ==="
echo "=================================================================="

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${BEST_MODEL}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${REPORT_DIR}" \
    --val_subjects "BEH0335,BEH0314,BEH0086" \
    --device "cuda"

echo "[STAGE 2 SUCCESS] Evaluation completed. Visuals & metrics in: ${ASSETS_DIR}"

# ----------------------------------------------------------------------
# 5. Stage 3: Automated Clinical Report & PDF Compilation (Same Job)
# ----------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "=== STAGE 3: Executive Report Generation & PDF Compilation    ==="
echo "=================================================================="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_job_${SLURM_JOB_ID}_${TIMESTAMP}.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_job_${SLURM_JOB_ID}_${TIMESTAMP}.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "${SLURM_JOB_ID}" \
    --checkpoint_name "rnfl_volumetric_${SLURM_JOB_ID}"

echo ""
echo "=================================================================="
echo "=== Pipeline Completed Successfully at $(date) ==="
echo "=== Checkpoints: ${CHECKPOINT_DIR} ==="
echo "=== Generated Report & Figures: ${REPORT_DIR} ==="
echo "=================================================================="
