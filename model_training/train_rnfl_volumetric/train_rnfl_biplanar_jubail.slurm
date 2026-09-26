#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: Upgraded Bi-Planar Orthogonal RNFL Network (Jubail)
# Features:
#   - Orthogonal Bi-Planar Slicing (Horizontal + Vertical)
#   - Focal Tversky Loss (alpha=0.3, beta=0.7) + ReLayNet-style Boundary Weighting
#   - Differentiable Column-Thickness Consistency Loss (L_thick)
#   - Bi-Planar Multi-View Fusion Evaluation across all 46 Solix volumes
# Partition: nvidia (NVIDIA A100 GPU, 40GB VRAM)
# ==============================================================================

#SBATCH -p nvidia                      # Request GPU partition (nvidia)
#SBATCH --gres=gpu:a100:1              # Request 1 NVIDIA A100 GPU (40 GB VRAM)
#SBATCH -n 1                           # 1 task
#SBATCH -c 8                           # 8 CPU cores for data loading
#SBATCH --mem=32G                      # 32 GB system memory
#SBATCH -t 04:00:00                    # Wall clock time limit (4 hours)
#SBATCH -J rnfl_biplanar               # Job name
#SBATCH -o slurm_rnfl_biplanar_%j.out  # Standard output log (%j expands to jobID)
#SBATCH -e slurm_rnfl_biplanar_%j.err  # Standard error log

set -e

echo "=================================================================="
echo "=== NYUAD HPC Job Started on $(hostname) at $(date) ==="
echo "=== Job ID: ${SLURM_JOB_ID} | Allocated GPU: ${CUDA_VISIBLE_DEVICES} ==="
echo "=== Upgraded Bi-Planar Orthogonal Volumetric RNFL Architecture ==="
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

DATASET_ROOT="${SCRATCH_DIR}/deidentified"
REPO_DIR="${HOME}/train-cnn-models"
TRAIN_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/train.py"
EVAL_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/batch_cohort_evaluator.py"
REPORT_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/build_cohort_report.py"

if [ ! -d "${DATASET_ROOT}" ]; then
    echo "[ERROR] Dataset not found at ${DATASET_ROOT}!"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_biplanar_${SLURM_JOB_ID}"
REPORT_DIR="${SCRATCH_DIR}/reports/report_biplanar_${SLURM_JOB_ID}"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${ASSETS_DIR}"

echo ""
echo "##################################################################"
echo "### STEP 1: TRAINING UPGRADED BI-PLANAR VOLUMETRIC U-NET       ###"
echo "##################################################################"

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
    --precision "bf16" \
    --enable_orthogonal \
    --device "cuda" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --log_interval 25

BEST_MODEL="${CHECKPOINT_DIR}/best_volumetric_rnfl_net.pt"
if [ ! -f "${BEST_MODEL}" ]; then
    BEST_MODEL="${CHECKPOINT_DIR}/latest_volumetric_rnfl_net.pt"
fi
echo "[Step 1 SUCCESS] Best Bi-Planar Model saved to: ${BEST_MODEL}"

echo ""
echo "##################################################################"
echo "### STEP 2: HIGH-THROUGHPUT BI-PLANAR FUSION COHORT EVALUATION  ###"
echo "##################################################################"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${BEST_MODEL}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${REPORT_DIR}" \
    --val_subjects "BEH0335,BEH0314,BEH0086" \
    --batch_size 8 \
    --device "cuda"

echo "[Step 2 SUCCESS] Bi-Planar Evaluation finished across 46 paired volumes."

echo ""
echo "##################################################################"
echo "### STEP 3: COMPILING EXECUTIVE CLINICAL COHORT REPORT         ###"
echo "##################################################################"

OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_biplanar_job_${SLURM_JOB_ID}_${TIMESTAMP}.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_biplanar_job_${SLURM_JOB_ID}_${TIMESTAMP}.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "${SLURM_JOB_ID}" \
    --checkpoint_name "rnfl_biplanar_${SLURM_JOB_ID}" \
    --model_variant "Bi-Planar Orthogonal Heavy" \
    --model_desc "Bi-Planar 2.5D Fusion, Tversky + Boundary BCE + Thickness Integral, base_channels=32 (~6.69M params)"

echo "[Step 3 SUCCESS] Executive Report Compiled: ${OUTPUT_PDF}"
echo ""
echo "=================================================================="
echo "=== Job Completed Successfully at $(date) ==="
echo "=================================================================="
