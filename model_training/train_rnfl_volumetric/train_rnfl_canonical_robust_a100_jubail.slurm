#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: Canonical Robust Bi-Planar RNFL Network (A100 Jubail HPC)
# Features:
#   - Constrained Spatial Transformer (STN) for continuous in-plane tilt canonicalization
#   - Physics-Informed OCT Robustness Augmentations (Speckle, Vessel Shadows, Roll-off)
#   - Orthogonal Bi-Planar Slicing (Horizontal + Vertical)
#   - Optimized for large A100 pool on Jubail for fast queue scheduling
# Partition: nvidia (NVIDIA A100 GPU, 40GB/80GB VRAM)
# ==============================================================================

#SBATCH -p nvidia                      # Request GPU partition (nvidia)
#SBATCH --gres=gpu:a100:1              # Request 1 NVIDIA A100 GPU
#SBATCH -n 1                           # 1 task
#SBATCH -c 8                           # 8 CPU cores for data loading
#SBATCH --mem=48G                      # 48 GB system memory
#SBATCH -t 12:00:00                    # Wall clock time limit (12 hours)
#SBATCH -J rnfl_canon_a100             # Job name
#SBATCH -o slurm_rnfl_canon_a100_%j.out  # Standard output log
#SBATCH -e slurm_rnfl_canon_a100_%j.err  # Standard error log

set -e

echo "=================================================================="
echo "=== NYUAD HPC Job Started on $(hostname) at $(date) ==="
echo "=== Job ID: ${SLURM_JOB_ID} | Allocated GPU: ${CUDA_VISIBLE_DEVICES} ==="
echo "=== Canonical Robust Bi-Planar Architecture on A100 ==="
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

DATASET_ROOT="${SCRATCH_DIR}/deidentified-new"
if [ ! -d "${DATASET_ROOT}" ]; then
    echo "[ERROR] Dataset not found at: ${DATASET_ROOT}!"
    exit 1
fi

REPO_DIR="${HOME}/train-cnn-models"
TRAIN_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/train.py"
EVAL_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/batch_cohort_evaluator.py"
REPORT_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/build_cohort_report.py"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_canonical_robust_${SLURM_JOB_ID}"
REPORT_DIR="${SCRATCH_DIR}/reports/report_canonical_robust_${SLURM_JOB_ID}"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${ASSETS_DIR}"

SPLIT_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/stratified_held_out_v2.json"
TRAIN_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/phase1_train_manifest.json"

echo "Validation Split Manifest: ${SPLIT_MANIFEST}"
echo "Training Cohort Manifest:   ${TRAIN_MANIFEST}"

echo ""
echo "##################################################################"
echo "### STEP 1: TRAINING CANONICAL ROBUST BI-PLANAR U-NET WITH STN ###"
echo "##################################################################"

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --dataset_root "${DATASET_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --train_manifest "${TRAIN_MANIFEST}" \
    --epochs 20 \
    --base_channels 32 \
    --batch_size 32 \
    --accum_steps 2 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --context_slices 5 \
    --num_workers 8 \
    --precision "bf16" \
    --enable_orthogonal \
    --use_stn \
    --robust_augment \
    --device "cuda" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --log_interval 25

BEST_MODEL="${CHECKPOINT_DIR}/best_volumetric_rnfl_net.pt"
if [ ! -f "${BEST_MODEL}" ]; then
    BEST_MODEL="${CHECKPOINT_DIR}/latest_volumetric_rnfl_net.pt"
fi
echo "[Step 1 SUCCESS] Best Canonical Robust Model saved to: ${BEST_MODEL}"

echo ""
echo "##################################################################"
echo "### STEP 2: HIGH-THROUGHPUT BI-PLANAR FUSION COHORT EVALUATION  ###"
echo "##################################################################"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${BEST_MODEL}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${REPORT_DIR}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --os_orientation_mode "corrected" \
    --batch_size 8 \
    --device "cuda"

echo "[Step 2 SUCCESS] Bi-Planar Evaluation finished across cohort volumes."

echo ""
echo "##################################################################"
echo "### STEP 3: COMPILING EXECUTIVE CLINICAL COHORT REPORT         ###"
echo "##################################################################"

OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_canonical_robust_job_${SLURM_JOB_ID}_${TIMESTAMP}.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_canonical_robust_job_${SLURM_JOB_ID}_${TIMESTAMP}.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "${SLURM_JOB_ID}" \
    --checkpoint_name "rnfl_canonical_robust_${SLURM_JOB_ID}" \
    --model_variant "Canonical Robust Bi-Planar (STN, A100)" \
    --model_desc "Bi-Planar 2.5D with Constrained STN Canonicalizer and Physics-Informed Optical Noise, base_channels=32 (~6.75M params)"

echo "[Step 3 SUCCESS] Executive Report Compiled: ${OUTPUT_PDF}"
echo "=================================================================="
echo "=== Job Completed Successfully at $(date) ==="
echo "=================================================================="
