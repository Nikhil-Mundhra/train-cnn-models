#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: 2x NVIDIA A100 Distributed (DDP) Canonical Robust RNFL (Jubail)
# Features:
#   - 2x NVIDIA A100 GPUs via PyTorch DDP (torchrun --nproc_per_node=2)
#   - ~1.85x Throughput Acceleration over single A100
#   - Constrained Spatial Transformer (STN) tilt canonicalization
#   - Physics-Informed OCT Robustness Augmentations (Speckle, Shadows, Roll-off)
# Partition: nvidia (2x NVIDIA A100 GPUs, 80GB/160GB aggregate VRAM)
# ==============================================================================

#SBATCH -p nvidia                      # Request GPU partition (nvidia)
#SBATCH --gres=gpu:a100:2              # Request 2 NVIDIA A100 GPUs
#SBATCH -n 1                           # 1 task
#SBATCH -c 16                          # 16 CPU cores (8 cores per GPU)
#SBATCH --mem=64G                      # 64 GB system memory
#SBATCH -t 06:00:00                    # Wall clock limit (6 hours sufficient with 2x A100)
#SBATCH -J rnfl_canon_2xa100           # Job name
#SBATCH -o slurm_rnfl_canon_2xa100_%j.out  # Standard output log
#SBATCH -e slurm_rnfl_canon_2xa100_%j.err  # Standard error log

set -e

echo "=================================================================="
echo "=== NYUAD HPC Job Started on $(hostname) at $(date) ==="
echo "=== Job ID: ${SLURM_JOB_ID} | Allocated GPUs: ${CUDA_VISIBLE_DEVICES} ==="
echo "=== 2x A100 Distributed Data Parallel (DDP) Volumetric RNFL ==="
echo "=================================================================="

# ----------------------------------------------------------------------
# 1. Environment & Paths Configuration
# ----------------------------------------------------------------------
module purge

NETID="${USER}"
SCRATCH_DIR="/scratch/${NETID}"
PYTHON_BIN="${SCRATCH_DIR}/envs/oct-env/bin/python"
TORCHRUN_BIN="${SCRATCH_DIR}/envs/oct-env/bin/torchrun"

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

CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_canonical_robust_2xa100_${SLURM_JOB_ID}"
REPORT_DIR="${SCRATCH_DIR}/reports/report_canonical_robust_2xa100_${SLURM_JOB_ID}"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${ASSETS_DIR}"

SPLIT_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/stratified_held_out_v2.json"
TRAIN_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/phase1_train_manifest.json"

echo "Validation Split Manifest: ${SPLIT_MANIFEST}"
echo "Training Cohort Manifest:   ${TRAIN_MANIFEST}"

echo ""
echo "##################################################################"
echo "### STEP 1: DISTRIBUTED 2x A100 TRAINING WITH TORCHRUN DDP    ###"
echo "##################################################################"

# Master port randomized to prevent port collisions on shared nodes
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

"${TORCHRUN_BIN}" \
    --nproc_per_node=2 \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_SCRIPT}" \
    --dataset_root "${DATASET_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --train_manifest "${TRAIN_MANIFEST}" \
    --epochs 20 \
    --base_channels 32 \
    --batch_size 32 \
    --accum_steps 1 \
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
echo "[Step 1 SUCCESS] Best Model saved to: ${BEST_MODEL}"

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
    --batch_size 16 \
    --device "cuda"

echo "[Step 2 SUCCESS] Bi-Planar Evaluation finished across cohort volumes."

echo ""
echo "##################################################################"
echo "### STEP 3: COMPILING EXECUTIVE CLINICAL COHORT REPORT         ###"
echo "##################################################################"

OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_canonical_robust_2xa100_job_${SLURM_JOB_ID}_${TIMESTAMP}.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_canonical_robust_2xa100_job_${SLURM_JOB_ID}_${TIMESTAMP}.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "${SLURM_JOB_ID}" \
    --checkpoint_name "rnfl_canonical_robust_2xa100_${SLURM_JOB_ID}" \
    --model_variant "Canonical Robust Bi-Planar (2x A100 DDP)" \
    --model_desc "2x A100 DDP Multi-GPU, Constrained STN Canonicalizer & Optical Noise, base_channels=32 (~6.75M params)"

echo "[Step 3 SUCCESS] Executive Report Compiled: ${OUTPUT_PDF}"
echo "=================================================================="
echo "=== Job Completed Successfully at $(date) ==="
echo "=================================================================="
