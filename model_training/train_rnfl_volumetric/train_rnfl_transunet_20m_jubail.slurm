#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: TransUNet Balanced Hybrid 20M Architecture (A100 Jubail HPC)
# Iso-Capacity Architectural Standardization Experiment (~19.97M parameters)
# Features:
#   - Balanced CNN Pyramid: (64 -> 128 -> 256) preserving inductive bias & boundary edges
#   - Scaled Vision Transformer: L=8, d=384, heads=12, mlp=1536 (~14.93M ViT params)
#   - Cascaded Upsampler (CUP) with dynamic shape interpolation (256 -> 128 -> 64 -> 32)
#   - Multi-Scale Boundary Head: Direct 64-channel stem feature injection
#   - Constrained Spatial Transformer (STN) canonicalizer (theta tilt, t_y)
#   - Multi-slice 2.5D volumetric context (5 channels) + Bi-planar orthogonal fusion
#   - Zero-leakage held-out validation evaluation on NYUAD Jubail
# Partition: nvidia (NVIDIA A100 GPU, 40GB/80GB VRAM)
# ==============================================================================

#SBATCH -p nvidia                      # Request GPU partition (nvidia)
#SBATCH --gres=gpu:a100:1              # Request 1 NVIDIA A100 GPU
#SBATCH -n 1                           # 1 task
#SBATCH -c 8                           # 8 CPU cores for data loading
#SBATCH --mem=48G                      # 48 GB system memory
#SBATCH -t 14:00:00                    # Wall clock time limit (14 hours)
#SBATCH -J rnfl_trans_20m              # Job name
#SBATCH -o slurm_rnfl_trans_20m_%j.out # Standard output log
#SBATCH -e slurm_rnfl_trans_20m_%j.err # Standard error log

set -e

echo "=================================================================="
echo "=== NYUAD HPC Job Started on $(hostname) at $(date) ==="
echo "=== Job ID: ${SLURM_JOB_ID} | Allocated GPU: ${CUDA_VISIBLE_DEVICES} ==="
echo "=== TransUNet Balanced Hybrid 20M (Iso-Capacity ~19.97M Params) on A100 ==="
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

CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_transunet_20m_${SLURM_JOB_ID}"
REPORT_DIR="${SCRATCH_DIR}/reports/report_transunet_20m_${SLURM_JOB_ID}"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${ASSETS_DIR}"

SPLIT_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/stratified_held_out_v2.json"
TRAIN_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/phase1_train_manifest.json"

echo "Validation Split Manifest: ${SPLIT_MANIFEST}"
echo "Training Cohort Manifest:   ${TRAIN_MANIFEST}"

echo ""
echo "##################################################################"
echo "### STEP 1: TRAINING BALANCED HYBRID 20M TRANSUNET MODEL       ###"
echo "##################################################################"

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --dataset_root "${DATASET_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --train_manifest "${TRAIN_MANIFEST}" \
    --arch "transunet" \
    --transunet_stage_channels "64,128,256" \
    --base_channels 32 \
    --transunet_hidden_size 384 \
    --transunet_layers 8 \
    --transunet_heads 12 \
    --transunet_mlp_dim 1536 \
    --epochs 20 \
    --batch_size 8 \
    --accum_steps 4 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --context_slices 5 \
    --num_workers 8 \
    --precision "bf16" \
    --enable_orthogonal \
    --use_stn \
    --use_laterality_embedding \
    --device "cuda" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --log_interval 25

BEST_MODEL="${CHECKPOINT_DIR}/best_volumetric_rnfl_net.pt"
if [ ! -f "${BEST_MODEL}" ]; then
    BEST_MODEL="${CHECKPOINT_DIR}/latest_volumetric_rnfl_net.pt"
fi
echo "[Step 1 SUCCESS] Best TransUNet 20M Model saved to: ${BEST_MODEL}"

echo ""
echo "##################################################################"
echo "### STEP 2: HIGH-THROUGHPUT BATCH COHORT EVALUATION             ###"
echo "##################################################################"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${BEST_MODEL}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${REPORT_DIR}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --os_orientation_mode "corrected" \
    --disc_cut_mode "cup" \
    --batch_size 8 \
    --device "cuda"

echo "[Step 2 SUCCESS] Evaluation finished across cohort volumes."

echo ""
echo "##################################################################"
echo "### STEP 3: COMPILING EXECUTIVE CLINICAL COHORT REPORT         ###"
echo "##################################################################"

OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_transunet_20m_job_${SLURM_JOB_ID}_${TIMESTAMP}.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_transunet_20m_job_${SLURM_JOB_ID}_${TIMESTAMP}.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "${SLURM_JOB_ID}" \
    --checkpoint_name "rnfl_transunet_20m_${SLURM_JOB_ID}" \
    --model_variant "TransUNet Balanced Hybrid 20M (CNN-Transformer, A100)" \
    --model_desc "Iso-Capacity TransUNet with balanced CNN pyramid (64-128-256) and ViT-8/384 encoder (~19.97M params)"

echo "[Step 3 SUCCESS] Executive Report Compiled: ${OUTPUT_PDF}"
echo "=================================================================="
echo "=== Job Completed Successfully at $(date) ==="
echo "=================================================================="
