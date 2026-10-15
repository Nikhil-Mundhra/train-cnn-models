#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: Fast GPU Evaluation for TransUNet 20M Epoch 12 Checkpoint
# Evaluates 20 held-out validation subjects (40 volumes) using latest_volumetric_rnfl_net.pt
# ==============================================================================

#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -J eval_ep12
#SBATCH -o /home/nm4358/train-cnn-models/slurm_eval_transunet_20m_ep12_%j.out
#SBATCH -e /home/nm4358/train-cnn-models/slurm_eval_transunet_20m_ep12_%j.err

set -euo pipefail

echo "=================================================================="
echo "=== NYUAD HPC Job Started on $(hostname) at $(date) ==="
echo "=== Job ID: ${SLURM_JOB_ID} | Allocated GPU: ${CUDA_VISIBLE_DEVICES:-all} ==="
echo "=== Evaluating TransUNet Balanced Hybrid 20M Epoch 12 Checkpoint ==="
echo "=================================================================="

module purge

NETID="${USER}"
SCRATCH_DIR="/scratch/${NETID}"
PYTHON_BIN="${SCRATCH_DIR}/envs/oct-env/bin/python"
DATASET_ROOT="${SCRATCH_DIR}/deidentified-new"

REPO_DIR="${HOME}/train-cnn-models"
EVAL_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/batch_cohort_evaluator.py"
REPORT_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/build_cohort_report.py"

SPLIT_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/stratified_held_out_v2.json"
CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_transunet_20m_18777225"
REPORT_DIR="${SCRATCH_DIR}/reports/report_transunet_20m_18777225_ep12"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"

mkdir -p "${REPORT_DIR}" "${ASSETS_DIR}"

MODEL_CKPT="${CHECKPOINT_DIR}/latest_volumetric_rnfl_net.pt"

echo "[Model Checkpoint] ${MODEL_CKPT}"
echo "[Report Output] ${REPORT_DIR}"

export PYTHONUNBUFFERED=1

echo ""
echo "##################################################################"
echo "### STEP 1: HIGH-THROUGHPUT BATCH COHORT EVALUATION (EPOCH 12) ###"
echo "##################################################################"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${MODEL_CKPT}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${REPORT_DIR}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --os_orientation_mode corrected \
    --disc_cut_mode cup \
    --batch_size 8 \
    --device cuda

echo ""
echo "##################################################################"
echo "### STEP 2: COMPILING EXECUTIVE CLINICAL COHORT REPORT         ###"
echo "##################################################################"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_transunet_20m_ep12_${TIMESTAMP}.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_transunet_20m_ep12_${TIMESTAMP}.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "18777225_ep12" \
    --checkpoint_name "rnfl_transunet_20m_18777225_ep12" \
    --model_variant "TransUNet Balanced Hybrid 20M (Epoch 12)" \
    --model_desc "Iso-Capacity TransUNet with balanced CNN pyramid (64-128-256) and ViT-8/384 encoder (~19.97M params) - Epoch 12 Checkpoint"

echo "[Step 2 SUCCESS] Evaluation and report finished."
echo "=================================================================="
echo "=== Job Completed Successfully at $(date) ==="
echo "=================================================================="
