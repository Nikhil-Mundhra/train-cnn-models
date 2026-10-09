#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: Fast GPU Evaluation & Report Compilation for TransUNet (Job 18724310)
# Evaluates 40 held-out validation scans and compiles Markdown + PDF clinical reports
# Partition: nvidia (1x NVIDIA V100 GPU on dn001/dn002)
# ==============================================================================

#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -J eval_trans_18724310
#SBATCH -o /home/nm4358/train-cnn-models/slurm_eval_transunet_18724310.out
#SBATCH -e /home/nm4358/train-cnn-models/slurm_eval_transunet_18724310.err

set -euo pipefail

echo "=================================================================="
echo "=== NYUAD HPC Job Started on $(hostname) at $(date) ==="
echo "=== Job ID: ${SLURM_JOB_ID} | GPU: ${CUDA_VISIBLE_DEVICES:-all} ==="
echo "=== Evaluating TransUNet Checkpoint (Job 18724310) across 40 Scans ==="
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
TRAIN_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/phase1_train_manifest.json"

CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_transunet_4v100_18724310"
REPORT_DIR="${SCRATCH_DIR}/reports/report_transunet_4v100_18724310"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"

mkdir -p "${REPORT_DIR}" "${ASSETS_DIR}"

BEST_MODEL="${CHECKPOINT_DIR}/best_volumetric_rnfl_net.pt"
if [ ! -f "${BEST_MODEL}" ]; then
    BEST_MODEL="${CHECKPOINT_DIR}/latest_volumetric_rnfl_net.pt"
fi

echo "[Model Checkpoint] ${BEST_MODEL}"
echo "[Report Output] ${REPORT_DIR}"

echo ""
echo "##################################################################"
echo "### STEP 1: HIGH-THROUGHPUT BATCH COHORT EVALUATION             ###"
echo "##################################################################"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${BEST_MODEL}" \
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
OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_transunet_4v100_job_18724310_${TIMESTAMP}.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_transunet_4v100_job_18724310_${TIMESTAMP}.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --train_manifest "${TRAIN_MANIFEST}" \
    --training_stage "Phase 1 broad-cohort Anisotropic TransUNet training (4x V100)" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "18724310" \
    --checkpoint_name "rnfl_transunet_4v100_18724310" \
    --model_variant "Anisotropic TransUNet Hybrid (4x V100)" \
    --model_desc "Anisotropic TransUNet hybrid with axial-preserving tokenization, multi-scale boundary injection, and STN canonicalizer (~3.76M params)"

echo "=================================================================="
echo "=== TransUNet Evaluation Complete: ${OUTPUT_MD} ==="
echo "=================================================================="
