#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: Phase 2 Fine-Tuning on Human-Audited Subjects (Jubail)
# Dataset: Strictly /scratch/${USER}/deidentified-new
# Split:
#   - Train: frozen Phase 2 audited manifest (20 subjects, 40 scans)
#   - Val: frozen stratified held-out v2 manifest (20 subjects, 40 scans)
#   - ZERO OVERLAP GUARANTEE between train and val
# Partition: nvidia (NVIDIA A100 GPU, 40GB VRAM)
# ==============================================================================

#SBATCH -p nvidia                      # Request GPU partition (nvidia)
#SBATCH --gres=gpu:a100:1              # Request 1 NVIDIA A100 GPU (40 GB VRAM)
#SBATCH -n 1                           # 1 task
#SBATCH -c 8                           # 8 CPU cores for data loading
#SBATCH --mem=32G                      # 32 GB system memory
#SBATCH -t 02:00:00                    # Wall clock time limit (2 hours sufficient for 10 epochs)
#SBATCH -J rnfl_finetune_audited       # Job name
#SBATCH -o slurm_rnfl_finetune_%j.out  # Standard output log (%j expands to jobID)
#SBATCH -e slurm_rnfl_finetune_%j.err  # Standard error log

set -e

echo "=================================================================="
echo "=== NYUAD HPC Job Started on $(hostname) at $(date) ==="
echo "=== Job ID: ${SLURM_JOB_ID} | Allocated GPU: ${CUDA_VISIBLE_DEVICES} ==="
echo "=== Phase 2: Human-Audited Fine-Tuning (Strict Subject Separation) ==="
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

# Pre-trained Phase 1 checkpoint evaluated in the 20-subject report.
BASE_CHECKPOINT="${SCRATCH_DIR}/checkpoints/rnfl_biplanar_18563914/best_volumetric_rnfl_net.pt"

if [ ! -f "${BASE_CHECKPOINT}" ]; then
    echo "[ERROR] Base checkpoint not found at: ${BASE_CHECKPOINT}"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_biplanar_finetuned_${SLURM_JOB_ID}"
REPORT_DIR="${SCRATCH_DIR}/reports/report_biplanar_finetuned_${SLURM_JOB_ID}"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${ASSETS_DIR}"

SPLIT_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/stratified_held_out_v2.json"
PRETRAIN_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/phase1_train_manifest.json"
TRAIN_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/phase2_audited_train_manifest.json"

echo ""
echo "##################################################################"
echo "### STEP 1: FINE-TUNING ON 20 HUMAN-AUDITED TRAINING SUBJECTS  ###"
echo "##################################################################"
echo "Base Checkpoint:   ${BASE_CHECKPOINT}"
echo "Training Manifest: ${TRAIN_MANIFEST}"
echo "Val Manifest:      ${SPLIT_MANIFEST}"

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --dataset_root "${DATASET_ROOT}" \
    --resume_checkpoint "${BASE_CHECKPOINT}" \
    --train_manifest "${TRAIN_MANIFEST}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --epochs 10 \
    --base_channels 32 \
    --batch_size 32 \
    --accum_steps 2 \
    --lr 2e-5 \
    --weight_decay 1e-4 \
    --context_slices 5 \
    --num_workers 6 \
    --precision "bf16" \
    --enable_orthogonal \
    --no_augment \
    --device "cuda" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --log_interval 20

BEST_MODEL="${CHECKPOINT_DIR}/best_volumetric_rnfl_net.pt"
if [ ! -f "${BEST_MODEL}" ]; then
    BEST_MODEL="${CHECKPOINT_DIR}/latest_volumetric_rnfl_net.pt"
fi
echo "[Step 1 SUCCESS] Best Fine-Tuned Model saved to: ${BEST_MODEL}"

echo ""
echo "##################################################################"
echo "### STEP 2: HIGH-THROUGHPUT BI-PLANAR FUSION EVALUATION        ###"
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

echo "[Step 2 SUCCESS] Fine-Tuned Evaluation completed."

echo ""
echo "##################################################################"
echo "### STEP 3: COMPILING EXECUTIVE CLINICAL COHORT REPORT         ###"
echo "##################################################################"

OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_finetuned_job_${SLURM_JOB_ID}_${TIMESTAMP}.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_finetuned_job_${SLURM_JOB_ID}_${TIMESTAMP}.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --train_manifest "${PRETRAIN_MANIFEST}" \
    --fine_tune_manifest "${TRAIN_MANIFEST}" \
    --training_stage "Phase 1 pretraining and Phase 2 audited fine-tuning" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "${SLURM_JOB_ID}" \
    --checkpoint_name "rnfl_biplanar_finetuned_${SLURM_JOB_ID}" \
    --model_variant "Bi-Planar Orthogonal Fine-Tuned" \
    --model_desc "Bi-Planar 2.5D ResNet (6.6M), Phase 1 119-subject pretraining + Phase 2 20-subject audited fine-tuning (lr=2e-5)"

echo "[Step 3 SUCCESS] Executive Report Compiled: ${OUTPUT_PDF}"
echo "=================================================================="
echo "=== Job Completed Successfully at $(date) ==="
echo "=================================================================="
