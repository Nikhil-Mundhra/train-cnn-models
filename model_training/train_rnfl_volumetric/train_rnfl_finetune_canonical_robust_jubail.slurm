#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: Phase 2 Canonical Robust Fine-Tuning on Audited Cohort (Jubail)
# Features:
#   - Resumes from Phase 1 Pre-Trained Canonical Robust Model (STN + Robust Augmentations)
#   - Trains for 10 epochs exclusively on 10 human-audited training subjects
#   - Reduced learning rate (lr = 2e-5) to preserve global morphology
#   - Strict Zero-Leakage Guarantee: held-out validation cohort isolated
# Partition: nvidia (NVIDIA A100 GPU, 40GB VRAM)
# ==============================================================================

#SBATCH -p nvidia                      # Request GPU partition (nvidia)
#SBATCH --gres=gpu:a100:1              # Request 1 NVIDIA A100 GPU (40 GB VRAM)
#SBATCH -n 1                           # 1 task
#SBATCH -c 8                           # 8 CPU cores for data loading
#SBATCH --mem=32G                      # 32 GB system memory
#SBATCH -t 02:30:00                    # Wall clock time limit (2.5 hours sufficient for 10 epochs)
#SBATCH -J rnfl_finetune_canonical     # Job name
#SBATCH -o slurm_rnfl_finetune_canonical_%j.out  # Standard output log (%j expands to jobID)
#SBATCH -e slurm_rnfl_finetune_canonical_%j.err  # Standard error log

set -e

echo "=================================================================="
echo "=== NYUAD HPC Job Started on $(hostname) at $(date) ==="
echo "=== Job ID: ${SLURM_JOB_ID} | Allocated GPU: ${CUDA_VISIBLE_DEVICES} ==="
echo "=== Phase 2: Canonical Robust Fine-Tuning on Human-Audited Data ==="
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

# Pre-trained base weights: allow user override or auto-discover latest Phase 1 canonical run
if [ -n "$1" ] && [ -f "$1" ]; then
    BASE_CHECKPOINT="$1"
else
    # Auto-discover most recent Phase 1 canonical robust checkpoint
    BASE_CHECKPOINT=$(find "${SCRATCH_DIR}/checkpoints" -name "best_volumetric_rnfl_net.pt" -path "*canonical_robust*" 2>/dev/null | sort -V | tail -n 1)
    if [ -z "${BASE_CHECKPOINT}" ] || [ ! -f "${BASE_CHECKPOINT}" ]; then
        # Fallback to standard biplanar production checkpoint
        BASE_CHECKPOINT="${SCRATCH_DIR}/checkpoints/rnfl_biplanar_18563914/best_volumetric_rnfl_net.pt"
    fi
fi

if [ ! -f "${BASE_CHECKPOINT}" ]; then
    echo "[ERROR] Base checkpoint not found at: ${BASE_CHECKPOINT}"
    echo "Usage: sbatch $0 [PATH_TO_PHASE1_CHECKPOINT]"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CHECKPOINT_DIR="${SCRATCH_DIR}/checkpoints/rnfl_canonical_finetuned_${SLURM_JOB_ID}"
REPORT_DIR="${SCRATCH_DIR}/reports/report_canonical_finetuned_${SLURM_JOB_ID}"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${ASSETS_DIR}"

TRAIN_AUDITED="BEH0181,BEH0185,BEH0298,BEH0321,BEH0323,BEH0348,BEH0349,BEH0350,BEH0352,BEH0362"
VAL_SUBJECTS="BEH0335,BEH0314,BEH0174,BEH0310,BEH0354,BEH0086,BEH0090,BEH0096,BEH0241,BEH0284"

echo ""
echo "##################################################################"
echo "### STEP 1: FINE-TUNING ON 10 HUMAN-AUDITED TRAINING SUBJECTS  ###"
echo "##################################################################"
echo "Base Checkpoint:   ${BASE_CHECKPOINT}"
echo "Training Subjects: ${TRAIN_AUDITED}"
echo "Val Subjects:      ${VAL_SUBJECTS}"

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --dataset_root "${DATASET_ROOT}" \
    --resume_checkpoint "${BASE_CHECKPOINT}" \
    --train_subjects "${TRAIN_AUDITED}" \
    --val_subjects "${VAL_SUBJECTS}" \
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
    --use_stn \
    --robust_augment \
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
    --val_subjects "${VAL_SUBJECTS}" \
    --os_orientation_mode "corrected" \
    --disc_cut_mode "cup" \
    --batch_size 8 \
    --device "cuda"

echo "[Step 2 SUCCESS] Fine-Tuned Evaluation completed."

echo ""
echo "##################################################################"
echo "### STEP 3: COMPILING EXECUTIVE CLINICAL COHORT REPORT         ###"
echo "##################################################################"

OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_canonical_finetuned_job_${SLURM_JOB_ID}_${TIMESTAMP}.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_canonical_finetuned_job_${SLURM_JOB_ID}_${TIMESTAMP}.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "${SLURM_JOB_ID}" \
    --checkpoint_name "rnfl_canonical_finetuned_${SLURM_JOB_ID}" \
    --model_variant "Canonical Robust Fine-Tuned (STN)" \
    --model_desc "Bi-Planar 2.5D with STN Canonicalizer & Optical Noise, Phase 1 Pre-trained + Phase 2 10-Epoch Audited Fine-Tuned (lr=2e-5)"

echo "[Step 3 SUCCESS] Executive Report Compiled: ${OUTPUT_PDF}"
echo "=================================================================="
echo "=== Job Completed Successfully at $(date) ==="
echo "=================================================================="
