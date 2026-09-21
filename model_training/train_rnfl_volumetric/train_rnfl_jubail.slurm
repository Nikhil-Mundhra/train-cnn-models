#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: Multi-Model Benchmark & Comparative Clinical Pipeline (Jubail)
# Trains, Evaluates, and Compiles Reports for BOTH Model Architectures:
#   1. LIGHT MODEL:  base_channels=16 (~1.67M params) -> Train -> Eval -> PDF Report
#   2. HEAVY MODEL:  base_channels=32 (~6.69M params) -> Train -> Eval -> PDF Report
# Partition: nvidia (NVIDIA A100 GPU)
# Reference: https://crc-docs.abudhabi.nyu.edu/hpc/jobs/quick_start.html
# ==============================================================================

#SBATCH -p nvidia                      # Request GPU partition (nvidia)
#SBATCH --gres=gpu:a100:1              # Request 1 NVIDIA A100 GPU (40 GB VRAM)
#SBATCH -n 1                           # 1 task
#SBATCH -c 8                           # 8 CPU cores for data loading
#SBATCH --mem=32G                      # 32 GB system memory
#SBATCH -t 04:00:00                    # Wall clock time limit (4 hours)
#SBATCH -J rnfl_dual_models            # Job name
#SBATCH -o slurm_rnfl_%j.out           # Standard output log (%j expands to jobID)
#SBATCH -e slurm_rnfl_%j.err           # Standard error log

set -e

echo "=================================================================="
echo "=== NYUAD HPC Job Started on $(hostname) at $(date) ==="
echo "=== Job ID: ${SLURM_JOB_ID} | Allocated GPU: ${CUDA_VISIBLE_DEVICES} ==="
echo "=================================================================="

# ----------------------------------------------------------------------
# 1. Environment & Paths Configuration
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

# ==============================================================================
# PIPELINE PART 1: LIGHT MODEL (base_channels=16, ~1.67M params)
# ==============================================================================
echo ""
echo "##################################################################"
echo "### PART 1: LIGHT MODEL (base_channels=16, ~1.67M parameters)  ###"
echo "##################################################################"

CHECKPOINT_DIR_LIGHT="${SCRATCH_DIR}/checkpoints/rnfl_light_${SLURM_JOB_ID}"
REPORT_DIR_LIGHT="${SCRATCH_DIR}/reports/report_light_${SLURM_JOB_ID}"
ASSETS_DIR_LIGHT="${REPORT_DIR_LIGHT}/assets/executive_cohort_report"
mkdir -p "${CHECKPOINT_DIR_LIGHT}"
mkdir -p "${ASSETS_DIR_LIGHT}"

echo "[Step 1.1] Training Light Volumetric U-Net (20 Epochs)..."
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --dataset_root "${DATASET_ROOT}" \
    --val_subjects "BEH0335,BEH0314,BEH0086" \
    --epochs 20 \
    --base_channels 16 \
    --batch_size 32 \
    --accum_steps 2 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --context_slices 5 \
    --num_workers 6 \
    --device "cuda" \
    --checkpoint_dir "${CHECKPOINT_DIR_LIGHT}" \
    --log_interval 25

BEST_MODEL_LIGHT="${CHECKPOINT_DIR_LIGHT}/best_volumetric_rnfl_net.pt"
if [ ! -f "${BEST_MODEL_LIGHT}" ]; then
    BEST_MODEL_LIGHT="${CHECKPOINT_DIR_LIGHT}/latest_volumetric_rnfl_net.pt"
fi
echo "[Step 1.1 SUCCESS] Light Model saved to: ${BEST_MODEL_LIGHT}"

echo "[Step 1.2] Evaluating Light Model on GPU across full cohort (46 volumes)..."
"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${BEST_MODEL_LIGHT}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${REPORT_DIR_LIGHT}" \
    --val_subjects "BEH0335,BEH0314,BEH0086" \
    --device "cuda"

echo "[Step 1.2 SUCCESS] Light Model evaluation finished. Generating PDF Report..."
OUTPUT_MD_LIGHT="${REPORT_DIR_LIGHT}/executive_cohort_rnfl_report_light_job_${SLURM_JOB_ID}_${TIMESTAMP}.md"
OUTPUT_PDF_LIGHT="${REPORT_DIR_LIGHT}/executive_cohort_rnfl_report_light_job_${SLURM_JOB_ID}_${TIMESTAMP}.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR_LIGHT}/cohort_evaluation_metrics.json" \
    --assets_dir "${ASSETS_DIR_LIGHT}" \
    --output_md "${OUTPUT_MD_LIGHT}" \
    --output_pdf "${OUTPUT_PDF_LIGHT}" \
    --job_id "${SLURM_JOB_ID}" \
    --checkpoint_name "rnfl_light_${SLURM_JOB_ID}" \
    --model_variant "Light" \
    --model_desc "base_channels=16, 1.67M params, batch_size=32"

echo "[Step 1.3 SUCCESS] Light Model Report compiled: ${OUTPUT_PDF_LIGHT}"

# ==============================================================================
# PIPELINE PART 2: HEAVY / DENSE MODEL (base_channels=32, ~6.69M params)
# ==============================================================================
echo ""
echo "##################################################################"
echo "### PART 2: HEAVY MODEL (base_channels=32, ~6.69M parameters)  ###"
echo "##################################################################"

CHECKPOINT_DIR_HEAVY="${SCRATCH_DIR}/checkpoints/rnfl_heavy_${SLURM_JOB_ID}"
REPORT_DIR_HEAVY="${SCRATCH_DIR}/reports/report_heavy_${SLURM_JOB_ID}"
ASSETS_DIR_HEAVY="${REPORT_DIR_HEAVY}/assets/executive_cohort_report"
mkdir -p "${CHECKPOINT_DIR_HEAVY}"
mkdir -p "${ASSETS_DIR_HEAVY}"

echo "[Step 2.1] Training Heavy Volumetric U-Net (20 Epochs)..."
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
    --checkpoint_dir "${CHECKPOINT_DIR_HEAVY}" \
    --log_interval 25

BEST_MODEL_HEAVY="${CHECKPOINT_DIR_HEAVY}/best_volumetric_rnfl_net.pt"
if [ ! -f "${BEST_MODEL_HEAVY}" ]; then
    BEST_MODEL_HEAVY="${CHECKPOINT_DIR_HEAVY}/latest_volumetric_rnfl_net.pt"
fi
echo "[Step 2.1 SUCCESS] Heavy Model saved to: ${BEST_MODEL_HEAVY}"

echo "[Step 2.2] Evaluating Heavy Model on GPU across full cohort (46 volumes)..."
"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${BEST_MODEL_HEAVY}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${REPORT_DIR_HEAVY}" \
    --val_subjects "BEH0335,BEH0314,BEH0086" \
    --device "cuda"

echo "[Step 2.2 SUCCESS] Heavy Model evaluation finished. Generating PDF Report..."
OUTPUT_MD_HEAVY="${REPORT_DIR_HEAVY}/executive_cohort_rnfl_report_heavy_job_${SLURM_JOB_ID}_${TIMESTAMP}.md"
OUTPUT_PDF_HEAVY="${REPORT_DIR_HEAVY}/executive_cohort_rnfl_report_heavy_job_${SLURM_JOB_ID}_${TIMESTAMP}.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR_HEAVY}/cohort_evaluation_metrics.json" \
    --assets_dir "${ASSETS_DIR_HEAVY}" \
    --output_md "${OUTPUT_MD_HEAVY}" \
    --output_pdf "${OUTPUT_PDF_HEAVY}" \
    --job_id "${SLURM_JOB_ID}" \
    --checkpoint_name "rnfl_heavy_${SLURM_JOB_ID}" \
    --model_variant "Heavy" \
    --model_desc "base_channels=32, 6.69M params, batch_size=32"

echo "[Step 2.3 SUCCESS] Heavy Model Report compiled: ${OUTPUT_PDF_HEAVY}"

# ==============================================================================
# PIPELINE COMPLETION SUMMARY
# ==============================================================================
echo ""
echo "=================================================================="
echo "=== DUAL-MODEL PIPELINE COMPLETED SUCCESSFULLY AT $(date) ==="
echo "=================================================================="
echo "1. LIGHT MODEL (1.67M params):"
echo "   - Weights:     ${CHECKPOINT_DIR_LIGHT}"
echo "   - Report PDF:  ${OUTPUT_PDF_LIGHT}"
echo "   - Report MD:   ${OUTPUT_MD_LIGHT}"
echo "   - Figures/Dir: ${REPORT_DIR_LIGHT}"
echo ""
echo "2. HEAVY MODEL (6.69M params):"
echo "   - Weights:     ${CHECKPOINT_DIR_HEAVY}"
echo "   - Report PDF:  ${OUTPUT_PDF_HEAVY}"
echo "   - Report MD:   ${OUTPUT_MD_HEAVY}"
echo "   - Figures/Dir: ${REPORT_DIR_HEAVY}"
echo "=================================================================="
