#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: Fast GPU Evaluation & PDF Report for Light Model (18043443)
# Duration: ~90 seconds
# ==============================================================================

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -J eval_light
#SBATCH -o slurm_eval_light_%j.out
#SBATCH -e slurm_eval_light_%j.err

set -e

module purge

NETID="${USER}"
SCRATCH_DIR="/scratch/${NETID}"
PYTHON_BIN="${SCRATCH_DIR}/envs/oct-env/bin/python"

DATASET_ROOT="${SCRATCH_DIR}/deidentified"
CHECKPOINT="${SCRATCH_DIR}/checkpoints/rnfl_light_18043443/best_volumetric_rnfl_net.pt"
REPORT_DIR="${SCRATCH_DIR}/reports/report_light_18043443"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"

REPO_DIR="${HOME}/train-cnn-models"
EVAL_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/batch_cohort_evaluator.py"
REPORT_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/build_cohort_report.py"

mkdir -p "${ASSETS_DIR}"

echo "=================================================================="
echo "=== Evaluating Pre-Trained Light Model (18043443) on GPU ==="
echo "=================================================================="

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${REPORT_DIR}" \
    --val_subjects "BEH0335,BEH0314,BEH0086" \
    --device "cuda"

echo "[SUCCESS] Evaluation finished. Compiling PDF Report..."

OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_light_18043443.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_light_18043443.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "18043443" \
    --checkpoint_name "rnfl_light_18043443" \
    --model_variant "Light" \
    --model_desc "base_channels=16, 1.67M params, batch_size=32"

echo "=================================================================="
echo "=== Light Model Report Generated Successfully! ==="
echo "PDF Path: ${OUTPUT_PDF}"
echo "=================================================================="
