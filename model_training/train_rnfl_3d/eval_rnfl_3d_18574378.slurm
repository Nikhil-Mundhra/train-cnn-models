#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: Fast GPU Evaluation & Report for Job 18574378 (3D U-Net)
# Evaluates 20 held-out validation subjects (40 volumes) using AnisotropicRNFLUNet3D
# ==============================================================================

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J eval_rnfl_3d_18574378
#SBATCH -o /home/nm4358/train-cnn-models/slurm_eval_rnfl_3d_18574378.out
#SBATCH -e /home/nm4358/train-cnn-models/slurm_eval_rnfl_3d_18574378.err

set -e

module purge

NETID="${USER}"
SCRATCH_DIR="/scratch/${NETID}"
PYTHON_BIN="${SCRATCH_DIR}/envs/oct-env/bin/python"
DATASET_ROOT="${SCRATCH_DIR}/deidentified-new"
CHECKPOINT="${SCRATCH_DIR}/checkpoints/rnfl_3d_18574378/best_rnfl_3d_net.pt"
REPORT_DIR="${SCRATCH_DIR}/reports/report_rnfl_3d_18574378"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"

REPO_DIR="${HOME}/train-cnn-models"
EVAL_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/batch_cohort_evaluator.py"
REPORT_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/build_cohort_report.py"
SPLIT_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/stratified_held_out_v2.json"

export PYTHONUNBUFFERED=1

echo "=================================================================="
echo "=== Evaluating Dense Anisotropic 3D U-Net (Job 18574378)       ==="
echo "=== 20 Held-Out Validation Subjects (40 Paired 3D Volumes)     ==="
echo "=================================================================="

rm -rf "${REPORT_DIR}"
mkdir -p "${REPORT_DIR}"

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${REPORT_DIR}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --os_orientation_mode "corrected" \
    --model_family "3d" \
    --val_stride 64 \
    --batch_size 4 \
    --device "cuda"

echo "[SUCCESS] Evaluation finished. Compiling Markdown & PDF Assets..."

OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_3d_job_18574378.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_3d_job_18574378.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "18574378" \
    --checkpoint_name "rnfl_3d_18574378" \
    --model_variant "Dense Anisotropic 3D U-Net" \
    --model_desc "Dense Anisotropic 3D U-Net (Phase 0 Candidate), 3D sliding-window batched inference, stride=64, base_channels=32 (~16.4M params)"

echo "[SUCCESS] Evaluation & Report generation complete."
