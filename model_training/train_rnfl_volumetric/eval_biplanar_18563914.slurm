#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: Fast GPU Evaluation & Report for Job 18563914 with Adaptive Deep-Dives
# Duration: ~8 minutes
# ==============================================================================

#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:25:00
#SBATCH -J eval_biplanar_18563914
#SBATCH -o /home/nm4358/train-cnn-models/slurm_eval_biplanar_18563914.out
#SBATCH -e /home/nm4358/train-cnn-models/slurm_eval_biplanar_18563914.err

set -e

module purge

NETID="${USER}"
SCRATCH_DIR="/scratch/${NETID}"
PYTHON_BIN="${SCRATCH_DIR}/envs/oct-env/bin/python"
DATASET_ROOT="${SCRATCH_DIR}/deidentified-new"
CHECKPOINT="${SCRATCH_DIR}/checkpoints/rnfl_biplanar_18563914/best_volumetric_rnfl_net.pt"
REPORT_DIR="${SCRATCH_DIR}/reports/report_biplanar_18563914"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"

REPO_DIR="${HOME}/train-cnn-models"
EVAL_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/batch_cohort_evaluator.py"
REPORT_SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/build_cohort_report.py"
SPLIT_MANIFEST="${REPO_DIR}/model_training/train_rnfl_3d/manifests/stratified_held_out_v2.json"

echo "=================================================================="
echo "=== Evaluating Model (18563914) with Adaptive Deep-Dive Panels ==="
echo "=================================================================="

"${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${REPORT_DIR}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --os_orientation_mode "corrected" \
    --batch_size 8 \
    --device "cuda"

echo "[SUCCESS] Evaluation finished. Compiling Markdown & PDF Assets..."

OUTPUT_MD="${REPORT_DIR}/executive_cohort_rnfl_report_biplanar_job_18563914_20261003_000432.md"
OUTPUT_PDF="${REPORT_DIR}/executive_cohort_rnfl_report_biplanar_job_18563914_20261003_000432.pdf"

"${PYTHON_BIN}" "${REPORT_SCRIPT}" \
    --metrics_json "${ASSETS_DIR}/cohort_evaluation_metrics.json" \
    --assets_dir "${ASSETS_DIR}" \
    --output_md "${OUTPUT_MD}" \
    --output_pdf "${OUTPUT_PDF}" \
    --job_id "18563914" \
    --checkpoint_name "rnfl_biplanar_18563914" \
    --model_variant "Bi-Planar Orthogonal Heavy" \
    --model_desc "Bi-Planar 2.5D Fusion, Tversky + Boundary BCE + Thickness Integral, base_channels=32 (~6.69M params)"

echo "[SUCCESS] Complete."
