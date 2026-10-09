#!/bin/bash -l
# ==============================================================================
# SLURM Job Script: High-Throughput Statistical Comparison (New 61s vs Old 61s)
# Target Cohort: All 78 unseen subjects (156 scans) in deidentified-new
# Evaluates: Parametric t-test, Wilcoxon signed-rank, TOST equivalence, Cohen's d
# Partition: nvidia (NVIDIA V100 GPU)
# ==============================================================================

#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -J eval_61s_compare
#SBATCH -o slurm_eval_61s_compare_%j.out
#SBATCH -e slurm_eval_61s_compare_%j.err

set -euo pipefail

echo "=================================================================="
echo "=== NYUAD HPC Job Started on $(hostname) at $(date) ==="
echo "=== Job ID: ${SLURM_JOB_ID} | GPU: ${CUDA_VISIBLE_DEVICES:-all} ==="
echo "=== Comparing New 61s vs Old 61s across all 78 Unseen Subjects ==="
echo "=================================================================="

module purge

NETID="${USER}"
SCRATCH_DIR="/scratch/${NETID}"
PYTHON_BIN="${SCRATCH_DIR}/envs/oct-env/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[ERROR] Python not found at: ${PYTHON_BIN}" >&2
    exit 1
fi

DATASET_ROOT="${SCRATCH_DIR}/deidentified-new"
REPO_DIR="${HOME}/train-cnn-models"
SCRIPT="${REPO_DIR}/model_training/train_rnfl_volumetric/compare_61s_models_unseen_cohort.py"

TRAIN_MANIFEST_NEW="${REPO_DIR}/model_training/train_rnfl_3d/manifests/train_manifest_61subj_v2.json"
SPLIT_MANIFEST_V2="${REPO_DIR}/model_training/train_rnfl_3d/manifests/stratified_held_out_v2.json"

OLD_CHECKPOINT="${REPO_DIR}/checkpoints/jubail/rnfl_biplanar_18266075/best_volumetric_rnfl_net.pt"
if [ ! -f "${OLD_CHECKPOINT}" ]; then
    OLD_CHECKPOINT="${SCRATCH_DIR}/checkpoints/rnfl_biplanar_18266075/best_volumetric_rnfl_net.pt"
fi

# Pick newest 61s checkpoint:
NEW_CHECKPOINT=$(ls -td ${SCRATCH_DIR}/checkpoints/rnfl_biplanar_61subj_* 2>/dev/null | head -n 1)/best_volumetric_rnfl_net.pt
if [ ! -f "${NEW_CHECKPOINT}" ]; then
    echo "[ERROR] New 61s checkpoint not found at: ${NEW_CHECKPOINT}" >&2
    exit 1
fi

OUTPUT_DIR="${SCRATCH_DIR}/reports/statistical_comparison_61s_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

echo "[Model A - Old 61s] ${OLD_CHECKPOINT}"
echo "[Model B - New 61s] ${NEW_CHECKPOINT}"
echo "[Output Directory] ${OUTPUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT}" \
    --old_checkpoint "${OLD_CHECKPOINT}" \
    --new_checkpoint "${NEW_CHECKPOINT}" \
    --dataset_root "${DATASET_ROOT}" \
    --train_manifest_new "${TRAIN_MANIFEST_NEW}" \
    --split_manifest_v2 "${SPLIT_MANIFEST_V2}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "cuda" \
    --batch_size 8

echo "=================================================================="
echo "=== Statistical Evaluation Completed at $(date) ==="
echo "=== Output Report: ${OUTPUT_DIR}/statistical_saturation_report_61s_models.md ==="
echo "=================================================================="
