#!/bin/bash
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="train-cnn-models/reports/baseline_check_${TIMESTAMP}"
mkdir -p "${REPORT_DIR}"
CHECKPOINT="train-cnn-models/checkpoints/rnfl_biplanar_18563914/best_volumetric_rnfl_net.pt"
DATASET_ROOT="/Users/nikhilmundhra/Library/CloudStorage/Box-Box/OCT_Segmentations_Solix/deidentified-new"
SUBJECTS="BEH0314,BEH0335"

echo "=== [1/2] Starting Baseline Evaluation Check for Subjects: ${SUBJECTS} ==="
python3 train-cnn-models/model_training/train_rnfl_volumetric/batch_cohort_evaluator.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${REPORT_DIR}" \
    --subjects "${SUBJECTS}" \
    --os_orientation_mode "corrected" \
    --batch_size 4 \
    --device "mps"

echo "=== [2/2] Compiling Executive Clinical Report ==="
METRICS_JSON="${REPORT_DIR}/assets/executive_cohort_report/cohort_evaluation_metrics.json"
ASSETS_DIR="${REPORT_DIR}/assets/executive_cohort_report"
OUTPUT_MD="${REPORT_DIR}/baseline_executive_report.md"

if [ -f "${METRICS_JSON}" ]; then
    python3 train-cnn-models/model_training/train_rnfl_volumetric/build_cohort_report.py \
        --metrics_json "${METRICS_JSON}" \
        --assets_dir "${ASSETS_DIR}" \
        --output_md "${OUTPUT_MD}" \
        --job_id "baseline_${TIMESTAMP}" \
        --checkpoint_name "rnfl_biplanar_18563914" \
        --model_variant "Bi-Planar Orthogonal Heavy (Baseline)" \
        --model_desc "Mutual core held-out audited baseline evaluation on BEH0314, BEH0335" || true
    echo "=== Baseline Report successfully saved at ${OUTPUT_MD} ==="
else
    echo "[ERROR] Metrics JSON was not generated at ${METRICS_JSON}"
fi
