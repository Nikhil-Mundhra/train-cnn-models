# RNFL Evaluation, Orientation Audit, and Quality Control

This workflow keeps the evaluation population, eye orientation, comparator
availability, and manual-review decisions explicit. It is an engineering
evaluation framework, not a validated clinical decision system.

## Corrected OS orientation

Training presents OS data in a standardized orientation. Horizontal inference
flips OS inputs and restores its output to native coordinates. Vertical
inference must likewise write each standardized prediction back to its mirrored
native column before horizontal/vertical fusion.

`--os_orientation_mode corrected` is the production default. The other modes
exist only for controlled experiments:

- `legacy_vertical_mirror` reproduces the historical vertical write-index bug.
- `native` disables OS standardization and is an ablation, not a deployment mode.

Run the paired orientation/fusion experiment on selected subjects with:

```bash
python model_training/train_rnfl_volumetric/run_orientation_ablation.py \
  --checkpoint checkpoints/rnfl_biplanar_18223981/best_volumetric_rnfl_net.pt \
  --dataset_root /path/to/deidentified \
  --output_dir /path/to/orientation_ablation \
  --subjects BEH0086,BEH0314,BEH0335 \
  --eyes OD,OS
```

The JSON and CSV outputs preserve scan-level metrics for every variant. A valid
regression result should show no meaningful OD change between `corrected` and
`legacy_biplanar`, because the correction is OS-specific. OS corrected results
should then be compared with both the legacy and horizontal-only variants.

## Subject-disjoint or external evaluation

Assign each subject exactly once in a JSON or CSV manifest. See
`external_cohort_manifest.example.json`. CSV files require `subject` and `split`
columns. Recognized evaluation split names include `validation`, `held_out`,
`test`, `external`, and `external_test`.

```bash
python model_training/train_rnfl_volumetric/batch_cohort_evaluator.py \
  --checkpoint /path/to/best_volumetric_rnfl_net.pt \
  --dataset_root /path/to/external_deidentified \
  --output_dir /path/to/external_report \
  --split_manifest model_training/train_rnfl_volumetric/external_cohort_manifest.example.json \
  --os_orientation_mode corrected
```

Subjects absent from the manifest are excluded. The manifest should be frozen
before evaluation, and no subject in an evaluation split may have contributed
B-scans, annotations, augmentation sources, threshold tuning, or checkpoint
selection during training.

## Automatic review routing

Each run writes `manual_review_queue.json` and `manual_review_queue.csv`.
Prediction-only checks cover:

- ILM/NFL surface-order violations;
- expected-mask dropout;
- discontinuous NFL boundaries;
- implausible cup fraction;
- horizontal/vertical boundary and probability disagreement.

When reference annotations are available, the queue also applies the report's
operational Dice, MABE, and cup-IoU limits. These defaults are explicitly
engineering thresholds and require calibration on an independent clinical
cohort before they can support clinical use.

## Commercial comparator availability

Every run also writes `commercial_pairing_audit.csv`. It distinguishes genuinely
paired commercial annotations from missing annotations and excluded unedited
mirror files. Paired effect estimates must use only the `paired` rows and should
report the number improved, tied, and worsened for each metric. A second-reader
comparison cannot be estimated until independently edited second-reader
annotations are supplied.

Every run additionally writes `audit_correction_analysis.csv`. The primary
raw-versus-audit comparison is restricted to NFL columns displaced by at least
1 pixel during human audit. It reports raw and U-Net MABE on those edited
columns, normalized correction gain, the fraction of edits recovered, and
preservation within 1 pixel where the raw boundary was accepted. Cup-region
outputs include presence recall and conditional left-edge, right-edge, and width
errors along the fast axis. Whole-mask raw-versus-audit Dice remains descriptive
because the audited annotation is derived from the raw commercial result.
