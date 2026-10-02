# Phase 0: 3D RNFL Experiment Contract

This phase must pass before production model or cohort-report integration.

## Frozen comparison population

- Dataset: `deidentified-new` only.
- Primary successor manifest: `manifests/successor_held_out_v1.json` (10 subjects held out by the current 61-subject BiPlanar model and planned audited fine-tuning run).
- Cross-legacy manifest: `manifests/mutual_held_out_v1.json` (`BEH0314` and `BEH0335`, the intersection held out by every retained historical checkpoint). Use this only when an older two-subject-validation checkpoint is part of the comparison.
- Annotation tiers are frozen from dataset structure: paired `tsv/bad` + `tsv/good` subjects are human-audited/corrected; `tsv/good`-only subjects are accepted-as-segmented.
- Both eyes remain in evaluation. `BEH0335 OS` is an accepted unedited mirror, not an independently corrected comparator.
- Successor training, fine-tuning, threshold calibration, and checkpoint selection must exclude every subject in the primary successor manifest.

## Spatial and metric contract

- Array order: `(slow_z, axial_y, fast_x)`.
- Spacing: `(18.81, 3.12367, 18.75)` micrometres.
- Whole-volume metrics: Dice, Hausdorff, HD95, and mean surface distance from `seg-metrics==1.2.8`; bounded volume similarity is computed explicitly as $1-|V_p-V_g|/(V_p+V_g)$ because the package's `vs` field is a signed relative volume difference.
- Surface connectivity: face-connected (`fully_connected=False`).
- Empty masks: both empty is a perfect match with zero distance; one empty gives zero overlap and `NaN` distance metrics.
- Clinical metrics remain separate: ILM MABE/P95, NFL MABE/P95 outside annotation-derived NFL-absence columns, and TSNIT-sector deltas.
- “Cup” targets derived from missing NFL coordinates must be described as **NFL-absence regions**, not independent anatomical cup segmentations.

Install and run the metric contract tests:

```bash
.venv/bin/python -m pip install -r model_training/train_rnfl_3d/requirements-evaluation.txt
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python -m unittest \
  model_training/train_rnfl_3d/tests/test_evaluation_metrics.py
```

## A100 feasibility gate

Run this inside a 15-minute interactive Jubail allocation before implementing the production network:

```bash
salloc -p nvidia --gres=gpu:1 -c 4 -t 00:15:00
/scratch/nm4358/envs/oct-env/bin/python \
  model_training/train_rnfl_3d/profile_feasibility.py \
  --output /scratch/nm4358/rnfl_3d_phase0_profile.json
```

The default matrix tests `32x768x32`, `48x768x48`, and `64x768x64` at batch sizes 1 and 2 using bfloat16. A candidate passes only when:

1. forward, backward, and optimizer step complete without OOM;
2. peak reserved memory leaves at least 20% of the allocated GPU memory free;
3. three measured steps complete with finite loss;
4. the result JSON is retained with the experiment artifacts.

If `64x768x64` fails, test an anatomy-preserving axial ROI with a stored global axial offset before reducing transverse context. Forward-pass count is not a speed claim: end-to-end inference latency must be measured separately. With a `64x64` transverse window and 50% overlap, a `320x320` volume requires 81 patches.
