# Phase 2 audited fine-tuning regression: Jubail job 18703058

**Status: failed candidate. Do not promote its checkpoint over the Phase 1 model.** This record covers the two-V100 canonical robust bi-planar experiment submitted on 7 October 2026. It does not establish that all audited fine-tuning recipes fail.

## Reproduction record

- Recipe: [`train_rnfl_finetune_canonical_v100_jubail.slurm`](../train_rnfl_finetune_canonical_v100_jubail.slurm), submitted as job `18703058` on `dn008` (two 32 GiB V100s, distributed training).
- Phase 1 starting checkpoint: `/scratch/nm4358/checkpoints/rnfl_canonical_robust_18697275/best_volumetric_rnfl_net.pt`.
- Training: [`phase2_audited_train_manifest.json`](../../train_rnfl_3d/manifests/phase2_audited_train_manifest.json), 20 human-audited subjects / 40 scans. This is a subset of the Phase 1 training cohort (119 subjects / 237 scans: 20 audited / 40 scans and 99 accepted-as-segmented / 197 scans).
- Validation: [`stratified_held_out_v2.json`](../../train_rnfl_3d/manifests/stratified_held_out_v2.json), 20 disjoint subjects / 40 scans. Subject isolation was checked. The same 40 subject-eye IDs were used for the Phase 1 and fine-tuned evaluations.
- Training settings: five epochs, learning rate `2e-5`, weight decay `1e-4`, batch size four per rank, accumulation eight, effective batch 64, FP16, orthogonal bi-planar model with STN, `--no_augment`.
- Evaluation: the selected best fine-tuned checkpoint (epoch four), corrected OS orientation, cup disc cut, bi-planar fusion, and the same held-out subject-eye IDs used for the Phase 1 evaluation. Both metrics files record corrected orientation and bi-planar fusion. The older Phase 1 metrics file does not record its disc-cut setting, so exact provenance for that setting remains to be checked against its original evaluator command.

The training loop selected epoch four at validation peripapillary NFL MABE 28.06 µm; epoch five measured 31.69 µm. The starting checkpoint's historical training-loop baseline was 11.01 µm. These training-loop values have a different aggregation and inference path from the cohort evaluator below and should not be mixed with its scores. One non-primary distributed rank printed `inf` as its best score; rank zero saved and evaluated the epoch-four checkpoint.

## Paired held-out evaluation

| Endpoint | Phase 1 mean / median | Fine-tuned mean / median | Mean change | Scans improved / worsened |
| --- | ---: | ---: | ---: | ---: |
| Peripapillary boundary MABE (µm; lower is better) | 10.12 / 9.77 | 54.18 / 54.60 | +44.07 | 0 / 40 |
| RNFL Dice (higher is better) | 0.86 / 0.86 | 0.70 / 0.72 | -0.16 | 0 / 40 |
| P95 boundary error (µm; lower is better) | 25.28 / 23.39 | 110.53 / 112.01 | +85.25 | 0 / 40 |
| Cup IoU (higher is better) | 0.89 / 0.91 | 0.84 / 0.86 | -0.05 | 9 / 31 |

The largest peripapillary MABE increase was BEH0043 OS, from 7.16 to 99.71 µm (+92.55 µm). Every held-out scan had higher MABE and P95 and lower Dice after fine-tuning. This is a broad regression, not an isolated outlier.

On the 10 held-out scans with materially edited manual-reference columns, the fine-tuned model beat the raw commercial boundary on one scan. Across 106,689 edited columns, pooled boundary MABE was 42.63 µm for the model and 23.57 µm for the raw commercial boundary. In the 10 full-cube signed thickness-map pairs, the model's mean per-eye thickness MAE was 7.79 µm versus 2.86 µm for the commercial estimate; it won on zero eyes. Boundary MABE and thickness MAE are different endpoints. A near-zero commercial error on an accepted-as-segmented region can be a comparison with its own source, so that region is not independent evidence of commercial accuracy.

Both evaluations flagged 40/40 scans for manual review under current operational defaults. The count of flags alone therefore cannot distinguish the two models, and these thresholds are not clinically validated.

## Decision and pipeline treatment

1. **Keep the script and checkpoints as a historical failed experiment.** Do not stash or remove the committed recipe: it records the exact settings needed to reproduce and investigate this run. The script header marks its status. Do not submit the unchanged recipe as a prospective improvement or deploy its checkpoint.
2. **Change the *next* Phase 2 candidate pipeline before promotion.** Evaluate the Phase 1 starting checkpoint and every candidate on the same subject-eye manifest and evaluator settings, and record those settings in the metrics. Save a paired comparison. Require improvement on the prespecified primary endpoint (held-out peripapillary MABE) and review Dice, P95, and audited-only performance before calling a candidate successful. Selecting the lowest MABE *among fine-tuned epochs* is insufficient when every epoch may be worse than the starting model. A report should state `failed candidate` when this gate fails.
3. **Keep other Phase 2 jobs separate.** The existing one-H200 canonical robust script and the older biplanar audited script use different settings/checkpoints. Their outcomes require their own paired evaluations; this result does not adjudicate them. Do not cancel or modify those jobs on the basis of job `18703058` alone.

The cause of the regression is not established. Possible factors to test one at a time include the unaugmented five-epoch schedule, learning rate or update count, and the effect of optimizing a narrow audited subset. Avoid attributing it to the audited labels or to V100 hardware without a controlled comparison.

The compute node produced Markdown and all report assets but lacked the PDF compilation engine. The PDF was rebuilt locally from those assets; that packaging issue is separate from the model regression.

## Evidence locations

- Fine-tuned cohort metrics: `/scratch/nm4358/reports/report_canonical_finetuned_v100_18703058/assets/executive_cohort_report/cohort_evaluation_metrics.json`.
- Fine-tuned checkpoints: `/scratch/nm4358/checkpoints/rnfl_canonical_finetuned_v100_18703058/`.
- Downloaded paired analysis: `Capstone/scratch/rnfl_finetune_v100_18703058/evaluation_comparison_18703058.md` (local workspace artifact, not tracked in this repository).
- Downloaded report and rebuilt PDF: `Capstone/scratch/rnfl_finetune_v100_18703058/report_canonical_finetuned_v100_18703058/` (local workspace artifact).
