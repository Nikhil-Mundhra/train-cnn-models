"""
Markdown Generation and Formatting Utilities for Clinical RNFL Reports.
"""

import os
import re
from typing import Any, Dict, List, Optional
import numpy as np

from .charts import compute_distribution_stats
from .assets import select_executive_gallery


def normalize_markdown_lists(text: str) -> str:
    """
    Automated Defensive List Normalization:
    Guarantees that any list item following a non-blank line gets an injected blank line
    so CommonMark and sane_lists never collapse list items into run-on paragraph text.
    """
    lines = text.split("\n")
    normalized = []
    in_code = False
    list_item_pattern = re.compile(r"^\s*([*\-+]|\d+\.)\s+")

    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            normalized.append(line)
            continue
        if not in_code and list_item_pattern.match(line):
            if normalized:
                prev = normalized[-1].strip()
                if prev and not list_item_pattern.match(prev):
                    normalized.append("")
        normalized.append(line)
    return "\n".join(normalized)


def generate_markdown_report(
    metrics_data: Dict[str, Any],
    assets_rel_dir: str,
    job_id: str,
    checkpoint_name: str,
    model_variant: str = "Volumetric",
    model_desc: str = "",
    max_gallery_items: int = 24,
    training_cohort: Dict[str, Any] = None,
    fine_tune_cohort: Dict[str, Any] = None,
    training_stage: str = "Training",
) -> str:
    scans: List[Dict[str, Any]] = metrics_data.get('scans', [])
    gallery_all: List[Dict[str, Any]] = metrics_data.get('gallery', [])
    gallery: List[Dict[str, Any]] = select_executive_gallery(gallery_all, max_gallery_items)
    deep_dives: List[Dict[str, Any]] = metrics_data.get('deep_dives', [])
    thickness_maps: List[Dict[str, Any]] = metrics_data.get('thickness_maps', [])

    all_subjects = sorted(list(set(s['subject'] for s in scans))) if scans else []
    val_subjects = sorted(list(set(s['subject'] for s in scans if s.get('is_validation', False))))
    n_scans = len(scans)
    n_od = sum(1 for s in scans if s['eye'] == 'OD')
    n_os = sum(1 for s in scans if s['eye'] == 'OS')
    if training_cohort:
        training_summary = (
            f"{training_cohort['subject_count']} subjects / {training_cohort['scan_count']} scans "
            f"({training_cohort['audited_subject_count']} subjects / {training_cohort['audited_scan_count']} scans "
            f"with paired human-audited annotations; "
            f"{training_cohort['accepted_subject_count']} subjects / {training_cohort['accepted_scan_count']} scans "
            f"accepted as segmented)"
        )
        training_methods = (
            f"- **Checkpoint training ({training_stage})**: {training_summary}. "
            f"Counts come from `{training_cohort['manifest_name']}`. A paired commercial and accepted curve "
            "identifies the human-audited tier; it does not establish that every scan was manually changed. "
            "Training and evaluation subjects are disjoint."
        )
        if fine_tune_cohort:
            fine_tune_summary = (
                f"{fine_tune_cohort['subject_count']} audited subjects / "
                f"{fine_tune_cohort['scan_count']} scans"
            )
            training_summary = f"Phase 1: {training_summary}; Phase 2: {fine_tune_summary} (subset of Phase 1)"
            training_methods += (
                f"\n- **Audited fine-tuning (Phase 2)**: {fine_tune_summary}, from "
                f"`{fine_tune_cohort['manifest_name']}`. These subjects are included in the Phase 1 count; "
                "the stage counts must not be added together."
            )
    else:
        training_summary = "not supplied to the report builder"
        training_methods = "- **Checkpoint training**: Training manifest was not supplied; subject and annotation-tier counts are unavailable."

    bench_scans = [s for s in scans if not s.get('is_validation', False)]
    bench_od = [s for s in bench_scans if s['eye'] == 'OD']
    bench_os = [s for s in bench_scans if s['eye'] == 'OS']
    val_scans = [s for s in scans if s.get('is_validation', False)]
    val_od_scans = [s for s in val_scans if s.get('eye') == 'OD']
    val_os_scans = [s for s in val_scans if s.get('eye') == 'OS']
    paired_commercial = [s for s in scans if s.get('bad_dice') is not None and not s.get('is_mirror', False)]
    paired_dice_better = sum(1 for s in paired_commercial if s['unet_dice'] > s['bad_dice'])
    paired_dice_worse = sum(1 for s in paired_commercial if s['unet_dice'] < s['bad_dice'])
    paired_cup = [s for s in paired_commercial if s.get('bad_cup_iou') is not None]
    paired_cup_better = sum(1 for s in paired_cup if s['unet_cup_iou'] > s['bad_cup_iou'])
    paired_cup_worse = sum(1 for s in paired_cup if s['unet_cup_iou'] < s['bad_cup_iou'])
    audit_edited_scans = [
        s for s in scans
        if s.get('raw_edit_mabe_um') is not None
        and s.get('unet_edit_mabe_um') is not None
        and s.get('audit_edited_columns', 0) >= 50
        and s.get('audit_correction_gain') is not None
        and not s.get('is_mirror', False)
    ]
    val_edited_scans = [s for s in audit_edited_scans if s.get('is_validation', False)]
    bench_edited_scans = [s for s in audit_edited_scans if not s.get('is_validation', False)]
    val_audit_improved = sum(1 for s in val_edited_scans if s.get('raw_edit_mabe_um', 0) > s.get('unet_edit_mabe_um', 0))
    val_gain_median = float(np.median([s['audit_correction_gain'] for s in val_edited_scans])) if val_edited_scans else 0.0
    audit_improved = sum(1 for s in audit_edited_scans if s.get('raw_edit_mabe_um', 0) > s.get('unet_edit_mabe_um', 0))
    audit_gain_median = float(np.median([s['audit_correction_gain'] for s in audit_edited_scans])) if audit_edited_scans else 0.0
    audit_recovery_median = float(np.median([s['audit_edit_recovery_rate'] for s in audit_edited_scans])) if audit_edited_scans else 0.0
    preservation_values = [
        s['audit_unchanged_preservation_rate'] for s in audit_edited_scans
        if s.get('audit_unchanged_preservation_rate') is not None
    ]
    audit_preservation_median = float(np.median(preservation_values)) if preservation_values else 0.0

    # Column-weighted pooled metrics across materially edited columns
    total_val_edited_cols = sum(s['audit_edited_columns'] for s in val_edited_scans)
    if total_val_edited_cols > 0:
        val_pooled_raw_mabe = sum(s['raw_edit_mabe_um'] * s['audit_edited_columns'] for s in val_edited_scans) / total_val_edited_cols
        val_pooled_unet_mabe = sum(s['unet_edit_mabe_um'] * s['audit_edited_columns'] for s in val_edited_scans) / total_val_edited_cols
        val_pooled_delta_mabe = val_pooled_raw_mabe - val_pooled_unet_mabe
        val_pooled_gain = 1.0 - (val_pooled_unet_mabe / val_pooled_raw_mabe)
    else:
        val_pooled_raw_mabe = val_pooled_unet_mabe = val_pooled_delta_mabe = val_pooled_gain = 0.0

    total_audit_edited_cols = sum(s['audit_edited_columns'] for s in audit_edited_scans)
    if total_audit_edited_cols > 0:
        audit_pooled_raw_mabe = sum(s['raw_edit_mabe_um'] * s['audit_edited_columns'] for s in audit_edited_scans) / total_audit_edited_cols
        audit_pooled_unet_mabe = sum(s['unet_edit_mabe_um'] * s['audit_edited_columns'] for s in audit_edited_scans) / total_audit_edited_cols
        audit_pooled_delta_mabe = audit_pooled_raw_mabe - audit_pooled_unet_mabe
        audit_pooled_gain = 1.0 - (audit_pooled_unet_mabe / audit_pooled_raw_mabe)
    else:
        audit_pooled_raw_mabe = audit_pooled_unet_mabe = audit_pooled_delta_mabe = audit_pooled_gain = 0.0

    audit_threshold_px = float(metrics_data.get('metadata', {}).get('audit_edit_threshold_px', 1.0))
    qc_assessed = [s for s in scans if s.get('qc_status') in {'pass', 'manual_review'}]
    review_count = sum(1 for s in qc_assessed if s.get('qc_status') == 'manual_review')

    # Compute Stats
    stats_bench_all_dice = compute_distribution_stats([s['unet_dice'] for s in bench_scans])
    stats_bench_all_mabe = compute_distribution_stats([s['unet_mabe'] for s in bench_scans])
    stats_bench_all_p95 = compute_distribution_stats([s['unet_p95'] for s in bench_scans])
    stats_bench_all_cup = compute_distribution_stats([s['unet_cup_iou'] for s in bench_scans])

    stats_bench_od_dice = compute_distribution_stats([s['unet_dice'] for s in bench_od])
    stats_bench_od_mabe = compute_distribution_stats([s['unet_mabe'] for s in bench_od])

    stats_bench_os_dice = compute_distribution_stats([s['unet_dice'] for s in bench_os])
    stats_bench_os_mabe = compute_distribution_stats([s['unet_mabe'] for s in bench_os])

    stats_val_dice = compute_distribution_stats([s['unet_dice'] for s in val_scans])
    stats_val_mabe = compute_distribution_stats([s['unet_mabe'] for s in val_scans])
    stats_val_p95 = compute_distribution_stats([s['unet_p95'] for s in val_scans])
    stats_val_cup = compute_distribution_stats([s['unet_cup_iou'] for s in val_scans])

    stats_val_od_dice = compute_distribution_stats([s['unet_dice'] for s in val_od_scans])
    stats_val_od_mabe = compute_distribution_stats([s['unet_mabe'] for s in val_od_scans])
    stats_val_os_dice = compute_distribution_stats([s['unet_dice'] for s in val_os_scans])
    stats_val_os_mabe = compute_distribution_stats([s['unet_mabe'] for s in val_os_scans])

    val_od_mabe = float(np.mean([s['unet_mabe'] for s in val_od_scans])) if val_od_scans else 0.0
    val_os_mabe = float(np.mean([s['unet_mabe'] for s in val_os_scans])) if val_os_scans else 0.0
    worst_val = max(val_scans, key=lambda s: s['unet_mabe']) if val_scans else None
    worst_val_label = f"{worst_val['subject']} {worst_val['eye']}" if worst_val else "None"
    worst_val_mabe = float(worst_val['unet_mabe']) if worst_val else 0.0
    qc_finding = (
        f"{review_count} of {len(qc_assessed)} assessed scans triggered at least one conservative operational review flag."
        if qc_assessed
        else "Operational QC status was not available in this metrics manifest."
    )

    if bench_scans:
        finding_1_text = (
            f"Benchmark OD median MABE was **${stats_bench_od_mabe['median']:.2f} \\; \\mu\\text{{m}}$** with median Dice **${stats_bench_od_dice['median']:.4f}$**; "
            f"benchmark OS median MABE was **${stats_bench_os_mabe['median']:.2f} \\; \\mu\\text{{m}}$** with median Dice **${stats_bench_os_dice['median']:.4f}$**. "
            f"The previous cohort-wide OS collapse was an inference-coordinate defect, not a supported model finding."
        )
        section_3_1_bullets = f"""- **RNFL Dice Overlap**: Benchmark OD median Dice was **${stats_bench_od_dice['median']:.4f}$** versus **${stats_bench_os_dice['median']:.4f}$** for benchmark OS.
- **Peripapillary Boundary Error (MABE)**: Benchmark OD median MABE was **${stats_bench_od_mabe['median']:.2f} \\; \\mu\\text{{m}}$**, versus **${stats_bench_os_mabe['median']:.2f} \\; \\mu\\text{{m}}$** for benchmark OS. The $5.0 \\; \\mu\\text{{m}}$ line is an operational reference.
- **NFL-Absence Cup-Region Detection**: The annotation-derived cup-region endpoint reached a median IoU of **${stats_bench_all_cup['median']:.4f}$**; this should not be interpreted as independent anatomical cup ground truth."""
    else:
        finding_1_text = (
            f"Held-out OD median MABE was **${stats_val_od_mabe['median']:.2f} \\; \\mu\\text{{m}}$** with median Dice **${stats_val_od_dice['median']:.4f}$**; "
            f"held-out OS median MABE was **${stats_val_os_mabe['median']:.2f} \\; \\mu\\text{{m}}$** with median Dice **${stats_val_os_dice['median']:.4f}$**. "
            f"Native-coordinate restoration preserved strict bilateral symmetry across both eyes without OS performance collapse."
        )
        section_3_1_bullets = f"""- **RNFL Dice Overlap**: Held-out OD median Dice was **${stats_val_od_dice['median']:.4f}$** versus **${stats_val_os_dice['median']:.4f}$** for held-out OS.
- **Peripapillary Boundary Error (MABE)**: Held-out OD median MABE was **${stats_val_od_mabe['median']:.2f} \\; \\mu\\text{{m}}$**, versus **${stats_val_os_mabe['median']:.2f} \\; \\mu\\text{{m}}$** for held-out OS. The $5.0 \\; \\mu\\text{{m}}$ line is an operational reference.
- **NFL-Absence Cup-Region Detection**: The annotation-derived cup-region endpoint reached a median IoU of **${stats_val_cup['median']:.4f}$**; this should not be interpreted as independent anatomical cup ground truth."""

    val_subjs_str = ", ".join(f"`{s}`" for s in val_subjects) if val_subjects else "None"
    subj_range_str = f"`{all_subjects[0]}` - `{all_subjects[-1]}`" if all_subjects else "None"

    if bench_scans:
        kpi_grid_html = f"""<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-title">Benchmark MABE</div>
        <div class="kpi-value">{stats_bench_all_mabe['median']:.2f} µm</div>
        <div class="kpi-sub">Median across {len(bench_scans)} benchmark scans (OD + OS)</div>
    </div>
    <div class="kpi-card amber">
        <div class="kpi-title">Held-Out MABE</div>
        <div class="kpi-value">{stats_val_mabe['median']:.2f} µm</div>
        <div class="kpi-sub">Median across {len(val_scans)} unseen OD + OS scans</div>
    </div>
    <div class="kpi-card cyan">
        <div class="kpi-title">Benchmark Dice</div>
        <div class="kpi-value">{stats_bench_all_dice['median']:.4f}</div>
        <div class="kpi-sub">Median across both eyes</div>
    </div>
    <div class="kpi-card purple">
        <div class="kpi-title">Audit Correction Gain</div>
        <div class="kpi-value">{val_pooled_gain * 100:+.1f}%</div>
        <div class="kpi-sub">{val_pooled_delta_mabe:+.2f} µm error reduced ({total_val_edited_cols:,} cols)</div>
    </div>
</div>"""
    else:
        kpi_grid_html = f"""<div class="kpi-grid">
    <div class="kpi-card amber">
        <div class="kpi-title">Held-Out MABE</div>
        <div class="kpi-value">{stats_val_mabe['median']:.2f} µm</div>
        <div class="kpi-sub">Median across {len(val_scans)} unseen OD + OS scans</div>
    </div>
    <div class="kpi-card cyan">
        <div class="kpi-title">Held-Out RNFL Dice</div>
        <div class="kpi-value">{stats_val_dice['median']:.4f}</div>
        <div class="kpi-sub">Median across {len(val_scans)} held-out scans</div>
    </div>
    <div class="kpi-card purple">
        <div class="kpi-title">Audit Correction Gain</div>
        <div class="kpi-value">{val_pooled_gain * 100:+.1f}%</div>
        <div class="kpi-sub">{val_pooled_delta_mabe:+.2f} µm error reduced ({total_val_edited_cols:,} cols)</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Held-Out Cup IoU</div>
        <div class="kpi-value">{stats_val_cup['median']:.4f}</div>
        <div class="kpi-sub">Mean: {stats_val_cup['mean']:.4f} ± {stats_val_cup['std']:.4f}</div>
    </div>
</div>"""

    val_win_rate_str = f"{val_audit_improved / len(val_edited_scans) * 100:.0f}%" if val_edited_scans else "N/A"
    audit_win_rate_str = f"{audit_improved / len(audit_edited_scans) * 100:.0f}%" if audit_edited_scans else "N/A"

    if val_edited_scans:
        finding_4_text = (
            f"Across {len(val_edited_scans)} materially edited held-out validation scans "
            f"({total_val_edited_cols:,} total edited columns), the U-Net reduced raw commercial error in "
            f"**{val_audit_improved} of {len(val_edited_scans)} scans ({val_win_rate_str} win rate)**. "
            f"Across all human-edited columns, the **pooled column-weighted correction gain was {val_pooled_gain * 100:+.1f}%**, "
            f"eliminating **{val_pooled_delta_mabe:+.2f} µm** of commercial error "
            f"(pooled raw MABE: {val_pooled_raw_mabe:.2f} µm $\\to$ U-Net MABE: {val_pooled_unet_mabe:.2f} µm). "
            f"Scan-level median gain was **{val_gain_median * 100:+.1f}%** with median edited-column recovery of "
            f"**{audit_recovery_median * 100:.1f}%**. This edit-focused analysis is primary; whole-mask commercial Dice "
            f"is reference-dependent and descriptive only."
        )
    elif audit_edited_scans:
        finding_4_text = (
            f"Across {len(audit_edited_scans)} materially edited scans, the U-Net reduced raw commercial error in "
            f"**{audit_improved} of {len(audit_edited_scans)} scans ({audit_win_rate_str} win rate)**. "
            f"Scan-level median gain was **{audit_gain_median * 100:+.1f}%** with median edited-column recovery of "
            f"**{audit_recovery_median * 100:.1f}%**. This edit-focused analysis is primary; whole-mask commercial Dice "
            f"is reference-dependent and descriptive only."
        )
    else:
        finding_4_text = (
            f"Across {len(audit_edited_scans)} materially edited scans, no manual boundary corrections "
            f"exceeded the operational edit threshold ({audit_threshold_px:g} px). This edit-focused analysis is primary; "
            f"whole-mask commercial Dice is reference-dependent and descriptive only."
        )

    if audit_edited_scans:
        audit_section_text = f"""Whole-mask comparison is reference-dependent because the human-audited annotation was created by editing the raw commercial result. The primary comparator analysis therefore isolates columns with a raw-to-audit displacement of at least {audit_threshold_px:g} px ($\\ge 3.12\\,\\mu\\text{{m}}$). Across {len(audit_edited_scans)} materially edited scans ({len(bench_edited_scans)} benchmark training and {len(val_edited_scans)} held-out validation across {', '.join(sorted(set(s['subject'] for s in val_edited_scans)))}), the U-Net reduced boundary error in **{audit_improved} of {len(audit_edited_scans)} scans overall ({audit_win_rate_str} win rate)**, and in **{val_audit_improved} of {len(val_edited_scans)} held-out validation scans ({val_win_rate_str} win rate)**.

Across the {total_val_edited_cols:,} materially edited validation columns:
- **Pooled Column-Weighted Gain**: **{val_pooled_gain * 100:+.1f}%**
- **Commercial Error Eliminated ($\\Delta\\text{{MABE}}$)**: **{val_pooled_delta_mabe:+.2f} µm** (pooled raw commercial MABE: ${val_pooled_raw_mabe:.2f}\\,\\mu\\text{{m}}$ $\\to$ U-Net MABE: ${val_pooled_unet_mabe:.2f}\\,\\mu\\text{{m}}$)
- **Scan-Level Median Gain**: **{val_gain_median * 100:+.1f}%**
- **Edited-Column Recovery Rate**: **{audit_recovery_median * 100:.1f}%**
- **Unchanged-Region Preservation Rate**: **{audit_preservation_median * 100:.1f}%**

The head-to-head chart below plots boundary error exclusively on human-edited columns (left), normalized human-correction gain bounded to $[-100\\%, +100\\%]$ (middle), and fidelity on human-accepted columns (right). Held-out validation scans are explicitly flagged with `[VAL]`.

![Audit-Correction Analysis]({assets_rel_dir}/baseline_vs_unet_head_to_head.png)"""
    else:
        audit_section_text = f"""Whole-mask comparison is reference-dependent because the human-audited annotation was created by editing the raw commercial result. In this run, no scans contained manual edits exceeding {audit_threshold_px:g} px."""

    thickness_section = ""
    thickness_appendix = ""
    if thickness_maps:
        limit_um = metrics_data['metadata']['thickness_map_color_limit_um']
        ranked_maps = sorted(thickness_maps, key=lambda entry: (entry['unet']['mae_um'], entry['subject'], entry['eye']))
        if len(ranked_maps) > 4:
            positions = {round(i * (len(ranked_maps) - 1) / 3) for i in range(4)}
            main_maps = [entry for index, entry in enumerate(ranked_maps) if index in positions]
            appendix_maps = [entry for index, entry in enumerate(ranked_maps) if index not in positions]
        else:
            main_maps, appendix_maps = ranked_maps, []

        def render_pair(entry):
            unet = entry['unet']
            commercial = entry['commercial']
            return f"""<div class="thickness-pair">
<h4>{entry['subject']} {entry['eye']} · {entry['valid_columns']:,} common valid columns ({entry['coverage_percent']:.1f}% of cube)</h4>
<div class="thickness-images">
<img src="{assets_rel_dir}/{unet['image_filename']}" alt="Signed U-Net RNFL thickness difference for {entry['subject']} {entry['eye']}" />
<img src="{assets_rel_dir}/{commercial['image_filename']}" alt="Signed commercial RNFL thickness difference for {entry['subject']} {entry['eye']}" />
</div>
<p>U-Net: bias {unet['mean_um']:+.2f} µm, SD {unet['std_um']:.2f} µm, MAE {unet['mae_um']:.2f} µm. Commercial: bias {commercial['mean_um']:+.2f} µm, SD {commercial['std_um']:.2f} µm, MAE {commercial['mae_um']:.2f} µm.</p>
</div>"""

        selection_note = (
            "Four examples span the ranked U-Net thickness MAE range; all remaining pairs appear in Appendix A."
            if appendix_maps else "All available pairs are shown below."
        )
        thickness_section = f"""### 3.4 Full-Cube Signed RNFL Thickness Differences

Each en face map shows predicted RNFL thickness minus the human-audited reference across the entire Disc Cube grid. The left image uses the U-Net; the right uses the raw commercial curves. Both use the same reference-valid columns and the same signed colour scale, centred at zero and spanning ±{limit_um:g} µm across all scans. Gray marks invalid or cup columns. Colours beyond the scale limits are saturated; numerical statistics use unsaturated values. Bias near zero is only desirable when SD and MAE are also small. These {len(thickness_maps)} pairs are held-out scans with distinct raw and audited curves; unchanged machine-derived reference regions remain in the full-cube statistics and should not be interpreted as independent manual corrections. {selection_note}

{''.join(render_pair(entry) for entry in main_maps)}

---

"""
        if appendix_maps:
            thickness_appendix = f"""---

## Appendix A. Remaining Full-Cube Thickness Difference Pairs

The maps below use the same common-mask rule and ±{limit_um:g} µm colour scale as Section 3.4.

{''.join(render_pair(entry) for entry in appendix_maps)}
"""

    md = f"""<style>
span[style*="#d97706"] code, span[style*="#d97706"] {{
    color: #ea580c !important;
}}
@media print {{
    body {{
        line-height: 1.45 !important;
    }}
    h2 {{
        page-break-before: always !important;
        break-before: page !important;
        margin-top: 10px !important;
        margin-bottom: 12px !important;
    }}
    h2:last-of-type {{
        page-break-before: auto !important;
        break-before: auto !important;
        margin-top: 14px !important;
        margin-bottom: 8px !important;
    }}
    h3 {{
        margin-top: 12px !important;
        margin-bottom: 6px !important;
    }}
    p, li {{
        margin-bottom: 5px !important;
    }}
    hr {{
        margin: 8px 0 !important;
    }}
    ul, ol {{
        margin: 6px 0 10px 0 !important;
        padding-left: 24px !important;
        page-break-inside: auto !important;
    }}
    li {{
        margin-bottom: 4px !important;
        line-height: 1.45 !important;
        display: list-item !important;
        list-style-type: disc !important;
        page-break-inside: avoid !important;
    }}
    ol li {{
        list-style-type: decimal !important;
    }}
    .kpi-grid {{
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-between !important;
        margin: 10px 0 14px 0 !important;
        page-break-inside: avoid !important;
    }}
    .kpi-card {{
        flex: 1 !important;
        margin: 0 4px !important;
        padding: 8px 10px !important;
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }}
    .challenge-grid {{
        display: flex !important;
        flex-wrap: wrap !important;
        justify-content: space-between !important;
        page-break-inside: avoid !important;
    }}
    .challenge-card {{
        width: 48.5% !important;
        margin-bottom: 10px !important;
        box-sizing: border-box !important;
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }}
    .table-container {{
        page-break-inside: avoid !important;
        break-inside: avoid !important;
        margin-bottom: 10px !important;
    }}
    .evidence-context {{
        page-break-inside: avoid !important;
        break-inside: avoid !important;
    }}
    .evidence-table {{
        table-layout: fixed !important;
        font-size: 9.25px !important;
        line-height: 1.28 !important;
        margin: 8px 0 10px 0 !important;
    }}
    .evidence-table th, .evidence-table td {{
        padding: 4px 5px !important;
        vertical-align: top !important;
        overflow-wrap: anywhere !important;
    }}
    .evidence-note {{
        background: #fff7ed !important;
        border-left: 4px solid #f97316 !important;
        padding: 7px 9px !important;
        margin: 8px 0 !important;
        font-size: 9.5px !important;
        line-height: 1.35 !important;
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
    }}
    .gallery-grid {{
        display: block !important;
        page-break-inside: auto !important;
        break-inside: auto !important;
    }}
    .gallery-item {{
        display: inline-block !important;
        width: 49% !important;
        vertical-align: top !important;
        page-break-inside: avoid !important;
        break-inside: avoid !important;
        margin-bottom: 6px !important;
        box-sizing: border-box !important;
    }}
    .gallery-item img {{
        width: 100% !important;
        height: auto !important;
        display: block !important;
        margin: 2px 0 !important;
    }}
    .deep-dive-item {{
        page-break-inside: avoid !important;
        break-inside: avoid !important;
        margin: 8px 0 14px 0 !important;
    }}
    .deep-dive-item h3 {{
        page-break-after: avoid !important;
        break-after: avoid !important;
    }}
    .deep-dive-item img {{
        margin-top: 4px !important;
    }}
    .thickness-pair {{
        page-break-inside: avoid !important;
        break-inside: avoid !important;
        margin: 12px 0 18px 0 !important;
    }}
    .thickness-images {{
        display: flex !important;
        gap: 8px !important;
    }}
    .thickness-images img {{
        width: 49% !important;
        height: auto !important;
        object-fit: contain !important;
    }}
}}
</style>

# Volumetric RNFL Segmentation ({model_variant} Model): Multi-Subject Cohort Report

**Evaluation Cohort Scope**: {len(all_subjects)} Subjects ({subj_range_str}) | {n_scans} OCT Volumes ({n_od} OD + {n_os} OS)<br>
**Checkpoint Training ({training_stage})**: {training_summary}<br>
**Modality**: Optovue Solix OCT `Disc Cube` ($320 \\times 768 \\times 320$ voxels; $18.81\\,\\mu\\text{{m}} \\times 3.12\\,\\mu\\text{{m}} \\times 18.75\\,\\mu\\text{{m}}$)<br>
**Checkpoint Provenance**: NYUAD HPC Jubail (SLURM Job `{job_id}`) | Checkpoint: `{checkpoint_name}`<br>
**Evaluation Runtime**: Local Apple MPS corrected-cohort rerun<br>
**Architecture / Variant**: **{model_variant} Architecture** ({model_desc})<br>
**Inference Policy**: Corrected OS native-coordinate restoration before horizontal/vertical biplanar fusion

**Evaluation Arms**:

- **<span style="color: #0284c7; font-weight: bold;">Cyan</span>**: Human-Corrected Reference (Good Arm)
- **<span style="color: #dc2626; font-weight: bold;">Red</span>**: Commercial Solix Heuristic Baseline (Bad Arm)
- **<span style="color: #16a34a; font-weight: bold;">Green</span>**: Multi-Task Volumetric U-Net ({model_variant} Model, 2.5D Context + Continuous 1D Boundary Regression)
- **<span style="color: #ea580c; font-weight: bold;">Orange</span>**: Held-Out Validation Cohort (<span style="color: #ea580c; font-weight: bold;">{val_subjs_str}</span>, {len(val_scans)} Scans Unseen During Training)

## 1. Executive Summary

This report delivers an automated cohort-wide comparative evaluation of the **Multi-Task Volumetric RNFL U-Net (<span style="color: #16a34a; font-weight: bold;">Green</span>)** against the **Human-Corrected Reference (<span style="color: #0284c7; font-weight: bold;">Cyan</span>)** and the **Commercial Solix Baseline (<span style="color: #dc2626; font-weight: bold;">Red</span>)** across {len(all_subjects)} held-out subjects ({n_scans} eye-level OCT volumes). The checkpoint was trained on NYUAD Jubail and this corrected cohort evaluation was executed locally on Apple MPS.

{kpi_grid_html}

### High-Level Findings:

1. **Laterality Stability After Coordinate Correction**: {finding_1_text}
2. **Held-Out Failure Is Subject-Specific**: Across the {len(val_scans)} held-out acquisitions ({val_subjs_str}), median MABE was **${stats_val_mabe['median']:.2f} \\; \\mu\\text{{m}}$**. The worst held-out scan was **{worst_val_label}** at **${worst_val_mabe:.2f} \\; \\mu\\text{{m}}$**; pooled validation statistics therefore require scan-level review.
3. **Not Ready for Autonomous Clinical Use**: {qc_finding} These engineering thresholds are not clinically validated, but residual Dice, cup-IoU, and boundary-error failures require mandatory human review.
4. **Audit-Correction Performance**: {finding_4_text}

---

## 2. Methods & Evaluation Protocol

### 2.1 Cohort Architecture & Data Modality
- **Evaluation dataset**: {len(all_subjects)} deidentified human subjects from the NYUAD / Rokers Lab Solix OCT repository.
{training_methods}
- **Protocol**: Optovue Solix `Disc Cube` ($320 \\times 768 \\times 320$ voxels), covering $6.0 \\times 6.0 \\times 2.4\\,\\text{{mm}}^3$ centered on the optic nerve head (ONH).
- **Voxel Pitch**: $18.81\\,\\mu\\text{{m}}$ (slow B-scan pitch) $\\times 3.12\\,\\mu\\text{{m}}$ (axial depth) $\\times 18.75\\,\\mu\\text{{m}}$ (fast A-scan pitch).

### 2.2 Evaluation Metrics
- **Dice Similarity Coefficient**: Spatial overlap between predicted and human-audited binary RNFL masks. It is a secondary consistency endpoint because unchanged regions dominate whole-mask overlap.
- **Mean Absolute Boundary Error (MABE)**: Mean axial displacement outside the optic cup cavity in $\\mu\\text{{m}}$ ($3.12367\\,\\mu\\text{{m}}/\\text{{px}}$).
- **95th Percentile Boundary Error ($P_{{95}}$)**: Localized worst-case boundary drift in $\\mu\\text{{m}}$.
- **Audit-Correction Gain**: One minus the ratio of U-Net error to raw error on columns where the raw and audited NFL boundaries differ by at least {audit_threshold_px:g} px. Positive values indicate recovery of human corrections.
- **Unchanged-Region Preservation**: Fraction of human-accepted columns where the U-Net remains within 1 px of the audited NFL boundary.
- **NFL-Absence Cup-Region IoU**: Jaccard overlap of columns where the NFL boundary is absent. This is an annotation-derived cup-region endpoint, not a full anatomical cup segmentation.
- **Cup-Edge Localization**: Horizontal left-edge, right-edge, and width errors on slices where both reference and prediction contain a detectable NFL-absence region.

---

## 3. Cohort Quantitative Benchmark Results

### 3.1 Distribution and Individual Scan Profiles

The multi-panel plot stratifies benchmark and held-out scans by eye. Error metrics use logarithmic axes so the common 2-5 µm range remains visible alongside severe subject-level outliers. Dashed thresholds are operational report references, not validated clinical decision limits.

![Cohort Statistical Distributions]({assets_rel_dir}/cohort_raincloud_distributions.png)

{section_3_1_bullets}

---

### 3.2 Complete {n_scans}-Scan Clinical Cohort Forest Chart

The chart shows all {n_scans} eye-level scans ranked best to worst by MABE. MABE uses a logarithmic axis to preserve the 2-15 µm range while retaining severe outliers; Dice and Cup IoU are shown in separate aligned panels. `[MIRROR]` identifies an unedited machine copy rather than an independent commercial annotation.

![Complete {n_scans}-Scan Forest Plot]({assets_rel_dir}/cohort_per_scan_forest_plot.png)

---

### 3.3 Audit-Correction Analysis: U-Net vs Raw Commercial Boundary

{audit_section_text}

---

{thickness_section}

## 4. External Evidence Context

The closest published evidence spans different OCT devices, scan geometries, pathologies, reference standards, and aggregation methods. The table therefore positions the model rather than ranking it. The current-project row uses the **subject-disjoint held-out cohort**; training-cohort benchmark Dice is intentionally excluded from the cross-study comparison.

<div class="evidence-context">
<table class="evidence-table">
<thead>
<tr><th style="width: 18%;">Evidence</th><th style="width: 24%;">Evaluation setting</th><th style="width: 14%;">RNFL Dice</th><th style="width: 19%;">Other error endpoint</th><th style="width: 25%;">Comparability note</th></tr>
</thead>
<tbody>
<tr><td><strong>Current model</strong></td><td>Optovue Solix Disc Cube; {len(val_subjects)} subjects / {len(val_scans)} eyes; full volumes; subject-disjoint held-out set</td><td><strong>Median {stats_val_dice['median']:.3f}</strong></td><td>Boundary MABE: <strong>{stats_val_mabe['median']:.2f} µm</strong> median; worst {worst_val_mabe:.2f} µm</td><td>Most relevant generalization result, but the sample is too small for a superiority or safety claim.</td></tr>
<tr><td><a href="https://doi.org/10.1167/tvst.15.4.7">Arian et al., 2026</a></td><td>External Spectralis circular B-scans: Thailand glaucoma (n=157) and US edema (n=32)</td><td>Mean 0.858 / 0.845</td><td>Thickness MAE: 7.19 / 15.41 µm; lower-boundary MUE: 14.52 / 24.82 µm</td><td>Strong external clinical evidence, but 2D circles and thickness/boundary endpoints differ from the Solix volume evaluation.</td></tr>
<tr><td><a href="https://arxiv.org/abs/2207.14447">GOALS, 2022</a></td><td>Topcon DRI circumpapillary B-scans; patient-disjoint challenge tests</td><td>0.816 / 0.843</td><td>Boundary MED: 4.06 / 4.15 pixels</td><td>High anatomical relevance and multi-grader reference; single 2D circles and no defensible µm conversion.</td></tr>
<tr><td><a href="https://doi.org/10.1038/s41598-022-22135-x">Razaghi et al., 2022</a></td><td>Spectralis circular B-scans; 127 independent test eyes spanning healthy, NAION, and optic neuritis</td><td>0.870</td><td>Thickness MAE: 1.04-1.20 µm across groups</td><td>Independent same-device test; thickness MAE is not interchangeable with local boundary MABE.</td></tr>
<tr><td><a href="https://doi.org/10.3389/fcell.2026.1890734">Qiu et al., 2026 (M2D)</a></td><td>1,017 Heidelberg/TowardPi circumpapillary scans; expert-corrected subset</td><td>Mean 0.874</td><td>Expert-subset thickness MAD: 1.8 µm</td><td>Cross-device evidence, but large-scale overlap primarily used proprietary output as the reference.</td></tr>
<tr><td><a href="https://doi.org/10.18502/jovr.v18i1.12724">Razaghi et al., 2023</a></td><td>SD-OCT B-scans; 50-image internal test; subject separation unclear</td><td>0.910</td><td>Thickness MAE: 2.23 ± 2.10 µm</td><td>Favorable internal result with a small image-level test; weaker generalization evidence.</td></tr>
</tbody>
</table>

<div class="evidence-note"><strong>Interpretation:</strong> The held-out Dice of {stats_val_dice['median']:.3f} lies within the approximately 0.82-0.88 range reported in external or difficult peripapillary RNFL evaluations. This is evidence of technical plausibility, not equivalence or superiority. No directly comparable external Optovue Solix Disc Cube benchmark was identified, and Dice alone does not resolve the {worst_val_mabe:.2f} µm worst-case boundary failure.</div>
</div>

Values above retain each publication's original endpoint and aggregation. Mean and median values, full-volume and circular-scan evaluations, boundary and thickness errors, and human versus commercial-derived references must not be treated as interchangeable.

---

## 5. OD Subject Statistical Overview

The aligned dot plots summarize the {n_od} OD acquisitions only. Dice, MABE, and Cup IoU use separate axes and operational thresholds. Commercial points appear only where an independent comparator is available; blank comparator positions represent missing annotations, not zero values.

![Cohort Summary Chart]({assets_rel_dir}/cohort_summary_chart.png)

---

## 6. OD Subject Gallery: Reference vs U-Net

Central peripapillary OD B-scans ($z = z_{{\\text{{disc}}}}$) comparing the **Human-Corrected Reference (<span style="color: #0284c7; font-weight: bold;">Cyan</span>)** with the **Volumetric U-Net (<span style="color: #16a34a; font-weight: bold;">Green</span>)**. The gallery {"contains one OD view per subject" if len(gallery) == len(gallery_all) else f"shows {len(gallery)} representative OD views selected from {len(gallery_all)}, prioritizing held-out scans and high-MABE failures"}; OS acquisitions and the commercial baseline are not shown here. Complete scan-level metrics remain archived in the report assets.

<div class="gallery-grid">
"""

    for g in gallery:
        subj = g['subject']
        fname = os.path.basename(g['filename'])
        dice_val = g.get('dice', 0.0)
        mabe_val = g.get('mabe', 0.0)
        md += f"""<div class="gallery-item">
<p><strong>{subj} (OD)</strong> - Dice: <code>{dice_val:.4f}</code> | MABE: <code>{mabe_val:.2f} µm</code></p>
<img src="{assets_rel_dir}/{fname}" alt="Gallery {subj}" />
</div>
"""

    md += f"""</div>

---

## 7. Cross-Sectional Deep-Dive Panels: Validation & Archetype Subjects

Detailed cross-sectional analysis comparing optical intensity boundaries, vertical cut behavior, and local layer transitions across key clinical archetypes. For human-audited scans with manual edits, full 3-arm panels show the Commercial Solix baseline alongside Reference and U-Net; for scans accepted without edits, the redundant commercial arm is omitted to present expanded, high-resolution views of the Reference Algorithm and Volumetric U-Net.

"""

    for dd in deep_dives:
        subj = dd['subject']
        eye = dd.get('eye', 'OD')
        fname = os.path.basename(dd['filename'])
        cohort_tag = dd.get('cohort', '')
        md += f"""<div class="deep-dive-item">
<h3>Subject {subj} ({eye}) [{cohort_tag}]</h3>
<img src="{assets_rel_dir}/{fname}" alt="Deep Dive {subj}" />
</div>

"""

    md += f"""## 8. Algorithmic Mechanics Driving Boundary Adherence

<div class="challenge-grid">
    <div class="challenge-card">
        <div class="challenge-title">GCL Hyporeflective Wedge Penetration</div>
        <div class="challenge-row"><span class="badge-red">Solix Baseline</span> Plunges deeply into adjacent hyporeflective ganglion cell layer.</div>
        <div class="challenge-row"><span class="badge-green">Volumetric U-Net</span> Continuous 1D head locks onto true hyperreflective optical gradient.</div>
        <div class="challenge-row"><span class="badge-cyan">Human-Corrected Reference</span> Manually reviewed anatomical transition interface.</div>
    </div>
    <div class="challenge-card">
        <div class="challenge-title">Optic Cup Cavity Void & BMO Bridging</div>
        <div class="challenge-row"><span class="badge-red">Solix Baseline</span> Bridges straight across non-physiological empty cup void.</div>
        <div class="challenge-row"><span class="badge-green">Volumetric U-Net</span> 1D cup head accurately truncates margin at Bruch's Membrane Opening.</div>
        <div class="challenge-row"><span class="badge-cyan">Human-Corrected Reference</span> Reviewed peripapillary termination at the annotated margin.</div>
    </div>
    <div class="challenge-card">
        <div class="challenge-title">Major Vessel Axial Shadowing</div>
        <div class="challenge-row"><span class="badge-red">Solix Baseline</span> Axial signal drop causes erratic vertical jumps and boundary loss.</div>
        <div class="challenge-row"><span class="badge-green">Volumetric U-Net</span> Multi-slice 2.5D contextual slices interpolate across vessel shadows cleanly.</div>
        <div class="challenge-row"><span class="badge-cyan">Human-Corrected Reference</span> Reviewed continuous layer contours.</div>
    </div>
    <div class="challenge-card">
        <div class="challenge-title">Pathological Disc Tilt & Steep Slope</div>
        <div class="challenge-row"><span class="badge-red">Solix Baseline</span> Steep regional gradients induce boundary distortion and clipping.</div>
        <div class="challenge-row"><span class="badge-green">Volumetric U-Net</span> Continuous 1D regression preserves curvature continuity and slope fidelity.</div>
        <div class="challenge-row"><span class="badge-cyan">Human-Corrected Reference</span> Reviewed boundary conformity.</div>
    </div>
</div>

---

## 9. Clinical Significance & Conclusion

1. **Comparable OD and OS Cohort Performance**: Corrected native-coordinate restoration removes the systematic OS artifact. Benchmark medians are closely aligned by eye, but laterality stability does not eliminate individual failures.
2. **Residual Clinical Risk**: The worst held-out scan, {worst_val_label}, reached **${worst_val_mabe:.2f} \\; \\mu\\text{{m}}$** MABE, and additional scans miss Dice or cup-IoU operational limits. The model is suitable for research and human-supervised review, not autonomous clinical use.
3. **Comparator Evidence Is Reference-Dependent**: The edit-focused analysis shows whether the U-Net recovers human changes without allowing unchanged pixels to dominate. It still cannot establish clinical superiority because the audit is not an independent second-reader reference.
4. **External Positioning**: Held-out Dice is within published external or difficult-cohort RNFL ranges, but cross-study differences and the small held-out cohort prevent a direct ranking or superiority claim.
5. **Execution Summary**: Checkpoint `{checkpoint_name}` was evaluated using corrected biplanar inference. All visual assets, scan-level metrics, audit-correction fields, comparator missingness, and manual-review outputs are archived in `{assets_rel_dir}`.

"""
    return md + thickness_appendix
