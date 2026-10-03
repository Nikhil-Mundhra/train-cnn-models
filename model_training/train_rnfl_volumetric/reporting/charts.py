"""
reporting/charts.py
===================
Publication-ready scientific chart generators for cohort evaluation:
1. Raincloud & distribution quad-plots
2. Ranked complete-scan forest plots
3. Commercial baseline vs. U-Net head-to-head comparison
4. OD subject summary dot plots
"""

import os
import sys
from typing import Dict, List, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .theme import get_theme_palette


def compute_distribution_stats(values: List[float]) -> Dict[str, Any]:
    """Computes summary statistics (mean, std, median, IQR, min, max) for metric arrays."""
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'q1': 0.0, 'q3': 0.0, 'iqr': 0.0, 'min': 0.0, 'max': 0.0}
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    median = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    return {
        'mean': mean,
        'std': std,
        'median': median,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'min': min_val,
        'max': max_val
    }


def render_statistical_raincloud_chart(scans: List[Dict[str, Any]], out_path: str, theme: str = "light") -> str:
    """
    Figure 1: Eye-stratified metric distributions with individual scans.
    Supports clean publication white theme (for PDFs) and dark theme.
    """
    pal = get_theme_palette(theme)
    groups = [
        ("OD benchmark", [s for s in scans if s['eye'] == 'OD' and not s.get('is_validation', False)], pal["bench_od_color"], "o"),
        ("OD held-out", [s for s in scans if s['eye'] == 'OD' and s.get('is_validation', False)], pal["val_color"], "D"),
        ("OS benchmark", [s for s in scans if s['eye'] == 'OS' and not s.get('is_validation', False)], pal["bench_os_color"], "o"),
        ("OS held-out", [s for s in scans if s['eye'] == 'OS' and s.get('is_validation', False)], pal["val_color"], "D"),
    ]

    metrics = [
        ("RNFL Dice similarity", "unet_dice", "Dice score", False, 0.85, "Operational reference: 0.85"),
        ("Peripapillary MABE", "unet_mabe", "MABE (µm, log scale)", True, 5.0, "Operational reference: 5 µm"),
        ("Worst-case P95 error", "unet_p95", "P95 error (µm, log scale)", True, None, None),
        ("NFL-absence cup-region IoU", "unet_cup_iou", "Cup-region IoU", False, 0.90, "Operational reference: 0.90"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=pal["fig_face"])
    axes = axes.flatten()

    for ax_idx, (m_title, m_key, y_label, use_log, thresh_val, thresh_label) in enumerate(metrics):
        ax = axes[ax_idx]
        ax.set_facecolor(pal["ax_face"])
        x_positions = np.arange(len(groups))

        for g_idx, (g_name, g_scans, g_color, marker) in enumerate(groups):
            vals = np.array([s[m_key] for s in g_scans if s.get(m_key) is not None])
            if len(vals) == 0:
                continue

            ax.boxplot(
                [vals], positions=[g_idx], widths=0.45,
                patch_artist=True, showmeans=False, showfliers=False,
                boxprops=dict(facecolor=g_color, color=g_color if theme == "light" else "white", alpha=0.35, linewidth=1.4),
                whiskerprops=dict(color=pal["subtext"], linewidth=1.2),
                capprops=dict(color=pal["subtext"], linewidth=1.2),
                medianprops=dict(color=pal["text"], linewidth=2.5)
            )

            np.random.seed(42 + g_idx)
            jitter = np.random.uniform(-0.14, 0.14, size=len(vals))
            ax.scatter(
                g_idx + jitter, vals,
                color=g_color, marker=marker,
                edgecolors=pal["val_color"] if marker == "D" else ("white" if theme == "dark" else pal["ax_face"]),
                linewidths=1.4 if marker == "D" else 0.8,
                s=72 if marker == "D" else 48, alpha=0.94, zorder=4
            )

            med_val = np.median(vals)
            q1, q3 = np.percentile(vals, [25, 75])
            if m_key in ("unet_dice", "unet_cup_iou"):
                stat_str = f"n={len(vals)}  {med_val:.3f}\nIQR {q1:.3f}-{q3:.3f}"
            else:
                stat_str = f"n={len(vals)}  {med_val:.1f} µm\nIQR {q1:.1f}-{q3:.1f}"
            ax.text(
                g_idx, 0.975, stat_str, transform=ax.get_xaxis_transform(),
                color=pal["text"], fontsize=8.2, fontweight="bold", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc=pal["badge_fc"], ec=g_color, lw=1.3, alpha=0.92)
            )

        if thresh_val is not None:
            ax.axhline(thresh_val, color=pal["threshold_line"], linestyle="--", linewidth=1.5, alpha=0.85, zorder=2)
            ax.text(0.99, thresh_val, f" {thresh_label}", transform=ax.get_yaxis_transform(),
                    color=pal["threshold_text"], fontsize=8.5, fontweight="bold", va="bottom", ha="right")

        ax.set_title(m_title, color=pal["text"], fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel(y_label, color=pal["subtext"], fontsize=11, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{g[0]}\n(n={len(g[1])})" for g in groups], color=pal["text"], fontsize=9.2, fontweight="bold")
        if use_log:
            ax.set_yscale("log")
            all_vals = [s[m_key] for _, group, _, _ in groups for s in group if s.get(m_key) is not None and s[m_key] > 0]
            ax.set_ylim(max(1.0, min(all_vals) * 0.75), max(all_vals) * 2.0)
        else:
            ax.set_ylim(0.50, 1.05)
        ax.grid(axis="y", color=pal["grid"], linestyle="--", alpha=0.7)
        ax.tick_params(colors=pal["tick"])
        for spine in ax.spines.values():
            spine.set_color(pal["spine"])

    plt.suptitle("RNFL Cohort Distributions by Eye and Evaluation Cohort", color=pal["text"], fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, "Circles: benchmark scans. Diamonds with orange outline: held-out scans. Thresholds are operational report references, not validated clinical decision limits.",
             color=pal["subtext"], fontsize=9, ha="center")
    plt.tight_layout(rect=[0, 0.025, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=200, facecolor=pal["fig_face"], bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_complete_scan_forest_chart(scans: List[Dict[str, Any]], out_path: str, theme: str = "light") -> str:
    """
    Figure 2: Complete 46-Scan Ranked Forest / Lollipop Plot.
    Supports clean publication white theme (for PDFs) and dark theme.
    """
    pal = get_theme_palette(theme)
    sorted_scans = sorted(scans, key=lambda s: s['unet_mabe'])
    n_scans = len(sorted_scans)
    y_pos = np.arange(n_scans)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(18, 17), sharey=True, facecolor=pal["fig_face"],
        gridspec_kw={"width_ratios": [1.45, 0.8, 0.8], "wspace": 0.05}
    )
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(pal["ax_face"])

    labels = []
    label_colors = []
    for s in sorted_scans:
        tag = f"{s['subject']} ({s['eye']})"
        if s.get('is_validation', False):
            label_colors.append(pal["val_color"])
        elif s.get('is_mirror', False):
            tag += " [MIRROR]"
            label_colors.append("#64748b" if theme == "light" else "#94a3b8")
        elif s['eye'] == 'OD':
            label_colors.append(pal["bench_od_color"])
        else:
            label_colors.append(pal["bench_os_color"])
        labels.append(tag)

    mabes = [s['unet_mabe'] for s in sorted_scans]
    dices = [s['unet_dice'] for s in sorted_scans]
    cups = [s['unet_cup_iou'] for s in sorted_scans]

    # --- Left: MABE lollipop. A log scale preserves the clinically relevant OD range. ---
    for i in range(n_scans):
        is_v = sorted_scans[i].get('is_validation', False)
        eye = sorted_scans[i]['eye']
        color = pal["val_color"] if is_v else (pal["unet_color"] if eye == "OD" else pal["bench_os_color"])
        marker = "D" if is_v else ("o" if eye == "OD" else "s")
        size = 85 if is_v else 65

        ax1.hlines(y_pos[i], 1.0, mabes[i], color=color, alpha=0.6, linewidth=1.5)
        ax1.scatter(mabes[i], y_pos[i], color=color, s=size, marker=marker, edgecolors="white" if theme == "dark" else pal["ax_face"], linewidths=1.0, zorder=4)
        if mabes[i] >= 10 or is_v:
            ax1.annotate(f"{mabes[i]:.1f}", (mabes[i], y_pos[i]), xytext=(5, 0), textcoords="offset points",
                         color=pal["text"], fontsize=8.0, va="center", fontweight="bold")

    ax1.axvline(5.0, color=pal["threshold_line"], linestyle="--", linewidth=1.8, alpha=0.9, zorder=3, label="Operational reference: 5 µm")
    ax1.set_title("Peripapillary MABE", color=pal["text"], fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("MABE (µm, log scale)", color=pal["subtext"], fontsize=10.5, fontweight="bold")
    ax1.set_yticks(y_pos)
    ytick_objs = ax1.set_yticklabels(labels, fontsize=9.5, fontweight="bold")
    for idx, c in enumerate(label_colors):
        ytick_objs[idx].set_color(c)
    ax1.set_xscale("log")
    ax1.set_xlim(1.0, max(mabes) * 1.35)
    ax1.set_ylim(-0.8, n_scans - 0.2)
    ax1.grid(axis="x", color=pal["grid"], linestyle="--", alpha=0.7)
    ax1.legend(facecolor=pal["legend_fc"], edgecolor=pal["legend_ec"], labelcolor=pal["legend_text"], loc="lower right", fontsize=10)
    ax1.tick_params(colors=pal["tick"])
    for spine in ax1.spines.values():
        spine.set_color(pal["spine"])

    # --- Separate panels avoid implying a change between unlike metrics. ---
    for i in range(n_scans):
        is_v = sorted_scans[i].get('is_validation', False)
        eye = sorted_scans[i]['eye']
        node_color = pal["val_color"] if is_v else (pal["unet_color"] if eye == "OD" else pal["bench_os_color"])
        marker = "D" if is_v else ("o" if eye == "OD" else "s")
        size = 85 if is_v else 60
        ax2.scatter(dices[i], y_pos[i], color=node_color, s=size, marker=marker, edgecolors="white" if theme == "dark" else pal["ax_face"], linewidths=0.8, zorder=4)
        ax3.scatter(cups[i], y_pos[i], color=node_color if is_v else pal["cup_color"], s=size, marker=marker if is_v else "d", edgecolors="white" if theme == "dark" else pal["ax_face"], linewidths=0.8, zorder=4)

    custom_legend = [
        Line2D([0], [0], color=pal["val_color"], lw=3, label='Held-Out Validation'),
        Line2D([0], [0], color=pal["unet_color"], lw=3, label='Benchmark OD'),
        Line2D([0], [0], color=pal["bench_os_color"], lw=3, label='Benchmark OS'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor="#64748b", markersize=8, label='Mirror-derived scan'),
    ]
    for ax, title, xlabel in [
        (ax2, "RNFL Dice", "Dice (higher is better)"),
        (ax3, "NFL-absence cup region", "IoU (higher is better)"),
    ]:
        ax.set_title(title, color=pal["text"], fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel, color=pal["subtext"], fontsize=10.5, fontweight="bold")
        ax.set_yticks(y_pos)
        ax.tick_params(axis="y", labelleft=False)
        ax.set_xlim(0.48, 1.02)
        ax.set_ylim(-0.8, n_scans - 0.2)
        ax.grid(axis="x", color=pal["grid"], linestyle="--", alpha=0.7)
        ax.tick_params(colors=pal["tick"])
        for spine in ax.spines.values():
            spine.set_color(pal["spine"])
    ax1.invert_yaxis()
    fig.legend(handles=custom_legend, facecolor=pal["legend_fc"], edgecolor=pal["legend_ec"],
               labelcolor=pal["legend_text"], loc="lower center", bbox_to_anchor=(0.72, 0.027),
               fontsize=8.2, ncol=2)

    plt.suptitle(f"All {n_scans} Eye-Level Scans, Ranked Best to Worst by MABE", color=pal["text"], fontsize=16, fontweight="bold", y=0.995)
    fig.text(0.5, 0.006, "Mirror-derived scans are unedited machine copies and are not independent commercial comparisons.",
             color=pal["subtext"], fontsize=9, ha="center")
    plt.tight_layout(rect=[0, 0.055, 1, 0.98])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=200, facecolor=pal["fig_face"], bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_baseline_comparison_chart(scans: List[Dict[str, Any]], out_path: str, theme: str = "light") -> Optional[str]:
    """
    Figure 3: Edit-focused audit correction analysis.
    Supports clean publication white theme (for PDFs) and dark theme.
    """
    pal = get_theme_palette(theme)
    edited_scans = [
        s for s in scans
        if s.get('raw_edit_mabe_um') is not None
        and s.get('unet_edit_mabe_um') is not None
        and s.get('audit_edited_columns', 0) >= 50
        and s.get('audit_correction_gain') is not None
        and not s.get('is_mirror', False)
    ]
    if not edited_scans:
        return None

    edited_scans = sorted(edited_scans, key=lambda s: s['audit_correction_gain'], reverse=True)
    n = len(edited_scans)
    y_pos = np.arange(n)
    labels = [
        f"{'[VAL] ' if s.get('is_validation') else '      '}{s['subject']} ({s['eye']})  n={s['audit_edited_columns']:,}"
        for s in edited_scans
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(18, max(7.5, 0.58 * n + 2.8)), sharey=True,
        facecolor=pal["fig_face"], gridspec_kw={"width_ratios": [1.35, 1.0, 1.0], "wspace": 0.08}
    )
    ax1, ax2, ax3 = axes
    for ax in axes:
        ax.set_facecolor(pal["ax_face"])

    raw_errors = [s['raw_edit_mabe_um'] for s in edited_scans]
    unet_errors = [s['unet_edit_mabe_um'] for s in edited_scans]
    gains = [s['audit_correction_gain'] for s in edited_scans]
    preservation = [s.get('audit_unchanged_preservation_rate') for s in edited_scans]

    # --- Left: error only where the human reviewer materially edited the raw surface. ---
    for i in range(n):
        is_v = edited_scans[i].get('is_validation', False)
        improved = unet_errors[i] < raw_errors[i]
        delta_um = raw_errors[i] - unet_errors[i]
        if is_v:
            line_color = pal["val_color"] if improved else pal["bad_color"]
            unet_color = pal["val_color"]
            unet_marker = "D"
            unet_size = 92
        else:
            line_color = pal["unet_color"] if improved else pal["bad_color"]
            unet_color = pal["unet_color"]
            unet_marker = "s"
            unet_size = 82

        ax1.hlines(y_pos[i], raw_errors[i], unet_errors[i], color=line_color, alpha=0.75, linewidth=2.0)
        ax1.scatter(raw_errors[i], y_pos[i], color=pal["bad_color"], s=72, marker="o", zorder=3)
        ax1.scatter(unet_errors[i], y_pos[i], color=unet_color, s=unet_size, marker=unet_marker, zorder=4)
        ax1.text(
            max(raw_errors[i], unet_errors[i]) + 0.8, y_pos[i],
            f"Δ:{delta_um:+.1f}µm", color=line_color, fontsize=8.0, va="center", fontweight="bold"
        )

    ax1.set_title("Boundary error on human-edited columns", color=pal["text"], fontsize=11.5, fontweight="bold", pad=12)
    ax1.set_xlabel("MABE (µm; lower is better)", color=pal["subtext"], fontsize=10.5, fontweight="bold")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, color=pal["text"], fontsize=9.5, fontweight="bold")
    ax1.set_xlim(0, max(raw_errors + unet_errors) * 1.25)
    ax1.grid(axis="x", color=pal["grid"], linestyle="--", alpha=0.7)

    # --- Middle: normalized correction gain bounded to [-100%, +100%]. ---
    for i in range(n):
        is_v = edited_scans[i].get('is_validation', False)
        gain_val = gains[i]
        clamped_gain = max(-1.0, min(1.0, gain_val))
        if gain_val > 0:
            bar_color = pal["val_color"] if is_v else pal["unet_color"]
        else:
            bar_color = pal["bad_color"]
        ax2.barh(y_pos[i], clamped_gain * 100.0, color=bar_color, alpha=0.85, height=0.55)
        text_color = pal["val_color"] if is_v else pal["text"]
        label_txt = f" {gain_val * 100:+.0f}%" if gain_val >= -1.0 else " <-100%"
        ax2.text(clamped_gain * 100.0, y_pos[i], label_txt, color=text_color, fontsize=8.5, va="center", fontweight="bold" if is_v else "normal")
    ax2.axvline(0, color=pal["spine"], linewidth=1.4)
    ax2.set_xlim(-110, 115)
    ax2.set_title("Human-correction gain", color=pal["text"], fontsize=11.5, fontweight="bold", pad=12)
    ax2.set_xlabel("1 - (U-Net error / raw error), %", color=pal["subtext"], fontsize=10.0, fontweight="bold")
    ax2.grid(axis="x", color=pal["grid"], linestyle="--", alpha=0.7)

    # --- Right: fidelity where the human reviewer accepted the raw boundary. ---
    for i, value in enumerate(preservation):
        if value is None:
            continue
        is_v = edited_scans[i].get('is_validation', False)
        edge_kwargs = {"edgecolor": pal["val_color"], "linewidth": 1.5} if is_v else {}
        ax3.barh(y_pos[i], value * 100.0, color=pal["cup_color"], alpha=0.85, height=0.55, **edge_kwargs)
        text_color = pal["val_color"] if is_v else pal["text"]
        ax3.text(value * 100.0, y_pos[i], f" {value * 100:.0f}%", color=text_color, fontsize=8.5, va="center", fontweight="bold" if is_v else "normal")
    ax3.set_xlim(0, 105)
    ax3.set_title("Unchanged-region preservation", color=pal["text"], fontsize=11.5, fontweight="bold", pad=12)
    ax3.set_xlabel("Columns within 1 px of audit, %", color=pal["subtext"], fontsize=10.0, fontweight="bold")
    ax3.grid(axis="x", color=pal["grid"], linestyle="--", alpha=0.7)

    for ax in axes:
        ax.set_yticks(y_pos)
        if ax is not ax1:
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(colors=pal["tick"])
        for spine in ax.spines.values():
            spine.set_color(pal["spine"])
    ax1.invert_yaxis()

    # Color validation ytick labels in bold orange AFTER tick_params & invert_yaxis
    for s, lbl in zip(edited_scans, ax1.get_yticklabels()):
        if s.get("is_validation"):
            lbl.set_color(pal["val_color"])
            lbl.set_fontweight("bold")
        else:
            lbl.set_color(pal["text"])

    legend = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=pal["bad_color"], markersize=8, label='Raw commercial'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor=pal["unet_color"], markersize=8, label='Volumetric U-Net (Benchmark)'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor=pal["val_color"], markersize=8, label='Volumetric U-Net (Held-Out [VAL])'),
    ]
    fig.legend(handles=legend, facecolor=pal["legend_fc"], edgecolor=pal["legend_ec"],
               labelcolor=pal["legend_text"], loc="lower center", bbox_to_anchor=(0.5, 0.035), fontsize=9, ncol=3)

    threshold = edited_scans[0].get('audit_edit_threshold_px', 1.0)
    plt.suptitle(f"Audit-Correction Analysis: {n} Scans with Material Human Edits", color=pal["text"], fontsize=15, fontweight="bold", y=0.99)
    fig.text(0.5, 0.008, f"Edited columns differ by at least {threshold:g} px between raw and audit. Positive correction gain favours the U-Net; n is the number of edited columns.",
             color=pal["subtext"], fontsize=9, ha="center")
    plt.tight_layout(rect=[0, 0.075, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=200, facecolor=pal["fig_face"], bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_cohort_summary_chart(scans: List[Dict[str, Any]], out_path: str, theme: str = "light") -> str:
    """
    Cohort Summary Chart (OD Dice, MABE & Cup IoU).
    Supports clean publication white theme (for PDFs) and dark theme.
    """
    pal = get_theme_palette(theme)
    od_scans = [s for s in scans if s.get('eye') == "OD"]
    od_scans = sorted(od_scans, key=lambda s: s['unet_mabe'])
    subjects = [s['subject'] for s in od_scans]
    y = np.arange(len(od_scans))
    fig, axes = plt.subplots(1, 3, figsize=(16, 10.5), sharey=True, facecolor=pal["fig_face"], gridspec_kw={"wspace": 0.06})
    metrics = [
        ("unet_dice", "bad_dice", "RNFL Dice", "Dice (higher is better)", (0.78, 1.01), 0.85),
        ("unet_mabe", "bad_mabe", "Boundary error", "MABE (µm; lower is better)", (0.0, max(s['unet_mabe'] for s in od_scans) * 1.12), 5.0),
        ("unet_cup_iou", "bad_cup_iou", "NFL-absence cup region", "IoU (higher is better)", (0.55, 1.01), 0.90),
    ]

    for ax, (u_key, b_key, title, xlabel, xlim, threshold) in zip(axes, metrics):
        ax.set_facecolor(pal["ax_face"])
        for idx, scan in enumerate(od_scans):
            marker = "D" if scan.get('is_validation', False) else "o"
            edge = pal["val_color"] if scan.get('is_validation', False) else ("white" if theme == "dark" else pal["ax_face"])
            b_val = scan.get(b_key)
            if b_val is not None and np.isfinite(b_val) and not scan.get('is_mirror', False):
                ax.hlines(y[idx], min(scan[u_key], b_val), max(scan[u_key], b_val), color=pal["spine"], linewidth=1.4, alpha=0.8)
                ax.scatter(b_val, y[idx], color=pal["bad_color"], marker="s", s=52, zorder=3)
            ax.scatter(scan[u_key], y[idx], color=pal["unet_color"], marker=marker, s=72,
                       edgecolors=edge, linewidths=1.5 if marker == "D" else 0.7, zorder=4)
        ax.axvline(threshold, color=pal["threshold_line"], linestyle="--", linewidth=1.4, alpha=0.8)
        ax.set_title(title, color=pal["text"], fontsize=12.5, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel, color=pal["subtext"], fontsize=10, fontweight="bold")
        ax.set_xlim(*xlim)
        ax.set_yticks(y)
        ax.grid(axis="x", color=pal["grid"], linestyle="--", alpha=0.7)
        ax.tick_params(colors=pal["tick"])
        for spine in ax.spines.values():
            spine.set_color(pal["spine"])

    axes[0].set_yticklabels(subjects, fontsize=9.5, fontweight="bold")
    axes[0].invert_yaxis()
    for label, scan in zip(axes[0].get_yticklabels(), od_scans):
        label.set_color(pal["val_color"] if scan.get('is_validation', False) else pal["tick"])
    axes[2].legend(handles=[
        Line2D([0], [0], marker='o', color='none', markerfacecolor=pal["unet_color"], markersize=8, label='U-Net benchmark'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor=pal["unet_color"], markeredgecolor=pal["val_color"], markersize=8, label='U-Net held-out'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor=pal["bad_color"], markersize=8, label='Commercial (where available)'),
    ], facecolor=pal["legend_fc"], edgecolor=pal["legend_ec"], labelcolor=pal["legend_text"], loc="lower left", fontsize=8.5)

    n_commercial = sum(1 for s in od_scans if s.get('bad_dice') is not None and not s.get('is_mirror', False))
    fig.suptitle("OD Subject Overview: Separate Clinical Endpoints", color=pal["text"], fontsize=16, fontweight="bold", y=0.995)
    fig.text(0.5, 0.015, f"OD only (n={len(od_scans)}). Commercial paired annotations available for {n_commercial} scans; absent comparators are left blank. Dashed lines are operational report references.",
             color=pal["subtext"], fontsize=8.8, ha="center")

    out_path_abs = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path_abs), exist_ok=True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.965])
    plt.savefig(out_path, dpi=200, facecolor=pal["fig_face"], bbox_inches="tight")
    plt.close(fig)
    return out_path
