#!/usr/bin/env python3
"""
Executive Clinical RNFL Report Builder.

Parses evaluation metrics JSON, generates dual-theme publication charts (dark for markdown,
clean publication white for PDF), renders structured Markdown reports, and compiles
stand-alone PDF documents.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

try:
    from model_training.train_rnfl_volumetric.reporting import (
        get_theme_palette,
        compute_distribution_stats,
        render_statistical_raincloud_chart,
        render_complete_scan_forest_chart,
        render_baseline_comparison_chart,
        render_cohort_summary_chart,
        normalize_markdown_lists,
        generate_markdown_report,
        find_browser,
        convert_md_to_html,
        compile_report_pdf,
    )
except ImportError:
    try:
        from .reporting import (
            get_theme_palette,
            compute_distribution_stats,
            render_statistical_raincloud_chart,
            render_complete_scan_forest_chart,
            render_baseline_comparison_chart,
            render_cohort_summary_chart,
            normalize_markdown_lists,
            generate_markdown_report,
            find_browser,
            convert_md_to_html,
            compile_report_pdf,
        )
    except (ImportError, ValueError):
        from reporting import (
            get_theme_palette,
            compute_distribution_stats,
            render_statistical_raincloud_chart,
            render_complete_scan_forest_chart,
            render_baseline_comparison_chart,
            render_cohort_summary_chart,
            normalize_markdown_lists,
            generate_markdown_report,
            find_browser,
            convert_md_to_html,
            compile_report_pdf,
        )

__all__ = [
    "get_theme_palette",
    "compute_distribution_stats",
    "render_statistical_raincloud_chart",
    "render_complete_scan_forest_chart",
    "render_baseline_comparison_chart",
    "render_cohort_summary_chart",
    "normalize_markdown_lists",
    "generate_markdown_report",
    "find_browser",
    "convert_md_to_html",
    "compile_report_pdf",
]


def main():
    parser = argparse.ArgumentParser(description="Build Executive Clinical RNFL Report from Metrics JSON")
    parser.add_argument("--metrics_json", type=str, required=True, help="Path to cohort_evaluation_metrics.json")
    parser.add_argument("--assets_dir", type=str, required=True, help="Path to directory containing output images")
    parser.add_argument("--output_md", type=str, required=True, help="Path for destination Markdown report")
    parser.add_argument("--output_pdf", type=str, default=None, help="Optional path for destination PDF report")
    parser.add_argument("--job_id", type=str, default="local", help="SLURM Job ID")
    parser.add_argument("--checkpoint_name", type=str, default="latest", help="Checkpoint description")
    parser.add_argument("--model_variant", type=str, default="Volumetric", help="Model variant label (e.g. Light, Heavy)")
    parser.add_argument("--model_desc", type=str, default="", help="Detailed architecture description")
    args = parser.parse_args()

    if not os.path.exists(args.metrics_json):
        print(f"[ERROR] Metrics file not found: {args.metrics_json}")
        sys.exit(1)

    with open(args.metrics_json, "r") as f:
        metrics_data = json.load(f)

    md_dir = os.path.dirname(os.path.abspath(args.output_md))
    assets_abs = os.path.abspath(args.assets_dir)
    try:
        assets_rel = os.path.relpath(assets_abs, md_dir)
    except ValueError:
        assets_rel = assets_abs

    # Generate publication charts (Dual-Theme: Dark for .md, Light for PDF)
    scans = metrics_data.get('scans', [])
    if scans:
        # Dark theme (default for Markdown viewing in dark IDE/GitHub)
        print("[Report Builder] Generating Dark Theme Figures for Markdown...")
        render_statistical_raincloud_chart(scans, os.path.join(assets_abs, "cohort_raincloud_distributions.png"), theme="dark")
        render_complete_scan_forest_chart(scans, os.path.join(assets_abs, "cohort_per_scan_forest_plot.png"), theme="dark")
        render_baseline_comparison_chart(scans, os.path.join(assets_abs, "baseline_vs_unet_head_to_head.png"), theme="dark")
        render_cohort_summary_chart(scans, os.path.join(assets_abs, "cohort_summary_chart.png"), theme="dark")

        # Light theme (publication-standard white for PDF)
        print("[Report Builder] Generating Clean Publication White Figures for PDF...")
        render_statistical_raincloud_chart(scans, os.path.join(assets_abs, "cohort_raincloud_distributions_light.png"), theme="light")
        render_complete_scan_forest_chart(scans, os.path.join(assets_abs, "cohort_per_scan_forest_plot_light.png"), theme="light")
        render_baseline_comparison_chart(scans, os.path.join(assets_abs, "baseline_vs_unet_head_to_head_light.png"), theme="light")
        render_cohort_summary_chart(scans, os.path.join(assets_abs, "cohort_summary_chart_light.png"), theme="light")

    print(f"[Report Builder] Rendering Markdown report for Job {args.job_id} ({args.model_variant})...")
    md_content = generate_markdown_report(
        metrics_data=metrics_data,
        assets_rel_dir=assets_rel,
        job_id=args.job_id,
        checkpoint_name=args.checkpoint_name,
        model_variant=args.model_variant,
        model_desc=args.model_desc
    )

    os.makedirs(md_dir, exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Report Builder] SUCCESS: Markdown report written to {args.output_md}")

    # Compile PDF if requested
    if args.output_pdf:
        compile_report_pdf(
            md_content=md_content,
            md_dir=md_dir,
            title=os.path.basename(args.output_md),
            pdf_path=args.output_pdf,
            theme="light",
            md_path=args.output_md
        )


if __name__ == "__main__":
    main()
