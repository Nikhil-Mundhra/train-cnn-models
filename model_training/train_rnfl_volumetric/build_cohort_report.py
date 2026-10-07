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
    from model_training.train_rnfl_volumetric.reporting.training_cohort import summarize_training_cohort
except ImportError:
    from reporting.training_cohort import summarize_training_cohort

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
        select_executive_gallery,
        generate_pdf_image_variants,
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
            select_executive_gallery,
            generate_pdf_image_variants,
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
            select_executive_gallery,
            generate_pdf_image_variants,
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
    "select_executive_gallery",
    "generate_pdf_image_variants",
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
    parser.add_argument("--train_manifest", type=str, default=None, help="Exact training manifest for checkpoint provenance")
    parser.add_argument("--fine_tune_manifest", type=str, default=None, help="Additional audited fine-tuning manifest, if applied")
    parser.add_argument("--training_stage", type=str, default="Training", help="Training phase applied to this checkpoint")
    parser.add_argument(
        "--max_gallery_items",
        type=int,
        default=24,
        help="Maximum representative OD gallery panels; <=0 disables the cap",
    )
    parser.add_argument(
        "--pdf_jpeg_quality",
        type=int,
        default=84,
        help="JPEG quality for PDF-only gallery and deep-dive images",
    )
    args = parser.parse_args()

    if not os.path.exists(args.metrics_json):
        print(f"[ERROR] Metrics file not found: {args.metrics_json}")
        sys.exit(1)

    with open(args.metrics_json, "r") as f:
        metrics_data = json.load(f)

    training_cohort = None
    fine_tune_cohort = None
    if args.train_manifest:
        with open(args.train_manifest, "r", encoding="utf-8") as f:
            train_manifest = json.load(f)
        training_cohort = summarize_training_cohort(train_manifest, metrics_data.get("scans", []))
    if args.fine_tune_manifest:
        if not args.train_manifest:
            raise ValueError("--fine_tune_manifest requires --train_manifest")
        with open(args.fine_tune_manifest, "r", encoding="utf-8") as f:
            fine_tune_manifest = json.load(f)
        fine_tune_cohort = summarize_training_cohort(fine_tune_manifest, metrics_data.get("scans", []))
        pretrain_subjects = {scan["subject"] for scan in train_manifest["scans"]}
        fine_tune_subjects = {scan["subject"] for scan in fine_tune_manifest["scans"]}
        if not fine_tune_subjects <= pretrain_subjects:
            raise ValueError("Fine-tuning subjects must be contained in the pretraining cohort")

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

    selected_gallery = select_executive_gallery(
        metrics_data.get('gallery', []),
        args.max_gallery_items,
    )

    if args.output_pdf:
        print(
            "[Report Builder] Generating PDF-optimized clinical images "
            f"(gallery={len(selected_gallery)}, deep_dives={len(metrics_data.get('deep_dives', []))})..."
        )
        generate_pdf_image_variants(
            selected_gallery,
            assets_abs,
            max_width=1200,
            quality=args.pdf_jpeg_quality,
        )
        generate_pdf_image_variants(
            metrics_data.get('deep_dives', []),
            assets_abs,
            max_width=1800,
            quality=args.pdf_jpeg_quality,
        )

    print(f"[Report Builder] Rendering Markdown report for Job {args.job_id} ({args.model_variant})...")
    md_content = generate_markdown_report(
        metrics_data=metrics_data,
        assets_rel_dir=assets_rel,
        job_id=args.job_id,
        checkpoint_name=args.checkpoint_name,
        model_variant=args.model_variant,
        model_desc=args.model_desc,
        max_gallery_items=args.max_gallery_items,
        training_cohort=training_cohort,
        fine_tune_cohort=fine_tune_cohort,
        training_stage=args.training_stage,
    )

    os.makedirs(md_dir, exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Report Builder] SUCCESS: Markdown report written to {args.output_md}")

    # Compile PDF if requested
    if args.output_pdf:
        compiled = compile_report_pdf(
            md_content=md_content,
            md_dir=md_dir,
            title=os.path.basename(args.output_md),
            pdf_path=args.output_pdf,
            theme="light",
            md_path=args.output_md
        )
        if not compiled:
            raise RuntimeError(f"PDF report was not regenerated: {args.output_pdf}")


if __name__ == "__main__":
    main()
