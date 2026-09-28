"""
test_reporting.py
=================
Automated unit tests for the reporting and visualization suite:
- Palette definitions (theme.py)
- Distribution statistics and chart rendering (charts.py)
- Markdown normalization and report generation (markdown.py)
- HTML conversion and browser discovery (compiler.py)
- Top-level re-exports (build_cohort_report.py)
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import matplotlib
from PIL import Image

matplotlib.use("Agg")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TESTS_DIR = Path(__file__).resolve().parent
MODULE_DIR = TESTS_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import build_cohort_report
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
    pdf_variant_filename,
    select_executive_gallery,
    create_pdf_image_variant,
)


class TestThemePalette(unittest.TestCase):
    """Verifies light and dark theme palette generation."""

    def test_get_theme_palette_light(self):
        pal = get_theme_palette("light")
        self.assertEqual(pal["fig_face"], "#ffffff")
        self.assertEqual(pal["ax_face"], "#ffffff")
        self.assertEqual(pal["text"], "#0f172a")
        self.assertIn("bench_od_color", pal)
        self.assertIn("unet_color", pal)

    def test_get_theme_palette_dark(self):
        pal = get_theme_palette("dark")
        self.assertEqual(pal["fig_face"], "#0f172a")
        self.assertEqual(pal["ax_face"], "#1e293b")
        self.assertEqual(pal["text"], "#ffffff")
        self.assertIn("bench_os_color", pal)
        self.assertIn("bad_color", pal)

    def test_get_theme_palette_default(self):
        pal = get_theme_palette()
        self.assertEqual(pal["fig_face"], "#ffffff")


class TestDistributionStats(unittest.TestCase):
    """Verifies robust summary statistics calculation."""

    def test_empty_list(self):
        stats = compute_distribution_stats([])
        self.assertEqual(stats["mean"], 0.0)
        self.assertEqual(stats["std"], 0.0)
        self.assertEqual(stats["median"], 0.0)
        self.assertEqual(stats["iqr"], 0.0)
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 0.0)

    def test_populated_list(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_distribution_stats(values)
        self.assertAlmostEqual(stats["mean"], 30.0)
        self.assertAlmostEqual(stats["median"], 30.0)
        self.assertAlmostEqual(stats["min"], 10.0)
        self.assertAlmostEqual(stats["max"], 50.0)
        self.assertGreater(stats["std"], 0.0)
        self.assertGreater(stats["iqr"], 0.0)


class TestMarkdownUtilities(unittest.TestCase):
    """Verifies list normalization and report template compilation."""

    def test_normalize_markdown_lists_injects_blank_line(self):
        raw = "Heading text\n- item 1\n- item 2"
        normalized = normalize_markdown_lists(raw)
        self.assertIn("Heading text\n\n- item 1\n- item 2", normalized)

    def test_normalize_markdown_lists_preserves_code_blocks(self):
        raw = "Text\n```\n- item inside code\n```"
        normalized = normalize_markdown_lists(raw)
        self.assertEqual(raw, normalized)

    def test_normalize_markdown_lists_idempotent_on_clean_lists(self):
        clean = "Text\n\n- item 1\n- item 2"
        normalized = normalize_markdown_lists(clean)
        self.assertEqual(clean, normalized)

    def test_generate_markdown_report_structure(self):
        metrics_data = {
            "scans": [
                {
                    "subject": "BEH0001",
                    "eye": "OD",
                    "is_validation": False,
                    "unet_dice": 0.88,
                    "unet_mabe": 4.5,
                    "unet_p95": 12.0,
                    "unet_cup_iou": 0.92,
                    "bad_dice": 0.82,
                    "bad_cup_iou": 0.80,
                    "is_mirror": False,
                },
                {
                    "subject": "BEH0002",
                    "eye": "OS",
                    "is_validation": True,
                    "unet_dice": 0.70,
                    "unet_mabe": 9.2,
                    "unet_p95": 25.0,
                    "unet_cup_iou": 0.75,
                    "bad_dice": 0.65,
                    "bad_cup_iou": 0.70,
                    "is_mirror": False,
                },
            ],
            "gallery": [
                {
                    "subject": "BEH0001",
                    "filename": "gallery_BEH0001.png",
                    "dice": 0.88,
                    "mabe": 4.5,
                }
            ],
            "deep_dives": [
                {
                    "subject": "BEH0001",
                    "filename": "deep_dive_BEH0001.png",
                    "cohort": "Benchmark",
                }
            ],
        }

        md = generate_markdown_report(
            metrics_data=metrics_data,
            assets_rel_dir="assets",
            job_id="12345",
            checkpoint_name="best_model.pt",
            model_variant="Heavy",
            model_desc="Bi-Planar Orthogonal U-Net",
        )

        self.assertIn("# Volumetric RNFL Segmentation (Heavy Model): Multi-Subject Cohort Report", md)
        self.assertIn("NYUAD HPC Jubail (SLURM Job `12345`)", md)
        self.assertIn("gallery_BEH0001.png", md)
        self.assertIn("deep_dive_BEH0001.png", md)
        self.assertIn("Benchmark MABE", md)
        self.assertIn("## 4. External Evidence Context", md)
        self.assertIn("Current model", md)
        self.assertIn("Arian et al., 2026", md)
        self.assertIn("subject-disjoint held-out cohort", md)
        self.assertIn("## 9. Clinical Significance & Conclusion", md)

    def test_executive_gallery_is_bounded_and_keeps_failure_coverage(self):
        gallery = [
            {
                "subject": f"BEH{i:04d}",
                "cohort": "Validation (Held-Out)" if i in {2, 17, 28} else "Training / Benchmark",
                "mabe": float(i),
            }
            for i in range(30)
        ]
        selected = select_executive_gallery(gallery, max_items=12)
        self.assertEqual(len(selected), 12)
        self.assertIn("BEH0029", {row["subject"] for row in selected})
        self.assertTrue(any("Validation" in row["cohort"] for row in selected))

    def test_generate_markdown_report_describes_sampled_gallery(self):
        metrics_data = {
            "scans": [
                {
                    "subject": "BEH0001",
                    "eye": "OD",
                    "is_validation": False,
                    "unet_dice": 0.9,
                    "unet_mabe": 3.0,
                    "unet_p95": 8.0,
                    "unet_cup_iou": 0.93,
                }
            ],
            "gallery": [
                {
                    "subject": f"BEH{i:04d}",
                    "filename": f"gallery_{i}.png",
                    "dice": 0.9,
                    "mabe": float(i),
                }
                for i in range(30)
            ],
        }
        md = generate_markdown_report(
            metrics_data,
            "assets",
            "1",
            "best.pt",
            max_gallery_items=10,
        )
        self.assertIn("shows 10 representative OD views selected from 30", md)


class TestPdfImageAssets(unittest.TestCase):
    def test_pdf_variant_filename(self):
        self.assertEqual(pdf_variant_filename("gallery/example.png"), "example_pdf.jpg")

    def test_create_pdf_image_variant_resizes_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.png"
            destination = Path(temp_dir) / "source_pdf.jpg"
            Image.new("RGB", (2400, 800), (80, 120, 160)).save(source)
            original_size = source.stat().st_size

            create_pdf_image_variant(str(source), str(destination), max_width=1200, quality=84)

            self.assertTrue(source.exists())
            self.assertEqual(source.stat().st_size, original_size)
            with Image.open(destination) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (1200, 400))


class TestHtmlCompiler(unittest.TestCase):
    """Verifies HTML rendering, KaTeX preservation, and theme asset replacement."""

    def test_convert_md_to_html_basic(self):
        md = "# Report Title\n\nThis is a paragraph with $\\mu = 5.0\\,\\mu\\text{m}$."
        html = convert_md_to_html(md, "file:///fake/path/", "Test Title", theme="dark")
        self.assertIn("<title>Test Title</title>", html)
        self.assertIn("katex", html)
        self.assertIn("Report Title", html)

    def test_convert_md_to_html_theme_asset_replacement(self):
        md = "\n".join([
            "![Distribution](assets/cohort_raincloud_distributions.png)",
            "![Gallery](assets/gallery_BEH0001.png)",
            '<img src="assets/deep_dive_BEH0001.png" />',
        ])
        html = convert_md_to_html(md, "file:///fake/path/", "Test", theme="light")
        self.assertIn("cohort_raincloud_distributions_light.png", html)
        self.assertIn("gallery_BEH0001_pdf.jpg", html)
        self.assertIn("deep_dive_BEH0001_pdf.jpg", html)

    def test_find_browser_safe_execution(self):
        # find_browser should execute without crashing and return str or None
        browser = find_browser()
        self.assertTrue(browser is None or isinstance(browser, str))


class TestChartRendering(unittest.TestCase):
    """Verifies that all 4 publication charts render valid PNG files."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dummy_scans = [
            {
                "subject": "BEH0001",
                "eye": "OD",
                "is_validation": False,
                "is_mirror": False,
                "unet_dice": 0.89,
                "unet_mabe": 4.1,
                "unet_p95": 11.2,
                "unet_cup_iou": 0.94,
                "bad_dice": 0.81,
                "bad_mabe": 6.2,
                "bad_p95": 16.5,
                "bad_cup_iou": 0.82,
                "audit_edit_threshold_px": 1.0,
                "audit_edited_columns": 120,
                "raw_edit_mabe_um": 8.0,
                "unet_edit_mabe_um": 4.0,
                "audit_correction_gain": 0.5,
                "audit_unchanged_preservation_rate": 0.80,
            },
            {
                "subject": "BEH0002",
                "eye": "OS",
                "is_validation": False,
                "is_mirror": False,
                "unet_dice": 0.85,
                "unet_mabe": 5.2,
                "unet_p95": 14.0,
                "unet_cup_iou": 0.90,
                "bad_dice": 0.78,
                "bad_mabe": 7.5,
                "bad_p95": 19.0,
                "bad_cup_iou": 0.79,
                "audit_edit_threshold_px": 1.0,
                "audit_edited_columns": 80,
                "raw_edit_mabe_um": 6.0,
                "unet_edit_mabe_um": 9.0,
                "audit_correction_gain": -0.5,
                "audit_unchanged_preservation_rate": 0.65,
            },
            {
                "subject": "BEH0003",
                "eye": "OD",
                "is_validation": True,
                "is_mirror": False,
                "unet_dice": 0.87,
                "unet_mabe": 4.6,
                "unet_p95": 12.5,
                "unet_cup_iou": 0.91,
                "bad_dice": None,
                "bad_mabe": None,
                "bad_p95": None,
                "bad_cup_iou": None,
            },
            {
                "subject": "BEH0004",
                "eye": "OS",
                "is_validation": True,
                "is_mirror": True,
                "unet_dice": 0.72,
                "unet_mabe": 8.9,
                "unet_p95": 24.1,
                "unet_cup_iou": 0.78,
                "bad_dice": 0.70,
                "bad_mabe": 9.5,
                "bad_p95": 26.0,
                "bad_cup_iou": 0.75,
            },
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_render_statistical_raincloud_chart(self):
        out_png = os.path.join(self.temp_dir.name, "raincloud.png")
        result = render_statistical_raincloud_chart(self.dummy_scans, out_png, theme="light")
        self.assertTrue(os.path.exists(out_png))
        self.assertGreater(os.path.getsize(out_png), 1000)
        self.assertEqual(result, out_png)

    def test_render_complete_scan_forest_chart(self):
        out_png = os.path.join(self.temp_dir.name, "forest.png")
        result = render_complete_scan_forest_chart(self.dummy_scans, out_png, theme="dark")
        self.assertTrue(os.path.exists(out_png))
        self.assertGreater(os.path.getsize(out_png), 1000)
        self.assertEqual(result, out_png)

    def test_render_baseline_comparison_chart(self):
        out_png = os.path.join(self.temp_dir.name, "baseline_cmp.png")
        result = render_baseline_comparison_chart(self.dummy_scans, out_png, theme="light")
        self.assertTrue(os.path.exists(out_png))
        self.assertGreater(os.path.getsize(out_png), 1000)
        self.assertEqual(result, out_png)

    def test_render_cohort_summary_chart(self):
        out_png = os.path.join(self.temp_dir.name, "cohort_summary.png")
        result = render_cohort_summary_chart(self.dummy_scans, out_png, theme="dark")
        self.assertTrue(os.path.exists(out_png))
        self.assertGreater(os.path.getsize(out_png), 1000)
        self.assertEqual(result, out_png)


class TestReExportsAndCLI(unittest.TestCase):
    """Verifies backward compatibility re-exports from build_cohort_report."""

    def test_re_exported_symbols(self):
        expected_symbols = [
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
        for sym in expected_symbols:
            self.assertTrue(
                hasattr(build_cohort_report, sym),
                f"Missing expected re-export {sym} in build_cohort_report",
            )
            self.assertTrue(callable(getattr(build_cohort_report, sym)))


if __name__ == "__main__":
    unittest.main()
