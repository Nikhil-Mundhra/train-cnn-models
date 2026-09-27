"""
Clinical RNFL Reporting and Visualization Suite.
"""

from .theme import get_theme_palette
from .charts import (
    compute_distribution_stats,
    render_statistical_raincloud_chart,
    render_complete_scan_forest_chart,
    render_baseline_comparison_chart,
    render_cohort_summary_chart,
)
from .markdown import (
    normalize_markdown_lists,
    generate_markdown_report,
)
from .compiler import (
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
