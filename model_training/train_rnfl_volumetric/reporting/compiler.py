"""
HTML and PDF Compilation Utilities for Clinical RNFL Reports.
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

from .markdown import normalize_markdown_lists


def find_browser() -> Optional[str]:
    """Locate a headless Chromium or Chrome binary across platforms."""
    if sys.platform == 'darwin':
        playwright_candidates = glob.glob(
            os.path.expanduser("~/Library/Caches/ms-playwright/**/chrome-headless-shell"),
            recursive=True
        )
        mac_candidates = playwright_candidates + [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
        for c in mac_candidates:
            if os.path.exists(c) and os.access(c, os.X_OK):
                return c

    linux_playwright = glob.glob(
        os.path.expanduser("~/.cache/ms-playwright/**/chrome-headless-shell"),
        recursive=True
    ) + glob.glob(
        os.path.expanduser("~/.cache/ms-playwright/**/chrome"),
        recursive=True
    )
    for c in linux_playwright:
        if os.path.exists(c) and os.access(c, os.X_OK):
            return c

    for c in ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"]:
        found = shutil.which(c)
        if found:
            return found

    return None


def convert_md_to_html(md_text: str, md_dir_uri: str, title: str, theme: str = "light") -> str:
    """
    Converts markdown text to standalone HTML with KaTeX and print-optimized styles.
    Replaces chart filenames with _light versions when theme == 'light'.
    """
    # 1. Protect code blocks
    code_blocks = []
    code_inlines = []

    def save_code_block(m):
        idx = len(code_blocks)
        code_blocks.append(m.group(0))
        return f"@@CODE_BLOCK_{idx}@@"

    def save_code_inline(m):
        idx = len(code_inlines)
        code_inlines.append(m.group(0))
        return f"@@CODE_INLINE_{idx}@@"

    md_text = re.sub(r"(```[\s\S]*?```)", save_code_block, md_text)
    md_text = re.sub(r"(`[^`\n]+?`)", save_code_inline, md_text)

    # 2. Protect LaTeX math expressions
    math_blocks = []
    math_inlines = []

    def save_math_block(m):
        idx = len(math_blocks)
        math_blocks.append(m.group(0))
        return f"@@MATH_BLOCK_{idx}@@"

    def save_math_inline(m):
        idx = len(math_inlines)
        math_inlines.append(m.group(0))
        return f"@@MATH_INLINE_{idx}@@"

    md_text = re.sub(r"(?<!\\)\$\$([\s\S]+?)(?<!\\)\$\$", save_math_block, md_text)
    md_text = re.sub(r"(?<!\\)\\\[([\s\S]+?)(?<!\\)\\\]", save_math_block, md_text)
    md_text = re.sub(r"(?<!\\)\\\(([\s\S]+?)(?<!\\)\\\)", save_math_inline, md_text)
    md_text = re.sub(r"(?<![\$\\])\$(?!\s)([^\$\n]+?)(?<!\s)(?<!\\)\$", save_math_inline, md_text)

    # 3. Restore code blocks
    for i, c in enumerate(code_blocks):
        md_text = md_text.replace(f"@@CODE_BLOCK_{i}@@", c)
    for i, c in enumerate(code_inlines):
        md_text = md_text.replace(f"@@CODE_INLINE_{i}@@", c)

    # 3.5. Automated Defensive List Normalization:
    md_text = normalize_markdown_lists(md_text)

    # 3.6. Theme-Aware Asset Selection for PDF
    if theme == "light":
        for chart_base in ["cohort_raincloud_distributions", "cohort_per_scan_forest_plot", "baseline_vs_unet_head_to_head", "cohort_summary_chart"]:
            md_text = re.sub(rf"({chart_base})(\.png)", r"\1_light\2", md_text)
        # Clinical panels use compact JPEG derivatives only in the PDF. Their
        # original lossless PNGs remain untouched for archival and Markdown use.
        md_text = re.sub(
            r"((?:gallery|deep_dive)[^\s\)\"']+)\.png",
            r"\1_pdf.jpg",
            md_text,
        )

    # 4. Markdown conversion
    try:
        import markdown
        html_body = markdown.markdown(md_text, extensions=['extra', 'sane_lists', 'md_in_html'])
    except ImportError:
        # Fallback if markdown package isn't installed
        html_body = f"<pre>{md_text}</pre>"

    # 5. Restore math expressions
    for i, m in enumerate(math_blocks):
        clean_m = re.sub(r'\\\\([a-zA-Z])', r'\\\1', m)
        html_body = html_body.replace(f"@@MATH_BLOCK_{i}@@", clean_m)
    for i, m in enumerate(math_inlines):
        clean_m = re.sub(r'\\\\([a-zA-Z])', r'\\\1', m)
        html_body = html_body.replace(f"@@MATH_INLINE_{i}@@", clean_m)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <base href="{md_dir_uri}">
    <title>{title}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
    <script>
        window.addEventListener("load", function() {{
            if (typeof renderMathInElement === "function") {{
                renderMathInElement(document.body, {{
                    delimiters: [
                        {{left: "$$", right: "$$", display: true}},
                        {{left: "\\\\[", right: "\\\\]", display: true}},
                        {{left: "$", right: "$", display: false}},
                        {{left: "\\\\(", right: "\\\\)", display: false}}
                    ],
                    throwOnError: false
                }});
            }}
        }});
    </script>
    <style>
        @page {{
            margin: 0.65in 0.55in;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            color: #24292f;
            line-height: 1.45;
            padding: 0;
            padding-bottom: 25px;
            max-width: 960px;
            margin: 0 auto;
        }}
        ul, ol {{
            margin: 8px 0 12px 0;
            padding-left: 26px;
        }}
        li {{
            margin-bottom: 5px;
            line-height: 1.5;
            list-style-type: disc;
        }}
        ol li {{
            list-style-type: decimal;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin: 16px 0 20px 0;
        }}
        .kpi-card {{
            background: #0f172a;
            color: white;
            border-radius: 8px;
            padding: 12px 14px;
            border-left: 4px solid #10b981;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        }}
        .kpi-card.amber {{ border-left-color: #f59e0b; }}
        .kpi-card.cyan {{ border-left-color: #06b6d4; }}
        .kpi-card.purple {{ border-left-color: #a855f7; }}
        .kpi-title {{
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            font-weight: 700;
            margin-bottom: 3px;
        }}
        .kpi-value {{
            font-size: 22px;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.1;
            margin-bottom: 3px;
        }}
        .kpi-sub {{
            font-size: 9.5px;
            color: #cbd5e1;
        }}
        .challenge-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin: 14px 0;
        }}
        .challenge-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
        }}
        .challenge-title {{
            font-size: 12px;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 8px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
        }}
        .challenge-row {{
            font-size: 10.5px;
            margin-bottom: 5px;
            line-height: 1.4;
        }}
        .badge-red {{
            background: #fee2e2;
            color: #dc2626;
            font-weight: 700;
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 9.5px;
        }}
        .badge-green {{
            background: #dcfce7;
            color: #16a34a;
            font-weight: 700;
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 9.5px;
        }}
        .badge-cyan {{
            background: #e0f2fe;
            color: #0284c7;
            font-weight: 700;
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 9.5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 12px 0;
            font-size: 11px;
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        th, td {{
            border: 1px solid #d0d7de;
            padding: 5px 8px;
            text-align: left;
        }}
        th {{
            background-color: #f6f8fa;
            font-weight: 600;
        }}
        tr:nth-child(2n) {{
            background-color: #f8fafc;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            margin: 12px 0;
            display: block;
        }}
        hr {{
            height: 0.2em;
            margin: 20px 0;
            background-color: #d0d7de;
            border: 0;
        }}
    </style>
</head>
<body>
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; text-align: center; font-size: 9pt; color: #6e7781; background: white;">
        {title} | Corrected Biplanar Cohort Evaluation
    </div>
    {html_body}
</body>
</html>
"""
    return html_content


def compile_report_pdf(
    md_content: str,
    md_dir: str,
    title: str,
    pdf_path: str,
    theme: str = "light",
    md_path: Optional[str] = None
) -> bool:
    """
    Compiles markdown content to PDF using headless Chrome/Chromium or pandoc/weasyprint.
    Returns True if the PDF was compiled successfully, False otherwise.
    """
    pdf_path = os.path.abspath(pdf_path)
    pdf_dir = os.path.dirname(pdf_path)
    os.makedirs(pdf_dir, exist_ok=True)

    browser_path = find_browser()
    if browser_path:
        print(f"[Report Builder] Found browser for PDF rendering: {browser_path}")
        md_dir_uri = 'file:///' + md_dir.replace(os.sep, '/') + '/'
        html_text = convert_md_to_html(md_content, md_dir_uri, title, theme=theme)
        fd, temp_html = tempfile.mkstemp(suffix=".html", text=True)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(html_text)

        print("[Report Builder] Executing headless print-to-pdf...")
        res = subprocess.run([
            browser_path,
            '--headless',
            '--disable-gpu',
            '--no-pdf-header-footer',
            '--virtual-time-budget=5000',
            '--run-all-compositor-stages-before-draw',
            f'--print-to-pdf={pdf_path}',
            temp_html
        ], capture_output=True, text=True, timeout=60)

        if os.path.exists(temp_html):
            os.remove(temp_html)

        if res.returncode == 0 and os.path.exists(pdf_path):
            size_kb = os.path.getsize(pdf_path) // 1024
            print(f"[Report Builder] SUCCESS: Compiled PDF report to {pdf_path} ({size_kb} KB)")
            return True
        else:
            print(f"[Report Builder] Warning: Headless browser exited with returncode {res.returncode}")

    # Fallback to Pandoc + Weasyprint if source markdown file is available
    if md_path and os.path.exists(md_path) and shutil.which("pandoc") and shutil.which("weasyprint"):
        print("[Report Builder] Attempting PDF compilation via Pandoc + WeasyPrint...")
        cmd = ["pandoc", md_path, "-o", pdf_path, "--pdf-engine=weasyprint"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(pdf_path):
            print(f"[Report Builder] SUCCESS: Compiled PDF via Pandoc to {pdf_path}")
            return True

    print("[Report Builder] NOTE: PDF compilation engine not present on this node.")
    if md_path:
        print(f"[Report Builder] Markdown report and assets are complete and ready at: {md_path}")
        print("[Report Builder] To compile PDF locally, run:")
        print(f"  python3 .agents/skills/document-manipulation/scripts/md_to_pdf.py -i {md_path} -o {pdf_path}")
    return False
