"""
reporting/theme.py
==================
Color palettes and styling definitions for clinical cohort visualizations.
Supports high-contrast light mode (publication PDF) and dark mode (IDE / GitHub markdown).
"""

from typing import Dict, Any


def get_theme_palette(theme: str = "light") -> Dict[str, Any]:
    """Provides high-contrast color palettes for light (publication PDF) and dark (IDE/markdown) rendering."""
    if theme == "light":
        return {
            "fig_face": "#ffffff",
            "ax_face": "#ffffff",
            "ax_face_alt": "#f8fafc",
            "text": "#0f172a",
            "subtext": "#475569",
            "grid": "#e2e8f0",
            "spine": "#cbd5e1",
            "tick": "#334155",
            "badge_fc": "#ffffff",
            "badge_ec": "#cbd5e1",
            "threshold_line": "#dc2626",
            "threshold_text": "#dc2626",
            "val_color": "#d97706",
            "bench_od_color": "#0284c7",
            "bench_os_color": "#4f46e5",
            "unet_color": "#059669",
            "bad_color": "#dc2626",
            "cup_color": "#7c3aed",
            "legend_fc": "#ffffff",
            "legend_ec": "#cbd5e1",
            "legend_text": "#0f172a",
        }
    else:
        return {
            "fig_face": "#0f172a",
            "ax_face": "#1e293b",
            "ax_face_alt": "#1e293b",
            "text": "#ffffff",
            "subtext": "#cbd5e1",
            "grid": "#334155",
            "spine": "#475569",
            "tick": "#cbd5e1",
            "badge_fc": "#0f172a",
            "badge_ec": "#475569",
            "threshold_line": "#f87171",
            "threshold_text": "#f87171",
            "val_color": "#f59e0b",
            "bench_od_color": "#06b6d4",
            "bench_os_color": "#6366f1",
            "unet_color": "#10b981",
            "bad_color": "#ef4444",
            "cup_color": "#a855f7",
            "legend_fc": "#0f172a",
            "legend_ec": "#475569",
            "legend_text": "#ffffff",
        }
