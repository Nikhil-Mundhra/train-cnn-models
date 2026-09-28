"""Image selection and PDF-specific asset optimization for RNFL reports."""

import os
from typing import Any, Dict, Iterable, List

from PIL import Image


def pdf_variant_filename(filename: str) -> str:
    """Return the JPEG filename used only by the rendered PDF."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return f"{stem}_pdf.jpg"


def _evenly_spaced(items: List[Any], count: int) -> List[Any]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    indices = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    return [items[i] for i in indices]


def select_executive_gallery(
    gallery: Iterable[Dict[str, Any]],
    max_items: int = 24,
) -> List[Dict[str, Any]]:
    """
    Bound the gallery while retaining held-out, high-error, and typical scans.

    Returned rows keep their original report order. A non-positive limit disables
    the cap.
    """
    rows = list(gallery)
    if max_items <= 0 or len(rows) <= max_items:
        return rows

    indexed = list(enumerate(rows))
    ranked = sorted(
        indexed,
        key=lambda pair: float(pair[1].get("mabe") or 0.0),
        reverse=True,
    )
    validation = [pair for pair in ranked if "Validation" in str(pair[1].get("cohort", ""))]
    selected_indices = set()

    # Reserve up to one third for held-out scans. If there are more held-out
    # scans than slots, sample their MABE range rather than showing only failures.
    validation_budget = min(len(validation), max(1, max_items // 3))
    for idx, _ in _evenly_spaced(validation, validation_budget):
        selected_indices.add(idx)

    # Reserve another third for the largest errors across the full cohort.
    target_after_failures = min(max_items, validation_budget + max(1, max_items // 3))
    for idx, _ in ranked:
        if len(selected_indices) >= target_after_failures:
            break
        selected_indices.add(idx)

    # Fill remaining positions with evenly spaced performance quantiles.
    remaining = max_items - len(selected_indices)
    candidates = [pair for pair in ranked if pair[0] not in selected_indices]
    for idx, _ in _evenly_spaced(candidates, remaining):
        selected_indices.add(idx)

    return [row for idx, row in indexed if idx in selected_indices][:max_items]


def create_pdf_image_variant(
    source_path: str,
    destination_path: str,
    max_width: int,
    quality: int = 84,
) -> str:
    """Create a compact print-resolution JPEG while preserving the source PNG."""
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    if not 1 <= quality <= 95:
        raise ValueError("quality must be between 1 and 95")

    with Image.open(source_path) as image:
        image.load()
        if image.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", image.size, "white")
            background.paste(image.convert("RGB"), mask=image.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")

        if image.width > max_width:
            new_height = max(1, round(image.height * max_width / image.width))
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

        os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)
        image.save(
            destination_path,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
            subsampling=2,
        )
    return destination_path


def generate_pdf_image_variants(
    items: Iterable[Dict[str, Any]],
    assets_dir: str,
    max_width: int,
    quality: int = 84,
) -> List[str]:
    """Generate PDF-only JPEG variants for report items with ``filename`` keys."""
    outputs = []
    for item in items:
        filename = item.get("filename")
        if not filename:
            continue
        source = filename if os.path.isabs(filename) else os.path.join(assets_dir, filename)
        if not os.path.exists(source):
            raise FileNotFoundError(f"Report image not found: {source}")
        destination = os.path.join(assets_dir, pdf_variant_filename(filename))
        outputs.append(create_pdf_image_variant(source, destination, max_width, quality))
    return outputs
