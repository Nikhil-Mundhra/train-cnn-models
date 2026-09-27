"""Coordinate transforms shared by RNFL training, inference, and audit tools."""

from typing import Tuple


OS_ORIENTATION_MODES = (
    "corrected",
    "legacy_vertical_mirror",
    "native",
)


def validate_orientation_mode(mode: str) -> str:
    if mode not in OS_ORIENTATION_MODES:
        choices = ", ".join(OS_ORIENTATION_MODES)
        raise ValueError(f"Unknown OS orientation mode {mode!r}; expected one of: {choices}")
    return mode


def horizontal_requires_flip(eye: str, mode: str = "corrected") -> bool:
    """Whether a horizontal B-scan should be standardized before inference."""
    validate_orientation_mode(mode)
    return eye.upper() == "OS" and mode != "native"


def vertical_source_and_destination(
    standardized_index: int,
    n_columns: int,
    eye: str,
    mode: str = "corrected",
) -> Tuple[int, int]:
    """
    Return ``(raw_source_column, native_destination_column)`` for a vertical pane.

    OS training samples enumerate a standardized left-to-right index but read the
    mirrored raw A-scan.  Correct inference must therefore write the prediction
    back to that mirrored raw/native column before fusing it with horizontal
    predictions.  The legacy mode reproduces the historical mismatched write and
    exists only for controlled regression experiments.
    """
    validate_orientation_mode(mode)
    if not 0 <= standardized_index < n_columns:
        raise IndexError(f"column {standardized_index} outside [0, {n_columns})")

    if eye.upper() != "OS" or mode == "native":
        return standardized_index, standardized_index

    raw_source = n_columns - 1 - standardized_index
    native_destination = standardized_index if mode == "legacy_vertical_mirror" else raw_source
    return raw_source, native_destination
