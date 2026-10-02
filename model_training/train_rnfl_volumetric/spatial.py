"""Authoritative Solix Disc Cube spatial calibration.

Array order throughout the RNFL pipeline is ``(slow_z, axial_y, fast_x)``.
Keep physical-unit conversions centralized here so training and evaluation do
not silently report different values for the same voxel displacement.
"""

SLOW_UM: float = 18.8100
AXIAL_UM: float = 3.12367
FAST_UM: float = 18.7500

SOLIX_SPACING_UM = (SLOW_UM, AXIAL_UM, FAST_UM)
SOLIX_VOLUME_SHAPE = (320, 768, 320)

