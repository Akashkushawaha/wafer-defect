"""Wafer-map defect pattern classification.

Wafer maps follow the WM-811K encoding: 0 = no die (off-wafer), 1 = good die,
2 = failed die.
"""

__version__ = "0.1.0"

OFF_WAFER, GOOD, FAIL = 0, 1, 2

# The nine WM-811K classes, in a fixed order used everywhere in the package.
LABELS = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
)
