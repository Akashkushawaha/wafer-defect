"""Synthetic wafer maps with the nine WM-811K pattern types.

Used for tests and for ``wafer demo``, so the pipeline can be exercised
without the 2 GB dataset. These maps are idealised: a model's score on them
says nothing about its score on real fab data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import FAIL, GOOD, LABELS, OFF_WAFER


def _grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    dy, dx = (y - cy) / (h / 2), (x - cx) / (w / 2)
    r = np.hypot(dx, dy)  # 0 at centre, ~1 at the wafer edge
    theta = np.arctan2(dy, dx)
    return r, theta, np.stack([dx, dy])


def _angle_diff(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def make_map(pattern: str, shape: tuple[int, int] = (32, 32), rng: np.random.Generator | None = None) -> np.ndarray:
    """Return one wafer map (uint8, 0/1/2) showing ``pattern``."""
    if pattern not in LABELS:
        raise ValueError(f"unknown pattern {pattern!r}")
    rng = rng or np.random.default_rng()
    r, theta, (dx, dy) = _grid(shape)
    on_wafer = r <= 1.0
    p = np.full(shape, rng.uniform(0.0, 0.03))  # background random fails

    if pattern == "Center":
        r0 = rng.uniform(0.2, 0.4)
        p = np.maximum(p, rng.uniform(0.7, 0.95) * (r < r0))
    elif pattern == "Donut":
        a, w = rng.uniform(0.3, 0.5), rng.uniform(0.15, 0.25)
        p = np.maximum(p, rng.uniform(0.6, 0.9) * ((r > a) & (r < a + w)))
    elif pattern == "Edge-Ring":
        p = np.maximum(p, rng.uniform(0.6, 0.95) * (r > rng.uniform(0.82, 0.9)))
    elif pattern == "Edge-Loc":
        phi, half = rng.uniform(-np.pi, np.pi), np.deg2rad(rng.uniform(15, 40))
        p = np.maximum(p, rng.uniform(0.6, 0.9) * ((r > rng.uniform(0.7, 0.8)) & (_angle_diff(theta, phi) < half)))
    elif pattern == "Loc":
        rad, phi = rng.uniform(0.0, 0.55), rng.uniform(-np.pi, np.pi)
        cx, cy = rad * np.cos(phi), rad * np.sin(phi)
        size = rng.uniform(0.1, 0.2)
        p = np.maximum(p, rng.uniform(0.6, 0.9) * (np.hypot(dx - cx, dy - cy) < size))
    elif pattern == "Scratch":
        phi = rng.uniform(0, np.pi)
        nx, ny = -np.sin(phi), np.cos(phi)  # normal to the scratch direction
        offset = rng.uniform(-0.5, 0.5)
        along = dx * np.cos(phi) + dy * np.sin(phi)
        start = rng.uniform(-0.8, 0.0)
        length = rng.uniform(0.6, 1.4)
        width = 1.2 / max(shape)
        line = (np.abs(dx * nx + dy * ny - offset) < width) & (along > start) & (along < start + length)
        p = np.maximum(p, 0.9 * line)
    elif pattern == "Random":
        p = np.full(shape, rng.uniform(0.15, 0.35))
    elif pattern == "Near-full":
        p = np.full(shape, rng.uniform(0.8, 0.97))

    fails = rng.random(shape) < p
    out = np.full(shape, OFF_WAFER, dtype=np.uint8)
    out[on_wafer] = GOOD
    out[on_wafer & fails] = FAIL
    return out


def make_dataset(n_per_class: int = 50, seed: int = 0, lots: int = 20) -> pd.DataFrame:
    """A tidy frame shaped like :func:`waferdefect.data.load_lswmd` output."""
    rng = np.random.default_rng(seed)
    rows = []
    for label in LABELS:
        for _ in range(n_per_class):
            side = int(rng.integers(20, 60))
            shape = (side, int(side * rng.uniform(0.8, 1.2)))
            rows.append(
                {
                    "wafer_map": make_map(label, shape, rng),
                    "label": label,
                    "lot": f"lot{int(rng.integers(lots)):03d}",
                    "wafer_index": int(rng.integers(1, 26)),
                    "split": None,
                    "die_size": shape[0] * shape[1],
                }
            )
    return pd.DataFrame(rows)
