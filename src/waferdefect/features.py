"""Hand-built features for wafer-map pattern recognition.

The feature set follows Wu, Jang & Chen (2015), "Wafer Map Failure Pattern
Recognition and Similarity Ranking for Large-Scale Data Sets", IEEE Trans.
Semiconductor Manufacturing 28(1) — the paper that released WM-811K — with
two changes:

* every feature is normalised so it does not depend on die count
  (maps range from ~6x21 to 300x202 dies), and
* a radial failure profile is added. Most WM-811K classes are defined by
  *where on the radius* the failures sit (centre, mid-radius ring, edge), and
  that is where the process physics lives — so the radius gets its own block.

All features are computed on one map at a time; nothing is learned here.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from skimage.measure import label, regionprops
from skimage.transform import radon

from . import FAIL, OFF_WAFER

N_RADON = 20
N_RINGS = 5

DENSITY_NAMES = [f"density_{name}" for name in ("edge_top", "edge_right", "edge_bottom", "edge_left")] + [
    f"density_inner_{r}{c}" for r in range(3) for c in range(3)
]
RADIAL_NAMES = [f"radial_ring_{i}" for i in range(N_RINGS)]
RADON_NAMES = [f"radon_mean_{i}" for i in range(N_RADON)] + [f"radon_std_{i}" for i in range(N_RADON)]
GEOMETRY_NAMES = [
    "geom_area",
    "geom_perimeter",
    "geom_max_dist",
    "geom_major_axis",
    "geom_minor_axis",
    "geom_eccentricity",
    "geom_solidity",
]
GLOBAL_NAMES = ["fail_fraction", "n_clusters", "largest_cluster_share"]

FEATURE_NAMES = DENSITY_NAMES + RADIAL_NAMES + RADON_NAMES + GEOMETRY_NAMES + GLOBAL_NAMES


def _masks(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(m)
    return m != OFF_WAFER, m == FAIL


def density_features(m: np.ndarray) -> np.ndarray:
    """Failure density in 13 zones: 4 edge bands and a 3x3 interior grid.

    Density is failed dies / on-wafer dies in the zone, so off-wafer corners
    do not dilute it.
    """
    wafer, fail = _masks(m)
    h, w = fail.shape
    ri = np.linspace(0, h, 6).astype(int)
    ci = np.linspace(0, w, 6).astype(int)

    def dens(rs: slice, cs: slice) -> float:
        n = wafer[rs, cs].sum()
        return float(fail[rs, cs].sum() / n) if n else 0.0

    zones = [
        (slice(ri[0], ri[1]), slice(None)),  # top edge band
        (slice(None), slice(ci[4], ci[5])),  # right
        (slice(ri[4], ri[5]), slice(None)),  # bottom
        (slice(None), slice(ci[0], ci[1])),  # left
    ]
    zones += [(slice(ri[1 + r], ri[2 + r]), slice(ci[1 + c], ci[2 + c])) for r in range(3) for c in range(3)]
    return np.array([dens(rs, cs) for rs, cs in zones])


def _radius(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    return np.hypot((y - (h - 1) / 2) / (h / 2), (x - (w - 1) / 2) / (w / 2))


def radial_features(m: np.ndarray, n_rings: int = N_RINGS) -> np.ndarray:
    """Failure density in equal-area concentric rings, centre outwards.

    Equal *area* (not equal width) so every ring holds a similar number of
    dies and the outer rings are not over-resolved. Center, Donut and
    Edge-Ring differ mainly in which ring fails (ablation: +0.018 macro F1).
    """
    wafer, fail = _masks(m)
    r = _radius(fail.shape)
    r_max = r[wafer].max() if wafer.any() else 1.0
    r = r / (r_max or 1.0)
    edges = np.sqrt(np.linspace(0, 1, n_rings + 1))
    edges[-1] = np.inf
    out = np.zeros(n_rings)
    for i in range(n_rings):
        zone = wafer & (r >= edges[i]) & (r < edges[i + 1])
        n = zone.sum()
        out[i] = fail[zone].sum() / n if n else 0.0
    return out


def _resample(v: np.ndarray, n: int) -> np.ndarray:
    if v.size == n:
        return v.astype(float)
    x = np.linspace(0, 1, v.size)
    kind = "cubic" if v.size >= 4 else "linear"
    return interp1d(x, v, kind=kind)(np.linspace(0, 1, n))


def radon_features(m: np.ndarray, n: int = N_RADON) -> np.ndarray:
    """Mean and spread of the Radon transform, resampled to ``n`` points each.

    The Radon transform sums failures along lines at every angle. A scratch is
    a line, so it produces a sharp peak at one angle; rings and centre blobs
    are rotation-invariant and produce flat, angle-independent projections.
    Line sums grow with wafer diameter, so values are divided by it.
    """
    _, fail = _masks(m)
    img = fail.astype(float)
    if not img.any():
        return np.zeros(2 * n)
    theta = np.linspace(0.0, 180.0, max(img.shape), endpoint=False)
    sino = radon(img, theta=theta, circle=False) / max(img.shape)
    mean = _resample(sino.mean(axis=1), n)
    std = _resample(sino.std(axis=1), n)
    return np.concatenate([mean, std])


def geometry_features(m: np.ndarray) -> np.ndarray:
    """Shape of the largest connected failure region, scaled by wafer size."""
    wafer, fail = _masks(m)
    if not fail.any():
        return np.zeros(len(GEOMETRY_NAMES))
    labels = label(fail, connectivity=2)
    region = max(regionprops(labels), key=lambda p: p.area)
    h, w = fail.shape
    diameter = max(h, w)
    r = _radius(fail.shape)
    coords = region.coords
    return np.array(
        [
            region.area / max(wafer.sum(), 1),
            region.perimeter / (np.pi * diameter),
            float(r[coords[:, 0], coords[:, 1]].max()),
            region.axis_major_length / diameter,
            region.axis_minor_length / diameter,
            region.eccentricity,
            region.solidity,
        ]
    )


def global_features(m: np.ndarray) -> np.ndarray:
    wafer, fail = _masks(m)
    n_fail = fail.sum()
    if not n_fail:
        return np.zeros(len(GLOBAL_NAMES))
    labels, n = label(fail, connectivity=2, return_num=True)
    largest = np.bincount(labels.ravel())[1:].max()
    return np.array([n_fail / max(wafer.sum(), 1), n / max(wafer.sum(), 1), largest / n_fail])


def extract(m: np.ndarray) -> np.ndarray:
    """All features for one wafer map, in :data:`FEATURE_NAMES` order."""
    m = np.asarray(m)
    vec = np.concatenate(
        [density_features(m), radial_features(m), radon_features(m), geometry_features(m), global_features(m)]
    )
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def extract_many(maps, n_jobs: int = -1, verbose: int = 0) -> np.ndarray:
    """Feature matrix for a sequence of maps (parallel over CPU cores)."""
    maps = list(maps)
    if n_jobs == 1 or len(maps) < 200:
        rows = [extract(m) for m in maps]
    else:
        rows = Parallel(n_jobs=n_jobs, batch_size=256, verbose=verbose)(delayed(extract)(m) for m in maps)
    return np.vstack(rows) if rows else np.empty((0, len(FEATURE_NAMES)))
