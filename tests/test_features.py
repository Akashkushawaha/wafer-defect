import numpy as np
import pytest

from waferdefect.features import (
    FEATURE_NAMES,
    density_features,
    extract,
    extract_many,
    geometry_features,
    radial_features,
    radon_features,
)
from waferdefect.synthetic import make_map


def test_vector_length_and_finite(center_map):
    v = extract(center_map)
    assert v.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(v).all()


def test_feature_names_unique():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_no_failures_gives_zeros():
    m = make_map("none", (30, 30), np.random.default_rng(0))
    m[m == 2] = 1
    v = extract(m)
    assert np.isfinite(v).all()
    assert not v.any()


@pytest.mark.parametrize("shape", [(6, 21), (5, 5), (300, 202)])
def test_extreme_sizes(shape):
    m = make_map("Loc", shape, np.random.default_rng(3))
    assert extract(m).shape == (len(FEATURE_NAMES),)


def test_density_in_unit_interval(rng):
    for pattern in ("Center", "Near-full", "Random"):
        d = density_features(make_map(pattern, (40, 40), rng))
        assert ((d >= 0) & (d <= 1)).all()


def test_radial_profile_separates_center_and_edge_ring(rng):
    center = radial_features(make_map("Center", (50, 50), rng))
    ring = radial_features(make_map("Edge-Ring", (50, 50), rng))
    assert center[0] > center[-1]  # failures concentrated inside
    assert ring[-1] > ring[0]  # failures concentrated outside


def test_radial_profile_is_size_invariant():
    a = radial_features(make_map("Donut", (30, 30), np.random.default_rng(7)))
    b = radial_features(make_map("Donut", (120, 120), np.random.default_rng(7)))
    # same pattern parameters (same seed) at 16x the die count
    assert np.abs(a - b).max() < 0.35


def test_radon_detects_scratch_anisotropy(rng):
    # A ring looks the same from every angle, so its projections barely vary
    # with angle (low std); a scratch only lines up at one angle (high std).
    scratch = radon_features(make_map("Scratch", (60, 60), rng))
    ring = radon_features(make_map("Edge-Ring", (60, 60), rng))
    aniso = lambda v: v[20:].mean() / (v[:20].mean() + 1e-9)  # noqa: E731
    assert aniso(scratch) > 2 * aniso(ring)


def test_scratch_is_elongated(rng):
    g = geometry_features(make_map("Scratch", (60, 60), rng))
    eccentricity = g[5]
    assert eccentricity > 0.9


def test_extract_many_matches_extract(synth_df):
    maps = synth_df["wafer_map"].iloc[:5]
    X = extract_many(maps, n_jobs=1)
    assert np.allclose(X[2], extract(maps.iloc[2]))
