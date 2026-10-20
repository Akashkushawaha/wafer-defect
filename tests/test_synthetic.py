import numpy as np
import pytest

from waferdefect import FAIL, LABELS, OFF_WAFER
from waferdefect.synthetic import make_dataset, make_map


@pytest.mark.parametrize("pattern", LABELS)
def test_map_values_and_shape(pattern, rng):
    m = make_map(pattern, (30, 36), rng)
    assert m.shape == (30, 36)
    assert set(np.unique(m)) <= {0, 1, 2}
    assert m[0, 0] == OFF_WAFER  # corners are off the round wafer


def test_near_full_is_mostly_fail(rng):
    m = make_map("Near-full", (40, 40), rng)
    on = m != OFF_WAFER
    assert (m[on] == FAIL).mean() > 0.75


def test_unknown_pattern():
    with pytest.raises(ValueError):
        make_map("Spiral")


def test_dataset_shape():
    df = make_dataset(n_per_class=3, seed=0)
    assert len(df) == 3 * len(LABELS)
    assert set(df["label"]) == set(LABELS)
    assert {"wafer_map", "label", "lot", "wafer_index", "split"} <= set(df.columns)
