import numpy as np
import pytest

from waferdefect.synthetic import make_dataset, make_map


@pytest.fixture(scope="session")
def synth_df():
    return make_dataset(n_per_class=40, seed=1, lots=24)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def center_map(rng):
    return make_map("Center", (40, 40), rng)
