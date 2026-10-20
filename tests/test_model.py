import numpy as np
import pytest

from waferdefect import LABELS
from waferdefect import model as M
from waferdefect.causes import SIGNATURES, explain
from waferdefect.features import extract_many


@pytest.fixture(scope="module")
def xy(synth_df):
    X = extract_many(synth_df["wafer_map"], n_jobs=1)
    return X, synth_df["label"].to_numpy(), synth_df["lot"].to_numpy()


def test_lot_split_has_no_shared_lots(xy):
    _, y, lots = xy
    tr, te = M.split_indices(y, lots, how="lot")
    assert not set(lots[tr]) & set(lots[te])
    assert len(tr) + len(te) == len(y)


def test_official_split():
    y = np.array(["none"] * 4)
    split = np.array(["Training", "Test", "Training", None], dtype=object)
    tr, te = M.split_indices(y, np.zeros(4), how="official", official=split)
    assert list(tr) == [0, 2] and list(te) == [1]


def test_bad_split_name():
    with pytest.raises(ValueError):
        M.split_indices(np.array(["none"]), np.zeros(1), how="kfold")


@pytest.mark.parametrize("kind", ["hgb", "rf"])
def test_learns_synthetic_patterns(xy, kind):
    X, y, lots = xy
    tr, te = M.split_indices(y, lots, how="lot", seed=0)
    clf = M.build_model(kind).fit(X[tr], y[tr])
    ev = M.evaluate(clf, X[te], y[te])
    # Synthetic patterns are clean; this is a pipeline sanity check, not a benchmark.
    assert ev.macro_f1 > 0.8, ev.summary()


def test_save_load_roundtrip(xy, tmp_path):
    X, y, _ = xy
    clf = M.build_model("rf").fit(X, y)
    path = tmp_path / "m.joblib"
    M.save(clf, path, meta={"note": "t"})
    payload = M.load(path)
    assert (payload["model"].predict(X[:5]) == clf.predict(X[:5])).all()
    assert payload["meta"]["note"] == "t"


def test_every_class_has_a_signature():
    assert set(SIGNATURES) == set(LABELS)
    assert explain("Scratch").first_suspects
    with pytest.raises(KeyError):
        explain("Spiral")
