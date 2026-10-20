"""Loader tests, including a hand-built pickle in the pandas 0.x layout."""

import copyreg
import pickle
import sys
import types

import numpy as np
import pandas as pd
import pytest

from waferdefect.data import clean_tag, load_lswmd, tidy
from waferdefect.legacy import read_legacy_pickle


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.array([["Center"]], dtype=object), "Center"),
        (np.array([], dtype=object).reshape(0, 0), None),
        (np.array([["Training"]]), "Training"),
        ([[b"Edge-Ring"]], "Edge-Ring"),
        ("none", "none"),
        (None, None),
        ("", None),
    ],
)
def test_clean_tag(value, expected):
    assert clean_tag(value) == expected


def _raw_frame():
    maps = [np.array([[0, 1, 1], [1, 2, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1]]) for _ in range(4)]
    maps.append(np.array([[1, 2]]))  # degenerate, must be dropped
    tag = lambda s: np.array([[s]], dtype=object) if s else np.zeros((0, 0))  # noqa: E731
    return pd.DataFrame(
        {
            "waferMap": maps,
            "dieSize": [15.0] * 5,
            "lotName": ["lot1", "lot1", "lot2", "lot2", "lot3"],
            "waferIndex": [1.0, 2.0, 1.0, 2.0, 1.0],
            "trianTestLabel": [tag("Training"), tag("Test"), tag(""), tag("Training"), tag("Test")],
            "failureType": [tag("Center"), tag("none"), tag(""), tag("Loc"), tag("Scratch")],
        }
    )


def test_tidy_labeled_only():
    df = tidy(_raw_frame())
    assert list(df["label"]) == ["Center", "none", "Loc"]
    assert df["wafer_map"].iloc[0].dtype == np.uint8


def test_tidy_keeps_untagged():
    df = tidy(_raw_frame(), labeled_only=False)
    assert df["label"].isna().sum() == 1


def test_tidy_rejects_unknown_label():
    raw = _raw_frame()
    raw.loc[0, "failureType"] = np.array([["Spiral"]], dtype=object)
    with pytest.raises(ValueError, match="Spiral"):
        tidy(raw)


def test_load_modern_pickle(tmp_path):
    path = tmp_path / "LSWMD.pkl"
    _raw_frame().to_pickle(path)
    assert len(load_lswmd(path)) == 3


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        load_lswmd(tmp_path / "nope.pkl")


# ---------------------------------------------------------------- legacy layout
#
# LSWMD.pkl was written by pandas 0.x. We recreate that object layout with
# stand-in classes registered under a fake top-level package, pickle with
# protocol 2 (newline-terminated GLOBAL opcodes), then rename the package to
# "pandas" in the byte stream. The result references module paths that do
# not exist in modern pandas — the situation the legacy reader has to handle.

FAKE = "zzpandas"


def _install_fake_pandas():
    mods = {}
    for name in ("", ".core", ".core.frame", ".core.internals", ".indexes", ".indexes.base", ".indexes.numeric"):
        mods[name] = types.ModuleType(FAKE + name)
        sys.modules[FAKE + name] = mods[name]

    def _new_Index(cls, d):  # noqa: N802 - name must match pandas 0.x
        raise AssertionError("never called while pickling")

    _new_Index.__module__ = FAKE + ".indexes.base"
    _new_Index.__qualname__ = "_new_Index"
    mods[".indexes.base"]._new_Index = _new_Index

    def make_cls(mod, name, reduce):
        cls = type(name, (), {"__reduce_ex__": reduce, "__module__": FAKE + mod, "__qualname__": name})
        setattr(mods[mod], name, cls)
        return cls

    def index_reduce(self, proto):
        return (_new_Index, (type(self), self.d))

    Index = make_cls(".indexes.base", "Index", index_reduce)
    Int64Index = make_cls(".indexes.numeric", "Int64Index", index_reduce)

    def newobj_reduce(self, proto):
        return (copyreg.__newobj__, (type(self),), self.state)

    BlockManager = make_cls(".core.internals", "BlockManager", newobj_reduce)
    DataFrame = make_cls(".core.frame", "DataFrame", newobj_reduce)
    return Index, Int64Index, BlockManager, DataFrame


def _legacy_bytes(raw: pd.DataFrame) -> bytes:
    Index, Int64Index, BlockManager, DataFrame = _install_fake_pandas()
    try:
        cols = list(raw.columns)
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(raw[c])]
        obj_cols = [c for c in cols if c not in num_cols]

        def index(cls, data):
            ix = object.__new__(cls)
            ix.d = {"data": np.asarray(data, dtype=object if cls is Index else np.int64), "name": None}
            return ix

        columns = index(Index, cols)
        rows = index(Int64Index, np.arange(len(raw)))
        obj_vals = np.empty((len(obj_cols), len(raw)), dtype=object)
        for i, c in enumerate(obj_cols):
            for j, v in enumerate(raw[c]):
                obj_vals[i, j] = v
        num_vals = raw[num_cols].to_numpy(dtype=float).T
        blocks = [
            {"values": obj_vals, "mgr_locs": np.array([cols.index(c) for c in obj_cols])},
            {"values": num_vals, "mgr_locs": np.array([cols.index(c) for c in num_cols])},
        ]
        bm = object.__new__(BlockManager)
        bm.state = (
            [columns, rows],
            [b["values"] for b in blocks],
            [index(Index, [cols[i] for i in b["mgr_locs"]]) for b in blocks],
            {"0.14.1": {"axes": [columns, rows], "blocks": blocks}},
        )
        df = object.__new__(DataFrame)
        df.state = {"_data": bm, "_typ": "dataframe", "_metadata": []}
        data = pickle.dumps(df, protocol=2)
    finally:
        for name in [m for m in sys.modules if m == FAKE or m.startswith(FAKE + ".")]:
            del sys.modules[name]
    assert b"zzpandas" in data
    return data.replace(FAKE.encode() + b".", b"pandas.")


def test_legacy_pickle_fails_in_modern_pandas(tmp_path):
    path = tmp_path / "legacy.pkl"
    path.write_bytes(_legacy_bytes(_raw_frame()))
    with pytest.raises(Exception):
        pd.read_pickle(path)


def test_legacy_reader_roundtrip(tmp_path):
    raw = _raw_frame()
    path = tmp_path / "legacy.pkl"
    path.write_bytes(_legacy_bytes(raw))
    back = read_legacy_pickle(path)
    assert list(back.columns) == list(raw.columns)
    assert list(back.index) == list(range(len(raw)))
    assert list(back["lotName"]) == list(raw["lotName"])
    assert np.array_equal(back["waferMap"].iloc[1], raw["waferMap"].iloc[1])
    assert list(back["waferIndex"]) == list(raw["waferIndex"])


def test_load_lswmd_falls_back_to_legacy(tmp_path):
    path = tmp_path / "LSWMD.pkl"
    path.write_bytes(_legacy_bytes(_raw_frame()))
    df = load_lswmd(path)
    assert list(df["label"]) == ["Center", "none", "Loc"]
    assert list(df["split"]) == ["Training", "Test", "Training"]
