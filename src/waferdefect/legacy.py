"""Read the 2015-era ``LSWMD.pkl`` without the pandas version that wrote it.

WM-811K is distributed as a DataFrame pickled by pandas 0.x. Those pickles
reference module paths (``pandas.indexes.*``, ``Int64Index``, the 0.14.1
BlockManager state tuple) that modern pandas no longer ships, so
``pandas.read_pickle`` fails on pandas >= 2.

Instead of shimming pandas internals, this module unpickles every pandas
object into an inert stand-in that just records its state, then rebuilds a
plain DataFrame from the recorded column blocks. Nothing pandas-specific is
executed during unpickling, so it does not depend on the installed version.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class _Stub:
    """Placeholder for any pandas class found in the pickle."""

    _qualname = "?"
    # Defaults for objects created via copyreg._reconstructor (object.__new__).
    args: tuple = ()
    state: Any = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "_Stub":
        obj = object.__new__(cls)
        obj.args = args
        obj.state = None
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        self.state = state

    def __repr__(self) -> str:
        return f"<legacy {self._qualname}>"


class _LegacyIndex:
    """Result of the legacy ``_new_Index(cls, d)`` reconstructor."""

    def __init__(self, cls: Any, d: dict) -> None:
        self.cls = cls
        self.d = d if isinstance(d, dict) else {}

    def values(self) -> np.ndarray:
        d = self.d
        if "data" in d:
            data = d["data"]
            if isinstance(data, _LegacyIndex):
                return data.values()
            return np.asarray(data)
        if {"start", "stop"} <= d.keys():  # RangeIndex
            return np.arange(d["start"], d["stop"], d.get("step", 1))
        raise ValueError(f"cannot reconstruct index from keys {sorted(d)}")


def _new_index(cls: Any, d: dict | None = None) -> _LegacyIndex:
    return _LegacyIndex(cls, d or {})


class LegacyUnpickler(pickle.Unpickler):
    """Unpickler that turns pandas objects into stand-ins and keeps numpy."""

    _stub_cache: dict[tuple[str, str], type] = {}

    def find_class(self, module: str, name: str) -> Any:
        if module == "pandas" or module.startswith("pandas."):
            if name == "_new_Index":
                return _new_index
            key = (module, name)
            if key not in self._stub_cache:
                self._stub_cache[key] = type(name, (_Stub,), {"_qualname": f"{module}.{name}"})
            return self._stub_cache[key]
        if module.startswith("numpy.core"):
            # numpy 2 moved numpy.core -> numpy._core
            module = "numpy._core" + module[len("numpy.core"):]
        return super().find_class(module, name)


def _axis_values(ax: Any) -> np.ndarray:
    if isinstance(ax, _LegacyIndex):
        return ax.values()
    if isinstance(ax, _Stub):  # an Index pickled via NEWOBJ/BUILD
        state = ax.state
        if isinstance(state, dict) and "data" in state:
            return np.asarray(state["data"])
        if isinstance(state, tuple) and state:
            return np.asarray(state[0])
    return np.asarray(ax)


def _locs(locs: Any, n_items: int) -> np.ndarray:
    if isinstance(locs, slice):
        return np.arange(n_items)[locs]
    if isinstance(locs, _Stub):  # BlockPlacement stand-in
        return _locs(locs.args[0] if locs.args else locs.state, n_items)
    return np.asarray(locs, dtype=int).ravel()


def frame_from_stub(obj: Any) -> pd.DataFrame:
    """Rebuild a DataFrame from a stand-in produced by :class:`LegacyUnpickler`."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if not isinstance(obj, _Stub) or not isinstance(obj.state, dict) or "_data" not in obj.state:
        raise ValueError("pickle does not contain a legacy pandas DataFrame")
    mgr = obj.state["_data"]
    state = mgr.state
    if not isinstance(state, tuple) or len(state) < 2:
        raise ValueError(f"unrecognised BlockManager state: {type(state).__name__}")

    extra = state[3] if len(state) >= 4 and isinstance(state[3], dict) else {}
    if "0.14.1" in extra:
        axes = extra["0.14.1"]["axes"]
        blocks = [(b["values"], b["mgr_locs"]) for b in extra["0.14.1"]["blocks"]]
        columns = _axis_values(axes[0])
        blocks = [(vals, _locs(locs, len(columns))) for vals, locs in blocks]
    else:  # pre-0.14.1: (axes, block_values, block_items)
        axes, values, items = state[0], state[1], state[2]
        columns = _axis_values(axes[0])
        col_pos = {c: i for i, c in enumerate(columns)}
        blocks = [(v, np.array([col_pos[c] for c in _axis_values(it)])) for v, it in zip(values, items)]

    index = _axis_values(axes[1])
    data: dict[Any, Any] = {}
    for vals, locs in blocks:
        vals = np.asarray(vals)
        if vals.ndim == 1:
            vals = vals[None, :]
        for row, pos in zip(vals, locs):
            data[columns[pos]] = row
    return pd.DataFrame({c: data[c] for c in columns}, index=index)


def read_legacy_pickle(path: str | Path) -> pd.DataFrame:
    with open(path, "rb") as fh:
        obj = LegacyUnpickler(fh, encoding="latin1").load()
    return frame_from_stub(obj)
