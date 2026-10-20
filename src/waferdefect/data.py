"""Loading and cleaning WM-811K (``LSWMD.pkl``)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import LABELS
from .legacy import read_legacy_pickle

log = logging.getLogger(__name__)

# Column names as they appear in LSWMD.pkl (including the dataset's own typo).
RAW_COLUMNS = {
    "waferMap": "wafer_map",
    "dieSize": "die_size",
    "lotName": "lot",
    "waferIndex": "wafer_index",
    "trianTestLabel": "split",
    "failureType": "label",
}


def clean_tag(value) -> str | None:
    """WM-811K stores tags as nested arrays: ``[['Center']]`` or ``[]`` if untagged."""
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    arr = np.asarray(value, dtype=object).ravel()
    if arr.size == 0:
        return None
    tag = arr[0]
    if isinstance(tag, bytes):
        tag = tag.decode("latin1")
    tag = str(tag).strip()
    return tag or None


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read the pickle with modern pandas if possible, else the legacy reader."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download LSWMD.pkl from "
            "https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map"
        )
    try:
        return pd.read_pickle(path)
    except Exception as exc:  # pandas >= 2 cannot read the 0.x pickle
        log.info("pandas.read_pickle failed (%s); using legacy reader", type(exc).__name__)
        return read_legacy_pickle(path)


def tidy(raw: pd.DataFrame, labeled_only: bool = True) -> pd.DataFrame:
    """Rename columns, decode tags, and drop maps that cannot be used."""
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"not a WM-811K frame; missing columns {sorted(missing)}")
    df = raw[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS).copy()
    df["label"] = df["label"].map(clean_tag)
    df["split"] = df["split"].map(clean_tag)
    df["lot"] = df["lot"].astype(str)
    df["wafer_index"] = pd.to_numeric(df["wafer_index"], errors="coerce")

    unknown = set(df["label"].dropna()) - set(LABELS)
    if unknown:
        raise ValueError(f"unexpected failure types {sorted(unknown)}")

    # A handful of maps are degenerate (a single row/column); they carry no pattern.
    shapes = df["wafer_map"].map(lambda m: np.asarray(m).shape)
    ok = shapes.map(lambda s: len(s) == 2 and min(s) >= 3)
    if (~ok).any():
        log.info("dropping %d degenerate maps", int((~ok).sum()))
    df = df[ok]
    if labeled_only:
        df = df[df["label"].notna()]
    df["wafer_map"] = df["wafer_map"].map(lambda m: np.asarray(m, dtype=np.uint8))
    return df.reset_index(drop=True)


def load_lswmd(path: str | Path, labeled_only: bool = True) -> pd.DataFrame:
    """Load ``LSWMD.pkl`` as a tidy frame.

    Columns: ``wafer_map`` (uint8 array, 0/1/2), ``label`` (one of
    :data:`waferdefect.LABELS` or None), ``lot``, ``wafer_index``, ``split``
    (``Training``/``Test``/None) and ``die_size``.
    """
    return tidy(read_raw(path), labeled_only=labeled_only)


def summarize(df: pd.DataFrame) -> str:
    lines = [f"{len(df):,} wafer maps from {df['lot'].nunique():,} lots"]
    counts = df["label"].fillna("<untagged>").value_counts()
    for label, n in counts.items():
        lines.append(f"  {label:<12} {n:>9,}  {100 * n / len(df):5.1f}%")
    sizes = df["wafer_map"].map(lambda m: m.shape)
    rows = sizes.map(lambda s: s[0])
    lines.append(
        f"map size: {rows.min()}..{rows.max()} rows, median {int(rows.median())}; "
        f"{sizes.nunique():,} distinct shapes"
    )
    if df["split"].notna().any():
        lines.append("official split: " + ", ".join(f"{k}={v:,}" for k, v in df["split"].value_counts().items()))
    return "\n".join(lines)
