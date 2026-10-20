"""See the RAW WM-811K data as pictures, before any cleaning.

Run from the project folder:
    python scripts/explore_raw.py                    # uses data/sample.pkl
    python scripts/explore_raw.py data/LSWMD.pkl     # full file (slower)

Prints what each raw column holds and opens three figures
(also saved in reports/explore/):
  1. raw_table.png  – 8 rows of the raw table, wafer maps drawn as thumbnails
  2. raw_row.png    – one row taken apart: the map, its numbers, every field
  3. raw_labels.png – the forms the raw label column takes, and how often
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from waferdefect.data import read_raw

WAFER = ListedColormap(["#ffffff", "#b9c4d2", "#c2570c"])  # 0 no die, 1 pass, 2 fail
OUT = Path("reports/explore")


def describe(v) -> str:
    """How a raw cell is actually stored, in a few words."""
    if isinstance(v, np.ndarray):
        return f"ndarray {v.shape} {v.dtype}"
    return f"{type(v).__name__} {v}"[:60]


def show_label(v) -> str:
    """The raw label exactly as stored, e.g. [['Center']] or []."""
    a = np.asarray(v, dtype=object)
    return "[]" if a.size == 0 else repr(a.tolist())


def main(path: str = "data/sample.pkl") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = read_raw(path)

    # ------------------------------------------------------------ 0. text summary
    print(f"\nRAW TABLE: {len(raw):,} rows x {raw.shape[1]} columns  ({path})\n")
    first = raw.iloc[0]
    for col in raw.columns:
        print(f"  {col:<15} {describe(first[col])}")
    print("\nFirst 5 rows (wafer maps shown by their shape):")
    preview = raw.head().copy()
    preview["waferMap"] = preview["waferMap"].map(lambda m: f"grid {m.shape}")
    preview["failureType"] = preview["failureType"].map(show_label)
    preview["trianTestLabel"] = preview["trianTestLabel"].map(show_label)
    print(preview.to_string())

    # ------------------------------------------------------------ 1. the table, drawn
    # Prefer a mix: some labelled, some not, so both raw label forms appear.
    labelled = raw[raw["failureType"].map(lambda v: np.asarray(v).size > 0)]
    unlabelled = raw[raw["failureType"].map(lambda v: np.asarray(v).size == 0)]
    rows = list(labelled.head(5).itertuples()) + list(unlabelled.head(3).itertuples())

    cols = ["waferMap", "dieSize", "lotName", "waferIndex", "trianTestLabel", "failureType"]
    fig, axes = plt.subplots(len(rows) + 1, len(cols), figsize=(15, 1.25 * (len(rows) + 1)),
                             gridspec_kw={"width_ratios": [1.1, 1, 1, 1, 1.6, 1.6]})
    for ax, name in zip(axes[0], cols):
        ax.axis("off")
        ax.text(0.5, 0.3, name, ha="center", fontsize=11, fontweight="bold", family="monospace")
    for r, row in enumerate(rows, start=1):
        for c, name in enumerate(cols):
            ax = axes[r, c]
            ax.axis("off")
            v = getattr(row, name)
            if name == "waferMap":
                ax.imshow(v, cmap=WAFER, vmin=0, vmax=2, interpolation="nearest")
            else:
                text = show_label(v) if name in ("failureType", "trianTestLabel") else str(v)
                ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=10, family="monospace",
                        color="#9a3412" if text == "[]" else "black")
    fig.suptitle("The raw table: one row = one wafer.  [] = nobody labelled it.", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "raw_table.png", dpi=130)

    # ------------------------------------------------------------ 2. one row taken apart
    row = labelled.iloc[0]
    m = row["waferMap"]
    fig = plt.figure(figsize=(14, 6.5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(m, cmap=WAFER, vmin=0, vmax=2, interpolation="nearest")
    ax1.set_title(f"waferMap as a picture  {m.shape}")
    ax1.set_xlabel("column")
    ax1.set_ylabel("row")

    ax2 = fig.add_subplot(1, 3, 2)
    h, w = min(16, m.shape[0]), min(16, m.shape[1])
    crop = m[:h, :w]
    ax2.imshow(crop, cmap=WAFER, vmin=0, vmax=2, interpolation="nearest")
    for i in range(h):
        for j in range(w):
            ax2.text(j, i, str(crop[i, j]), ha="center", va="center", fontsize=7,
                     color="white" if crop[i, j] == 2 else "black")
    ax2.set_title(f"top-left {h}x{w}: the actual numbers")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis("off")
    on, failed = int((m > 0).sum()), int((m == 2).sum())
    lines = [
        "One raw row, field by field",
        "",
        f"waferMap        grid {m.shape}, values {sorted(np.unique(m).tolist())}",
        "                0 = no die (outside the circle)",
        "                1 = die passed the test",
        "                2 = die failed",
        f"                {on} dies on the wafer, {failed} failed",
        "",
        f"dieSize         {row['dieSize']}",
        f"                (= dies on the wafer: {on})",
        f"lotName         '{row['lotName']}'",
        f"waferIndex      {row['waferIndex']}   (wafer no. within the lot)",
        f"trianTestLabel  {show_label(row['trianTestLabel'])}",
        f"failureType     {show_label(row['failureType'])}",
        "                (a label wrapped in two lists)",
    ]
    ax3.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "raw_row.png", dpi=130)

    # ------------------------------------------------------------ 3. what the label column holds
    forms = Counter(show_label(v) for v in raw["failureType"])
    names, counts = zip(*forms.most_common())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(names[::-1], counts[::-1], color=["#c2570c" if n == "[]" else "#0b5d8c" for n in names[::-1]])
    ax.bar_label(bars, labels=[f"{c:,}" for c in counts[::-1]], padding=3, fontsize=9)
    ax.set_title("Raw failureType values, exactly as stored ([] = unlabelled)")
    ax.set_xlabel("wafers")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "raw_labels.png", dpi=130)

    print(f"\nFigures saved in {OUT}/ — close the windows to finish.")
    plt.show()


if __name__ == "__main__":
    main(*sys.argv[1:])
