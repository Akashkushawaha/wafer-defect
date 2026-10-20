"""Figures for reports: confusion matrix and per-class example maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from . import LABELS  # noqa: E402

# off-wafer, good die, failed die
WAFER_CMAP = ListedColormap(["#ffffff", "#c9d3dc", "#c2410c"])


def confusion_figure(cm: np.ndarray, labels: list[str], path: str | Path, title: str = "") -> None:
    """Row-normalised confusion matrix (each row = recall for that true class)."""
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else "#1f2937")
    ax.set_title(title or "Confusion matrix (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def examples_figure(df: pd.DataFrame, path: str | Path, per_class: int = 4, seed: int = 0) -> None:
    """Grid of example maps, one row per class."""
    present = [lab for lab in LABELS if (df["label"] == lab).any()]
    fig, axes = plt.subplots(len(present), per_class, figsize=(per_class * 1.6, len(present) * 1.6), dpi=150,
                             squeeze=False)
    rng = np.random.default_rng(seed)
    for row, lab in zip(axes, present):
        sub = df[df["label"] == lab]
        picks = sub.iloc[rng.choice(len(sub), size=min(per_class, len(sub)), replace=False)]
        for ax in row:
            ax.axis("off")
        for ax, m in zip(row, picks["wafer_map"]):
            ax.imshow(m, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
        row[0].set_title(lab, loc="left", fontsize=9)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
