"""Model building, lot-aware splitting and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import LABELS, __version__
from .features import FEATURE_NAMES

MODELS = ("hgb", "rf", "svm")


def build_model(kind: str = "hgb", seed: int = 0) -> Pipeline:
    """Classifier with class weighting for WM-811K's heavy imbalance.

    ~85% of tagged maps are ``none`` and the rarest class (Near-full) is
    ~0.1%, so unweighted models learn to predict ``none``. Every option here
    weights classes inversely to frequency.
    """
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, class_weight="balanced", early_stopping=True, random_state=seed
        )
        return make_pipeline(clf)
    if kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
        return make_pipeline(clf)
    if kind == "svm":  # the classifier family used in the WM-811K paper
        return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced", probability=True))
    raise ValueError(f"model must be one of {MODELS}")


def split_indices(
    y: np.ndarray,
    lots: np.ndarray,
    how: str = "lot",
    official: np.ndarray | None = None,
    test_size: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Train/test indices.

    ``lot`` (default) keeps every wafer of a lot on one side. Wafers in a lot
    went through the same tools together and share their defects, so a
    random split puts near-duplicates in both train and test and overstates
    accuracy. ``official`` uses the dataset's own Training/Test tags;
    ``random`` is a plain stratified split, kept for comparison.
    """
    idx = np.arange(len(y))
    if how == "lot":
        n_splits = max(2, round(1 / test_size))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train, test = next(cv.split(idx, y, groups=lots))
        return train, test
    if how == "official":
        if official is None:
            raise ValueError("official split requested but no split column given")
        official = np.asarray(official, dtype=object)
        return idx[official == "Training"], idx[official == "Test"]
    if how == "random":
        return train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    raise ValueError("split must be 'lot', 'official' or 'random'")


@dataclass
class Evaluation:
    labels: list[str]
    macro_f1: float
    balanced_accuracy: float
    accuracy: float
    report: dict
    confusion: np.ndarray
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "labels": self.labels,
            "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "accuracy": self.accuracy,
            "per_class": {k: v for k, v in self.report.items() if k in self.labels},
            "confusion": self.confusion.tolist(),
            **self.extra,
        }

    def summary(self) -> str:
        support = {lab: self.report.get(lab, {}).get("support", 0) for lab in self.labels}
        none_share = support.get("none", 0) / max(sum(support.values()), 1)
        note = f"   ({none_share:.0%} of test maps are 'none'; do not headline this)" if none_share > 0.5 else ""
        lines = [
            f"macro F1          {self.macro_f1:.3f}",
            f"balanced accuracy {self.balanced_accuracy:.3f}",
            f"accuracy          {self.accuracy:.3f}{note}",
            "",
            f"{'class':<12}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}",
        ]
        for lab in self.labels:
            r = self.report.get(lab)
            if r:
                lines.append(
                    f"{lab:<12}{r['precision']:>10.3f}{r['recall']:>9.3f}{r['f1-score']:>8.3f}{int(r['support']):>9}"
                )
        return "\n".join(lines)


def evaluate(model: Pipeline, X: np.ndarray, y: np.ndarray) -> Evaluation:
    pred = model.predict(X)
    labels = [lab for lab in LABELS if lab in set(y) | set(pred)]
    return Evaluation(
        labels=labels,
        macro_f1=float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        accuracy=float(np.mean(pred == y)),
        report=classification_report(y, pred, labels=labels, output_dict=True, zero_division=0),
        confusion=confusion_matrix(y, pred, labels=labels),
    )


def save(model: Pipeline, path: str | Path, meta: dict | None = None) -> None:
    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "version": __version__,
        "meta": meta or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load(path: str | Path) -> dict:
    payload = joblib.load(path)
    if payload.get("feature_names") != FEATURE_NAMES:
        raise ValueError("model was trained on a different feature set; re-run `wafer featurize` and `wafer train`")
    return payload
