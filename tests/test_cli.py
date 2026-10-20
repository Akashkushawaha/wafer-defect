import json

import numpy as np
import pandas as pd

from waferdefect.cli import load_features, main
from waferdefect.synthetic import make_dataset


def _as_raw_pickle(df: pd.DataFrame, path):
    tag = lambda s: np.array([[s]], dtype=object) if s else np.zeros((0, 0))  # noqa: E731
    raw = pd.DataFrame(
        {
            "waferMap": df["wafer_map"],
            "dieSize": df["die_size"].astype(float),
            "lotName": df["lot"],
            "waferIndex": df["wafer_index"].astype(float),
            "trianTestLabel": [tag("Training") if i % 3 else tag("Test") for i in range(len(df))],
            "failureType": [tag(lab) for lab in df["label"]],
        }
    )
    raw.to_pickle(path)


def test_demo_end_to_end(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo", "-o", str(out), "--n-per-class", "20", "--n-jobs", "1", "--model", "rf"]) == 0
    for name in ("examples.png", "confusion.png", "metrics.json", "report.txt", "model.joblib"):
        assert (out / name).exists(), name
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["split"] == "lot" and 0 <= metrics["macro_f1"] <= 1


def test_featurize_train_predict(tmp_path, capsys):
    df = make_dataset(n_per_class=15, seed=3, lots=10)
    pkl = tmp_path / "LSWMD.pkl"
    _as_raw_pickle(df, pkl)

    assert main(["inspect", str(pkl)]) == 0
    assert "wafer maps" in capsys.readouterr().out

    feats = tmp_path / "features.npz"
    assert main(["featurize", str(pkl), "-o", str(feats), "--n-jobs", "1"]) == 0
    d = load_features(feats)
    assert d["X"].shape[0] == len(df)

    model = tmp_path / "model.joblib"
    rep = tmp_path / "reports"
    for split in ("lot", "official"):
        assert main(["train", str(feats), "-o", str(model), "--split", split, "--model", "rf",
                     "--reports", str(rep)]) == 0
    assert (rep / "confusion.png").exists()

    lot, idx = df.iloc[0]["lot"], int(df.iloc[0]["wafer_index"])
    capsys.readouterr()
    assert main(["predict", str(model), str(pkl), "--lot", lot, "--wafer", str(idx)]) == 0
    out = capsys.readouterr().out
    assert "first suspects" in out


def test_explain(capsys):
    assert main(["explain", "Edge-Ring"]) == 0
    assert "edge" in capsys.readouterr().out.lower()
