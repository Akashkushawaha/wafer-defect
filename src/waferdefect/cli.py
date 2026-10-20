"""Command-line interface: ``wafer <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import LABELS, __version__
from .causes import explain
from .features import FEATURE_NAMES, extract, extract_many

log = logging.getLogger("waferdefect")


# ---------------------------------------------------------------- feature files

def save_features(path: Path, df: pd.DataFrame, X: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X.astype(np.float32),
        y=df["label"].fillna("").astype(str).to_numpy(dtype="U16"),
        lot=df["lot"].astype(str).to_numpy(dtype="U32"),
        wafer_index=df["wafer_index"].fillna(-1).astype(int).to_numpy(),
        split=df["split"].fillna("").astype(str).to_numpy(dtype="U16"),
        feature_names=np.array(FEATURE_NAMES, dtype="U32"),
    )


def load_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if list(data["feature_names"]) != FEATURE_NAMES:
        raise SystemExit(f"{path} was built with a different feature set; re-run `wafer featurize`.")
    return data


def _cap_per_class(y: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    idx = np.arange(len(y))
    if not cap:
        return idx
    rng = np.random.default_rng(seed)
    keep = [rng.permutation(idx[y == lab])[:cap] for lab in np.unique(y)]
    return np.sort(np.concatenate(keep))


# ---------------------------------------------------------------- commands

def cmd_inspect(args: argparse.Namespace) -> int:
    from .data import load_lswmd, summarize

    df = load_lswmd(args.pkl, labeled_only=not args.all)
    print(summarize(df))
    if args.examples:
        from .plots import examples_figure

        examples_figure(df[df["label"].notna()], args.examples)
        print(f"example maps -> {args.examples}")
    return 0


def cmd_featurize(args: argparse.Namespace) -> int:
    from .data import load_lswmd

    t0 = time.time()
    df = load_lswmd(args.pkl, labeled_only=not args.all)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=0).reset_index(drop=True)
    print(f"loaded {len(df):,} maps in {time.time() - t0:.0f}s; extracting {len(FEATURE_NAMES)} features ...")
    t1 = time.time()
    X = extract_many(df["wafer_map"], n_jobs=args.n_jobs, verbose=5 if args.verbose else 0)
    save_features(Path(args.out), df, X)
    print(f"features {X.shape} in {time.time() - t1:.0f}s -> {args.out}")
    return 0


def _train_and_report(
    X: np.ndarray,
    y: np.ndarray,
    lots: np.ndarray,
    split: np.ndarray | None,
    args: argparse.Namespace,
) -> int:
    from . import model as M
    from .plots import confusion_figure

    tr, te = M.split_indices(y, lots, how=args.split, official=split, seed=args.seed)
    # Cap only the training side: the test set keeps the real class mix.
    tr = tr[_cap_per_class(y[tr], args.max_per_class, args.seed)]
    print(f"split={args.split}: train {len(tr):,} / test {len(te):,} "
          f"(lots shared: {len(set(lots[tr]) & set(lots[te]))})")

    clf = M.build_model(args.model, seed=args.seed)
    t0 = time.time()
    clf.fit(X[tr], y[tr])
    fit_s = time.time() - t0
    ev = M.evaluate(clf, X[te], y[te])
    ev.extra.update({"model": args.model, "split": args.split, "n_train": int(len(tr)), "n_test": int(len(te)),
                     "fit_seconds": round(fit_s, 1), "version": __version__})
    print(ev.summary())

    M.save(clf, args.out, meta=ev.extra)
    print(f"model -> {args.out}")
    if args.reports:
        rep = Path(args.reports)
        rep.mkdir(parents=True, exist_ok=True)
        (rep / "metrics.json").write_text(json.dumps(ev.to_dict(), indent=2))
        (rep / "report.txt").write_text(ev.summary() + "\n")
        confusion_figure(ev.confusion, ev.labels, rep / "confusion.png",
                         title=f"{args.model} · {args.split} split · macro F1 {ev.macro_f1:.3f}")
        print(f"reports -> {rep}/")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    d = load_features(Path(args.features))
    keep = d["y"] != ""
    if args.drop_none:
        keep &= d["y"] != "none"
    split = d["split"][keep]
    return _train_and_report(d["X"][keep], d["y"][keep], d["lot"][keep],
                             np.where(split == "", None, split), args)


def cmd_predict(args: argparse.Namespace) -> int:
    from . import model as M
    from .data import load_lswmd

    payload = M.load(args.model)
    clf = payload["model"]
    df = load_lswmd(args.pkl, labeled_only=False)
    sel = df[(df["lot"] == args.lot) & (df["wafer_index"] == args.wafer)]
    if sel.empty:
        raise SystemExit(f"no wafer {args.wafer} in lot {args.lot}")
    row = sel.iloc[0]
    proba = clf.predict_proba(extract(row["wafer_map"])[None, :])[0]
    order = np.argsort(proba)[::-1][:3]
    print(f"lot {args.lot} wafer {args.wafer}  (tagged: {row['label'] or 'untagged'})")
    for i in order:
        print(f"  {clf.classes_[i]:<10} {proba[i]:.2f}")
    _print_signature(clf.classes_[order[0]])
    return 0


def _print_signature(pattern: str) -> None:
    sig = explain(pattern)
    print(f"\n{sig.pattern}: {sig.looks_like}")
    print("first suspects:")
    for s in sig.first_suspects:
        print(f"  - {s}")
    print(f"check next: {sig.check_next}")


def cmd_explain(args: argparse.Namespace) -> int:
    _print_signature(args.pattern)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .plots import examples_figure
    from .synthetic import make_dataset

    out = Path(args.out)
    df = make_dataset(n_per_class=args.n_per_class, seed=args.seed)
    print(f"synthetic set: {len(df)} maps, {df['lot'].nunique()} lots (idealised; not a benchmark)")
    examples_figure(df, out / "examples.png")
    X = extract_many(df["wafer_map"], n_jobs=args.n_jobs)
    ns = argparse.Namespace(split="lot", seed=args.seed, max_per_class=None, model=args.model,
                            out=str(out / "model.joblib"), reports=str(out))
    return _train_and_report(X, df["label"].to_numpy(), df["lot"].to_numpy(), None, ns)


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wafer", description="Wafer-map defect pattern classification (WM-811K)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("inspect", help="load LSWMD.pkl and print class/size statistics")
    s.add_argument("pkl")
    s.add_argument("--all", action="store_true", help="include untagged maps")
    s.add_argument("--examples", metavar="PNG", help="also save a grid of example maps")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("featurize", help="extract features from LSWMD.pkl into an .npz file")
    s.add_argument("pkl")
    s.add_argument("-o", "--out", default="artifacts/features.npz")
    s.add_argument("--all", action="store_true", help="include untagged maps")
    s.add_argument("--limit", type=int, help="random subset, for a quick run")
    s.add_argument("--n-jobs", type=int, default=-1)
    s.set_defaults(func=cmd_featurize)

    def train_opts(s: argparse.ArgumentParser) -> None:
        s.add_argument("--model", choices=("hgb", "rf", "svm"), default="hgb")
        s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("train", help="train and evaluate a classifier on a feature file")
    s.add_argument("features")
    s.add_argument("-o", "--out", default="artifacts/model.joblib")
    s.add_argument("--split", choices=("lot", "official", "random"), default="lot")
    s.add_argument("--drop-none", action="store_true", help="classify defect patterns only")
    s.add_argument("--max-per-class", type=int, help="cap training maps per class (speeds up svm)")
    s.add_argument("--reports", default="reports", help="directory for metrics.json, report.txt, confusion.png")
    train_opts(s)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("predict", help="classify one wafer and print likely causes")
    s.add_argument("model")
    s.add_argument("pkl")
    s.add_argument("--lot", required=True)
    s.add_argument("--wafer", type=int, required=True, help="wafer index within the lot")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("explain", help="what a pattern usually points at in the fab")
    s.add_argument("pattern", choices=LABELS)
    s.set_defaults(func=cmd_explain)

    s = sub.add_parser("demo", help="run the full pipeline on synthetic maps (no dataset needed)")
    s.add_argument("-o", "--out", default="demo")
    s.add_argument("--n-per-class", type=int, default=80)
    s.add_argument("--n-jobs", type=int, default=-1)
    train_opts(s)
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
