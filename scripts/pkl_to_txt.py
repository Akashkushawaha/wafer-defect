"""Write a WM-811K pickle out as plain text you can open in any editor.

    python scripts/pkl_to_txt.py                               # data/sample.pkl -> data/sample.txt
    python scripts/pkl_to_txt.py data/LSWMD.pkl --limit 2000   # first 2000 wafers of the full file

Each wafer becomes one block: its raw fields exactly as stored, then the
wafer map as rows of digits (0 = no die, 1 = passed, 2 = failed). A second
file, <name>_index.csv, lists one line per wafer for opening in Excel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from waferdefect.data import read_raw


def raw_label(v) -> str:
    a = np.asarray(v, dtype=object)
    return "[]" if a.size == 0 else repr(a.tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", nargs="?", default="data/sample.pkl")
    ap.add_argument("-o", "--out", help="output .txt (default: next to the pickle)")
    ap.add_argument("--limit", type=int, help="only the first N wafers")
    args = ap.parse_args()

    src = Path(args.pkl)
    out = Path(args.out) if args.out else src.with_suffix(".txt")
    raw = read_raw(src)
    if args.limit:
        raw = raw.head(args.limit)

    with open(out, "w") as f, open(out.with_name(out.stem + "_index.csv"), "w") as idx:
        f.write(f"# {len(raw):,} wafers from {src.name}\n")
        f.write("# waferMap digits: 0 = no die (outside the wafer), 1 = die passed, 2 = die failed\n")
        f.write("# failureType [] = nobody labelled this wafer\n\n")
        idx.write("wafer_no,original_row,lotName,waferIndex,dieSize,rows,cols,failed_dies,trianTestLabel,failureType\n")
        for n, (row_id, r) in enumerate(raw.iterrows(), start=1):
            m = np.asarray(r["waferMap"])
            failed = int((m == 2).sum())
            label, split = raw_label(r["failureType"]), raw_label(r["trianTestLabel"])
            f.write(f"=== wafer {n} of {len(raw)}  (row {row_id} in the original file) ===\n")
            f.write(f"lotName         {r['lotName']}\n")
            f.write(f"waferIndex      {r['waferIndex']}\n")
            f.write(f"dieSize         {r['dieSize']}\n")
            f.write(f"trianTestLabel  {split}\n")
            f.write(f"failureType     {label}\n")
            f.write(f"waferMap        {m.shape[0]} rows x {m.shape[1]} columns, {failed} failed dies\n")
            for line in m:
                f.write("    " + "".join(str(int(v)) for v in line) + "\n")
            f.write("\n")
            idx.write(f"{n},{row_id},{r['lotName']},{r['waferIndex']},{r['dieSize']},{m.shape[0]},{m.shape[1]},"
                      f"{failed},\"{split}\",\"{label}\"\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and {out.stem}_index.csv")


if __name__ == "__main__":
    main()
