# wafer-defect

Classifying wafer-map failure patterns on **WM-811K** (811,457 real wafer maps
from 46,393 production lots), and saying what each pattern points at on the
fab floor.

A wafer map is the result of electrically testing every die on a wafer: pass
or fail, laid out where the die sat. When failures are random, that's normal
yield loss. When they form a *shape* — a ring at the edge, a blob in the
middle, a line — something systematic in the process caused them, and the
shape says which part of the process to look at. This project classifies the
shape and then names the first suspects.

```
$ wafer predict artifacts/model.joblib LSWMD.pkl --lot lot1 --wafer 7
lot lot1 wafer 7  (tagged: Edge-Ring)
  Edge-Ring  0.94
  Edge-Loc   0.04
  Loc        0.01

Edge-Ring: Failures all the way round the wafer edge.
first suspects:
  - Edge exclusion effects: film roll-off, edge-bead removal, etch-rate change at the edge
  - Lithography focus or exposure drift near the edge; wafer bow
check next: Check edge-die metrology and edge-bead / bevel-clean steps; a full ring means a rotationally symmetric cause.
```
<sub>Output format shown; numbers depend on your trained model.</sub>

## Approach

**Hand-built features, not a CNN.** Each map becomes 68 numbers that describe
*where* the failures are and *what shape* they make. The model then learns on
those. This runs on a laptop CPU in minutes, every feature has a physical
meaning you can inspect, and it is the approach of the paper that released
the dataset (Wu, Jang & Chen, *IEEE Trans. Semiconductor Manufacturing*, 2015).

| Block | # | What it captures |
|---|---|---|
| Zone density | 13 | Failure rate in 4 edge bands and a 3×3 interior grid |
| **Radial profile** | 5 | Failure rate in equal-area rings, centre → edge. *Added in this project* |
| Radon transform | 40 | Line sums at every angle. Lines (scratches) peak at one angle; rings don't |
| Geometry | 7 | Size, reach, elongation and solidity of the largest failure cluster |
| Global | 3 | Overall fail fraction, cluster count, how concentrated the failures are |

Two deliberate changes from the paper:

1. **Every feature is size-normalised.** Maps range from 6×21 to 300×202 dies.
   Raw counts would teach the model die size, not pattern.
2. **A radial profile block.** Center, Donut and Edge-Ring differ almost only
   in *which radius* fails — and radius is where process physics shows up
   (spin-coating, CMP pressure zones, edge effects are all radial).

### Evaluation choices that matter

- **Lot-aware split (default).** Wafers in a lot run through the same tools
  together and share defects. A random split puts near-duplicates on both
  sides and inflates the score. `--split lot` keeps each lot on one side;
  `--split official` uses the dataset's tags; `--split random` exists to show
  the difference.
- **Macro F1 is the headline, not accuracy.** ~85% of tagged maps are `none`,
  so predicting `none` every time scores ~85% accuracy and is useless.
- **Class weighting** throughout, because Near-full is ~0.1% of the data.

## Pattern → process: what the classes mean

| Pattern | Spatial signature | Usual first suspects |
|---|---|---|
| **Center** | Failures around the centre | Centre-out processes: spin-coat/develop, CMP pressure profile, chamber gas/heat distribution |
| **Donut** | Ring at mid-radius | Zone mismatch in a multi-zone heater, chuck or CMP carrier |
| **Edge-Ring** | Ring at the edge | Film roll-off, edge-bead removal, edge etch rate, litho focus near the edge |
| **Edge-Loc** | One patch of the edge | Mechanical contact — clamps, pins, robot end-effector |
| **Loc** | Interior cluster | Particle event or local process excursion |
| **Scratch** | Line or arc | CMP (pad debris, slurry agglomerates) or handling damage |
| **Random** | No structure | Random defect density; baseline yield loss below a threshold |
| **Near-full** | Almost all dies fail | Gross failure — wrong recipe, missed step — or a tester/probe-card problem |

These are first suspects, not diagnoses: a yield engineer confirms by
correlating with tool/chamber history and in-line metrology. Full notes in
[`src/waferdefect/causes.py`](src/waferdefect/causes.py), and `wafer explain <pattern>`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # 49 tests, ~20 s, no dataset needed
wafer demo                  # whole pipeline on synthetic maps -> demo/
```

### On the real data

Download `LSWMD.pkl` (2.1 GB) from
[Kaggle: WM-811K wafer map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
into `data/`.

```bash
wafer inspect data/LSWMD.pkl --examples reports/examples.png   # check it loads
wafer featurize data/LSWMD.pkl -o artifacts/features.npz       # ~173k tagged maps
wafer train artifacts/features.npz --split lot --reports reports/
wafer train artifacts/features.npz --split random --reports reports/random/   # compare
wafer predict artifacts/model.joblib data/LSWMD.pkl --lot <lot> --wafer <n>
```

Add `--limit 20000` to `featurize` for a quick first run.

## Results

*To be filled in from `reports/` after training on WM-811K: macro F1 and
per-class recall on the lot split, the confusion matrix, and the gap between
lot-aware and random splits.*

## Engineering notes

**Reading a 2015 pickle under pandas 3.** `LSWMD.pkl` was written by pandas
0.x and references modules that no longer exist (`pandas.indexes`,
`Int64Index`), so `pd.read_pickle` fails on any recent pandas. Rather than
pinning an ancient pandas or monkey-patching its internals,
[`legacy.py`](src/waferdefect/legacy.py) unpickles every pandas object into an
inert stand-in that just records its state, then rebuilds a DataFrame from the
recorded column blocks. Tests recreate the 0.x object layout byte-for-byte to
check this (`tests/test_data.py`).

**Layout**

```
src/waferdefect/
  data.py        load + clean WM-811K (tags are nested arrays like [['Center']])
  legacy.py      pandas-0.x pickle reader
  features.py    the 68 features
  model.py       classifiers, lot-aware split, evaluation
  causes.py      pattern -> process knowledge
  synthetic.py   synthetic maps for tests and the demo
  plots.py       confusion matrix, example grids
  cli.py         `wafer` command
tests/           pytest suite
```

## Data

WM-811K: Wu, M.-J., Jang, J.-S. R., & Chen, J.-L. (2015). *Wafer Map Failure
Pattern Recognition and Similarity Ranking for Large-Scale Data Sets.* IEEE
Transactions on Semiconductor Manufacturing, 28(1), 1–12. The dataset is not
included in this repository.
