# The answer follows the frame

Hold a patient, a CT level and a tumour fixed, and change only how the slice is cropped. Qwen2.5-VL's rate
of answering *yes* to "is there an abnormality" runs from **2.8% to 96.4%**. Its ability to tell that slice
from the same slice with the tumour painted out never moves: AUROC 0.617 to 0.702 throughout.

The evidence is there. The answer does not use it.

```
m(x, q) = b(presentation, q) + e(x, q)
```

`b` depends on the crop, the window and the wording; `e` is the image. In medical imaging `b` dominates, so
every metric read off a threshold — accuracy, sensitivity, specificity, and every silent-failure rate built
from them — is partly reporting how the images were cropped.

**The presentation is a nuisance, and it is one we control.** This repository removes it with three
training-free operators and measures what that is worth.

| operator | removes | cost | needs |
|---|---|---|---|
| `centre` — subtract the presentation's batch median | the mean of `b` | nothing | a batch |
| `marginalise` — average the margin over K presentations | the variance of `b` | K readings | nothing |
| `difference` — subtract a matched input read the same way | `b` exactly | 2× | a counterpart |

The only precondition is that **the same evidence can be presented more than once**. No labels, no training,
no masks, no paired organ, no prior study, no change to the weights.

## What it buys

Averaging the margin over twelve presentations of the same CT evidence, five models:

| K presentations | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| balanced accuracy | 64.2% | 66.1% | 68.9% | 70.0% | **70.4%** |
| AUROC | 0.763 | 0.806 | 0.837 | 0.857 | **0.863** |

(`analysis/marginalisation_curve.py` prints this. Values below K=12 are means over random subsets of the
twelve presentations, so they move a little between runs; K=12 and the floor below are exact.)

The number that matters is the floor, because a deployed system does not choose which presentation it is
handed: **the worst single presentation gives 55.7%, the average gives 70.4%** — 14.7 points — and most
of the gain is already there at four readings. MedGemma goes 76.2% → 84.9% and Lingshu 75.8% → 81.9%, both
above their own *best* single presentation.

The exception names the division of labour. InternVL3 answers yes to 85–100% of everything, so its `b` never
changes sign and averaging cannot cancel it (52.4% at K=1, 50.0% at K=12). The matched difference, which
removes `b`'s mean rather than its variance, takes it to 57.5%. **Average against the spread, difference
against the offset.**

## Why looking closer does not work here

The obvious remedy — crop in, zoom, more pixels — was measured and it fails on medical images while working
perfectly on natural ones. Same counterfactual control, same model, same question type:

| the finding covers | natural image | CT |
|---|---|---|
| 0.05–0.07% of frame | 0.542 | 0.523 |
| 0.19–0.2% | 0.758 | 0.620 |
| 0.9–1.0% | **0.914** | 0.655 |

Magnifying a tumour 64-fold, from 0.08% of the frame to 5.4%, moves grounding by **+0.009**.

## Layout

```
src/fh/            the framework: config, readout, presentations, operators
experiments/       the measurements, one file each
  framing.py              12 presentations x 5 models, the central measurement
  counterfactual.py       erase the finding and ask again -- the only sound control
  magnification.py        does looking closer help (no)
  natural_vs_medical.py   the same dial on COCO (yes)
  matched_counterpart.py  the difference operator, 4 organs, real-pixel negatives
  marginalise_vqa.py      the framework on VQA-RAD and PathVQA, no masks, no anatomy
  exploratory/            paths that were tried and refuted -- see METHOD.md
analysis/          tables from the runs in results/
results/           every run in this README, as JSON and per-item JSONL
METHOD.md          the full argument, including what was refuted and why
```

`results/` ships with the runs, so every table above can be re-derived without a GPU:

```bash
pip install -e .
python analysis/marginalisation_curve.py     # the K curve
python analysis/stability.py                 # floor and spread per readout
python analysis/framing_table.py             # per-model, per-axis breakdown
python analysis/scope.py                     # where the difference operator pays
```

Files prefixed `UNDERSIZED_LEVELS_`, `DISTANT_ERASE_LEVEL_` and `INVALID_` are kept deliberately: they are
measurements with a flaw that `METHOD.md` names. They are not inputs to any table.

## Running it

```bash
pip install -e .
bash scripts/fetch_data.sh                   # tells you what to download and fetches what it can
export FH_DATA=/path/to/data                 # default ./data
export FH_RESULTS=/path/to/results           # default ./results

# the central measurement: 12 presentations of the same evidence, one model
python experiments/framing.py --model Qwen/Qwen2.5-VL-7B-Instruct --cases 63 --levels 4

# the framework on ordinary medical VQA -- no masks, no anatomy
python experiments/marginalise_vqa.py --set vqa-rad --model google/medgemma-4b-it --n 400
```

One 40 GB GPU is enough for the 7–8B models. `framing.py` at 63 cases × 4 levels is about 12k readings,
roughly half an hour.

**Sampling matters and it bit us.** `framing.py --pick spread` takes levels evenly across a tumour's extent,
which always includes its first and last — its smallest cross-sections. That drew lesions 4.4× smaller by
median area than the other runs and put two thirds of them inside the quartile where grounding is exactly
chance. The default is `--pick largest`, which reads the model at its best case.

## The one control that works

Every control we built from *other levels* turned out to measure craniocaudal position instead, and three
results were discarded for it — one gave AUROC 0.838 that fell to 0.479 under a proper control, one gave
0.86, and one inverted to 0.012 because the far end of a chest CT is the neck or the upper abdomen. A fourth
pasted spine where a liver tumour had been and flipped the sign of the measurement.

A chest varies more along its own axis than a tumour varies from the tissue beside it. So the negative is
either the same slice with the finding painted over by its contralateral mirror, or the same window at the
*nearest* level the finding does not reach. `figures/discarded_distant_level_erase.png` shows what the
distant version looked like.

## Licence

MIT. Neither the Decathlon, COCO, nor the VQA sets are redistributed; see `scripts/fetch_data.sh`.
