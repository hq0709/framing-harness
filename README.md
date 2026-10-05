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

Five models, twelve presentations of the same CT evidence, scored against the
counterfactual in which the finding is erased:

| model | plain | `centre` | `marginalise` | **both** | says yes (plain) |
|---|---|---|---|---|---|
| InternVL3-8B | 50.1% | **65.3%** | 50.0% | **73.4%** | 96.3% |
| Qwen3-VL-8B | 60.3% | 71.9% | 68.5% | **83.7%** | 70.4% |
| Lingshu-7B | 75.8% | 77.1% | 81.9% | **87.9%** | 30.3% |
| MedGemma-4b | 76.2% | 74.9% | 84.9% | 83.3% | 37.5% |
| Qwen2.5-VL-7B | 59.8% | 62.3% | 66.7% | 69.0% | 16.2% |
| **mean** | **64.4%** | **70.3%** | **70.4%** | **79.5%** | |

The two operators are close to orthogonal, so they add: **+5.9 points for centring, +6.0 for averaging,
+15.0 for both.** Centring costs nothing — it needs a batch, not a second reading.

The prediction they were derived from is the row that makes the case. InternVL3 answers yes to 96.3% of
everything, so its `b` never changes sign and averaging has no spread to remove: 50.1% to 50.0%, nothing.
Centring moves the zero instead and takes it to 65.3%, and both together to 73.4%. **Average against the
spread, centre against the offset.**

On the floor — the single worst presentation, which a deployed system does not get to avoid:

| | plain | centred | averaged | centred and averaged |
|---|---|---|---|---|
| balanced accuracy | 55.7% | 61.3% | 70.4% | **79.5%** |

`analysis/operators.py` prints this; `analysis/marginalisation_curve.py` shows how the averaging gain
accumulates with K (most of it by four readings, AUROC 0.763 to 0.863 at twelve).


### Does centring need a balanced batch?

It subtracts a batch median, and the runs above pair every real slice with the same slice minus its finding,
so the batch is exactly half positive. Deployment is not. Resampling the same readings from 5% to 95%
prevalence (`analysis/prevalence.py`):

| positives in the batch | 5% | 10% | 20% | 35% | 50% | 65% | 80% | 95% |
|---|---|---|---|---|---|---|---|---|
| no centring | 70.2% | 70.5% | 70.4% | 70.5% | 70.2% | 70.6% | 70.3% | 70.1% |
| **centre by median** | 68.4% | 69.2% | **72.0%** | **76.3%** | **79.2%** | **80.5%** | **77.3%** | **74.3%** |
| centre by quantile, given the prevalence | 65.0% | 70.4% | 76.7% | 81.0% | 79.2% | 74.0% | 63.0% | **50.7%** |

The median costs at most 1.8 points, and only below 20% prevalence; above that it pays, up to +9.9. **Use
the median and do not use the prevalence** — the quantile version looks more principled and is worse,
because fixing the threshold by quantile forces the predicted positive rate to equal the prevalence, which
is a far stronger assumption than moving a zero, and an imperfect ranking does not survive it.

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

`bash scripts/check.sh` is the one to run first on a new machine. It exists because the first version of
this repository was pushed with four scripts that compiled, started, and answered `--help`, and then died
on an undefined name the moment they did any work -- a compile check cannot see a NameError, and these
scripts parse their arguments before they touch anything.

Files prefixed `UNDERSIZED_LEVELS_`, `DISTANT_ERASE_LEVEL_` and `INVALID_` are kept deliberately: they are
measurements with a flaw that `METHOD.md` names. They are not inputs to any table.

## Running it

```bash
pip install -e .
bash scripts/check.sh                        # static scan, every script starts, tables rebuild
python scripts/smoke.py                      # ~2 min on a GPU: does the readout work on this machine

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
