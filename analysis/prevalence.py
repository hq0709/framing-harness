"""Centring subtracts a batch median. What happens when the batch is not half positive?

The runs here pair every real slice with the same slice minus its finding, so the batch is exactly balanced
and the median sits where the two classes meet. Deployment is not balanced. If the operator only works at
50% prevalence it is a laboratory trick, so this resamples the same readings at prevalences from 5% to 95%
and watches it come apart -- or not.

The honest version of the operator is the quantile, not the median: subtract the (1 - prevalence) quantile,
which is the median only when half the cases are positive. Prevalence is a number a clinic knows about its
own population and no label is needed for it, so both are measured: the median as written, and the quantile
given a prevalence estimate, including a deliberately wrong one.

    python analysis/prevalence.py                # reads results/framing2_*.jsonl
"""
import glob
import json
import sys

import numpy as np

from fh import RESULTS, balanced_accuracy

pat = sys.argv[1] if len(sys.argv) > 1 else "framing2_*.jsonl"
F = sorted(glob.glob(str(RESULTS / pat)))
if not F:
    raise SystemExit(f"no runs matching {pat} in {RESULTS}")

PREV = [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95]
rng = np.random.default_rng(0)
rows = {k: {p: [] for p in PREV} for k in ("plain", "median", "quantile", "quantile_wrong")}

for f in F:
    data = [json.loads(l) for l in open(f)]
    keys = list(data[0]["f"])
    pos = {k: np.array([r["f"][k]["ar"] for r in data]) for k in keys}
    neg = {k: np.array([r["f"][k]["ac"] for r in data]) for k in keys}
    n = len(data)
    for p in PREV:
        for _ in range(30):
            npos = max(2, int(round(n * p)))
            ipos = rng.choice(n, npos, replace=True)
            ineg = rng.choice(n, max(2, n - npos), replace=True)
            lab = np.r_[np.ones(len(ipos), bool), np.zeros(len(ineg), bool)]
            batch = {k: np.r_[pos[k][ipos], neg[k][ineg]] for k in keys}
            cen_m = {k: v - np.median(v) for k, v in batch.items()}
            # the quantile the prevalence implies: the top p of a batch should be the positives
            cen_q = {k: v - np.quantile(v, 1 - p) for k, v in batch.items()}
            # a clinic that believes the prevalence is 50% when it is not
            cen_w = {k: v - np.quantile(v, 0.5) for k, v in batch.items()}
            rows["plain"][p].append(balanced_accuracy(np.mean(list(batch.values()), axis=0), lab))
            rows["median"][p].append(balanced_accuracy(np.mean(list(cen_m.values()), axis=0), lab))
            rows["quantile"][p].append(balanced_accuracy(np.mean(list(cen_q.values()), axis=0), lab))
            rows["quantile_wrong"][p].append(balanced_accuracy(np.mean(list(cen_w.values()), axis=0), lab))

print("Balanced accuracy after averaging over all presentations, by how many of the batch are positive.\n")
print(f"{'prevalence':>11}" + "".join(f"{int(p * 100):>8}%" for p in PREV))
print("-" * (11 + 9 * len(PREV)))
for k, lab in (("plain", "no centring"), ("median", "centre: median"),
               ("quantile", "centre: quantile"), ("quantile_wrong", "  ...assuming 50%")):
    print(f"{lab:>11}" + "".join(f"{np.mean(rows[k][p]) * 100:8.1f}%" for p in PREV))

m = np.array([np.mean(rows["median"][p]) for p in PREV])
q = np.array([np.mean(rows["quantile"][p]) for p in PREV])
b = np.array([np.mean(rows["plain"][p]) for p in PREV])
print(f"\n  median-centring beats no centring at {int((m > b).sum())}/{len(PREV)} prevalences"
      f"   (worst case {min(m - b) * 100:+.1f}pp, best {max(m - b) * 100:+.1f}pp)")
print(f"  quantile-centring, given the prevalence: {int((q > b).sum())}/{len(PREV)}"
      f"   (worst {min(q - b) * 100:+.1f}pp, best {max(q - b) * 100:+.1f}pp)")
lo = [i for i, x in enumerate(PREV) if m[i] <= b[i]]
print(f"\n  So: use the median, and do not use the prevalence. Fixing the threshold by quantile forces the"
      f"\n  predicted positive rate to equal the prevalence, which is a far stronger assumption than moving"
      f"\n  a zero, and an imperfect ranking cannot survive it -- at {int(PREV[-1] * 100)}% prevalence it"
      f" falls to {q[-1] * 100:.1f}%."
      f"\n  Median-centring costs at most {-min(m - b) * 100:.1f}pp, and only below "
      f"{int(PREV[max(lo) + 1] * 100) if lo and max(lo) + 1 < len(PREV) else 0}% prevalence; above that it pays,"
      f" up to {max(m - b) * 100:+.1f}pp.")
