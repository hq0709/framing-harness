"""All three operators on the same data, so the framework's claim is measured and not just stated.

`marginalise` and the matched `difference` were measured on the CT runs; `centre` was not, and it is the one
the framework leans on hardest, because it costs nothing and because it is the only one that can move a zero
that never changes sign. A model answering yes to 85-100% of everything cannot be helped by averaging --
there is no spread to average away -- and has no anatomy requirement that the difference operator could use
on an arbitrary benchmark. Centring is what is left, and it needs only a batch.

The batch statistic is taken over the readings themselves. No label is touched.

    python analysis/operators.py                 # reads results/framing2_*.jsonl
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

print("Balanced accuracy at the natural zero. Positives are real slices, negatives the same slice with the")
print("finding erased, so a readout that cannot separate them is not reading the finding.\n")
print(f"{'model':<24}{'plain':>8}{'centre':>9}{'marg':>8}{'centre+marg':>13}{'matched':>10}   says yes (plain)")
print("-" * 92)
agg = {k: [] for k in ("plain", "centre", "marg", "centre+marg", "matched")}
floor = {}
for f in F:
    rows = [json.loads(l) for l in open(f)]
    name = json.load(open(f.replace(".jsonl", ".json")))["model"].split("/")[-1]
    keys = list(rows[0]["f"])
    lab = np.r_[np.ones(len(rows), bool), np.zeros(len(rows), bool)]
    g = lambda k, t: np.array([r["f"][k][t] for r in rows])
    # each presentation's readings over the whole batch, positives and negatives together: that pooled
    # vector is what the batch median is taken over, which is exactly what deployment would have
    pooled = {k: np.r_[g(k, "ar"), g(k, "ac")] for k in keys}
    centred = {k: v - np.median(v) for k, v in pooled.items()}
    matched = {k: np.r_[g(k, "ar") - g(k, "mr"), g(k, "ac") - g(k, "mc")] for k in keys}

    worst = {"plain": min(balanced_accuracy(pooled[k], lab) for k in keys),
             "centre": min(balanced_accuracy(centred[k], lab) for k in keys)}
    v = {"plain": float(np.median([balanced_accuracy(pooled[k], lab) for k in keys])),
         "centre": float(np.median([balanced_accuracy(centred[k], lab) for k in keys])),
         "marg": balanced_accuracy(np.mean(list(pooled.values()), axis=0), lab),
         "centre+marg": balanced_accuracy(np.mean(list(centred.values()), axis=0), lab),
         "matched": float(np.median([balanced_accuracy(matched[k], lab) for k in keys]))}
    yes = float(np.median([(pooled[k] > 0).mean() for k in keys]))
    for k in agg:
        agg[k].append(v[k])
    for k, x in worst.items():
        floor.setdefault(k, []).append(x)
    print(f"{name[:23]:<24}" + "".join(f"{v[k] * 100:7.1f}% " for k in
          ("plain", "centre", "marg", "centre+marg", "matched")).replace("%  ", "%   ")
          + f"      {yes * 100:5.1f}%")

print("-" * 92)
print(f"{'mean':<24}" + "".join(f"{np.mean(agg[k]) * 100:7.1f}% " for k in
      ("plain", "centre", "marg", "centre+marg", "matched")).replace("%  ", "%   "))
print(f"\n  centring alone, costing nothing:          {np.mean(agg['plain']) * 100:.1f}% -> "
      f"{np.mean(agg['centre']) * 100:.1f}%  ({(np.mean(agg['centre']) - np.mean(agg['plain'])) * 100:+.1f}pp)")
print(f"  averaging alone, costing K readings:      {np.mean(agg['plain']) * 100:.1f}% -> "
      f"{np.mean(agg['marg']) * 100:.1f}%  ({(np.mean(agg['marg']) - np.mean(agg['plain'])) * 100:+.1f}pp)")
print("\n  the floor -- the single worst presentation, which deployment does not get to avoid:")
print(f"    plain {np.mean(floor['plain']) * 100:.1f}%   centred {np.mean(floor['centre']) * 100:.1f}%   "
      f"averaged over all 12 {np.mean(agg['marg']) * 100:.1f}%   centred and averaged "
      f"{np.mean(agg['centre+marg']) * 100:.1f}%")
print(f"  both:                                     {np.mean(agg['plain']) * 100:.1f}% -> "
      f"{np.mean(agg['centre+marg']) * 100:.1f}%  "
      f"({(np.mean(agg['centre+marg']) - np.mean(agg['plain'])) * 100:+.1f}pp)")
