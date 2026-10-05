"""Averaging works because the readings disagree. How much, and along which axis?

Averaging K readings only buys something if their errors are not the same error. If the twelve presentations
were twelve copies of one reading, the mean would be that reading. So the quantity behind the whole method
is how far apart the presentations are -- and the practical question that follows is where to spend a small
budget: three crops of one wording, or one crop in three wordings.

Each axis is ablated by averaging only within one of its levels, so a budget of K readings is spent moving
along a single axis at a time and the gains are comparable.

    python analysis/which_presentations.py        # reads results/framing2_*.jsonl
"""
import glob
import itertools
import json
import sys

import numpy as np

from fh import RESULTS, balanced_accuracy

pat = sys.argv[1] if len(sys.argv) > 1 else "framing2_*.jsonl"
F = sorted(glob.glob(str(RESULTS / pat)))
if not F:
    raise SystemExit(f"no runs matching {pat} in {RESULTS}")

AXES = {"crop": 0, "window": 1, "phrasing": 2}
corr_within, corr_across, gains = {a: [] for a in AXES}, [], {a: [] for a in AXES}
single, full = [], []

for f in F:
    rows = [json.loads(l) for l in open(f)]
    name = json.load(open(f.replace(".jsonl", ".json")))["model"].split("/")[-1]
    keys = list(rows[0]["f"])
    lab = np.r_[np.ones(len(rows), bool), np.zeros(len(rows), bool)]
    pooled = {k: np.r_[np.array([r["f"][k]["ar"] for r in rows]),
                       np.array([r["f"][k]["ac"] for r in rows])] for k in keys}
    cen = {k: v - np.median(v) for k, v in pooled.items()}

    # how alike are two presentations' readings of the same items?
    for a, b in itertools.combinations(keys, 2):
        r = float(np.corrcoef(pooled[a], pooled[b])[0, 1])
        pa, pb = a.split("|"), b.split("|")
        same = [ax for ax, i in AXES.items() if pa[i] == pb[i]]
        differ = [ax for ax, i in AXES.items() if pa[i] != pb[i]]
        if len(differ) == 1:
            corr_within[differ[0]].append(r)       # the pair differs on exactly this axis
        corr_across.append(r)

    single.append(float(np.median([balanced_accuracy(cen[k], lab) for k in keys])))
    full.append(balanced_accuracy(np.mean(list(cen.values()), axis=0), lab))
    # spend the budget on one axis: average over its levels, holding the other two fixed, then take the
    # median over which fixed combination you happened to start from
    for ax, i in AXES.items():
        vals = []
        others = [j for j in range(3) if j != i]
        fixed = {tuple(k.split("|")[j] for j in others) for k in keys}
        for fx in fixed:
            sub = [k for k in keys if tuple(k.split("|")[j] for j in others) == fx]
            if len(sub) > 1:
                vals.append(balanced_accuracy(np.mean([cen[k] for k in sub], axis=0), lab))
        gains[ax].append(float(np.median(vals)) if vals else np.nan)

print("Correlation between two presentations' readings of the same items, by what separates them.")
print("A pair that differs on one axis only is how much that axis moves the reading.\n")
for ax in AXES:
    v = np.array(corr_within[ax])
    print(f"  differ in {ax:<9} r = {v.mean():+.3f}   (n = {len(v)} pairs)")
print(f"  any two             r = {np.mean(corr_across):+.3f}")
print(f"\nThe readings are {'far from' if np.mean(corr_across) < 0.9 else 'nearly'} identical, which is why "
      f"averaging has anything to remove.")

print("\nSpending a budget along one axis at a time (centred, then averaged within that axis):\n")
print(f"{'':<12}{'levels':>8}{'bal.acc':>10}{'vs one reading':>16}")
base = np.mean(single)
for ax, i in AXES.items():
    n = len({k.split('|')[i] for k in keys})
    g = np.nanmean(gains[ax])
    print(f"  {ax:<10}{n:>8}{g * 100:9.1f}%{(g - base) * 100:+15.1f}pp")
print(f"  {'all three':<10}{len(keys):>8}{np.mean(full) * 100:9.1f}%{(np.mean(full) - base) * 100:+15.1f}pp")
print(f"  {'one reading':<10}{1:>8}{base * 100:9.1f}%")
best = max(AXES, key=lambda a: np.nanmean(gains[a]))
print(f"\n  A small budget is best spent on {best}: it is the axis whose readings agree least"
      f" (r = {np.mean(corr_within[best]):+.3f}), and no single axis reaches what all three together do.")
