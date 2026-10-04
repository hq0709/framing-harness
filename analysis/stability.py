"""What each readout does when the presentation is chosen badly.

Twelve presentations of the same evidence, scored two ways: the plain reading, and the plain reading minus
a matched counterpart read under the same presentation. The comparison gives up the best case and removes
the worst, which is the trade worth making when the presentation is not ours to choose.

    python analysis/stability.py                             # reads results/framing2_*.jsonl
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

print(f"{'model':<24}{'readout':<10}{'worst':>8}{'median':>8}{'best':>7}{'range':>8}{'sd':>7}")
print("-" * 74)
agg = {}
for f in F:
    rows = [json.loads(l) for l in open(f)]
    name = json.load(open(f.replace(".jsonl", ".json")))["model"].split("/")[-1]
    keys = list(rows[0]["f"])
    lab = np.r_[np.ones(len(rows), bool), np.zeros(len(rows), bool)]
    g = lambda k, t: np.array([r["f"][k][t] for r in rows])
    out = {"plain": [], "matched": []}
    for k in keys:
        out["plain"].append(balanced_accuracy(np.r_[g(k, "ar"), g(k, "ac")], lab))
        out["matched"].append(balanced_accuracy(
            np.r_[g(k, "ar") - g(k, "mr"), g(k, "ac") - g(k, "mc")], lab))
    first = True
    for ro, v in out.items():
        v = np.array(v)
        print(f"{name[:23] if first else '':<24}{ro:<10}{v.min() * 100:7.1f}%{np.median(v) * 100:7.1f}%"
              f"{v.max() * 100:6.1f}%{(v.max() - v.min()) * 100:7.1f}pp{v.std() * 100:6.1f}pp")
        agg.setdefault(ro, []).append(v); first = False
    print()

print("Across the models:")
for ro, vs in agg.items():
    a = np.concatenate(vs)
    print(f"  {ro:<8} worst {a.min() * 100:5.1f}%   median {np.median(a) * 100:5.1f}%   best {a.max() * 100:5.1f}%"
          f"   mean spread within a model {np.mean([v.max() - v.min() for v in vs]) * 100:5.1f}pp"
          f"   below 55% in {int((a < 0.55).sum())}/{len(a)} presentations")
