"""How many presentations the average needs, and what it does to the floor.

The framework's central claim measured on one axis: average the margin over K presentations of the same
evidence and the decision improves, while the evidence itself was never touched. The number that matters is
not the mean but the worst presentation, because a deployed system does not get to pick which presentation
it is handed.

    python analysis/marginalisation_curve.py                 # reads results/framing2_*.jsonl
"""
import glob
import json
import sys

import numpy as np

from fh import RESULTS, auroc, balanced_accuracy

pat = sys.argv[1] if len(sys.argv) > 1 else "framing2_*.jsonl"
F = sorted(glob.glob(str(RESULTS / pat)))
if not F:
    raise SystemExit(f"no runs matching {pat} in {RESULTS}")

Ks = [1, 2, 3, 4, 6, 8, 12]
rng = np.random.default_rng(0)
print("Balanced accuracy of the averaged margin, mean over 40 random subsets of presentations.\n")
print(f"{'model':<24}" + "".join(f"K={k:<6}" for k in Ks) + "  worst K=1   AUROC@max")
print("-" * 104)
curve = {k: [] for k in Ks}
acurve = {k: [] for k in Ks}
worst, full = [], []
for f in F:
    rows = [json.loads(l) for l in open(f)]
    meta = json.load(open(f.replace(".jsonl", ".json")))
    name = meta["model"].split("/")[-1]
    keys = list(rows[0]["f"])
    # positives are the real slices, negatives the same slice with the finding erased
    P = {k: np.array([r["f"][k]["ar"] for r in rows]) for k in keys}
    N = {k: np.array([r["f"][k]["ac"] for r in rows]) for k in keys}
    lab = np.r_[np.ones(len(rows), bool), np.zeros(len(rows), bool)]
    row = []
    for K in Ks:
        if K > len(keys):
            row.append(np.nan); continue
        b, a = [], []
        for _ in range(40):
            sub = list(rng.permutation(keys)[:K])
            m = np.r_[np.mean([P[k] for k in sub], axis=0), np.mean([N[k] for k in sub], axis=0)]
            b.append(balanced_accuracy(m, lab))
            a.append(auroc(m[lab], m[~lab]))
        row.append(np.mean(b)); curve[K].append(np.mean(b)); acurve[K].append(np.mean(a))
    w = min(balanced_accuracy(np.r_[P[k], N[k]], lab) for k in keys)
    m = np.r_[np.mean([P[k] for k in keys], axis=0), np.mean([N[k] for k in keys], axis=0)]
    worst.append(w); full.append(balanced_accuracy(m, lab))
    print(f"{name[:23]:<24}" + "".join(f"{v * 100:5.1f}% " for v in row)
          + f"   {w * 100:5.1f}%     {auroc(m[lab], m[~lab]):.3f}")

print("-" * 104)
print(f"{'mean':<24}" + "".join(f"{np.mean(curve[k]) * 100:5.1f}% " for k in Ks))
print(f"{'mean AUROC':<24}" + "".join(f"{np.mean(acurve[k]):5.3f} " for k in Ks))
g1, g4, g12 = np.mean(curve[1]), np.mean(curve[4]), np.mean(curve[max(Ks)])
print(f"\n  K=1 -> K=4 {(g4 - g1) * 100:+.1f}pp   K=4 -> K={max(Ks)} {(g12 - g4) * 100:+.1f}pp"
      f"   -- {(g4 - g1) / max(g12 - g1, 1e-9) * 100:.0f}% of the gain arrives by four presentations")
print(f"  worst single presentation {np.mean(worst) * 100:.1f}%  ->  averaged over all "
      f"{np.mean(full) * 100:.1f}%   ({(np.mean(full) - np.mean(worst)) * 100:+.1f} points on the floor)")
