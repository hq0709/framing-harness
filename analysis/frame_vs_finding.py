"""How much does the frame move the answer, next to how much the finding moves it?

Both are measured in the same unit and on the same images, so they divide. Within one framing, erasing the
tumour and asking again gives the finding's effect on the rate of answering Yes -- a difference between
paired readings of the same slice. Across framings, holding the slice and its tumour fixed, the spread of
that same rate gives the frame's effect. Their ratio says which of the two the answer is about.
"""
import glob, json, sys
import numpy as np
from scipy import stats
from fh import RESULTS

pat = sys.argv[1] if len(sys.argv) > 1 else "framing2_*.json"
F = sorted(glob.glob(str(RESULTS / pat)))
if not F:
    sys.exit(f"no results matching {pat}")

print(f"{'model':<24}{'px':>6} | {'FINDING':^30} | {'FRAME':^18} | {'ratio':>7}")
print(f"{'':<24}{'':>6} | {'yes|tumour':>11}{'yes|erased':>11}{'delta':>8} | {'min':>6}{'max':>6}{'range':>6} |")
print("-" * 98)
rat = []
for f in F:
    d = json.load(open(f)); bf = d["by_framing"]
    yr = np.array([v["says_yes_real"] for v in bf.values()])
    yc = np.array([v["says_yes_cf"] for v in bf.values()])
    finding = float(np.mean(yr - yc))                 # paired, within framing
    frame = float(yr.max() - yr.min())                # same slices, same tumours, framing alone
    r = frame / max(abs(finding), 1e-9)
    rat.append(r)
    print(f"{d['model'].split('/')[-1]:<24}{d.get('median_tumour_px', 0):6.0f} | {yr.mean()*100:10.1f}%"
          f"{yc.mean()*100:10.1f}%{finding*100:7.1f}pp | {yr.min()*100:5.1f}%{yr.max()*100:5.1f}%"
          f"{frame*100:5.1f}pp | {r:6.1f}x")

print(f"\nThe frame moves the answer {np.median(rat):.0f}x more than the finding does (median over "
      f"{len(F)} models; range {min(rat):.0f}-{max(rat):.0f}x).")

# and the same two effects on the readout that is supposed to be immune to one of them
print("\nSame two effects on the matched-difference readout (its decision is the side it names):")
print(f"{'model':<24}{'side|tumour':>12}{'range across framings':>24}")
for f in F:
    d = json.load(open(f)); bf = list(d["by_framing"].values())
    sa = np.array([v["side_acc"] for v in bf])
    n = d["levels"]
    p = stats.binomtest(int(round(sa.mean() * n)), n, 0.5, alternative="greater").pvalue
    print(f"{d['model'].split('/')[-1]:<24}{sa.mean()*100:11.1f}%  (p={p:7.1g}){sa.min()*100:12.1f}%-{sa.max()*100:.1f}%"
          f"  sd {sa.std()*100:.1f}pp")
