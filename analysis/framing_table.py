"""One table from the framing sweep: how far each model's answer travels when the evidence does not.

Twelve framings of the same tumour in the same slice -- three crops, two windows, two phrasings -- scored
against the counterfactual in which that tumour is painted out. For each model: the range its rate of
answering Yes covers, what its balanced accuracy and discriminability do meanwhile, and whether the matched
difference stays put while the answer moves. Also which of the three axes moves the answer most, because a
reader will want to know whether this is a crop artefact.
"""
import glob, json, sys
import numpy as np
from fh import RESULTS

pat = sys.argv[1] if len(sys.argv) > 1 else "framing2_*.json"
F = sorted(glob.glob(str(RESULTS / pat)))
if not F:
    sys.exit(f"no results matching {pat}")
print(f"{'model':<24}{'n':>5} | {'ABSOLUTE: says Yes':^26} | {'bal.acc':^13} {'AUROC':^13} | {'MATCHED: side':^20}")
print(f"{'':<24}{'':>5} | {'min':>7}{'max':>8}{'swing':>7}{'sd':>5} | {'min':>6}{'max':>7} {'min':>6}{'max':>7} | {'min':>6}{'max':>7}{'sd':>6}")
print("-" * 118)
rows = []
for f in F:
    d = json.load(open(f))
    s, bf = d["swing"], d["by_framing"]
    ba = [v["balanced_acc"] for v in bf.values()]
    au = [v["absolute_auroc"] for v in bf.values()]
    y, sd = s["absolute_says_yes"], s["matched_side_acc"]
    name = d["model"].split("/")[-1]
    print(f"{name:<24}{d['levels']:>5} | {y['min']*100:6.1f}%{y['max']*100:7.1f}%{y['ratio']:6.1f}x"
          f"{y['std']*100:5.1f} | {min(ba)*100:5.1f}%{max(ba)*100:6.1f}% {min(au):6.3f}{max(au):7.3f}"
          f" | {sd['min']*100:5.1f}%{sd['max']*100:6.1f}%{sd['std']*100:5.1f}")
    rows.append((name, d, y, sd, ba, au))

print("\nWhich axis moves the answer (mean rate of answering Yes within each level of the axis):")
for name, d, *_ in rows:
    parts = []
    for axis, g in d["axis_effect"].items():
        v = list(g.values())
        parts.append(f"{axis} {min(v)*100:.0f}-{max(v)*100:.0f}%")
    print(f"  {name:<24} " + "   ".join(parts))

print("\nAcross models:")
allsw = [r[2]["std"] for r in rows]
allsd = [r[3]["std"] for r in rows]
allba = [b for r in rows for b in r[4]]
allau = [a for r in rows for a in r[5]]
print(f"  rate of answering Yes moves by {np.mean(allsw)*100:.1f} pp (sd across framings), every model")
print(f"  the matched side call moves by  {np.mean(allsd)*100:.1f} pp")
print(f"  balanced accuracy stays in      {min(allba)*100:.1f}-{max(allba)*100:.1f}%")
print(f"  discriminability stays in       {min(allau):.3f}-{max(allau):.3f}")
sig = [(r[0], sum(v["side_p"] < 0.05 for v in r[1]["by_framing"].values()), len(r[1]["by_framing"]))
       for r in rows]
print("\nFramings where the side call beats chance at p<0.05: " +
      ", ".join(f"{n} {k}/{t}" for n, k, t in sig))


# ---- the internal control -------------------------------------------------------------------------
# Of the three axes, only the window changes what is in the picture: a lung nodule in a soft-tissue window
# is genuinely not visible. Crop and phrasing leave the evidence untouched. So the window should move
# discriminability and the other two should not -- and if that is what happens, the flatness measured along
# crop and phrasing is the instrument working, not the instrument being blind.
import collections
print("\n" + "=" * 118)
print("Does the measurement notice when the evidence really is removed?")
print(f"\n{'model':<24} {'axis':<10} {'level':<14} {'says Yes':>10} {'bal.acc':>9} {'AUROC':>8} {'side':>8}")
print("-" * 118)
for f in F:
    d = json.load(open(f)); name = d["model"].split("/")[-1]
    g = collections.defaultdict(list)
    for k, v in d["by_framing"].items():
        s, win, ph = k.split("|")
        for axis, lev in (("crop", s), ("window", win), ("phrasing", ph)):
            g[(axis, lev)].append(v)
    first = True
    for axis in ("window", "crop", "phrasing"):
        for lev in sorted({k[1] for k in g if k[0] == axis}, key=lambda x: (-len(x), x)):
            v = g[(axis, lev)]
            print(f"{name if first else '':<24} {axis if lev == sorted({k[1] for k in g if k[0]==axis}, key=lambda x:(-len(x),x))[0] else '':<10} {lev:<14}"
                  f" {np.mean([x['says_yes_real'] for x in v])*100:9.1f}%"
                  f" {np.mean([x['balanced_acc'] for x in v])*100:8.1f}%"
                  f" {np.mean([x['absolute_auroc'] for x in v]):8.3f}"
                  f" {np.mean([x['side_acc'] for x in v])*100:7.1f}%")
            first = False
    print()

print("Spread each axis produces, averaged over models (how much the level of that axis moves each number):")
sp = collections.defaultdict(lambda: collections.defaultdict(list))
for f in F:
    d = json.load(open(f))
    g = collections.defaultdict(list)
    for k, v in d["by_framing"].items():
        s, win, ph = k.split("|")
        for axis, lev in (("crop", s), ("window", win), ("phrasing", ph)):
            g[(axis, lev)].append(v)
    for axis in ("crop", "window", "phrasing"):
        levs = sorted({k[1] for k in g if k[0] == axis})
        for key, lab in (("says_yes_real", "says Yes"), ("absolute_auroc", "AUROC"), ("side_acc", "side")):
            m = [np.mean([x[key] for x in g[(axis, l)]]) for l in levs]
            sp[axis][lab].append(max(m) - min(m))
print(f"\n{'axis':<12}{'changes the picture?':<22}{'says Yes':>12}{'AUROC':>10}{'side':>10}")
for axis, changes in (("window", "yes"), ("crop", "no"), ("phrasing", "no")):
    v = sp[axis]
    print(f"{axis:<12}{changes:<22}{np.mean(v['says Yes'])*100:11.1f}pp{np.mean(v['AUROC']):10.3f}"
          f"{np.mean(v['side'])*100:9.1f}pp")
