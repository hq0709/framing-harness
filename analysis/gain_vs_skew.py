"""The harness buys back what the prior costs -- and nothing more.

On lung CT at a tight crop the plain reading already works: 84% sensitivity, 70% specificity, and reading
the window against a tumour-free one of the same organ makes it slightly worse. On pancreas and colon the
same model answers No to almost everything -- 10% and 16% sensitivity -- and the same comparison lifts it to
57% and 62%. Those look like opposite results until you notice what differs: not the organ, but how badly
the model's prior is placed for it.

A readout answering Yes to 84% of tumours and No to 70% of erasures has its zero roughly where the evidence
crosses over. A readout answering Yes to 10% has put its zero somewhere the evidence never reaches, and the
answer stops being about the image. Subtracting a matched reading moves the zero back to where the two
inputs are equal, which helps in proportion to how far out it had drifted and costs a little where it had
not drifted at all.

So the prediction is quantitative: plot the gain against the skew and they should line up. Skew is how
lopsided the plain readout's two error rates are; gain is what the comparison does to balanced accuracy.
If the line is flat, the harness is a coin flip dressed as a method. If it slopes, the harness is doing one
thing, doing it for a reason, and declaring in advance where it will and will not help.
"""
import glob, json, re
from collections import defaultdict
from fh import RESULTS

import numpy as np
from scipy import stats


def rates(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float((pos > 0).mean()), float((neg <= 0).mean())


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


cells = []
for f in sorted(glob.glob(str(RESULTS / "harness_Task*.json"))):
    meta = json.load(open(f))
    rows = [json.loads(l) for l in open(f.replace(".json", ".jsonl"))]
    if not rows:
        continue
    by = defaultdict(list)
    for r in rows:
        by[r["case"]].append(r)
    cases = list(by.values())

    def agg(fn):
        return (np.array([np.mean([fn(r)[0] for r in v]) for v in cases]),
                np.array([np.mean([fn(r)[1] for r in v]) for v in cases]))

    ap, an = agg(lambda r: (r["abs_real"], r["abs_cf"]))
    hp, hn = agg(lambda r: (r["abs_real"] - r["axial_ref"], r["abs_cf"] - r["axial_ref"]))
    asen, aspe = rates(ap, an)
    hsen, hspe = rates(hp, hn)
    cells.append({
        "organ": meta["organ"], "model": meta["model"].split("/")[-1], "n": len(cases),
        "abs_sens": asen, "abs_spec": aspe, "abs_bal": (asen + aspe) / 2,
        "h_sens": hsen, "h_spec": hspe, "h_bal": (hsen + hspe) / 2,
        "skew": abs(asen - aspe), "gain": (hsen + hspe) / 2 - (asen + aspe) / 2,
        "abs_auroc": auroc(ap, an), "h_auroc": auroc(hp, hn)})

if not cells:
    raise SystemExit("no harness results yet")

print(f"{'model':<24}{'organ':<10}{'n':>4} | {'PLAIN READING':^26} | {'HARNESS (axial)':^26} | {'skew':>6}{'gain':>7}")
print(f"{'':<24}{'':<10}{'':>4} | {'sens':>7}{'spec':>7}{'bal':>6}{'AUROC':>6} | {'sens':>7}{'spec':>7}{'bal':>6}{'AUROC':>6} |")
print("-" * 118)
for c in sorted(cells, key=lambda c: (c["model"], -c["skew"])):
    print(f"{c['model']:<24}{c['organ']:<10}{c['n']:>4} | {c['abs_sens']*100:6.1f}%{c['abs_spec']*100:6.1f}%"
          f"{c['abs_bal']*100:5.1f}%{c['abs_auroc']:6.3f} | {c['h_sens']*100:6.1f}%{c['h_spec']*100:6.1f}%"
          f"{c['h_bal']*100:5.1f}%{c['h_auroc']:6.3f} | {c['skew']*100:5.0f}%{c['gain']*100:+6.1f}pp")

sk = np.array([c["skew"] for c in cells]); gn = np.array([c["gain"] for c in cells])
print(f"\n{len(cells)} cells.")
if len(cells) >= 4:
    r, p = stats.pearsonr(sk, gn)
    rho, prho = stats.spearmanr(sk, gn)
    sl, ic = np.polyfit(sk, gn, 1)
    print(f"  gain against skew:  Pearson r = {r:+.3f} (p = {p:.3g})   Spearman rho = {rho:+.3f} (p = {prho:.3g})")
    print(f"  fit: gain = {sl:+.2f} x skew {ic:+.3f}   -- the harness breaks even at skew = "
          f"{-ic/sl*100:.0f}% and pays above it")
    big = [c for c in cells if c["skew"] > 0.5]
    small = [c for c in cells if c["skew"] <= 0.5]
    for tag, g in (("skew > 50%", big), ("skew <= 50%", small)):
        if g:
            print(f"  {tag:<12} n={len(g):2d}  mean gain {np.mean([c['gain'] for c in g])*100:+5.1f}pp  "
                  f"(plain {np.mean([c['abs_bal'] for c in g])*100:.1f}% -> harness "
                  f"{np.mean([c['h_bal'] for c in g])*100:.1f}%)")
