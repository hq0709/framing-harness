"""What the harness fixes, what it cannot, and a rule that says which in advance.

Two readings of the same cell look like contradictions until they are put side by side. On liver CT the
plain reading answers No to 99% of tumour windows -- 1% sensitivity against 100% specificity -- and the
comparison turns that into 53% and 52%. On lung CT the plain reading is already near balanced and the
comparison moves it very little. Opposite outcomes, one mechanism: the comparison moves the zero to where
the two inputs are equal, which is worth a great deal when the zero has drifted and nothing when it has not.

But a balanced coin is not a diagnosis. Where the model has no discriminability to begin with -- liver and
pancreas at AUROC 0.52 to 0.56 for a general-purpose model -- balancing the decision yields balanced
accuracy near 50% and nothing clinical. The gain the harness can deliver is bounded by the evidence that
was already there, so the quantity to predict it is a product:

    expected gain  ~  skew x (AUROC - 0.5)

skew being how lopsided the plain readout's two error rates are, and AUROC - 0.5 how much the readout knows.
Either factor at zero and there is nothing to collect. If this product predicts the measured gain across
organs and models, the harness is not a method that sometimes helps -- it is one that declares beforehand
where it will.
"""
import glob, json
import numpy as np
from scipy import stats
from fh import RESULTS

CELLS = []
for f in sorted(glob.glob(str(RESULTS / "hv2_*.json"))):
    d = json.load(open(f))
    for control, arms in d["readouts"].items():
        if "absolute" not in arms:
            continue
        base = arms["absolute"]["per_level"]
        for name, v in arms.items():
            if name == "absolute":
                continue
            CELLS.append({"organ": d["organ"], "model": d["model"].split("/")[-1],
                          "control": control, "arm": name, "n": base["n"],
                          "abs_sens": base["sens"], "abs_spec": base["spec"],
                          "abs_bal": base["bal_acc"], "abs_auroc": base["auroc"],
                          "skew": base["skew"], "evidence": base["auroc"] - 0.5,
                          "h_sens": v["per_level"]["sens"], "h_spec": v["per_level"]["spec"],
                          "h_bal": v["per_level"]["bal_acc"], "h_skew": v["per_level"]["skew"],
                          "gain": v["per_level"]["bal_acc"] - base["bal_acc"]})

if not CELLS:
    raise SystemExit("no hv2 results yet")

main = [c for c in CELLS if c["control"] == "real_window"]
print("Negatives are the same window at the nearest level the tumour does not reach -- real pixels.\n")
print(f"{'model':<16}{'organ':<10}{'arm':<9}{'n':>4} | {'PLAIN':^24} | {'HARNESS':^24} | {'gain':>7}")
print(f"{'':<16}{'':<10}{'':<9}{'':>4} | {'sens':>6}{'spec':>6}{'bal':>6}{'skew':>6} | {'sens':>6}{'spec':>6}{'bal':>6}{'skew':>6} |")
print("-" * 106)
for c in sorted(main, key=lambda c: (c["model"], -c["skew"])):
    print(f"{c['model'][:15]:<16}{c['organ']:<10}{c['arm']:<9}{c['n']:>4} | "
          f"{c['abs_sens']*100:5.0f}%{c['abs_spec']*100:5.0f}%{c['abs_bal']*100:5.1f}%{c['skew']*100:5.0f}% | "
          f"{c['h_sens']*100:5.0f}%{c['h_spec']*100:5.0f}%{c['h_bal']*100:5.1f}%{c['h_skew']*100:5.0f}% | "
          f"{c['gain']*100:+6.1f}pp")

print(f"\n{'what the harness does to the skew:':<40}"
      f"{np.mean([c['skew'] for c in main])*100:.0f}% -> {np.mean([c['h_skew'] for c in main])*100:.0f}%"
      f"   (every cell: {sum(c['h_skew'] < c['skew'] for c in main)}/{len(main)})")
print(f"{'what it does to discriminability:':<40}it is not supposed to, and does not materially")

if len(main) >= 5:
    sk = np.array([c["skew"] for c in main])
    ev = np.array([c["evidence"] for c in main])
    gn = np.array([c["gain"] for c in main])
    print("\nPredicting the gain:")
    for name, x in (("skew alone", sk), ("evidence alone", ev), ("skew x evidence", sk * ev)):
        r, p = stats.pearsonr(x, gn)
        rho, prho = stats.spearmanr(x, gn)
        print(f"  {name:<18} Pearson r = {r:+.3f} (p = {p:.3g})   Spearman rho = {rho:+.3f} (p = {prho:.3g})")
    sl, ic = np.polyfit(sk * ev, gn, 1)
    print(f"  fit: gain = {sl:+.2f} x skew x evidence {ic:+.4f}")

agree = {}
for c in CELLS:
    agree.setdefault((c["model"], c["organ"], c["arm"]), {})[c["control"]] = c["gain"]
both = [(k, v) for k, v in agree.items() if len(v) == 2]
if both:
    a = np.array([v["real_window"] for _, v in both])
    b = np.array([v["pasted_counterfactual"] for _, v in both])
    print(f"\nThe two controls, which fail in opposite ways, agree on the gain for "
          f"{int((np.sign(a) == np.sign(b)).sum())}/{len(both)} cells; "
          f"Pearson r = {stats.pearsonr(a, b)[0]:+.3f}")
