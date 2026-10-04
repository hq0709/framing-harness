"""Aiming the budget beats spending it evenly -- now without the mask doing the aiming.

Spreading ten views over a chest gave AUROC 0.47 while ten views on the tumour gave 0.63, so where the
budget goes is the whole effect. That test aimed with the mask, which no deployed system has. The question
left is whether the model's own comparison score can aim it.

The accounting works out for free, because scoring IS scanning. Reading every candidate level once costs
the same whether the decision then averages all of them or keeps only the strongest few: both see the same
images and spend the same forward passes. So the comparison is not between a cheap method and an expensive
one, it is between two ways of turning the same readings into one decision --

    average   mean over all candidate levels        what a uniform sweep amounts to
    focus     mean over the top k by |score|        label-free, the agent's aggregation
    oracle    mean over the k levels with tumour    upper bound, costs a mask

-- and every k, including the oracle, is read off one scan of each range, so the curve is one run.

The unit of decision is a scan range: does this stretch of the volume contain a tumour? A positive range
covers the tumour; a negative range of the same extent in the same patient does not. Selecting the maximum
and then reporting it inflates both classes equally, since both offer the same number of candidates, so the
inflation cannot produce the separation by itself.
"""
import argparse, json
from pathlib import Path
from fh import MSD, RESULTS, load_any, read_yesno, render

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import stats


Q = "Is there an abnormality in the lung shown in these images? Answer Yes or No."


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def half(img, left):
    w, h = img.size
    c = img.crop((0, 0, w // 2, h) if left else (w // 2, 0, w, h))
    if not left:                      # flip so both readings put the mediastinum on the same side
        c = c.transpose(Image.FLIP_LEFT_RIGHT)
    return c.resize((448, 448), Image.BILINEAR)


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--cases", type=int, default=63)
ap.add_argument("--window", type=int, default=120, help="levels per scan range")
ap.add_argument("--cands", type=int, default=24, help="candidate levels read per range")
ap.add_argument("--margin", type=int, default=15, help="levels a negative range keeps from any tumour")
ap.add_argument("--out", default=str(RESULTS / "focus_vs_average.json"))
a = ap.parse_args()

model, proc = load_any(a.model)
ranges = []

for lab in sorted((MSD / "labelsTr").glob("lung_*.nii.gz"))[: a.cases]:
    mask = np.asanyarray(nib.load(str(lab)).dataobj)
    if mask.max() == 0:
        continue
    vol = np.asanyarray(nib.load(str(MSD / "imagesTr" / lab.name)).dataobj)
    Z = mask.shape[2]
    per_z = mask.reshape(-1, Z).sum(axis=0)
    on = np.nonzero(per_z)[0]
    W = min(a.window, Z)
    # positive range: centred on the tumour. negative: the same extent, as far from it as the volume allows
    lo_p = int(np.clip(int(on.mean()) - W // 2, 0, Z - W))
    cand_lo = [s for s in range(0, Z - W + 1)
               if all(not (s <= o < s + W) for o in on)
               and min(abs(s + W // 2 - o) for o in on) > a.margin]
    if not cand_lo:
        continue
    lo_n = max(cand_lo, key=lambda s: min(abs(s + W // 2 - o) for o in on))
    xs = np.nonzero(mask.sum(axis=(1, 2)))[0]
    gt_left = bool(xs.mean() < mask.shape[0] / 2)   # which half holds the tumour, for localisation only
    case = lab.name.split(".")[0]

    for kind, lo in (("lesion", lo_p), ("clean", lo_n)):
        zs = [int(z) for z in np.linspace(lo, lo + W - 1, a.cands).round()]
        d, has = [], []
        for z in zs:
            f = render(vol, z)
            d.append(read_yesno(model, proc, [half(f, True)], Q) - read_yesno(model, proc, [half(f, False)], Q))
            has.append(bool(per_z[z] > 0))
        ranges.append({"case": case, "kind": kind, "lo": lo, "zs": zs, "signed": d,
                       "has_tumour": has, "gt_left": gt_left})
    print(f"  {case}", flush=True)

les = [r for r in ranges if r["kind"] == "lesion"]
cln = [r for r in ranges if r["kind"] == "clean"]
ks = [k for k in (1, 2, 3, 5, 8, 12, 24) if k <= a.cands]
R = {"ranges": {"lesion": len(les), "clean": len(cln)}, "cases": len({r["case"] for r in ranges}),
     "window": a.window, "candidates": a.cands, "reads_per_range": 2 * a.cands, "focus": {}, "uniform": {}}

for k in ks:
    foc = lambda r: float(np.mean(sorted((abs(x) for x in r["signed"]), reverse=True)[:k]))
    # a uniform sweep of only k levels: the same decision rule on fewer reads, to show that extra reads
    # help the focused decision and do nothing for the even one
    uni = lambda r: float(np.mean([abs(r["signed"][i]) for i in
                                   np.linspace(0, a.cands - 1, k).round().astype(int)]))
    R["focus"][str(k)] = auroc([foc(r) for r in les], [foc(r) for r in cln])
    R["uniform"][str(k)] = auroc([uni(r) for r in les], [uni(r) for r in cln])
R["average_all"] = R["focus"][str(a.cands)]      # top-k with k = every candidate is the plain mean

def orc(r, k):
    v = [abs(x) for x, h in zip(r["signed"], r["has_tumour"]) if h] or [abs(x) for x in r["signed"]]
    return float(np.mean(sorted(v, reverse=True)[:k]))
R["oracle"] = {str(k): auroc([orc(r, k) for r in les], [orc(r, k) for r in cln]) for k in ks}

best = max(ks, key=lambda k: R["focus"][str(k)])
fb = [np.mean(sorted((abs(x) for x in r["signed"]), reverse=True)[:best]) for r in les]
cb = [np.mean(sorted((abs(x) for x in r["signed"]), reverse=True)[:best]) for r in cln]
R["best_k"] = {"k": best, "auroc": R["focus"][str(best)],
               "mannwhitney_p": float(stats.mannwhitneyu(fb, cb, alternative="greater").pvalue),
               "vs_average_auroc": R["average_all"]}
# does the winning level's sign name the right lung? chance is a half, and it needs no mask to produce
sel = [(r, int(np.argmax([abs(x) for x in r["signed"]]))) for r in les]
hit = sum((r["signed"][i] > 0) == r["gt_left"] for r, i in sel)
R["side_from_sign"] = {"hits": f"{hit}/{len(sel)}", "acc": hit / len(sel),
                       "binom_p": float(stats.binomtest(hit, len(sel), 0.5, alternative="greater").pvalue)}
# and did the level it picked actually carry tumour?
on_t = sum(r["has_tumour"][i] for r, i in sel)
base = float(np.mean([np.mean(r["has_tumour"]) for r in les]))
R["picked_level_on_tumour"] = {"hits": f"{on_t}/{len(sel)}", "acc": on_t / len(sel), "chance": base,
                               "binom_p": float(stats.binomtest(on_t, len(sel), base, alternative="greater").pvalue)}

out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(R, indent=1) + "\n")
Path(str(out).replace(".json", ".jsonl")).write_text("\n".join(json.dumps(r) for r in ranges) + "\n")
print("\n" + json.dumps(R, indent=1))
