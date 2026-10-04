"""Where the budget goes, tested against the only control that works.

The first attempt at this put the negative range as far from the tumour as the volume allowed, which in a
three-hundred-level chest CT means the neck or the upper abdomen. Liver against stomach is wildly asymmetric
and mid-chest is nearly symmetric, so the left-right difference came out larger on the clean ranges than on
the tumour ranges and the measurement inverted -- AUROC 0.012 for the oracle, a near-perfect separator
pointing the wrong way. It was reading height. Every control we have built out of other levels has read
height; only erasing the tumour from the slice itself has not.

So the negative range here is the same range with the tumour painted over, level by level, with the mirrored
patch from the other lung. Positive and negative share every level that never carried tumour -- the same
images, the same readings -- and differ only in the few that did.

That is what makes this the test of aiming rather than of budget. A hundred-and-twenty-level window sampled
at twenty-four places catches about four tumour levels, so averaging all twenty-four dilutes the only
evidence there is by roughly six to one, while keeping the strongest four does not. If focusing the same
twenty-four readings onto the strongest few separates real from erased and averaging them does not, then
deciding where to look is doing the work, and it is doing it without a mask.
"""
import argparse, json
from pathlib import Path
from fh import MSD, RESULTS, load_any, read_yesno, render

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import stats


Q = "Is there an abnormality in the lung shown in these images? Answer Yes or No."
SIDE = 512


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def half(img, left):
    w, h = img.size
    c = img.crop((0, 0, w // 2, h) if left else (w // 2, 0, w, h))
    if not left:
        c = c.transpose(Image.FLIP_LEFT_RIGHT)
    return c.resize((448, 448), Image.BILINEAR)


def erase(img, m2d, pad):
    rr, cc = np.nonzero(m2d)
    r0, r1 = max(int(rr.min()) - pad, 0), min(int(rr.max()) + 1 + pad, SIDE)
    c0, c1 = max(int(cc.min()) - pad, 0), min(int(cc.max()) + 1 + pad, SIDE)
    w = c1 - c0
    a = np.array(img)
    patch = a[r0:r1, SIDE - c1:SIDE - c1 + w][:, ::-1]
    if patch.shape[:2] != (r1 - r0, w):
        return None
    a[r0:r1, c0:c1] = patch
    return Image.fromarray(a)


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--cases", type=int, default=63)
ap.add_argument("--window", type=int, default=120)
ap.add_argument("--cands", type=int, default=24)
ap.add_argument("--pad", type=int, default=8)
ap.add_argument("--min-area", type=int, default=12)
ap.add_argument("--out", default=str(RESULTS / "focus_counterfactual.json"))
a = ap.parse_args()

model, proc = load_any(a.model)
rows = []

for lab in sorted((MSD / "labelsTr").glob("lung_*.nii.gz"))[: a.cases]:
    mask = np.asanyarray(nib.load(str(lab)).dataobj)
    if mask.max() == 0:
        continue
    vol = np.asanyarray(nib.load(str(MSD / "imagesTr" / lab.name)).dataobj)
    Z = mask.shape[2]
    area = mask.reshape(-1, Z).sum(axis=0)
    on = np.nonzero(area)[0]
    W = min(a.window, Z)
    lo = int(np.clip(int(on.mean()) - W // 2, 0, Z - W))
    zs = [int(z) for z in np.linspace(lo, lo + W - 1, a.cands).round()]
    case = lab.name.split(".")[0]

    d_real, d_cf, a_real, a_cf, changed = [], [], [], [], []
    for z in zs:
        real = render(vol, z, side=SIDE)
        m2d = np.rot90(mask[:, :, z])
        cf = erase(real, m2d, a.pad) if m2d.sum() >= a.min_area else None
        dr = read_yesno(model, proc, [half(real, True)], Q) - read_yesno(model, proc, [half(real, False)], Q)
        ar = read_yesno(model, proc, [real], Q)
        d_real.append(dr); a_real.append(ar)
        if cf is None:                       # nothing to erase: the two ranges share this reading exactly
            d_cf.append(dr); a_cf.append(ar); changed.append(False)
        else:
            d_cf.append(read_yesno(model, proc, [half(cf, True)], Q) - read_yesno(model, proc, [half(cf, False)], Q))
            a_cf.append(read_yesno(model, proc, [cf], Q)); changed.append(True)
    rows.append({"case": case, "lo": lo, "zs": zs, "changed": changed,
                 "d_real": d_real, "d_cf": d_cf, "abs_real": a_real, "abs_cf": a_cf})
    print(f"  {case}  {sum(changed)}/{len(zs)} levels carried tumour", flush=True)

R = {"cases": len(rows), "window": a.window, "candidates": a.cands,
     "tumour_levels_per_range": float(np.mean([sum(r["changed"]) for r in rows])),
     "reads_per_range_pair": float(np.mean([2 * (2 * a.cands) + 2 * sum(r["changed"]) for r in rows]))}


def topk(v, k):
    return float(np.mean(sorted((abs(x) for x in v), reverse=True)[:k]))


ks = [k for k in (1, 2, 3, 4, 6, 8, 12, 24) if k <= a.cands]
for name, key_r, key_c, stat in (("comparison", "d_real", "d_cf", topk),
                                 ("absolute", "abs_real", "abs_cf", topk)):
    R[name] = {}
    for k in ks:
        pos = [stat(r[key_r], k) for r in rows]
        neg = [stat(r[key_c], k) for r in rows]
        R[name][str(k)] = {"auroc": auroc(pos, neg),
                           "wins": f"{sum(p > n for p, n in zip(pos, neg))}/{len(pos)}",
                           "wilcoxon_p": float(stats.wilcoxon(pos, neg, alternative="greater").pvalue)}
    # the mask's own choice of levels, as the ceiling that aiming could reach
    pos = [float(np.mean([abs(x) for x, c in zip(r[key_r], r["changed"]) if c] or [0.0])) for r in rows]
    neg = [float(np.mean([abs(x) for x, c in zip(r[key_c], r["changed"]) if c] or [0.0])) for r in rows]
    R[name]["oracle_tumour_levels"] = {
        "auroc": auroc(pos, neg), "wins": f"{sum(p > n for p, n in zip(pos, neg))}/{len(pos)}",
        "wilcoxon_p": float(stats.wilcoxon(pos, neg, alternative="greater").pvalue)}
    R[name]["average_all_vs_best_focus"] = {
        "average_all": R[name][str(a.cands)]["auroc"],
        "best_k": max(ks, key=lambda k: R[name][str(k)]["auroc"]),
        "best_auroc": max(R[name][str(k)]["auroc"] for k in ks)}

# does the label-free score put the budget on the levels that changed?
hit = sum(r["changed"][int(np.argmax([abs(x) for x in r["d_real"]]))] for r in rows)
base = float(np.mean([np.mean(r["changed"]) for r in rows]))
R["picked_level_carried_tumour"] = {
    "hits": f"{hit}/{len(rows)}", "acc": hit / len(rows), "chance": base,
    "binom_p": float(stats.binomtest(hit, len(rows), base, alternative="greater").pvalue)}

out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(R, indent=1) + "\n")
Path(str(out).replace(".json", ".jsonl")).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print("\n" + json.dumps(R, indent=1))
