"""Apparent size, not resolvability, is what the answer tracks -- so test whether magnifying fixes it.

Erasing the tumour and asking again gave a grounding that depends on nothing but how much of the frame the
tumour covers: AUROC 0.484 in the smallest quartile, which is exactly chance, rising to 0.655 in the
largest, with Spearman rho = +0.42 over three hundred and seventy-eight levels. The median tumour is one
thousandth of the slice. It is resolvable -- none falls below three pixels at the encoder's grid -- and it
is still invisible to the answer, because resolvable and salient are different things and we had measured
the wrong one.

If that reading is right, the fix is mechanical. Crop to the tumour and fill the frame with it and its
apparent size rises by two orders of magnitude, so grounding should rise with the magnification and keep
rising until the crop stops containing enough context to judge. If grounding is flat across scales then
apparent size was not the mechanism and the size correlation was standing in for something else -- large
tumours being denser, say, or sitting in easier places.

The control at every scale is the counterfactual crop: the same patient, level, window, framing and
magnification, with the tumour painted over by the mirrored patch from the other lung. Positive and negative
differ in the tumour and in nothing else, at each scale separately, so the comparison across scales is
between like and like. Beside each crop we read its mirrored counterpart at the same scale, which is the
module's own readout, to see whether the comparison keeps its advantage once the region is magnified.

Crops are centred with the mask here, which measures the mechanism and not yet a method; proposing the
region without the mask is the next question, and it only becomes worth asking if this comes out positive.
"""
import argparse, json
from pathlib import Path
from fh import MSD, RESULTS, load_any, read_yesno, render

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import stats


Q = "Is there an abnormality in the lung shown in these images? Answer Yes or No."
SIDE, GRID = 512, 448


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def box(cx, cy, s):
    """A square of side s centred on (cx, cy), slid back inside the slice rather than clipped, so every
    scale shows a window of exactly the same area and magnification is the only thing that changes."""
    x = int(np.clip(cx - s // 2, 0, SIDE - s))
    y = int(np.clip(cy - s // 2, 0, SIDE - s))
    return x, y, x + s, y + s


def cut(img, b, flip=False):
    c = img.crop(b)
    if flip:
        c = c.transpose(Image.FLIP_LEFT_RIGHT)
    return c.resize((GRID, GRID), Image.BICUBIC)


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
ap.add_argument("--levels", type=int, default=4)
ap.add_argument("--scales", default="512,256,128,64")
ap.add_argument("--pad", type=int, default=8)
ap.add_argument("--min-area", type=int, default=12)
ap.add_argument("--out", default=str(RESULTS / "magnify.json"))
a = ap.parse_args()

scales = [int(s) for s in a.scales.split(",")]
model, proc = load_any(a.model)
rows = []

for lab in sorted((MSD / "labelsTr").glob("lung_*.nii.gz"))[: a.cases]:
    mask = np.asanyarray(nib.load(str(lab)).dataobj)
    if mask.max() == 0:
        continue
    vol = np.asanyarray(nib.load(str(MSD / "imagesTr" / lab.name)).dataobj)
    area = mask.reshape(-1, mask.shape[2]).sum(axis=0)
    on = [int(z) for z in np.nonzero(area >= a.min_area)[0]]
    if not on:
        continue
    zs = [on[i] for i in np.linspace(0, len(on) - 1, min(a.levels, len(on))).round().astype(int)]
    case = lab.name.split(".")[0]

    for z in zs:
        real = render(vol, z, side=SIDE)
        m2d = np.rot90(mask[:, :, z])
        px = int(m2d.sum())
        if px < a.min_area:
            continue
        cf = erase(real, m2d, a.pad)
        if cf is None:
            continue
        rr, cc = np.nonzero(m2d)
        cy, cx = float(rr.mean()), float(cc.mean())
        r = {"case": case, "z": z, "px": px, "centre": [cx, cy], "scale": {}}
        for s in scales:
            b = box(cx, cy, s)
            mb = (SIDE - b[2], b[1], SIDE - b[0], b[3])      # the same window in the other lung
            r["scale"][str(s)] = {
                "abs_real": read_yesno(model, proc, [cut(real, b)], Q),
                "abs_cf": read_yesno(model, proc, [cut(cf, b)], Q),
                "mir_real": read_yesno(model, proc, [cut(real, mb, flip=True)], Q),
                "mir_cf": read_yesno(model, proc, [cut(cf, mb, flip=True)], Q),
                "apparent_frac": px / (s * s)}
        rows.append(r)
    print(f"  {case}", flush=True)

R = {"cases": len({r["case"] for r in rows}), "levels": len(rows), "scales": scales,
     "median_tumour_px": float(np.median([r["px"] for r in rows])), "by_scale": {}}
for s in scales:
    k = str(s)
    ar = [r["scale"][k]["abs_real"] for r in rows]
    ac = [r["scale"][k]["abs_cf"] for r in rows]
    cr = [r["scale"][k]["abs_real"] - r["scale"][k]["mir_real"] for r in rows]
    cc_ = [r["scale"][k]["abs_cf"] - r["scale"][k]["mir_cf"] for r in rows]
    R["by_scale"][k] = {
        "magnification": round(GRID / s, 2),
        "median_apparent_frac": float(np.median([r["scale"][k]["apparent_frac"] for r in rows])),
        "absolute_auroc": auroc(ar, ac), "comparison_auroc": auroc(cr, cc_),
        "absolute_shift": float(np.mean(np.array(ar) - np.array(ac))),
        "absolute_wilcoxon_p": float(stats.wilcoxon(ar, ac, alternative="greater").pvalue),
        "comparison_wilcoxon_p": float(stats.wilcoxon(cr, cc_, alternative="greater").pvalue),
        "says_yes_real": float(np.mean([x > 0 for x in ar])),
        "says_yes_cf": float(np.mean([x > 0 for x in ac]))}

a512, best = R["by_scale"][str(max(scales))], max(scales, key=lambda s: R["by_scale"][str(s)]["absolute_auroc"])
R["headline"] = {"full_slice_auroc": a512["absolute_auroc"], "best_scale": best,
                 "best_auroc": R["by_scale"][str(best)]["absolute_auroc"],
                 "gain": R["by_scale"][str(best)]["absolute_auroc"] - a512["absolute_auroc"]}
R["size_interaction"] = {}
px = np.array([r["px"] for r in rows], float)
small = px < np.median(px)
for s in scales:                       # the small tumours are the ones the full slice could not see at all
    k = str(s)
    for tag, m in (("small", small), ("large", ~small)):
        idx = np.nonzero(m)[0]
        R["size_interaction"].setdefault(tag, {})[k] = auroc(
            [rows[j]["scale"][k]["abs_real"] for j in idx], [rows[j]["scale"][k]["abs_cf"] for j in idx])

out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(R, indent=1) + "\n")
Path(str(out).replace(".json", ".jsonl")).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print("\n" + json.dumps(R, indent=1))
