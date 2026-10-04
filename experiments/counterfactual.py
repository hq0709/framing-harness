"""Remove the tumour and keep everything else, then ask again.

Two things need settling at once and one control settles both.

The plain reading of a whole slab separated tumour levels from clean levels at AUROC 0.84, which looks like
detection until you notice where the clean levels came from: twenty or more levels away, so a different
height in the chest. That is the confound that once inverted an earlier result of ours, and 0.84 is close
enough to the 0.86 it produced to be the same thing wearing a different hat.

Meanwhile the comparison readout, scored as a maximum over four quadrants, came out at chance -- because a
maximum over four noisy differences adds about 0.85 to both classes while the effect being chased is 0.25.
The proposition said aggregate and the module took an extremum. A mean over matched comparisons is the
estimator the proposition actually implies, and its noise falls as the comparisons accumulate.

The control for both: in the rendered slice, replace the tumour's patch with the mirrored patch from the
same place in the other lung. Same patient, same level, same window, same everything, no tumour. A negative
constructed this way cannot differ in height, in habitus or in scanner, so a readout that separates real
from counterfactual is reading the tumour and a readout that does not is reading something else. Labels
from fh import MSD, RESULTS, load_any, read_yesno, render
build the control; neither readout sees them.
"""
import argparse, json
from pathlib import Path

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
    if not left:                        # flip so both readings put the mediastinum on the same side
        c = c.transpose(Image.FLIP_LEFT_RIGHT)
    return c.resize((448, 448), Image.BILINEAR)


def erase(img, m2d, pad):
    """Paste the mirrored patch from the other lung over the tumour's bounding box.

    Local rather than whole-half, so the mediastinum and the heart stay where they are and the result is
    still a plausible chest -- a mirrored half would be perfectly symmetric, which no chest is, and the
    model could then separate real from counterfactual on symmetry alone without ever seeing the tumour."""
    rr, cc = np.nonzero(m2d)
    r0, r1 = max(int(rr.min()) - pad, 0), min(int(rr.max()) + 1 + pad, SIDE)
    c0, c1 = max(int(cc.min()) - pad, 0), min(int(cc.max()) + 1 + pad, SIDE)
    w = c1 - c0
    s0 = SIDE - c1                      # the mirrored column range, same rows
    a = np.array(img)
    patch = a[r0:r1, s0:s0 + w][:, ::-1]
    if patch.shape[:2] != (r1 - r0, w):
        return None
    a[r0:r1, c0:c1] = patch
    return Image.fromarray(a), (r0, r1, c0, c1)


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--cases", type=int, default=63)
ap.add_argument("--levels", type=int, default=6, help="tumour levels read per case")
ap.add_argument("--pad", type=int, default=8, help="pixels of margin around the tumour box")
ap.add_argument("--min-area", type=int, default=12, help="tumour pixels a level must have to be used")
ap.add_argument("--sanity", type=int, default=2, help="cases to dump real/counterfactual PNGs for")
ap.add_argument("--out", default=str(RESULTS / "counterfactual.json"))
a = ap.parse_args()

model, proc = load_any(a.model)
rows, dumped = [], 0

for lab in sorted((MSD / "labelsTr").glob("lung_*.nii.gz"))[: a.cases]:
    mask = np.asanyarray(nib.load(str(lab)).dataobj)
    if mask.max() == 0:
        continue
    vol = np.asanyarray(nib.load(str(MSD / "imagesTr" / lab.name)).dataobj)
    Z = mask.shape[2]
    area = mask.reshape(-1, Z).sum(axis=0)
    on = [int(z) for z in np.nonzero(area >= a.min_area)[0]]
    if not on:
        continue
    zs = [on[i] for i in np.linspace(0, len(on) - 1, min(a.levels, len(on))).round().astype(int)]
    case = lab.name.split(".")[0]

    for z in zs:
        real = render(vol, z, side=SIDE)
        m2d = np.rot90(mask[:, :, z])            # the same rotation render applies
        if m2d.sum() < a.min_area:
            continue
        made = erase(real, m2d, a.pad)
        if made is None:
            continue
        cf, box = made
        left = bool(np.nonzero(m2d)[1].mean() < SIDE / 2)   # which half holds it, for scoring only

        if dumped < a.sanity:
            o = RESULTS / f"sanity_cf_{case}_z{z}.png"
            strip = Image.new("RGB", (SIDE * 3, SIDE))
            d = Image.fromarray((np.abs(np.array(real).astype(int) - np.array(cf).astype(int))
                                 * 4).clip(0, 255).astype(np.uint8))
            for i, im in enumerate((real, cf, d)):
                strip.paste(im, (i * SIDE, 0))
            o.parent.mkdir(parents=True, exist_ok=True); strip.save(o)
            dumped += 1

        r = {"case": case, "z": z, "tumour_left": left, "box": box, "px": int(m2d.sum())}
        for tag, im in (("real", real), ("cf", cf)):
            r[f"abs_{tag}"] = read_yesno(model, proc, [im], Q)
            dl = read_yesno(model, proc, [half(im, True)], Q) - read_yesno(model, proc, [half(im, False)], Q)
            r[f"d_lr_{tag}"] = dl                                  # label-free: left minus right
            r[f"d_or_{tag}"] = dl if left else -dl                 # oriented, for the side check only
        rows.append(r)
    print(f"  {case} ({len(zs)} levels)", flush=True)

by_case = {}
for r in rows:
    by_case.setdefault(r["case"], []).append(r)
cases = [v for v in by_case.values() if v]
R = {"cases": len(cases), "levels_read": len(rows), "pad": a.pad,
     "mean_tumour_px": float(np.mean([r["px"] for r in rows]))}

# does either readout notice the tumour at all, level by level?
R["per_level_grounding_auroc"] = {
    "absolute": auroc([r["abs_real"] for r in rows], [r["abs_cf"] for r in rows]),
    "comparison": auroc([abs(r["d_lr_real"]) for r in rows], [abs(r["d_lr_cf"]) for r in rows])}
R["says_yes_rate"] = {"absolute_real": float(np.mean([r["abs_real"] > 0 for r in rows])),
                      "absolute_cf": float(np.mean([r["abs_cf"] > 0 for r in rows]))}
R["mean_shift_when_erased"] = {
    "absolute": float(np.mean([r["abs_real"] - r["abs_cf"] for r in rows])),
    "comparison_oriented": float(np.mean([r["d_or_real"] - r["d_or_cf"] for r in rows]))}

# and does averaging matched comparisons buy what taking a maximum over them threw away?
R["accumulation"] = {}
for k in (1, 2, 3, 4, 6):
    sub = [v[:k] for v in cases if len(v) >= k]
    if len(sub) < 10:
        continue
    mean = lambda v, key: float(np.mean([r[key] for r in v]))
    pos_c = [abs(mean(v, "d_lr_real")) for v in sub]
    neg_c = [abs(mean(v, "d_lr_cf")) for v in sub]
    pos_a = [mean(v, "abs_real") for v in sub]
    neg_a = [mean(v, "abs_cf") for v in sub]
    side = sum(mean(v, "d_or_real") > 0 for v in sub)
    R["accumulation"][str(k)] = {
        "n_cases": len(sub),
        "comparison_auroc": auroc(pos_c, neg_c), "absolute_auroc": auroc(pos_a, neg_a),
        "comparison_wilcoxon_p": float(stats.wilcoxon(pos_c, neg_c, alternative="greater").pvalue),
        "side_acc": side / len(sub),
        "side_binom_p": float(stats.binomtest(side, len(sub), 0.5, alternative="greater").pvalue)}

out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(R, indent=1) + "\n")
Path(str(out).replace(".json", ".jsonl")).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print("\n" + json.dumps(R, indent=1))
