"""The decision follows the frame. Does it do so for every model, and along every axis of framing?

One model, one axis: holding the patient, the level and the tumour fixed and changing only the crop scale
moved the rate of answering Yes from 10.3% to 93.3%, a nine-fold swing, while the ability to tell that slice
from the same slice with its tumour painted out stayed at AUROC 0.58 throughout. The evidence never moved.
The answer moved all the way.

If that is a property of these models rather than of one crop ladder, it should appear along any axis that
respecifies the question without changing what is in the image -- the window the radiologist would set, the
words the question is put in, how much is cropped -- and it should appear in every family. Twelve framings
of the same evidence, read by each model in turn, give both: how far the answer travels when nothing it is
about has changed, and whether the matched difference stays where it is while the answer travels.

Every framing is scored against the counterfactual: the same slice with the tumour replaced by the mirrored
patch from the other lung. It is the only negative we have that cannot differ in height, habitus or scanner,
and the three results we had to discard today all differed in height.
"""
import argparse, itertools, json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import stats
from fh import MSD, RESULTS, load_any, read_yesno, render


SIDE, GRID = 512, 448
SCALES = [256, 128, 64]
WINDOWS = {"lung": (-600.0, 1500.0), "soft": (50.0, 400.0)}
PHRASINGS = {
    "abnormality": "Is there an abnormality in the lung shown in these images? Answer Yes or No.",
    "nodule": "Does this image show a lung nodule or mass? Answer Yes or No."}


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])) if len(pos) and len(neg) else float("nan")


def box(cx, cy, s):
    x = int(np.clip(cx - s // 2, 0, SIDE - s))
    y = int(np.clip(cy - s // 2, 0, SIDE - s))
    return x, y, x + s, y + s


def cut(img, b, flip=False):
    c = img.crop(b)
    return (c.transpose(Image.FLIP_LEFT_RIGHT) if flip else c).resize((GRID, GRID), Image.BICUBIC)


def erase(img, m2d, pad=8):
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
ap.add_argument("--pick", default="largest", choices=["largest", "spread"],
                help="which tumour levels to read. 'spread' takes them evenly over the tumour's extent, "
                     "which puts the first and last -- its smallest cross-sections -- in every sample: the "
                     "first sweep did that and drew levels 4.4x smaller by median area than the runs it was "
                     "compared against, two thirds of them inside the quartile already measured as "
                     "ungrounded. 'largest' takes the biggest cross-sections instead, so the model is read "
                     "at its best case and a chance result cannot be blamed on the levels chosen.")
ap.add_argument("--min-area", type=int, default=12)
ap.add_argument("--out", default="")
a = ap.parse_args()

model, proc = load_any(a.model)
frames = [(s, w, p) for s, w, p in itertools.product(SCALES, WINDOWS, PHRASINGS)]
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
    pick = (sorted(on, key=lambda z: -area[z])[: a.levels] if a.pick == "largest"
            else [on[i] for i in np.linspace(0, len(on) - 1, min(a.levels, len(on))).round().astype(int)])
    for z in sorted(pick):
        m2d = np.rot90(mask[:, :, z])
        if m2d.sum() < a.min_area:
            continue
        rr, cc = np.nonzero(m2d)
        cy, cx = float(rr.mean()), float(cc.mean())
        r = {"case": lab.name.split(".")[0], "z": z, "px": int(m2d.sum()), "f": {}}
        for win, (wl, ww) in WINDOWS.items():
            real = render(vol, z, side=SIDE, wl=wl, ww=ww)
            cf = erase(real, m2d)
            if cf is None:
                continue
            for s in SCALES:
                b = box(cx, cy, s)
                mb = (SIDE - b[2], b[1], SIDE - b[0], b[3])
                ims = {"ar": cut(real, b), "ac": cut(cf, b),
                       "mr": cut(real, mb, flip=True), "mc": cut(cf, mb, flip=True)}
                for ph, q in PHRASINGS.items():
                    r["f"][f"{s}|{win}|{ph}"] = {k: read_yesno(model, proc, [im], q)
                                                 for k, im in ims.items()}
        if r["f"]:
            rows.append(r)
    print(f"  {lab.name.split('.')[0]}", flush=True)

R = {"model": a.model, "pick": a.pick, "median_tumour_px": float(np.median([r["px"] for r in rows])), "levels": len(rows), "cases": len({r["case"] for r in rows}),
     "framings": len(frames), "by_framing": {}}
yes, side = [], []
for s, win, ph in frames:
    k = f"{s}|{win}|{ph}"
    v = [r["f"][k] for r in rows if k in r["f"]]
    if not v:
        continue
    ar = np.array([x["ar"] for x in v]); ac = np.array([x["ac"] for x in v])
    dr = ar - np.array([x["mr"] for x in v]); dc = ac - np.array([x["mc"] for x in v])
    yr, yc = float((ar > 0).mean()), float((ac > 0).mean())
    sa = float((dr > 0).mean())
    R["by_framing"][k] = {
        "n": len(v), "says_yes_real": yr, "says_yes_cf": yc,
        "balanced_acc": (yr + 1 - yc) / 2, "absolute_auroc": auroc(ar, ac),
        "side_acc": sa, "side_p": float(stats.binomtest(int((dr > 0).sum()), len(dr), 0.5,
                                                        alternative="greater").pvalue),
        "comparison_auroc": auroc(dr, dc)}
    yes.append(yr); side.append(sa)

R["swing"] = {
    "absolute_says_yes": {"min": min(yes), "max": max(yes), "std": float(np.std(yes)),
                          "ratio": max(yes) / max(min(yes), 1e-9)},
    "matched_side_acc": {"min": min(side), "max": max(side), "std": float(np.std(side)),
                         "ratio": max(side) / max(min(side), 1e-9)},
    "absolute_auroc": {"min": min(v["absolute_auroc"] for v in R["by_framing"].values()),
                       "max": max(v["absolute_auroc"] for v in R["by_framing"].values())},
    "balanced_acc": {"min": min(v["balanced_acc"] for v in R["by_framing"].values()),
                     "max": max(v["balanced_acc"] for v in R["by_framing"].values())}}
# which axis moves the answer most: the spread of the yes-rate within each level of each axis
R["axis_effect"] = {}
for axis, pos in (("scale", 0), ("window", 1), ("phrasing", 2)):
    g = {}
    for s, win, ph in frames:
        k = f"{s}|{win}|{ph}"
        if k in R["by_framing"]:
            g.setdefault((s, win, ph)[pos], []).append(R["by_framing"][k]["says_yes_real"])
    R["axis_effect"][axis] = {str(kk): float(np.mean(vv)) for kk, vv in g.items()}

stem = str(RESULTS / (a.out or f"framing{'' if a.pick == 'spread' else '2'}"
                      f"_{a.model.split('/')[-1]}"))
Path(stem + ".json").parent.mkdir(parents=True, exist_ok=True)
Path(stem + ".json").write_text(json.dumps(R, indent=1) + "\n")
Path(stem + ".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print("\n" + json.dumps({k: R[k] for k in ("model", "levels", "swing", "axis_effect")}, indent=1))
