"""Negatives that are real pixels: the same window, the nearest level the finding does not reach.

The first version of this pasted a tumour-free patch over the tumour and used the result as the negative.
It drew that patch from the level furthest from the tumour, reasoning that distance guarantees the patch
carries none of it. Distance guarantees something else: in a seventy-five-level liver CT the furthest clean
level is not liver. The paste put spine and vessels where the tumour had been, the model read that as
abnormal, and the sign of the whole measurement flipped -- erasing the tumour made the slice read as *more*
diseased, by 0.6 of a logit.

That is the third time today a control built from a distant level turned out to measure craniocaudal
position. So this one does not edit pixels at all. For a window W on the tumour, the negative is W at the
nearest level where W holds no tumour: the same patient, the same (x, y), the same organ a few millimetres
along, and every pixel real. The harness's reference is a second clean level, kept apart from the negative
so the readout is never handed the very image it is scored against.

    positive   W at z            the tumour is in it
    negative   W at z_neg        nearest level where W is clear
    reference  W at z_ref        a different clear level, what the comparison subtracts

The pasted counterfactual is kept beside it, now drawn from z_neg and feathered at the edges, because the
two controls fail in opposite ways: a paste is perfectly level-matched and can leave an artefact, a real
window has no artefact and is a few levels off. Agreement between them is what makes either believable.
"""
import argparse, json
from pathlib import Path
from fh import MSD, RESULTS, load_any, read_yesno, render

import nibabel as nib
import numpy as np
from PIL import Image, ImageFilter
from scipy import stats


ROOT = MSD
WL = {"Task06_Lung": (-600.0, 1500.0), "Task03_Liver": (50.0, 400.0),
      "Task07_Pancreas": (50.0, 400.0), "Task10_Colon": (50.0, 400.0)}
ORGAN = {"Task06_Lung": "lung", "Task03_Liver": "liver",
         "Task07_Pancreas": "pancreas", "Task10_Colon": "colon"}
SIDE, GRID = 512, 448


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])) if len(pos) and len(neg) else float("nan")


def cut(img, b, flip=False):
    c = img.crop(b)
    return (c.transpose(Image.FLIP_LEFT_RIGHT) if flip else c).resize((GRID, GRID), Image.BICUBIC)


def feathered(real, src, rot, pad, blur):
    """Replace the tumour with the same place at a clean level, blended over a soft edge so the join is not
    itself a finding."""
    a = Image.new("L", (SIDE, SIDE), 0)
    rr, cc = np.nonzero(rot)
    box = (max(int(cc.min()) - pad, 0), max(int(rr.min()) - pad, 0),
           min(int(cc.max()) + 1 + pad, SIDE), min(int(rr.max()) + 1 + pad, SIDE))
    Image.Image.paste(a, Image.new("L", (box[2] - box[0], box[3] - box[1]), 255), box[:2])
    return Image.composite(src, real, a.filter(ImageFilter.GaussianBlur(blur)))


ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Task03_Liver")
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--cases", type=int, default=40)
ap.add_argument("--levels", type=int, default=3)
ap.add_argument("--scale", type=int, default=128)
ap.add_argument("--pad", type=int, default=6)
ap.add_argument("--blur", type=float, default=4.0)
ap.add_argument("--max-gap", type=int, default=40, help="levels the negative may sit from the tumour")
ap.add_argument("--min-area", type=int, default=12)
ap.add_argument("--sanity", type=int, default=2)
ap.add_argument("--out", default="")
a = ap.parse_args()

task = ROOT / a.task
wl, ww = WL[a.task]
Q = f"Is there an abnormality in the {ORGAN[a.task]} shown in this image? Answer Yes or No."
model, proc = load_any(a.model)
rows, dumped = [], 0

for lab in sorted((task / "labelsTr").glob("*.nii.gz"))[: a.cases]:
    m = np.asanyarray(nib.load(str(lab)).dataobj)
    mask = ((m >= 2) if m.max() > 1 else (m > 0)).astype(np.uint8)
    img = task / "imagesTr" / lab.name
    if mask.max() == 0 or not img.exists():
        continue
    vol = np.asanyarray(nib.load(str(img)).dataobj)
    Z = mask.shape[2]
    area = mask.reshape(-1, Z).sum(axis=0)
    on = [int(z) for z in np.nonzero(area >= a.min_area)[0]]
    if not on:
        continue
    case = lab.name.split(".")[0]

    for z in sorted(sorted(on, key=lambda t: -area[t])[: a.levels]):
        rot = np.rot90(mask[:, :, z])
        rr, cc = np.nonzero(rot)
        if len(rr) < a.min_area:
            continue
        cy, cx = float(rr.mean()), float(cc.mean())
        s = a.scale
        x0 = int(np.clip(cx - s // 2, 0, SIDE - s)); y0 = int(np.clip(cy - s // 2, 0, SIDE - s))
        b = (x0, y0, x0 + s, y0 + s)

        clear = sorted((zz for zz in range(Z)
                        if np.rot90(mask[:, :, zz])[y0:y0 + s, x0:x0 + s].sum() == 0),
                       key=lambda zz: abs(zz - z))
        if not clear or abs(clear[0] - z) > a.max_gap:
            continue
        z_neg = clear[0]
        z_ref = next((zz for zz in clear if abs(zz - z_neg) >= 3), None)
        if z_ref is None:
            continue

        real = render(vol, z, side=SIDE, wl=wl, ww=ww)
        neg = render(vol, z_neg, side=SIDE, wl=wl, ww=ww)
        ref = render(vol, z_ref, side=SIDE, wl=wl, ww=ww)
        cf = feathered(real, neg, rot, a.pad, a.blur)

        if dumped < a.sanity:
            strip = Image.new("RGB", (GRID * 4, GRID))
            for i, im in enumerate((real, cf, neg, ref)):
                strip.paste(cut(im, b), (i * GRID, 0))
            o = RESULTS / \
                f"sanity_v2_{a.task}_{case}_z{z}.png"
            strip.save(o); dumped += 1

        r = {"case": case, "z": z, "z_neg": z_neg, "z_ref": z_ref, "gap": abs(z_neg - z),
             "px": int(rot.sum()),
             "pos": read_yesno(model, proc, [cut(real, b)], Q),
             "neg": read_yesno(model, proc, [cut(neg, b)], Q),
             "ref": read_yesno(model, proc, [cut(ref, b)], Q),
             "cf": read_yesno(model, proc, [cut(cf, b)], Q)}
        if a.task == "Task06_Lung":
            mb = (SIDE - b[2], b[1], SIDE - b[0], b[3])
            r["mir_pos"] = read_yesno(model, proc, [cut(real, mb, flip=True)], Q)
            r["mir_neg"] = read_yesno(model, proc, [cut(neg, mb, flip=True)], Q)
        rows.append(r)
    print(f"  {case}", flush=True)

by = {}
for r in rows:
    by.setdefault(r["case"], []).append(r)
cases = list(by.values())

R = {"task": a.task, "organ": ORGAN[a.task], "model": a.model, "scale": a.scale,
     "cases": len(cases), "levels": len(rows),
     "median_tumour_px": float(np.median([r["px"] for r in rows])) if rows else 0,
     "median_gap_levels": float(np.median([r["gap"] for r in rows])) if rows else 0,
     "readouts": {}}


def score(pos, neg, agg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    se, sp = float((pos > 0).mean()), float((neg <= 0).mean())
    n, k = len(pos) * 2, int((pos > 0).sum() + (neg <= 0).sum())
    return {"n": len(pos), "sens": se, "spec": sp, "bal_acc": (se + sp) / 2,
            "auroc": auroc(pos, neg), "skew": abs(se - sp),
            "p": float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue)}


def arms(negkey):
    out = {"absolute": lambda r: (r["pos"], r[negkey]),
           "axial": lambda r: (r["pos"] - r["ref"], r[negkey] - r["ref"])}
    if rows and "mir_pos" in rows[0] and negkey == "neg":
        out["mirror"] = lambda r: (r["pos"] - r["mir_pos"], r["neg"] - r["mir_neg"])
    return out


for negkey, label in (("neg", "real_window"), ("cf", "pasted_counterfactual")):
    R["readouts"][label] = {}
    for name, fn in arms(negkey).items():
        per = score([fn(r)[0] for r in rows], [fn(r)[1] for r in rows], False)
        cas = score([np.mean([fn(r)[0] for r in v]) for v in cases],
                    [np.mean([fn(r)[1] for r in v]) for v in cases], True)
        R["readouts"][label][name] = {"per_level": per, "per_case_mean": cas}

stem = str(RESULTS / (a.out or f"hv2_{a.task}_{a.model.split('/')[-1]}"))
Path(stem + ".json").write_text(json.dumps(R, indent=1) + "\n")
Path(stem + ".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print("\n" + json.dumps(R, indent=1))
