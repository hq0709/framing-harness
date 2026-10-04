"""The same harness on four organs, two of which have no other side.

Reading a region against its mirror works for a lung. A liver has no mirror, and neither has a pancreas or a
colon, so a harness that only knows how to mirror is a lung harness. What the readout actually needs is a
second input that shares the framing and differs in the finding, and volumetric imaging always supplies one:
the same window a few levels away, where the organ continues and the tumour does not.

    mirror   the same box in the other lung                     paired organs
    axial    the same box at a level where the tumour is absent  any volume

Both are built from the image. The control is the same in either case and is built from the mask, which only
the scoring sees: paste a tumour-free patch over the tumour, from a level far enough away that it carries
none of it, and ask again. Positive and negative are then the same patient, level, window and framing, with
the tumour as the only difference -- the one control that cannot be reading craniocaudal position, which
three earlier designs turned out to be doing.

The erasure patch and the harness's counterpart are deliberately taken from different levels, so the readout
cannot win by being handed the very pixels the control used.
"""
import argparse, json
from pathlib import Path
from fh import MSD, RESULTS, load_any, read_yesno, render

import nibabel as nib
import numpy as np
from PIL import Image
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


ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Task03_Liver")
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--cases", type=int, default=40)
ap.add_argument("--levels", type=int, default=3, help="largest tumour cross-sections per case")
ap.add_argument("--scale", type=int, default=128, help="side of the window read, in slice pixels")
ap.add_argument("--pad", type=int, default=8)
ap.add_argument("--min-area", type=int, default=12)
ap.add_argument("--out", default="")
a = ap.parse_args()

task = ROOT / a.task
wl, ww = WL[a.task]
organ = ORGAN[a.task]
Q = f"Is there an abnormality in the {organ} shown in this image? Answer Yes or No."
model, proc = load_any(a.model)
rows = []

for lab in sorted((task / "labelsTr").glob("*.nii.gz"))[: a.cases]:
    m = np.asanyarray(nib.load(str(lab)).dataobj)
    mask = ((m >= 2) if m.max() > 1 else (m > 0)).astype(np.uint8)   # MSD marks organ 1, tumour 2
    if mask.max() == 0:
        continue
    img = task / "imagesTr" / lab.name
    if not img.exists():
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

        # levels where this same window holds no tumour at all: one for the control, a different one for
        # the harness, so the readout is never handed the pixels the control pasted
        clean = [zz for zz in range(Z)
                 if np.rot90(mask[:, :, zz])[y0:y0 + s, x0:x0 + s].sum() == 0]
        if len(clean) < 2:
            continue
        far = sorted(clean, key=lambda zz: -abs(zz - z))
        z_erase = far[0]
        z_ref = min((zz for zz in clean if abs(zz - z_erase) > s // 8), key=lambda zz: abs(zz - z),
                    default=None)
        if z_ref is None:
            continue

        real = render(vol, z, side=SIDE, wl=wl, ww=ww)
        # the counterfactual: the tumour's own box, replaced by the same box from a tumour-free level
        src = render(vol, z_erase, side=SIDE, wl=wl, ww=ww)
        rr2, cc2 = np.nonzero(rot)
        p0 = max(int(rr2.min()) - a.pad, 0), max(int(cc2.min()) - a.pad, 0)
        p1 = min(int(rr2.max()) + 1 + a.pad, SIDE), min(int(cc2.max()) + 1 + a.pad, SIDE)
        arr = np.array(real)
        arr[p0[0]:p1[0], p0[1]:p1[1]] = np.array(src)[p0[0]:p1[0], p0[1]:p1[1]]
        cf = Image.fromarray(arr)

        r = {"case": case, "z": z, "px": int(rot.sum()), "box": b,
             "z_erase": z_erase, "z_ref": z_ref,
             "abs_real": read_yesno(model, proc, [cut(real, b)], Q),
             "abs_cf": read_yesno(model, proc, [cut(cf, b)], Q)}
        # axial counterpart: the same window where the tumour is not
        ref = render(vol, z_ref, side=SIDE, wl=wl, ww=ww)
        r["axial_ref"] = read_yesno(model, proc, [cut(ref, b)], Q)
        if a.task == "Task06_Lung":
            mb = (SIDE - b[2], b[1], SIDE - b[0], b[3])
            r["mirror_real"] = read_yesno(model, proc, [cut(real, mb, flip=True)], Q)
            r["mirror_cf"] = read_yesno(model, proc, [cut(cf, mb, flip=True)], Q)
        rows.append(r)
    print(f"  {case}", flush=True)

by_case = {}
for r in rows:
    by_case.setdefault(r["case"], []).append(r)
cases = list(by_case.values())

R = {"task": a.task, "organ": organ, "model": a.model, "scale": a.scale,
     "cases": len(cases), "levels": len(rows),
     "median_tumour_px": float(np.median([r["px"] for r in rows])),
     "median_apparent_pct": float(np.median([r["px"] for r in rows])) / (a.scale ** 2) * 100}

def report(pos, neg, name):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return {"auroc": auroc(pos, neg), "mean_gap": float(np.mean(pos - neg)),
            "wins": f"{int((pos > neg).sum())}/{len(pos)}",
            "wilcoxon_p": float(stats.wilcoxon(pos, neg, alternative="greater").pvalue)}

R["per_level"] = {"absolute": report([r["abs_real"] for r in rows], [r["abs_cf"] for r in rows], "abs"),
                  "axial": report([r["abs_real"] - r["axial_ref"] for r in rows],
                                  [r["abs_cf"] - r["axial_ref"] for r in rows], "axial")}
if "mirror_real" in (rows[0] if rows else {}):
    R["per_level"]["mirror"] = report([r["abs_real"] - r["mirror_real"] for r in rows],
                                      [r["abs_cf"] - r["mirror_cf"] for r in rows], "mirror")
R["says_yes"] = {"real": float(np.mean([r["abs_real"] > 0 for r in rows])),
                 "cf": float(np.mean([r["abs_cf"] > 0 for r in rows]))}
R["per_case_mean_over_levels"] = {}
for name, f in (("absolute", lambda r: (r["abs_real"], r["abs_cf"])),
                ("axial", lambda r: (r["abs_real"] - r["axial_ref"], r["abs_cf"] - r["axial_ref"]))):
    p = [float(np.mean([f(r)[0] for r in v])) for v in cases]
    n = [float(np.mean([f(r)[1] for r in v])) for v in cases]
    R["per_case_mean_over_levels"][name] = report(p, n, name)

stem = str(RESULTS / (a.out or f"harness_{a.task}_{a.model.split('/')[-1]}"))
Path(stem + ".json").write_text(json.dumps(R, indent=1) + "\n")
Path(stem + ".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print("\n" + json.dumps({k: R[k] for k in R if k != "per_framing"}, indent=1))
