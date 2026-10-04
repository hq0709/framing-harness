"""Accumulation again, against a control matched for anatomy.

The first run had the negative set spanning the whole chest while the positive set sat inside the twenty
slices holding the tumour, so the clean sets were anatomically varied and the lesion sets were nearly
identical to each other. AUROC fell below chance as views were added, which reads as the model calling
variety abnormal rather than as evidence failing to accumulate. The windows arm pointed the same way: three
renderings of one slice stayed above chance while three slices fell below it, and renderings hold anatomy
fixed.

The control here is the other side of the same slice. Same z, same window, same lung anatomy at that level;
the lesion is the only difference, and it is the comparison a radiologist makes. Accumulation now means
adding slices on both sides at once, so variety cannot favour either arm.
"""
import argparse, json
from pathlib import Path
from fh import MSD, RESULTS, load_any, read_yesno, render

import nibabel as nib
import numpy as np


Q1 = "Is there an abnormality in the lung shown in this image? Answer Yes or No."
QN = "Is there an abnormality in the lung shown in these images? Answer Yes or No."


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])) if len(pos) and len(neg) else float("nan")


def half(img, left: bool):
    w, h = img.size
    return img.crop((0, 0, w // 2, h) if left else (w // 2, 0, w, h)).resize((448, 448))


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--cases", type=int, default=63)
ap.add_argument("--out", default=str(RESULTS / "matched_accumulation.json"))
a = ap.parse_args()

model, proc = load_any(a.model)
N_VIEWS = (1, 3, 5, 10)
ips, con = {n: [] for n in N_VIEWS}, {n: [] for n in N_VIEWS}
rec = []

for lab in sorted((MSD / "labelsTr").glob("lung_*.nii.gz"))[: a.cases]:
    mask = np.asanyarray(nib.load(str(lab)).dataobj)
    if mask.max() == 0:
        continue
    vol = np.asanyarray(nib.load(str(MSD / "imagesTr" / lab.name)).dataobj)
    Z = mask.shape[2]
    on = np.nonzero(mask.reshape(-1, Z).sum(axis=0))[0].tolist()
    if len(on) < 3:
        continue
    # which image-x half holds the lesion, after the rot90 the renderer applies
    xs = np.nonzero(mask.sum(axis=(1, 2)))[0]
    lesion_left_after_rot = xs.mean() < mask.shape[0] / 2
    case = lab.name.split(".")[0]
    for n in N_VIEWS:
        zs = [int(on[i]) for i in np.linspace(0, len(on) - 1, n).round().astype(int)]
        full = [render(vol, z) for z in zs]
        ipsi = [half(f, lesion_left_after_rot) for f in full]
        contra = [half(f, not lesion_left_after_rot) for f in full]
        q = Q1 if n == 1 else QN
        mi, mc = read_yesno(model, proc, ipsi, q), read_yesno(model, proc, contra, q)
        ips[n].append(mi); con[n].append(mc)
        rec.append({"case": case, "n": n, "ipsilateral": mi, "contralateral": mc, "slices": zs})
    print(f"  {case}", flush=True)

R = {str(n): {"auroc_ipsi_vs_contra": auroc(ips[n], con[n]), "n_cases": len(ips[n]),
              "mean_ipsi": float(np.mean(ips[n])), "mean_contra": float(np.mean(con[n])),
              "paired_win_rate": float(np.mean([i > c for i, c in zip(ips[n], con[n])]))} for n in N_VIEWS}
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(R, indent=1) + "\n")
Path(str(out).replace(".json", ".jsonl")).write_text("\n".join(json.dumps(r) for r in rec) + "\n")
print("\n" + json.dumps(R, indent=1))
