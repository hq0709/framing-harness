"""Arm A vs Arm B: does a bigger uniform slice budget fix what a small one misses?

The kill criterion for the whole viewport-as-action idea. A lung tumour occupies about 7% of the slices
of its volume, so uniform sampling at a small budget misses it outright -- measured, 8 slices miss the
lesion in 7 of 15 volumes. If simply raising the budget to 32 recovers the answers, then the fix is a
larger budget and there is nothing for an agent to do. If it does not, the question is which slices, not
how many, and that is a decision only something reasoning about the case can make.

One probe per side per volume: the side holding the tumour answers yes, the other no. The negative needs
no inpainting and it is the contralateral control a radiologist reads anyway.
"""
import argparse
import json
from pathlib import Path
from fh import MSD, RESULTS, read_yesno, render

import nibabel as nib
import numpy as np
import torch

LUNG = MSD / 'Task06_Lung'
LUNG_WL, LUNG_WW = -600.0, 1500.0


ap = argparse.ArgumentParser()
ap.add_argument("--budgets", default="8,32")
ap.add_argument("--cases", type=int, default=30)
ap.add_argument("--out", default=str(RESULTS / "slice_budget.jsonl"))
a = ap.parse_args()

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration     # noqa: E402
proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=512 * 512, max_pixels=512 * 512)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()

budgets = [int(b) for b in a.budgets.split(",")]
RESULTS.mkdir(parents=True, exist_ok=True)
rows = []
for lab_path in sorted((MSD / "labelsTr").glob("lung_*.nii.gz"))[: a.cases]:
    mask = np.asanyarray(nib.load(str(lab_path)).dataobj)
    if mask.max() == 0:
        continue
    vol = np.asanyarray(nib.load(str(MSD / "imagesTr" / lab_path.name)).dataobj)
    Z = mask.shape[2]
    on = set(np.nonzero(mask.reshape(-1, Z).sum(axis=0))[0].tolist())
    xs = np.nonzero(mask.sum(axis=(1, 2)))[0]
    lesion_left = xs.mean() > mask.shape[0] / 2          # image x; side label is nominal but consistent
    case = lab_path.name.split(".")[0]
    for n in budgets:
        idx = np.linspace(0, Z - 1, n).round().astype(int)
        imgs = [render(vol, int(z)) for z in idx]
        covered = bool(set(idx.tolist()) & on)
        for side_is_left in (True, False):
            side = "left" if side_is_left else "right"
            q = (f"These are {n} axial CT slices of the chest, ordered from superior to inferior. "
                 f"Is there a lung tumour in the {side} lung? Answer Yes or No.")
            m = read_yesno(model, proc, imgs, q)
            rows.append({"case": case, "budget": n, "side": side,
                         "gold": "yes" if side_is_left == lesion_left else "no",
                         "margin": round(m, 4), "pred": "yes" if m > 0 else "no",
                         "lesion_slices": len(on), "total_slices": Z, "sampled_covers_lesion": covered})
            print(json.dumps(rows[-1]), flush=True)

Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(f"\nwrote {len(rows)} probes to {a.out}")
