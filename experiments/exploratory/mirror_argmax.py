"""The method without the label: scan, compare, take the strongest asymmetry.

The matched control that rescued the accumulation result used the mask to decide which half of the chest
held the tumour. That is an oracle, so the result was about the model's capability and not about a method
anyone could run. Removing it is what turns the control into the module, and the way to remove it is to
stop asking where the lesion is and score every candidate region against its own anatomical counterpart.

A lung is paired. The counterpart of a region in one lung is the region at the same level in the other,
and in a body without disease the two look alike -- so whatever the model believes about chests in general
enters both readings equally and cancels in the difference. What survives is the asymmetry, which is what
a radiologist looks for in the first place.

That gives three numbers from one scan, none of which needs a mask:

  detection    max over regions of the signed difference -- large when some region differs from its mirror
  localisation argmax over regions -- which region, with four quadrants, so chance is a quarter
  grounding    the score IS a function of the chosen region, so masking that region must change it

and two baselines that isolate what the comparison contributes: the absolute reading of the whole slab,
which is how a VLM is normally asked, and the absolute reading of the best single region, which has the
scan but not the control.
"""
import argparse, json
from pathlib import Path
from fh import MSD, RESULTS, load_any, read_yesno, render

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import stats


Q = "Is there an abnormality in the lung shown in these images? Answer Yes or No."
# quadrant order is fixed so a row of scores can be read back: upper-left, upper-right, lower-left, lower-right
QUAD = [(0, 0), (0, 1), (1, 0), (1, 1)]


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def quad(img, row, col, mirror):
    """One quadrant, enlarged. Right-side quadrants are flipped so both readings show the mediastinum on
    the same side and the pair differs only in whose tissue it is."""
    w, h = img.size
    c = img.crop((col * w // 2, row * h // 2, (col + 1) * w // 2, (row + 1) * h // 2))
    if mirror:
        c = c.transpose(Image.FLIP_LEFT_RIGHT)
    return c.resize((448, 448), Image.BILINEAR)


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--cases", type=int, default=63)
ap.add_argument("--slab", type=int, default=3, help="adjacent slices per decision")
ap.add_argument("--per-case", type=int, default=3, help="positive slabs, and as many negative")
ap.add_argument("--margin", type=int, default=20, help="slices a negative slab must keep from any lesion")
ap.add_argument("--out", default=str(RESULTS / "mirror_argmax.json"))
a = ap.parse_args()

model, proc = load_any(a.model)
rows = []

for lab in sorted((MSD / "labelsTr").glob("lung_*.nii.gz"))[: a.cases]:
    mask = np.asanyarray(nib.load(str(lab)).dataobj)
    if mask.max() == 0:
        continue
    vol = np.asanyarray(nib.load(str(MSD / "imagesTr" / lab.name)).dataobj)
    Z = mask.shape[2]
    per_z = mask.reshape(-1, Z).sum(axis=0)
    on = np.nonzero(per_z)[0]
    if len(on) < a.slab:
        continue
    off = [z for z in range(Z) if min(abs(z - o) for o in on) > a.margin]
    if len(off) < a.slab:
        continue

    # ground-truth quadrant from the lesion centroid, in the same frame render() produces (a rot90)
    ij = np.nonzero(mask.sum(axis=2))
    cx, cy = ij[0].mean(), ij[1].mean()
    H, W = mask.shape[0], mask.shape[1]
    # rot90 on (x, y) sends (x, y) -> (row, col) = (W - 1 - y, x)
    gt_row, gt_col = int((W - 1 - cy) > H / 2), int(cx > W / 2)
    gt = QUAD.index((gt_row, gt_col))
    case = lab.name.split(".")[0]

    plans = []
    for i in np.linspace(0, len(on) - a.slab, a.per_case).round().astype(int):
        plans.append(("lesion", [int(z) for z in on[i:i + a.slab]]))
    for i in np.linspace(0, len(off) - a.slab, a.per_case).round().astype(int):
        plans.append(("clean", [int(z) for z in off[i:i + a.slab]]))

    for kind, zs in plans:
        full = [render(vol, z) for z in zs]
        p = [read_yesno(model, proc, [quad(f, r, c, mirror=(c == 1)) for f in full], Q) for r, c in QUAD]
        # each row of quadrants is one matched pair; a quadrant's score is how much it exceeds its mirror
        d_up, d_lo = p[0] - p[1], p[2] - p[3]
        score = [d_up, -d_up, d_lo, -d_lo]
        rows.append({"case": case, "kind": kind, "zs": zs, "gt_quad": gt,
                     "abs_quad": p, "mirror_score": score,
                     "pred_quad": int(np.argmax(score)),
                     "ours": float(max(score)),                 # comparison, label-free
                     "abs_region": float(max(p)),               # scan without the control
                     "abs_slab": read_yesno(model, proc, full, Q)})  # how a VLM is normally asked
    print(f"  {case}", flush=True)

les = [r for r in rows if r["kind"] == "lesion"]
cln = [r for r in rows if r["kind"] == "clean"]
R = {"slabs": {"lesion": len(les), "clean": len(cln)}, "cases": len({r["case"] for r in rows}),
     "detection_auroc": {k: auroc([r[k] for r in les], [r[k] for r in cln])
                         for k in ("ours", "abs_region", "abs_slab")},
     "says_yes_rate": {k: float(np.mean([r[k] > 0 for r in rows])) for k in ("abs_region", "abs_slab")}}

hit = sum(r["pred_quad"] == r["gt_quad"] for r in les)
R["localisation"] = {"hits": f"{hit}/{len(les)}", "acc": hit / len(les), "chance": 0.25,
                     "binom_p": float(stats.binomtest(hit, len(les), 0.25, alternative="greater").pvalue)}
hit_abs = sum(int(np.argmax(r["abs_quad"])) == r["gt_quad"] for r in les)
R["localisation_abs_baseline"] = {"hits": f"{hit_abs}/{len(les)}", "acc": hit_abs / len(les)}
# the asymmetry should be flat where there is nothing to be asymmetric about
R["clean_vs_lesion_max_score"] = {"lesion_mean": float(np.mean([r["ours"] for r in les])),
                                  "clean_mean": float(np.mean([r["ours"] for r in cln])),
                                  "mannwhitney_p": float(stats.mannwhitneyu(
                                      [r["ours"] for r in les], [r["ours"] for r in cln],
                                      alternative="greater").pvalue)}

out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(R, indent=1) + "\n")
Path(str(out).replace(".json", ".jsonl")).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print("\n" + json.dumps(R, indent=1))
