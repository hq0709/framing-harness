"""Does choosing the views beat spreading them, at the same budget?

The last place this idea can fail. Accumulation works: ten anatomy-matched views separate the lesion side
from the other side where one view does not. But if views spread uniformly through the volume accumulate
from fh import MSD, RESULTS, load_any, read_yesno, render
just as well as views aimed at the lesion, then nothing has to decide where to look, the budget is the whole
story, and the agent layer is decoration.

Both arms use the same contralateral control and the same number of images, so they differ only in which
slices are shown. Aimed views sit inside the slices carrying the tumour; spread views are spaced over the
whole volume, so most of them have no lesion on either side and carry no evidence either way.
"""
import argparse, json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import stats


QN = "Is there an abnormality in the lung shown in these images? Answer Yes or No."


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def half(img, left):
    w, h = img.size
    return img.crop((0, 0, w // 2, h) if left else (w // 2, 0, w, h)).resize((448, 448))


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--cases", type=int, default=63)
ap.add_argument("--budgets", default="5,10")
ap.add_argument("--out", default=str(RESULTS / "selection_matched.json"))
a = ap.parse_args()

model, proc = load_any(a.model)
budgets = [int(b) for b in a.budgets.split(",")]
arms = {b: {"aimed": ([], []), "spread": ([], [])} for b in budgets}
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
    xs = np.nonzero(mask.sum(axis=(1, 2)))[0]
    left = xs.mean() < mask.shape[0] / 2
    case = lab.name.split(".")[0]
    for b in budgets:
        plans = {"aimed": [int(on[i]) for i in np.linspace(0, len(on) - 1, b).round().astype(int)],
                 "spread": [int(z) for z in np.linspace(0, Z - 1, b).round().astype(int)]}
        for arm, zs in plans.items():
            full = [render(vol, z) for z in zs]
            mi = read_yesno(model, proc, [half(f, left) for f in full], QN)
            mc = read_yesno(model, proc, [half(f, not left) for f in full], QN)
            arms[b][arm][0].append(mi); arms[b][arm][1].append(mc)
            rec.append({"case": case, "budget": b, "arm": arm, "ipsi": mi, "contra": mc,
                        "slices_with_lesion": len(set(zs) & set(on))})
    print(f"  {case}", flush=True)

R = {}
for b in budgets:
    R[str(b)] = {}
    for arm in ("aimed", "spread"):
        ip, co = arms[b][arm]
        d = [i - c for i, c in zip(ip, co)]
        wins = sum(x > 0 for x in d)
        R[str(b)][arm] = {"auroc": auroc(ip, co), "paired_wins": f"{wins}/{len(d)}",
                          "win_rate": wins / len(d), "mean_diff": float(np.mean(d)),
                          "binom_p": float(stats.binomtest(wins, len(d), 0.5, alternative="greater").pvalue)}
    da = [i - c for i, c in zip(*arms[b]["aimed"])]
    ds = [i - c for i, c in zip(*arms[b]["spread"])]
    R[str(b)]["aimed_vs_spread_wilcoxon_p"] = float(stats.wilcoxon(da, ds, alternative="greater").pvalue)

out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(R, indent=1) + "\n")
Path(str(out).replace(".json", ".jsonl")).write_text("\n".join(json.dumps(r) for r in rec) + "\n")
print("\n" + json.dumps(R, indent=1))
