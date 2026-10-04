"""The framework on ordinary medical VQA: no masks, no anatomy, no training, standard accuracy.

A model's answer carries a term that depends on how the question was presented and not on the image. On CT
that term ran from answering Yes 2.8% of the time to 96.4% while discriminability never moved, and averaging
twelve presentations of the same evidence lifted balanced accuracy from 55.7% at the worst presentation to
70.4%. None of that used a mask, an organ or a paired structure -- which is the claim this script tests,
on benchmarks that have none of those things.

Three operators, in order of what they need:

    marginalise   average the margin over K presentations          removes the variance of the bias, K reads
    centre        subtract that presentation's batch median        removes its mean, costs nothing
    both

A presentation is anything that leaves the picture alone: a slightly tighter centre crop, a different input
resolution, the same question in different words. The batch median is the label-free part -- it is where the
model puts its zero for this presentation, estimated from the answers themselves, and no label is read.
"""
import argparse, glob, itertools, json
from pathlib import Path
from fh import HF_HOME, RESULTS, load_any, read_yesno

import numpy as np
from PIL import Image


CROPS = [1.00, 0.92, 0.85]
RES = [448, 336]
PHRASINGS = ["{q} Answer Yes or No.", "Question: {q}\nAnswer with Yes or No."]
SETS = {"vqa-rad": "flaviagiammarino--vqa-rad", "path-vqa": "flaviagiammarino--path-vqa"}


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])) if len(pos) and len(neg) else float("nan")


def bal(margin, label):
    margin, label = np.asarray(margin, float), np.asarray(label, bool)
    return (float((margin[label] > 0).mean()) + float((margin[~label] <= 0).mean())) / 2


def present(img, crop, res):
    w, h = img.size
    cw, ch = int(w * crop), int(h * crop)
    x, y = (w - cw) // 2, (h - ch) // 2
    return img.crop((x, y, x + cw, y + ch)).convert("RGB").resize((res, res), Image.BICUBIC)


ap = argparse.ArgumentParser()
ap.add_argument("--set", default="vqa-rad", choices=list(SETS))
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args()

from datasets import load_dataset
fs = sorted(glob.glob(f"{HF_HOME}/hub/datasets--{SETS[a.set]}"
                      f"/snapshots/*/**/*.parquet", recursive=True))
test = [f for f in fs if "test" in f.split("/")[-1].lower()] or fs[-1:]
ds = load_dataset("parquet", data_files=test, split="train")
idx = [i for i, ans in enumerate(ds["answer"]) if ans.strip().lower() in ("yes", "no")]
rng = np.random.default_rng(a.seed)
idx = sorted(rng.permutation(idx)[: a.n].tolist())

model, proc = load_any(a.model)
frames = list(itertools.product(CROPS, RES, range(len(PHRASINGS))))
rows = []
for n, i in enumerate(idx):
    ex = ds[i]
    img, q = ex["image"], ex["question"]
    r = {"i": i, "label": ex["answer"].strip().lower() == "yes", "m": {}}
    for c, res, ph in frames:
        r["m"][f"{c}|{res}|{ph}"] = read_yesno(model, proc, [present(img, c, res)],
                                               PHRASINGS[ph].format(q=q))
    rows.append(r)
    if n % 50 == 0:
        print(f"  {n}/{len(idx)}", flush=True)

keys = list(rows[0]["m"])
lab = np.array([r["label"] for r in rows])
M = {k: np.array([r["m"][k] for r in rows]) for k in keys}
C = {k: M[k] - np.median(M[k]) for k in keys}      # label-free: where this presentation puts its zero

R = {"set": a.set, "model": a.model, "n": len(rows), "framings": len(keys),
     "positives": int(lab.sum()), "per_framing": {}}
for k in keys:
    R["per_framing"][k] = {"says_yes": float((M[k] > 0).mean()), "bal_acc": bal(M[k], lab),
                           "auroc": auroc(M[k][lab], M[k][~lab])}

single = [v["bal_acc"] for v in R["per_framing"].values()]
R["single_framing"] = {"worst": min(single), "median": float(np.median(single)), "best": max(single),
                       "spread_pp": (max(single) - min(single)) * 100,
                       "auroc_median": float(np.median([v["auroc"] for v in R["per_framing"].values()]))}

R["curve"] = {}
for K in [k for k in (1, 2, 3, 4, 6, 8, 12) if k <= len(keys)]:
    mg, ce = [], []
    for _ in range(40):
        sub = list(rng.permutation(keys)[:K])
        mg.append(bal(np.mean([M[k] for k in sub], axis=0), lab))
        ce.append(bal(np.mean([C[k] for k in sub], axis=0), lab))
    R["curve"][str(K)] = {"marginalise": float(np.mean(mg)), "centre_and_marginalise": float(np.mean(ce))}

Mall = np.mean([M[k] for k in keys], axis=0)
Call = np.mean([C[k] for k in keys], axis=0)
R["all_framings"] = {
    "marginalise": {"bal_acc": bal(Mall, lab), "auroc": auroc(Mall[lab], Mall[~lab])},
    "centre_and_marginalise": {"bal_acc": bal(Call, lab), "auroc": auroc(Call[lab], Call[~lab])},
    "centre_only_median_framing": float(np.median([bal(C[k], lab) for k in keys]))}

stem = str(RESULTS / (a.out or f"vqa_{a.set}_{a.model.split('/')[-1]}"))
Path(stem + ".json").write_text(json.dumps(R, indent=1) + "\n")
Path(stem + ".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print("\n" + json.dumps({k: R[k] for k in ("set", "model", "n", "single_framing", "curve", "all_framings")}, indent=1))
