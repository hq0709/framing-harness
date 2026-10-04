"""The module in general form: a region is read against a matched reference set, and the prior cancels.

`mirror_argmax.py` differences one region against one mirror. Writing the second anatomy -- liver, which has
no contralateral half -- shows that the mirror was never the point. What the difference needs is a reference
that shares the prior, and a set of references serves better than one: a median over several is unmoved by a
reference that happens to be diseased itself, which is what defeats a single axial neighbour when a tumour
spans levels.

    s(r) = m(view(r), q) - median over r' in ref(r) of m(view(r'), q)

The prior b(q) cancels against the median exactly as it cancels against one reading, since it is the same
constant in every term. ref(r) is the only thing anatomy decides:

    mirror   {the same region in the other lung}                 paired organs
    axial    {the same region, several levels away}              liver, pancreas, colon

and both are built from the image alone. Detection is max over regions, localisation is argmax, and the
score is a function of the region that won -- so a right answer cannot rest on nothing.

    python counterpart.py --task Task06_Lung --ref mirror --model Qwen/Qwen2.5-VL-7B-Instruct
    python counterpart.py --task Task03_Liver --ref axial  --model lingshu-medical-mllm/Lingshu-7B
"""
import argparse, json
from pathlib import Path
from fh import MSD, RESULTS, load_any, render

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from scipy import stats


ROOT = MSD
Q = "Is there an abnormality in the {organ} shown in these images? Answer Yes or No."
QUAD = [(0, 0), (0, 1), (1, 0), (1, 1)]
# window and level per task: lung parenchyma for the chest, soft tissue for the abdomen
WL = {"Task06_Lung": (-600.0, 1500.0), "Task03_Liver": (50.0, 400.0),
      "Task07_Pancreas": (50.0, 400.0), "Task10_Colon": (50.0, 400.0)}
ORGAN = {"Task06_Lung": "lung", "Task03_Liver": "liver",
         "Task07_Pancreas": "pancreas", "Task10_Colon": "colon"}
PREFIX = {"Task06_Lung": "lung", "Task03_Liver": "liver",
          "Task07_Pancreas": "pancreas", "Task10_Colon": "colon"}


def load_any(name):
    """One loader for every family: the image-text-to-text auto class covers Qwen2.5-VL, Gemma 3,
    InternVL's HF port and LLaVA, and each ships the chat template its own readout needs."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
    for attr, px in (("min_pixels", 448 * 448), ("max_pixels", 448 * 448)):
        if hasattr(getattr(proc, "image_processor", None), attr):
            setattr(proc.image_processor, attr, px)
    model = AutoModelForImageTextToText.from_pretrained(
        name, dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True, attn_implementation="sdpa").eval()
    return model, proc


def read(model, proc, images, question) -> float:
    """log P(yes) - log P(no) at the answer position."""
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": question})
    text = proc.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True, tokenize=False)
    enc = proc(text=[text], images=images, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1].float()
    tok = getattr(proc, "tokenizer", proc)
    ids = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})
    y = torch.logsumexp(logits[ids(["Yes", " Yes", "yes", " yes"])], 0)
    n = torch.logsumexp(logits[ids(["No", " No", "no", " no"])], 0)
    return float(y - n)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def crop(img, row, col, flip=False):
    w, h = img.size
    c = img.crop((col * w // 2, row * h // 2, (col + 1) * w // 2, (row + 1) * h // 2))
    if flip:
        c = c.transpose(Image.FLIP_LEFT_RIGHT)
    return c.resize((448, 448), Image.BILINEAR)


ap = argparse.ArgumentParser()
ap.add_argument("--task", default="Task06_Lung")
ap.add_argument("--ref", default="mirror", choices=["mirror", "axial"])
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--cases", type=int, default=40)
ap.add_argument("--slab", type=int, default=3)
ap.add_argument("--per-case", type=int, default=2)
ap.add_argument("--margin", type=int, default=20, help="levels a clean slab keeps from any lesion")
ap.add_argument("--offsets", default="-40,-25,25,40", help="axial reference offsets, in levels")
ap.add_argument("--out", default="")
a = ap.parse_args()

task = ROOT / a.task
wl, ww = WL[a.task]
question = Q.format(organ=ORGAN[a.task])
offs = [int(o) for o in a.offsets.split(",")]
model, proc = load_any(a.model)
rows = []

for lab in sorted((task / "labelsTr").glob(f"{PREFIX[a.task]}_*.nii.gz"))[: a.cases]:
    mask = np.asanyarray(nib.load(str(lab)).dataobj)
    mask = (mask >= 2).astype(np.uint8) if mask.max() > 1 else mask  # MSD labels organ=1, tumour=2
    if mask.max() == 0:
        continue
    img = task / "imagesTr" / lab.name
    if not img.exists():
        continue
    vol = np.asanyarray(nib.load(str(img)).dataobj)
    Z = mask.shape[2]
    on = np.nonzero(mask.reshape(-1, Z).sum(axis=0))[0]
    if len(on) < a.slab:
        continue
    off = [z for z in range(Z) if min(abs(z - o) for o in on) > a.margin]
    if len(off) < a.slab:
        continue

    # read the ground-truth quadrant off the mask after the same rotation render() applies, rather than
    # deriving it in closed form: the closed form was right only because CT is square, and rot90 swaps the
    # index ranges, so the two thresholds belong to different axes
    rot = np.rot90(mask.sum(axis=2))
    rr, cc = np.nonzero(rot)
    gt = QUAD.index((int(rr.mean() > rot.shape[0] / 2), int(cc.mean() > rot.shape[1] / 2)))
    case = lab.name.split(".")[0]

    plans = [("lesion", [int(z) for z in on[i:i + a.slab]])
             for i in np.linspace(0, len(on) - a.slab, a.per_case).round().astype(int)]
    plans += [("clean", [int(z) for z in off[i:i + a.slab]])
              for i in np.linspace(0, len(off) - a.slab, a.per_case).round().astype(int)]

    for kind, zs in plans:
        view = lambda zz, r, c, f=False: [crop(render(vol, z, wl=wl, ww=ww), r, c, f) for z in zz]
        absr, score = [], []
        for r, c in QUAD:
            m_r = read(model, proc, view(zs, r, c), question)
            absr.append(m_r)
            if a.ref == "mirror":
                refs = [read(model, proc, view(zs, r, 1 - c, f=True), question)]
            else:
                refs = [read(model, proc, view([min(max(z + d, 0), Z - 1) for z in zs], r, c), question)
                        for d in offs if 0 <= zs[0] + d and zs[-1] + d < Z]
            score.append(m_r - float(np.median(refs)) if refs else float("nan"))
        rows.append({"case": case, "kind": kind, "zs": zs, "gt_quad": gt, "abs_quad": absr,
                     "score": score, "pred_quad": int(np.nanargmax(score)),
                     "ours": float(np.nanmax(score)), "abs_region": float(max(absr)),
                     "abs_slab": read(model, proc, [render(vol, z, wl=wl, ww=ww) for z in zs], question)})
    print(f"  {case}", flush=True)

les = [r for r in rows if r["kind"] == "lesion"]
cln = [r for r in rows if r["kind"] == "clean"]
R = {"task": a.task, "ref": a.ref, "model": a.model, "cases": len({r["case"] for r in rows}),
     "slabs": {"lesion": len(les), "clean": len(cln)},
     "detection_auroc": {k: auroc([r[k] for r in les], [r[k] for r in cln])
                         for k in ("ours", "abs_region", "abs_slab")},
     "says_yes_rate": {k: float(np.mean([r[k] > 0 for r in rows])) for k in ("abs_region", "abs_slab")}}
if les:
    h = sum(r["pred_quad"] == r["gt_quad"] for r in les)
    ha = sum(int(np.argmax(r["abs_quad"])) == r["gt_quad"] for r in les)
    R["localisation"] = {"ours": f"{h}/{len(les)}", "ours_acc": h / len(les),
                         "abs_baseline": f"{ha}/{len(les)}", "abs_acc": ha / len(les), "chance": 0.25,
                         "binom_p": float(stats.binomtest(h, len(les), 0.25, alternative="greater").pvalue)}

stem = str(RESULTS / (a.out or f"cp_{a.task}_{a.ref}_{a.model.split('/')[-1]}"))
out = Path(str(stem) + ".json"); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(R, indent=1) + "\n")
Path(str(stem) + ".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print("\n" + json.dumps(R, indent=1))
