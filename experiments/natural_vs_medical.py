"""One curve for both domains: how strong the evidence is decides whether the frame owns the answer.

A tumour covering a thousandth of a CT slice leaves the model's answer to the frame -- cropping differently
moved the rate of answering Yes from 10% to 93% while the ability to tell that slice from the same slice
with the tumour erased never moved. The obvious objection is that medical images are their own world. They
are not. They sit at one end of a dial that natural images occupy the rest of, and the dial is how much of
the frame the evidence covers.

So here the dial is turned by hand. A COCO object is cut out along its mask and composited onto a background
that does not contain its category, at sizes from a twentieth of a percent of the frame up to a fifth of it.
The counterfactual is that same background without the paste: identical in every pixel but the object, the
same control used on the CT slices. At each size the same question is put in several framings that leave the
picture alone -- two phrasings, the image once or twice -- so whatever the answer does across them, it does
without the evidence changing.

Resolution is swept too and kept separate, because halving it is the one move here that really does take
evidence away. It is the check that the measurement can see a change when there is one.

The prediction, if the law holds: grounding climbs with apparent size and the framing swing collapses, and
the CT measurements land on the left end of the same curve rather than beside it.
"""
import argparse, itertools, json, random
from collections import defaultdict
from pathlib import Path
from fh import COCO, RESULTS, load_any, read_yesno

import numpy as np
from PIL import Image, ImageDraw


ROOT = COCO


class Coco:
    """Just enough of the COCO API to cut objects out, without adding a dependency for it.

    Polygon segmentations rasterise with PIL; crowd regions are run-length encoded and are skipped here
    anyway, since a crowd has no single object to paste."""

    def __init__(self, path, split):
        d = json.loads(Path(path).read_text())
        here = set(x.name for x in (ROOT / split).iterdir())   # only part of the set was ever extracted
        self.split = split
        self.imgs = {i["id"]: i for i in d["images"] if i["file_name"] in here}
        self.cats = {c["name"]: c["id"] for c in d["categories"]}
        self.anns = [x for x in d["annotations"] if not x.get("iscrowd") and x["image_id"] in self.imgs]
        self.by_cat, self.cats_of_img = defaultdict(list), defaultdict(set)
        for x in self.anns:
            self.by_cat[x["category_id"]].append(x)
            self.cats_of_img[x["image_id"]].add(x["category_id"])

    def mask(self, ann):
        im = self.imgs[ann["image_id"]]
        m = Image.new("L", (im["width"], im["height"]), 0)
        d = ImageDraw.Draw(m)
        for poly in ann["segmentation"]:
            if len(poly) >= 6:
                d.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)], fill=255)
        return m

    def path(self, img_id):
        return ROOT / self.split / self.imgs[img_id]["file_name"]
PHRASINGS = {"isthere": "Is there a {c} in this image? Answer Yes or No.",
             "contain": "Does this image contain a {c}? Answer Yes or No."}
COPIES = [1, 2]
FRAC = [0.0005, 0.002, 0.01, 0.05, 0.2]


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])) if len(pos) and len(neg) else float("nan")


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--per-frac", type=int, default=60, help="composites per apparent size")
ap.add_argument("--side", type=int, default=448)
ap.add_argument("--lowres", type=int, default=224)
ap.add_argument("--cats", default="dog,car,clock,bird,cat,bus")
ap.add_argument("--split", default="train2017")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args()

rng = random.Random(a.seed)
coco = Coco(ROOT / "annotations" / f"instances_{a.split}.json", a.split)
print(f"{len(coco.imgs)} images on disk, {len(coco.anns)} usable annotations", flush=True)
cats = {c: coco.cats[c] for c in a.cats.split(",")}
model, proc = load_any(a.model)

# a cut-out needs a mask that is mostly object, or the paste carries its old background with it
cutouts = defaultdict(list)
for c, cid in cats.items():
    for ann in coco.by_cat[cid]:
        x, y, w, h = (int(v) for v in ann["bbox"])
        if w < 60 or h < 60 or ann["area"] < 8000:
            continue
        m = np.array(coco.mask(ann))[y:y + h, x:x + w]
        if m.size == 0 or m.mean() / 255 < 0.55:
            continue
        cutouts[c].append((ann, (x, y, w, h)))
    rng.shuffle(cutouts[c])

free = {c: [i for i in coco.imgs if cid not in coco.cats_of_img[i]] for c, cid in cats.items()}
for c in free:
    rng.shuffle(free[c])


def load(img_id):
    return Image.open(coco.path(img_id)).convert("RGB")


def composite(c, k, frac, side):
    """Paste one cut-out object onto a background with no instance of its category, scaled so the object
    covers `frac` of the frame. The background is the counterfactual: the same pixels, no object."""
    if k >= len(cutouts[c]) or k >= len(free[c]):
        return None
    ann, (x, y, w, h) = cutouts[c][k]
    src = load(ann["image_id"])
    m = coco.mask(ann)
    obj, om = src.crop((x, y, x + w, y + h)), m.crop((x, y, x + w, y + h))
    bg = load(free[c][k]).resize((side, side), Image.BICUBIC)
    target = frac * side * side
    s = float(np.sqrt(target / (w * h)))
    ow, oh = max(int(w * s), 2), max(int(h * s), 2)
    if ow >= side or oh >= side:
        return None
    obj, om = obj.resize((ow, oh), Image.BICUBIC), om.resize((ow, oh), Image.BILINEAR)
    px = rng.randint(0, side - ow); py = rng.randint(0, side - oh)
    out = bg.copy(); out.paste(obj, (px, py), om)
    return out, bg, free[c][k]


frames = list(itertools.product(PHRASINGS, COPIES, [a.side, a.lowres]))
rows = []
for frac in FRAC:
    made = 0
    for k in range(a.per_frac * 3):
        if made >= a.per_frac:
            break
        c = list(cats)[k % len(cats)]
        got = composite(c, k // len(cats), frac, a.side)
        if got is None:
            continue
        real, bg, bg_id = got
        # a pasted object leaves a seam, and a model answering Yes to the seam rather than to the dog would
        # look identical in the pair above. So the same composite is also asked about a category that is in
        # neither the object nor the background: if the seam were doing the work, that would answer Yes too.
        other = [w for w in cats if w != c and cats[w] not in coco.cats_of_img[bg_id]]
        w = rng.choice(other) if other else None
        r = {"cat": c, "frac": frac, "wrong_cat": w, "f": {}}
        for ph, cp, res in frames:
            ims = lambda im: [im.resize((res, res), Image.BICUBIC)] * cp
            e = {"real": read_yesno(model, proc, ims(real), PHRASINGS[ph].format(c=c)),
                 "cf": read_yesno(model, proc, ims(bg), PHRASINGS[ph].format(c=c))}
            if w:
                e["wrong"] = read_yesno(model, proc, ims(real), PHRASINGS[ph].format(c=w))
            r["f"][f"{ph}|{cp}|{res}"] = e
        rows.append(r); made += 1
    print(f"  frac={frac:.4f}: {made} composites", flush=True)

R = {"model": a.model, "n": len(rows), "fracs": FRAC, "framings": len(frames), "by_frac": {}}
for frac in FRAC:
    v = [r for r in rows if r["frac"] == frac]
    if not v:
        continue
    per = {}
    for key in v[0]["f"]:
        pr = [r["f"][key]["real"] for r in v]
        pc = [r["f"][key]["cf"] for r in v]
        pw = [r["f"][key]["wrong"] for r in v if "wrong" in r["f"][key]]
        per[key] = {"says_yes_real": float(np.mean([x > 0 for x in pr])),
                    "says_yes_cf": float(np.mean([x > 0 for x in pc])),
                    "says_yes_wrong": float(np.mean([x > 0 for x in pw])) if pw else float("nan"),
                    "auroc": auroc(pr, pc),
                    "auroc_vs_wrong": auroc(pr, pw) if pw else float("nan")}
    # the framings that leave the picture alone: phrasing and duplication, at full resolution
    keep = {k: p for k, p in per.items() if k.endswith(f"|{a.side}")}
    y = [p["says_yes_real"] for p in keep.values()]
    R["by_frac"][str(frac)] = {
        "n": len(v), "apparent_pct": frac * 100,
        "grounding_auroc": float(np.mean([p["auroc"] for p in keep.values()])),
        "says_yes_real": float(np.mean(y)), "frame_swing_pp": (max(y) - min(y)) * 100,
        "finding_effect_pp": float(np.mean([p["says_yes_real"] - p["says_yes_cf"] for p in keep.values()])) * 100,
        "lowres_auroc": float(np.mean([p["auroc"] for k, p in per.items() if k.endswith(f"|{a.lowres}")])),
        "auroc_vs_wrong_category": float(np.mean([p["auroc_vs_wrong"] for p in keep.values()])),
        "says_yes_wrong_category": float(np.mean([p["says_yes_wrong"] for p in keep.values()])),
        "per_framing": per}
    b = R["by_frac"][str(frac)]
    b["frame_over_finding"] = b["frame_swing_pp"] / max(abs(b["finding_effect_pp"]), 1e-9)

print(f"\n{'object covers':>14}{'grounding':>11}{'vs wrong cat':>14}{'frame swing':>13}"
      f"{'finding effect':>16}{'frame/finding':>15}{'lowres':>9}")
for frac in FRAC:
    b = R["by_frac"].get(str(frac))
    if b:
        print(f"{b['apparent_pct']:13.3f}%{b['grounding_auroc']:11.3f}{b['auroc_vs_wrong_category']:14.3f}"
              f"{b['frame_swing_pp']:12.1f}pp{b['finding_effect_pp']:15.1f}pp"
              f"{b['frame_over_finding']:14.1f}x{b['lowres_auroc']:9.3f}")
print("\nCT, same control, same question type: tumour covers 0.02% -> grounding 0.484; "
      "0.07% -> 0.523; 0.19% -> 0.620; 0.90% -> 0.655")

stem = str(RESULTS / (a.out or f"natural_vs_medical_{a.model.split('/')[-1]}"))
Path(stem + ".json").write_text(json.dumps(R, indent=1) + "\n")
Path(stem + ".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
