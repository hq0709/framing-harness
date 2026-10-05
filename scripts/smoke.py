"""Two minutes on a GPU, to find out whether this machine can run the rest.

It loads one model, reads one synthetic image under every presentation, and prints the margins. What it is
really checking is the thing that breaks first on a new machine: whether the chat template, the processor
and the yes/no token ids line up for that family, so the readout returns a number rather than a traceback.

The spread it prints across presentations is the phenomenon in miniature -- the same picture, a different
frame, a different answer.

    python scripts/smoke.py                                   # default model
    python scripts/smoke.py --model google/medgemma-4b-it
"""
import argparse

import numpy as np
from PIL import Image

from fh import PRESENTATIONS, load_any, read_yesno
from fh.presentations import present, question

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()

rng = np.random.default_rng(0)
img = Image.fromarray((rng.normal(128, 30, (512, 512, 3)).clip(0, 255)).astype(np.uint8))
img.paste(Image.new("RGB", (60, 60), (230, 230, 230)), (200, 220))   # one bright blob to ask about

print(f"loading {a.model} on {a.device} ...", flush=True)
model, proc = load_any(a.model, device=a.device)
q = "Is there an abnormality in this image?"

margins = []
for key in PRESENTATIONS:
    m = read_yesno(model, proc, [present(img, key)], question(q, key))
    margins.append(m)
    print(f"  {key:<16} margin {m:+7.3f}   says {'yes' if m > 0 else 'no'}")

m = np.array(margins)
print(f"\n{len(m)} presentations of one image: margin {m.min():+.2f} to {m.max():+.2f}, "
      f"says yes {int((m > 0).sum())}/{len(m)} times")
print(f"mean margin (what `marginalise` decides on): {m.mean():+.3f}")
print("\nreadout works" if np.isfinite(m).all() else "\nnon-finite margins -- the readout is wrong for this family")
