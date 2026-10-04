"""Loading any family behind one interface, and reading a yes/no answer out of it as a margin.

The margin is log P(yes) - log P(no) at the answer position, summed over the obvious spellings. It is the
quantity everything else in this repository operates on: the framework's claim is that this margin is a
presentation-dependent offset plus an image-dependent term, and the operators remove the first.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

LUNG_WL, LUNG_WW = -600.0, 1500.0
SOFT_WL, SOFT_WW = 50.0, 400.0


def load_any(name: str, device: str = "cuda:0", pixels: int = 448 * 448):
    """The image-text-to-text auto class covers Qwen2.5/3-VL, Gemma 3, InternVL's HF port and LLaVA, and
    each ships the chat template its own readout needs."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
    ip = getattr(proc, "image_processor", None)
    for attr in ("min_pixels", "max_pixels"):
        if hasattr(ip, attr):
            setattr(ip, attr, pixels)
    model = AutoModelForImageTextToText.from_pretrained(
        name, dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True, attn_implementation="sdpa").eval()
    return model, proc


def read_yesno(model, proc, images, question: str) -> float:
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


def render(vol: np.ndarray, z: int, side: int = 512, wl: float = LUNG_WL, ww: float = LUNG_WW) -> Image.Image:
    """One axial slice, windowed and rotated the way a radiologist views it."""
    lo, hi = wl - ww / 2, wl + ww / 2
    sl = np.clip((vol[:, :, z].astype(np.float32) - lo) / (hi - lo), 0, 1)
    img = Image.fromarray((np.rot90(sl) * 255).astype(np.uint8)).convert("RGB")
    return img.resize((side, side), Image.BILINEAR)
