"""Ways of presenting the same evidence, none of which changes what is in the picture.

A presentation is a crop, an input resolution and a wording. Changing any of them moved a model's rate of
answering Yes across the whole range from 2.8% to 96.4% while its ability to tell a slice from the same
slice with the tumour erased never moved. That gap is what the framework exploits: the presentation is a
nuisance we control, so we can average over it.

Crops stay at or above 85% of the frame deliberately. A tighter crop would start removing content, and then
it is no longer a presentation of the same evidence but a different image.
"""
from __future__ import annotations

import itertools

from PIL import Image

CROPS = (1.00, 0.92, 0.85)
RESOLUTIONS = (448, 336)
PHRASINGS = ("{q} Answer Yes or No.", "Question: {q}\nAnswer with Yes or No.")

PRESENTATIONS = [f"{c}|{r}|{p}" for c, r, p in itertools.product(CROPS, RESOLUTIONS, range(len(PHRASINGS)))]


def parse(key: str):
    c, r, p = key.split("|")
    return float(c), int(r), int(p)


def present(img: Image.Image, key: str) -> Image.Image:
    """The image under one presentation. The wording is applied by `question`."""
    crop, res, _ = parse(key)
    w, h = img.size
    cw, ch = int(w * crop), int(h * crop)
    x, y = (w - cw) // 2, (h - ch) // 2
    return img.crop((x, y, x + cw, y + ch)).convert("RGB").resize((res, res), Image.BICUBIC)


def question(q: str, key: str) -> str:
    return PHRASINGS[parse(key)[2]].format(q=q)
