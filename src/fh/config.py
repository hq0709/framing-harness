"""Every path this repository reads, taken from the environment so it runs somewhere else.

Defaults are relative to the working directory, so a clone with `data/` and `results/` beside it works with
no configuration at all. Set the variables when the data lives elsewhere:

    FH_DATA      parent of MSD/ and coco/            (default: ./data)
    FH_MSD       the Medical Segmentation Decathlon  (default: $FH_DATA/MSD)
    FH_COCO      COCO with instances_*.json          (default: $FH_DATA/coco)
    FH_RESULTS   where runs are written              (default: ./results)
    HF_HOME      the Hugging Face cache              (standard variable, models and VQA sets)
"""
import os
from pathlib import Path

DATA = Path(os.environ.get("FH_DATA", "data")).expanduser()
MSD = Path(os.environ.get("FH_MSD", DATA / "MSD")).expanduser()
COCO = Path(os.environ.get("FH_COCO", DATA / "coco")).expanduser()
RESULTS = Path(os.environ.get("FH_RESULTS", "results")).expanduser()
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")).expanduser()


def out(stem: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS / stem


def need(p: Path, what: str) -> Path:
    if not p.exists():
        raise SystemExit(f"{what} not found at {p}\nSet the matching FH_* variable; see src/fh/config.py "
                         f"and scripts/fetch_data.sh")
    return p
