#!/usr/bin/env bash
# What each experiment needs and where it expects to find it. Nothing is redistributed here: the Decathlon
# and COCO are downloaded from their own sources under their own terms.
#
#   $FH_DATA/MSD/Task06_Lung/{imagesTr,labelsTr}   medical-decathlon.com (also Task03/07/10)
#   $FH_DATA/coco/{train2017,annotations}          cocodataset.org
#   VQA-RAD, PathVQA, SLAKE                        Hugging Face, fetched below into $HF_HOME
#
# Set FH_DATA if the data lives elsewhere; see src/fh/config.py.
set -euo pipefail
: "${FH_DATA:=data}"
mkdir -p "$FH_DATA"

echo "== Medical Segmentation Decathlon"
echo "   Register at http://medicaldecathlon.com and unpack the task folders under $FH_DATA/MSD/."
echo "   Only imagesTr/ and labelsTr/ are read. Task06_Lung is enough for everything except"
echo "   experiments/matched_counterpart.py --task Task03_Liver|Task07_Pancreas|Task10_Colon."

echo "== COCO 2017 (only for experiments/natural_vs_medical.py)"
if [ -d "$FH_DATA/coco/annotations" ]; then
  echo "   already present"
else
  mkdir -p "$FH_DATA/coco"
  curl -L -o "$FH_DATA/coco/ann.zip" http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -q -o "$FH_DATA/coco/ann.zip" -d "$FH_DATA/coco" && rm "$FH_DATA/coco/ann.zip"
  curl -L -o "$FH_DATA/coco/train2017.zip" http://images.cocodataset.org/zips/train2017.zip
  unzip -q -o "$FH_DATA/coco/train2017.zip" -d "$FH_DATA/coco" && rm "$FH_DATA/coco/train2017.zip"
fi

echo "== Medical VQA sets (for experiments/marginalise_vqa.py)"
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("flaviagiammarino/vqa-rad", "flaviagiammarino/path-vqa", "BoKelvin/SLAKE"):
    try:
        print("  OK  ", repo, "->", snapshot_download(repo_id=repo, repo_type="dataset", max_workers=2))
    except Exception as e:
        print("  FAIL", repo, f"{type(e).__name__}: {str(e)[:160]}")
PY
