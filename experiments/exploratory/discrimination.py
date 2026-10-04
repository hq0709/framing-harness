"""Does the model's answer move at all between a slice with the lesion and one without?

The laterality probe found an answer that does not depend on the image, and the sanity check ruled out the
renderer and the readout: the model describes the anatomy correctly and says yes to other questions about
the same slice. What is left to separate is whether it cannot see the lesion or will not name it. Same
volume, same window, same size: one slice through the middle of the tumour, one well away from it, and
several wordings from the most clinical to the most permissive. If no wording separates the two, the
capability is absent and no viewport can supply it.
"""
import argparse
import json
from pathlib import Path
import nibabel as nib, numpy as np
from fh import MSD, RESULTS, load_any, read_yesno, render

QUESTIONS = {
    "tumour":      "Is there a lung tumour in this image? Answer Yes or No.",
    "mass":        "Is there a mass in the lung in this image? Answer Yes or No.",
    "nodule":      "Is there a pulmonary nodule in this image? Answer Yes or No.",
    "abnormal":    "Is there an abnormality in the lungs in this image? Answer Yes or No.",
    "not_normal":  "Are the lungs in this image normal? Answer Yes or No.",      # sign flips
}
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--cases", type=int, default=20)
a = ap.parse_args()
N = a.cases

model, proc = load_any(a.model)

rows = []
for lab_path in sorted((MSD / "labelsTr").glob("lung_*.nii.gz"))[:N]:
    mask = np.asanyarray(nib.load(str(lab_path)).dataobj)
    if mask.max() == 0:
        continue
    vol = np.asanyarray(nib.load(str(MSD / "imagesTr" / lab_path.name)).dataobj)
    Z = mask.shape[2]
    on = np.nonzero(mask.reshape(-1, Z).sum(axis=0))[0]
    z_les = int(on[len(on) // 2])
    far = [z for z in range(int(Z * 0.25), int(Z * 0.75)) if abs(z - z_les) > 40 and z not in set(on.tolist())]
    if not far:
        continue
    z_cln = int(far[len(far) // 2])
    for arm, z in (("lesion", z_les), ("clean", z_cln)):
        img = render(vol, z)
        for key, q in QUESTIONS.items():
            rows.append({"case": lab_path.name.split(".")[0], "arm": arm, "z": z,
                         "question": key, "margin": round(read_yesno(model, proc, [img], q), 4)})
            print(json.dumps(rows[-1]), flush=True)

out = Path(str(RESULTS / "discrimination.jsonl"))
out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(f"\nwrote {len(rows)} probes")
