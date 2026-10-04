"""Four tests of whether the harness logic holds in a medical setting.

The single-view probe left a specific picture: the model's evidence about a lesion is worth about 0.9
logits while its prior against naming one is worth about 3, so the answer never crosses. A harness cannot
change the prior, but it can change how much evidence reaches one decision. That gives four predictions,
each of which can fail.

  E1 accumulation   AUROC rises with the number of views the decision is made over. If it does not, there
                    is nothing for a harness to accumulate and the idea is dead.
  E2 selection      at a fixed budget, views chosen to contain the lesion beat views spread uniformly. If
                    they do not, raise the budget and skip the agent: which slices does not matter.
  E3 comparison     asking which of two images is more abnormal beats asking whether one is abnormal. The
                    first reads the ranking, which carries signal; the second reads the threshold, which
                    does not.
  E4 rendering      three windows of one slice carry evidence the way three slices do. If so, rendering is
                    a second independent axis for a harness to act on, not a cosmetic choice.

AUROC throughout, because accuracy on these probes measures the threshold rather than the capability.
"""
import argparse, json
from pathlib import Path
from fh import MSD, RESULTS, load_any, read_yesno, render

import nibabel as nib
import numpy as np


Q = "Is there an abnormality in the lungs in these images? Answer Yes or No."
Q1 = "Is there an abnormality in the lungs in this image? Answer Yes or No."
WINDOWS = {"lung": (-600.0, 1500.0), "soft": (50.0, 400.0), "bone": (400.0, 1800.0)}


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def cases(limit):
    for lab in sorted((MSD / "labelsTr").glob("lung_*.nii.gz"))[:limit]:
        m = np.asanyarray(nib.load(str(lab)).dataobj)
        if m.max() == 0:
            continue
        v = np.asanyarray(nib.load(str(MSD / "imagesTr" / lab.name)).dataobj)
        Z = m.shape[2]
        on = np.nonzero(m.reshape(-1, Z).sum(axis=0))[0].tolist()
        clean = [z for z in range(int(Z * 0.2), int(Z * 0.8)) if min(abs(z - o) for o in on) > 30]
        if len(on) < 3 or len(clean) < 12:
            continue
        yield lab.name.split(".")[0], v, on, clean, Z


def pick(xs, n):
    return [int(xs[i]) for i in np.linspace(0, len(xs) - 1, n).round().astype(int)]


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--cases", type=int, default=63)
ap.add_argument("--out", default=str(RESULTS / "harness_logic.json"))
a = ap.parse_args()

model, proc = load_any(a.model)
R = {"E1": {}, "E2": {}, "E3": {}, "E4": {}}
rec = []

# ---- E1 accumulation, and E2 selection at budget 5
acc_pos = {n: [] for n in (1, 3, 5, 10)}
acc_neg = {n: [] for n in (1, 3, 5, 10)}
sel_focus, sel_uniform = [], []
for case, vol, on, clean, Z in cases(a.cases):
    for n in (1, 3, 5, 10):
        p = [render(vol, z) for z in pick(on, n)]
        q = [render(vol, z) for z in pick(clean, n)]
        mp, mq = read_yesno(model, proc, p, Q if n > 1 else Q1), read_yesno(model, proc, q, Q if n > 1 else Q1)
        acc_pos[n].append(mp); acc_neg[n].append(mq)
        rec.append({"exp": "E1", "case": case, "n": n, "lesion": mp, "clean": mq})
    u = pick(list(range(Z)), 5)
    sel_uniform.append(read_yesno(model, proc, [render(vol, z) for z in u], Q))
    sel_focus.append(acc_pos[5][-1])
    rec.append({"exp": "E2", "case": case, "focused": sel_focus[-1], "uniform": sel_uniform[-1],
                "uniform_hits_lesion": bool(set(u) & set(on))})
    print(f"  {case} done", flush=True)

for n in (1, 3, 5, 10):
    R["E1"][n] = {"auroc": auroc(acc_pos[n], acc_neg[n]), "n_cases": len(acc_pos[n]),
                  "mean_lesion": float(np.mean(acc_pos[n])), "mean_clean": float(np.mean(acc_neg[n]))}
R["E2"] = {"auroc_focused_vs_clean": auroc(sel_focus, acc_neg[5]),
           "auroc_uniform_vs_clean": auroc(sel_uniform, acc_neg[5]),
           "mean_focused": float(np.mean(sel_focus)), "mean_uniform": float(np.mean(sel_uniform))}

# ---- E3 comparison beats assertion
corr = tot = 0
for i, (case, vol, on, clean, Z) in enumerate(cases(a.cases)):
    les, cln = render(vol, int(on[len(on) // 2])), render(vol, int(clean[len(clean) // 2]))
    first_is_lesion = i % 2 == 0
    imgs = [les, cln] if first_is_lesion else [cln, les]
    q = ("Two axial CT slices are shown. Is the FIRST image more abnormal than the second? "
         "Answer Yes or No.")
    m = read_yesno(model, proc, imgs, q)
    ok = (m > 0) == first_is_lesion
    corr += ok; tot += 1
    rec.append({"exp": "E3", "case": case, "first_is_lesion": first_is_lesion, "margin": m, "correct": bool(ok)})
R["E3"] = {"accuracy": corr / tot, "n": tot}

# ---- E4 rendering as a second evidence axis, three images either way
w_pos, w_neg, s_pos, s_neg = [], [], [], []
for case, vol, on, clean, Z in cases(a.cases):
    zl, zc = int(on[len(on) // 2]), int(clean[len(clean) // 2])
    w_pos.append(read_yesno(model, proc, [render(vol, zl, wl=w, ww=t) for w, t in WINDOWS.values()], Q))
    w_neg.append(read_yesno(model, proc, [render(vol, zc, wl=w, ww=t) for w, t in WINDOWS.values()], Q))
    s_pos.append(read_yesno(model, proc, [render(vol, z) for z in pick(on, 3)], Q))
    s_neg.append(read_yesno(model, proc, [render(vol, z) for z in pick(clean, 3)], Q))
    rec.append({"exp": "E4", "case": case, "win_lesion": w_pos[-1], "win_clean": w_neg[-1],
                "sli_lesion": s_pos[-1], "sli_clean": s_neg[-1]})
R["E4"] = {"auroc_three_windows_one_slice": auroc(w_pos, w_neg),
           "auroc_three_slices_one_window": auroc(s_pos, s_neg), "n_cases": len(w_pos)}

out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(R, indent=1) + "\n")
Path(str(out).replace(".json", ".jsonl")).write_text("\n".join(json.dumps(r) for r in rec) + "\n")
print("\n" + json.dumps(R, indent=1))
