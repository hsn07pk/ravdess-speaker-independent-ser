import os, sys, json, glob
import numpy as np
from ser.common import RESULTS, load_meta, result
from ser.probe import scores

src = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 else "wavlm_large_ft"
meta = load_meta()
y = meta["label"].values
pos = {f: i for i, f in enumerate(meta["file"])}
proba = np.full((len(meta), 8), np.nan)
folds = []
for jf in sorted(glob.glob(os.path.join(src, "fold_*.json"))):
    r = json.load(open(jf))
    p = np.load(os.path.join(src, f"proba_fold_{r['fold']:02d}.npy"))
    for f, row in zip(r["files"], p):
        proba[pos[f]] = row
    folds.append(r)
assert len(folds) == 24 and not np.isnan(proba).any(), (len(folds), int(np.isnan(proba).any(1).sum()))
res = RESULTS
np.save(result(f"proba_{name}.npy"), proba)
s = scores(y, proba)
s.update(name=name, outer="loso", inner="epoch picked on 2 validation actors", not_converged=0, kind="finetune",
         chosen=[[r["best_epoch"], r["best_val_uar"]] for r in folds])
with open(os.path.join(res, "summary.jsonl"), "a") as f:
    f.write(json.dumps(s) + "\n")
print(f"{name}: UAR {s['uar']:.4f} acc {s['acc']:.4f} F1 {s['f1']:.4f}; best epochs {[r['best_epoch'] for r in folds]}; minutes/fold {np.mean([r['minutes'] for r in folds]):.1f}")
