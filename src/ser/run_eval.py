import os, sys, json, time
import numpy as np
import pandas as pd
from ser.common import FEATURES, RESULTS, load_meta, result
from ser.probe import C_LR, run, scores, speaker_norm, permute_within

meta = load_meta()
y, g = meta["label"].values, meta["actor"].values
feat_dir, res_dir = FEATURES, RESULTS
os.makedirs(res_dir, exist_ok=True)
EMOBOX = [[1, 11, 20, 21], [4, 10, 12, 14], [8, 9, 19, 23], [3, 5, 7, 24], [2, 15, 16, 18], [6, 13, 17, 22]]
HAND = ["handcrafted", "egemaps", "compare"]
T0 = time.time()


def save(name, r, outer="loso", inner="group", labels=None):
    lab = y if labels is None else labels
    np.save(result(f"proba_{name}.npy"), r["proba"])
    np.save(result(f"inner_{name}.npy"), r["inner"])
    s = scores(lab, r["proba"])
    s.update(name=name, outer=outer if isinstance(outer, str) else "emobox6", inner=inner,
             chosen=[[None if l is None else int(l), c] for l, c in r["choice"]],
             not_converged=int(sum(not ok for ok in r["converged"])))
    with open(os.path.join(res_dir, "summary.jsonl"), "a") as f:
        f.write(json.dumps(s) + "\n")
    print(f"{name:34s} UAR {s['uar']:.4f}  acc {s['acc']:.4f}  F1 {s['f1']:.4f}  not_converged {s['not_converged']}  t={time.time() - T0:.0f}s", flush=True)


def table(name):
    df = pd.read_csv(os.path.join(feat_dir, f"{name}.csv"))
    assert (df["file"].values == meta["file"].values).all()
    return np.nan_to_num(df.drop(columns="file").values.astype(np.float64))


def ssl(name, pooling="mean"):
    X = np.load(os.path.join(feat_dir, f"{name}.npy"), mmap_mode="r")
    return np.ascontiguousarray(X[:, :, :X.shape[2] // 2]) if pooling == "mean" else np.array(X)


def rep(name):
    if name in HAND:
        return table(name), None
    X = ssl(name)
    return X, list(range(X.shape[1]))


def nested_layers(name):
    r = [json.loads(l) for l in open(os.path.join(res_dir, "summary.jsonl"))]
    return [l for l, c in [x for x in r if x["name"] == name][-1]["chosen"]]


step = sys.argv[1]

if step == "handcrafted":
    for feat in HAND:
        X = table(feat)
        for kind in ["lr", "svm", "rf"]:
            save(f"{feat}_{kind}", run(X, y, g, kind))

if step == "ssl":
    name = sys.argv[2]
    X, layers = rep(name)
    r = run(X, y, g, "lr", layers=layers, per_layer=True)
    save(f"{name}_nested", r)
    curve = [scores(y, r["layer_proba"][l])["uar"] for l in layers]
    for l in layers:
        np.save(result(f"layer_{name}_{l:02d}.npy"), r["layer_proba"][l])
    np.save(result(f"curve_{name}.npy"), np.array(curve))
    print(name, "per-layer LOSO UAR", [round(v, 4) for v in curve], "layer fits not converged", sum(not ok for ok in r["layer_converged"]), flush=True)
    L = len(layers)
    last = dict(proba=r["layer_proba"][L - 1], inner=r["inner"][:, L - 1:, :],
                choice=[(L - 1, C_LR[int(np.argmax(r["inner"][f, L - 1]))]) for f in range(len(r["choice"]))],
                converged=r["layer_converged"][L - 1::L])
    save(f"{name}_last", last)
    save(f"{name}_avg", run(X.mean(1), y, g, "lr"))

if step == "e2v":
    for name in sys.argv[2:]:
        save(f"{name}_lr", run(np.load(os.path.join(feat_dir, f"{name}.npy")), y, g, "lr"))

if step == "emobox":
    fold = np.zeros(len(g), int)
    for k, acts in enumerate(EMOBOX):
        fold[np.isin(g, acts)] = k
    for name in sys.argv[2:]:
        save(f"{name}_last_emobox", run(ssl(name)[:, -1], y, g, "lr", outer=fold), outer=fold)

if step == "best":
    name = sys.argv[2]
    X, _ = rep(name)
    fl = nested_layers(f"{name}_nested")
    for kind in ["svm", "rf"]:
        save(f"{name}_nested_{kind}", run(X, y, g, kind, fold_layers=fl))
    save(f"{name}_nested_egemaps_fusion", run(X, y, g, "lr", fold_layers=fl, extra=table("egemaps")))
    save(f"{name}_nested_meanstd", run(ssl(name, "meanstd"), y, g, "lr", fold_layers=fl))

if step == "inflation":
    for name in sys.argv[2:]:
        X, layers = rep(name)
        save(f"{name}_random5", run(X, y, g, "lr", outer="random5", inner="random", layers=layers), outer="random5", inner="random")
        save(f"{name}_spknorm", run(speaker_norm(X, g), y, g, "lr", layers=layers))

if step == "perm":
    name = sys.argv[2]
    X, layers = rep(name)
    for seed in [int(s) for s in sys.argv[3:]]:
        yp = permute_within(y, g, seed)
        np.save(result(f"labels_{name}_perm_s{seed}.npy"), yp)
        save(f"{name}_perm_s{seed}", run(X, yp, g, "lr", layers=layers), labels=yp)
