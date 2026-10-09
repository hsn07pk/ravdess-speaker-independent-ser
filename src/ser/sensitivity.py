import os, json, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ser.common import FEATURES, RESULTS, load_meta, result
from ser.probe import C_LR, JOBS, run, scores, model, rows, outer_folds, speaker_norm

meta = load_meta()
y, g = meta["label"].values, meta["actor"].values
res, feat = RESULTS, FEATURES
summ = {}
for line in open(os.path.join(res, "summary.jsonl")):
    r = json.loads(line)
    summ[r["name"]] = r
P = lambda n: np.load(result(f"proba_{n}.npy"))
uar = lambda p: scores(y, p)["uar"]
T0 = time.time()


def table(name):
    df = pd.read_csv(os.path.join(feat, f"{name}.csv"))
    assert (df["file"].values == meta["file"].values).all()
    return np.nan_to_num(df.drop(columns="file").values.astype(np.float64))


def ssl(name, pooling="mean"):
    X = np.load(os.path.join(feat, f"{name}.npy"), mmap_mode="r")
    return np.ascontiguousarray(X[:, :, :X.shape[2] // 2]) if pooling == "mean" else np.array(X)


layers_of = lambda n: [l for l, c in summ[n]["chosen"]]
out = []

r = run(ssl("wavlm_large"), y, g, "lr", layers=list(range(25)))
same = np.array_equal(r["proba"], P("wavlm_large_nested")) and [list(x) for x in r["choice"]] == summ["wavlm_large_nested"]["chosen"]
print(f"reproduce wavlm_large_nested with the current probe.py: bit-identical = {same}  t={time.time() - T0:.0f}s", flush=True)
out.append(dict(check="reproduce", system="wavlm_large_nested", uar_v2=summ["wavlm_large_nested"]["uar"], uar_alt=uar(r["proba"]),
                delta=uar(r["proba"]) - summ["wavlm_large_nested"]["uar"], folds_changed=int(sum(a != tuple(b) for a, b in zip(r["choice"], summ["wavlm_large_nested"]["chosen"]))),
                file="", note=f"bit-identical {same}"))

sel = pd.read_csv(os.path.join(res, "selection.csv"))
edge = sorted(sel[(sel.at_high_edge > 0) & ~sel.system.str.contains("_perm_s")].system)
EXT = C_LR + [1000.0]
cases = {
    "hubert_large_last": lambda: run(ssl("hubert_large")[:, -1], y, g, "lr", grid=EXT),
    "hubert_large_avg": lambda: run(ssl("hubert_large").mean(1), y, g, "lr", grid=EXT),
    "e2v_base_lr": lambda: run(np.load(os.path.join(feat, "e2v_base.npy")), y, g, "lr", grid=EXT),
    "wavlm_large_nested_meanstd": lambda: run(ssl("wavlm_large", "meanstd"), y, g, "lr", fold_layers=layers_of("wavlm_large_nested"), grid=EXT),
    "wavlm_large_random5": lambda: run(ssl("wavlm_large"), y, g, "lr", outer="random5", inner="random", layers=list(range(25)), grid=EXT),
    "hubert_large_random5": lambda: run(ssl("hubert_large"), y, g, "lr", outer="random5", inner="random", layers=list(range(25)), grid=EXT),
    "compare_spknorm": lambda: run(speaker_norm(table("compare"), g), y, g, "lr", grid=EXT),
}
assert sorted(cases) == edge, (sorted(cases), edge)
for n in edge:
    r = cases[n]()
    f = f"sens_grid1e3_{n}.npy"
    np.save(result(f), r["proba"])
    k = int(sum(c == 1000.0 for _, c in r["choice"]))
    out.append(dict(check="lr_grid_to_1000", system=n, uar_v2=summ[n]["uar"], uar_alt=uar(r["proba"]), delta=uar(r["proba"]) - summ[n]["uar"],
                    folds_changed=k, file=f, note="folds choosing C = 1000"))
    print(f"{n:30s} LR grid to 1000: UAR {summ[n]['uar']:.4f} -> {uar(r['proba']):.4f}; folds choosing 1000: {k}  t={time.time() - T0:.0f}s", flush=True)


def svm_fold(X, tr, te, l, c):
    m = model("svm", c).fit(rows(X, tr, l, None), y[tr])
    Z = rows(X, te, l, None)
    return te, m.predict(Z), m.predict_proba(Z)


for n, X in [("handcrafted_svm", table("handcrafted")), ("egemaps_svm", table("egemaps")), ("compare_svm", table("compare")), ("wavlm_large_nested_svm", ssl("wavlm_large"))]:
    folds = outer_folds(y, g, "loso")
    jobs = Parallel(n_jobs=JOBS)(delayed(svm_fold)(X, tr, te, l, c) for (tr, te), (l, c) in zip(folds, summ[n]["chosen"]))
    pred, proba = np.zeros(len(y), int), np.zeros((len(y), 8))
    for te, p, pr in jobs:
        pred[te], proba[te] = p, pr
    f = f"sens_svm_predict_{n}.npy"
    np.save(result(f), pred)
    u = float(np.mean([np.mean(pred[y == k] == k) for k in range(8)]))
    same = np.array_equal(proba, P(n))
    out.append(dict(check="svm_predict_vs_proba", system=n, uar_v2=summ[n]["uar"], uar_alt=u, delta=u - summ[n]["uar"],
                    folds_changed=int((pred != P(n).argmax(1)).sum()), file=f, note=f"clips whose label differs; stored probabilities reproduced bit-identically {same}"))
    print(f"{n:30s} SVM predict(): UAR {summ[n]['uar']:.4f} -> {u:.4f}; clips that differ {(pred != P(n).argmax(1)).sum()}; stored proba reproduced {same}  t={time.time() - T0:.0f}s", flush=True)

pd.DataFrame(out).to_csv(os.path.join(res, "sensitivity.csv"), index=False, float_format="%.10g")
print("sensitivity.csv written")
