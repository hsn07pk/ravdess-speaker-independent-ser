import os, json
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr, mannwhitneyu
from sklearn.metrics import balanced_accuracy_score
from ser.common import RESULTS, load_meta, result
from ser.probe import C_LR, C_SVM, scores, per_actor_uar, bootstrap_ci

meta = load_meta()
y, g = meta["label"].values, meta["actor"].values
res = RESULTS
rows = {}
for line in open(os.path.join(res, "summary.jsonl")):
    r = json.loads(line)
    rows[r["name"]] = r
SSL = ["wavlm_large", "xlsr_300m", "hubert_large", "whisper_large_v3"]
HAND = ["handcrafted", "egemaps", "compare"]
EMOBOX = [[1, 11, 20, 21], [4, 10, 12, 14], [8, 9, 19, 23], [3, 5, 7, 24], [2, 15, 16, 18], [6, 13, 17, 22]]
FMT = "%.10g"


def P(n):
    return np.load(result(f"proba_{n}.npy"))


def labels_for(n):
    if "_perm_s" in n:
        m, s = n.split("_perm_s")
        return np.load(result(f"labels_{m}_perm_s{s}.npy"))
    return y


def save(df, name):
    df.to_csv(os.path.join(res, name), index=False, float_format=FMT)
    print(f"\n== {name}\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


tab = []
for n in rows:
    lab, p = labels_for(n), P(n)
    s, pa = scores(lab, p), per_actor_uar(lab, p, g)
    lo, hi = bootstrap_ci(lab, p, g)
    tab.append(dict(system=n, uar=s["uar"], acc=s["acc"], f1=s["f1"], actor_sd=pa.std(ddof=1), ci_lo=lo, ci_hi=hi,
                    outer=rows[n]["outer"], inner=rows[n]["inner"], not_converged=rows[n]["not_converged"]))
save(pd.DataFrame(tab).sort_values("uar", ascending=False), "table_all.csv")

sel = []
for n, r in rows.items():
    if r.get("kind") == "finetune":
        continue
    ls = [l for l, c in r["chosen"] if l is not None]
    cs = [c for l, c in r["chosen"]]
    grid = C_LR if n.split("_")[-1] not in ("svm",) else C_SVM
    sel.append(dict(system=n, folds=len(cs), layers=json.dumps(sorted(Counter(ls).items())) if ls else "",
                    C=json.dumps(sorted(Counter(cs).items(), key=lambda t: (t[0] is None, t[0] or 0))),
                    at_low_edge=sum(c == grid[0] for c in cs), at_high_edge=sum(c == grid[-1] for c in cs)))
save(pd.DataFrame(sel), "selection.csv")

eb = []
for m in SSL:
    n = f"{m}_last_emobox"
    if n in rows:
        p = P(n).argmax(1)
        f = [balanced_accuracy_score(y[np.isin(g, s)], p[np.isin(g, s)]) for s in EMOBOX]
        eb.append(dict(system=n, pooled_uar=rows[n]["uar"], fold_mean_uar=np.mean(f), fold_sd=np.std(f, ddof=1)))
save(pd.DataFrame(eb), "emobox_anchor.csv")

inf = []
for m in SSL + HAND:
    base = f"{m}_nested" if m in SSL else f"{m}_lr"
    if base not in rows:
        continue
    d = dict(representation=m, loso_nested=rows[base]["uar"])
    if m in SSL:
        c = np.load(result(f"curve_{m}.npy"))
        d.update(best_layer_on_test=c.max(), best_layer=int(c.argmax()), last_layer=c[-1])
    for tag in ["random5", "spknorm"]:
        if f"{m}_{tag}" in rows:
            d[tag] = rows[f"{m}_{tag}"]["uar"]
    inf.append(d)
save(pd.DataFrame(inf), "inflation.csv")

cl = []
for m, names in [(f, [f"{f}_lr", f"{f}_svm", f"{f}_rf"]) for f in HAND] + \
        [("wavlm_large_nested", ["wavlm_large_nested", "wavlm_large_nested_svm", "wavlm_large_nested_rf"])]:
    d = dict(representation=m)
    for kind, n in zip(["lr", "svm", "rf"], names):
        if n in rows:
            d[kind] = rows[n]["uar"]
    cl.append(d)
save(pd.DataFrame(cl), "classifiers.csv")


def holm(p):
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


FAMILY = ([(f"{m}_nested", "compare_lr") for m in SSL] + [("e2v_base_lr", "compare_lr")]
          + [(f"{m}_nested", f"{m}_last") for m in SSL]
          + [(f"{a}_nested", f"{b}_nested") for i, a in enumerate(SSL) for b in SSL[i + 1:]]
          + [("wavlm_large_nested_egemaps_fusion", "wavlm_large_nested")]
          + [("compare_lr", "egemaps_lr"), ("compare_lr", "handcrafted_lr"), ("egemaps_lr", "handcrafted_lr")]
          + [("wavlm_large_nested", "wavlm_large_nested_svm"), ("wavlm_large_nested", "wavlm_large_nested_rf"),
             ("handcrafted_lr", "handcrafted_svm"), ("handcrafted_lr", "handcrafted_rf")])
out = []
for a, b in FAMILY:
    da, db = per_actor_uar(y, P(a), g), per_actor_uar(y, P(b), g)
    d = da - db
    out.append(dict(a=a, b=b, uar_a=rows[a]["uar"], uar_b=rows[b]["uar"], mean_diff=d.mean(), median_diff=np.median(d),
                    wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()), p=wilcoxon(da, db).pvalue))
cmp = pd.DataFrame(out)
cmp["p_holm"] = holm(cmp["p"].values)
save(cmp, "wilcoxon_holm.csv")

if "wavlm_large_ft" in rows:
    da, db = per_actor_uar(y, P("wavlm_large_ft"), g), per_actor_uar(y, P("wavlm_large_nested"), g)
    d = da - db
    save(pd.DataFrame([dict(a="wavlm_large_ft", b="wavlm_large_nested", uar_a=rows["wavlm_large_ft"]["uar"], uar_b=rows["wavlm_large_nested"]["uar"],
                            mean_diff=d.mean(), median_diff=np.median(d), wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                            ties=int((d == 0).sum()), p=wilcoxon(da, db).pvalue)]), "ft_vs_frozen.csv")

h = meta["human_acc"].values
tenths = np.rint(h * 10).astype(int)
BIN = np.select([tenths <= 3, tenths <= 6, tenths <= 9], ["0-30%", "40-60%", "70-90%"], "100%")
y7 = np.where(y == 1, 0, y)
EMO7 = {0: "neutral/calm", 2: "happy", 3: "sad", 4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"}


def pred7(p):
    q = p.copy()
    q[:, 0] += q[:, 1]
    q[:, 1] = -1.0
    return q.argmax(1)


hm, hb, he, hi = [], [], [], []
for n in [f"{m}_nested" for m in SSL] + ["compare_lr", "egemaps_lr", "handcrafted_lr"]:
    p = P(n)
    ok = pred7(p) == y7
    ptrue = np.where(np.isin(y, [0, 1]), p[:, 0] + p[:, 1], p[np.arange(len(y)), y])
    rho, pv = spearmanr(h, ptrue)
    hard = tenths <= 4
    hm.append(dict(system=n, acc7=ok.mean(), human=h.mean(), spearman=rho, p=pv, model_acc_on_hard=ok[hard].mean(), n_hard=int(hard.sum())))
    for b in ["0-30%", "40-60%", "70-90%", "100%"]:
        m = BIN == b
        hb.append(dict(system=n, human_bin=b, n=int(m.sum()), human=h[m].mean(), model=ok[m].mean()))
    for k, e in EMO7.items():
        m = y7 == k
        he.append(dict(system=n, emotion=e, human=h[m].mean(), model=ok[m].mean()))
    for it in ["normal", "strong"]:
        m = (meta["intensity"] == it).values
        hi.append(dict(system=n, intensity=it, human=h[m].mean(), model=ok[m].mean()))
hmdf = pd.DataFrame(hm)
hmdf.to_csv(os.path.join(res, "human_vs_model.csv"), index=False, float_format=FMT)
print("\n== human_vs_model.csv\n" + hmdf.to_string(index=False, formatters={"p": lambda v: f"{v:.3e}"}, float_format=lambda v: f"{v:.4f}"))
save(pd.DataFrame(hb), "human_bins.csv")
save(pd.DataFrame(he), "human_by_emotion.csv")
save(pd.DataFrame(hi), "human_by_intensity.csv")

ge = []
male = np.array([a % 2 == 1 for a in np.unique(g)])
for m in SSL:
    pa = per_actor_uar(y, P(f"{m}_nested"), g)
    ge.append(dict(system=f"{m}_nested", female=pa[~male].mean(), male=pa[male].mean(), p_mannwhitney=mannwhitneyu(pa[~male], pa[male]).pvalue))
save(pd.DataFrame(ge), "gender.csv")

perm = [dict(system=n, uar=rows[n]["uar"], acc=rows[n]["acc"]) for n in rows if "_perm_s" in n]
if perm:
    u = [d["uar"] for d in perm]
    js = dict(runs=perm, mean_uar=float(np.mean(u)), min_uar=float(np.min(u)), max_uar=float(np.max(u)), chance=0.125,
              note="labels permuted within each actor (class counts per actor kept); full nested layer x C selection repeated")
    json.dump(js, open(os.path.join(res, "sanity_permutation.json"), "w"), indent=1)
    print("\n== permutation sanity:", json.dumps(js))
