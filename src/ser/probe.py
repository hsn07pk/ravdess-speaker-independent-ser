import warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, PredefinedSplit, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

C_LR = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
C_SVM = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
GRID = {"lr": C_LR, "svm": C_SVM, "rf": [None]}
MAX_ITER = 5000
JOBS = 10


def model(kind, c, prob=True):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=MAX_ITER, class_weight="balanced"))
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=c, gamma="scale", class_weight="balanced", probability=prob, random_state=0))
    return RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0, n_jobs=1)


def outer_folds(y, g, outer):
    if isinstance(outer, np.ndarray):
        return list(PredefinedSplit(outer).split())
    if outer == "random5":
        return list(StratifiedKFold(5, shuffle=True, random_state=0).split(y, y))
    return list(LeaveOneGroupOut().split(y, y, g))


def inner_folds(y, g, inner):
    if inner == "random":
        return list(StratifiedKFold(5, shuffle=True, random_state=0).split(y, y))
    return list(GroupKFold(5).split(y, y, g))


def rows(X, idx, l, extra):
    Z = X[idx] if l is None else X[idx, l]
    return Z if extra is None else np.hstack([Z, extra[idx]])


def search(X, y, g, tr, l, kind, inner, extra, grid):
    Z, yt = rows(X, tr, l, extra), y[tr]
    splits = inner_folds(yt, g[tr], inner)
    s = np.zeros((len(splits), len(grid)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for i, (a, b) in enumerate(splits):
            if kind == "lr":
                sc = StandardScaler().fit(Z[a])
                A, B = sc.transform(Z[a]), sc.transform(Z[b])
                m = LogisticRegression(max_iter=MAX_ITER, class_weight="balanced", warm_start=True)
                for j, c in enumerate(grid):
                    s[i, j] = balanced_accuracy_score(yt[b], m.set_params(C=c).fit(A, yt[a]).predict(B))
            else:
                for j, c in enumerate(grid):
                    s[i, j] = balanced_accuracy_score(yt[b], model(kind, c, False).fit(Z[a], yt[a]).predict(Z[b]))
    return s.mean(0)


def final(X, y, tr, te, l, c, kind, extra):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = model(kind, c).fit(rows(X, tr, l, extra), y[tr])
    converged = True if kind != "lr" else bool(m[-1].n_iter_[0] < MAX_ITER)
    return m.predict_proba(rows(X, te, l, extra)), converged


def run(X, y, g, kind="lr", outer="loso", inner="group", layers=None, fold_layers=None, extra=None, per_layer=False, jobs=JOBS, grid=None):
    folds = outer_folds(y, g, outer)
    if not (isinstance(outer, str) and outer == "random5"):
        assert all(not np.isin(g[te], g[tr]).any() for tr, te in folds)
    grid = GRID[kind] if grid is None else grid
    if fold_layers is not None:
        cand = [[int(l)] for l in fold_layers]
    else:
        cand = [list(layers) if layers is not None else [None] for _ in folds]
    todo = [(f, l) for f in range(len(folds)) for l in cand[f]]
    if len(grid) > 1 or len(cand[0]) > 1:
        flat = Parallel(n_jobs=jobs)(delayed(search)(X, y, g, folds[f][0], l, kind, inner, extra, grid) for f, l in todo)
    else:
        flat = [np.zeros(1) for _ in todo]
    S = np.array(flat).reshape(len(folds), len(cand[0]), len(grid))
    choice = []
    for f in range(len(folds)):
        li, ci = np.unravel_index(np.argmax(S[f]), S[f].shape)
        choice.append((cand[f][li], grid[ci]))
    tasks = [(f, l, c) for f, (l, c) in enumerate(choice)]
    if per_layer:
        tasks += [(f, cand[f][li], grid[int(np.argmax(S[f, li]))]) for f in range(len(folds)) for li in range(len(cand[f]))]
    out = Parallel(n_jobs=jobs)(delayed(final)(X, y, folds[f][0], folds[f][1], l, c, kind, extra) for f, l, c in tasks)
    proba = np.zeros((len(y), 8))
    for (f, l, c), (p, ok) in zip(tasks[:len(folds)], out[:len(folds)]):
        proba[folds[f][1]] = p
    r = dict(proba=proba, choice=choice, inner=S, converged=[ok for _, ok in out[:len(folds)]])
    if per_layer:
        L = len(cand[0])
        lp = np.zeros((L, len(y), 8))
        for k, ((f, l, c), (p, ok)) in enumerate(zip(tasks[len(folds):], out[len(folds):])):
            lp[k % L][folds[f][1]] = p
        r["layer_proba"] = lp
        r["layer_converged"] = [ok for _, ok in out[len(folds):]]
    return r


def scores(y, proba):
    p = proba.argmax(1)
    return dict(uar=balanced_accuracy_score(y, p), acc=accuracy_score(y, p), f1=f1_score(y, p, average="macro"))


def per_actor_uar(y, proba, g):
    p = proba.argmax(1)
    return np.array([balanced_accuracy_score(y[g == a], p[g == a]) for a in np.unique(g)])


def bootstrap_ci(y, proba, g, reps=10000, seed=0):
    p = proba.argmax(1)
    actors = np.unique(g)
    hit = np.array([[np.sum((g == a) & (y == k) & (p == k)) for k in range(8)] for a in actors], float)
    tot = np.array([[np.sum((g == a) & (y == k)) for k in range(8)] for a in actors], float)
    idx = np.random.default_rng(seed).integers(0, len(actors), size=(reps, len(actors)))
    vals = (hit[idx].sum(1) / tot[idx].sum(1)).mean(1)
    return np.percentile(vals, [2.5, 97.5])


def speaker_norm(X, g):
    Z = X.copy()
    for a in np.unique(g):
        m = g == a
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    return Z


def permute_within(y, g, seed):
    rng = np.random.default_rng(seed)
    yp = y.copy()
    for a in np.unique(g):
        m = np.where(g == a)[0]
        yp[m] = y[m][rng.permutation(len(m))]
    return yp
