"""Experiment 1: leak-free repeated CV, honest baselines, epsilon-first DP sweep with exact accounting."""
import sys, json, time, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("absl").setLevel(logging.ERROR)
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, f1_score, log_loss
sys.path.insert(0, ".")
from data import load_raw, Prep
from dpnet import Net
from accounting import solve_sigma, eps_pld

DELTA = 1e-5
EPS_GRID = [0.25, 0.5, 1, 2, 4, 8]
CLIP, LR = 1.0, 0.5
BATCH = 64
_sig = {}
def sigma_for(name, n, eps):
    k = (name, n, eps)
    if k not in _sig:
        steps, q = cfg(name, n); _sig[k] = solve_sigma(eps, steps, DELTA, q)
    return _sig[k]
def cfg(name, n):
    return {"FB30": (30, None), "FB100": (100, None), "PO300": (300, BATCH/n)}[name]

def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    try: auc = roc_auc_score(y, p)
    except ValueError: auc = np.nan
    return dict(acc=accuracy_score(y, pred), auc=auc, bacc=balanced_accuracy_score(y, pred),
                mcc=matthews_corrcoef(y, pred) if len(set(pred)) > 1 else 0.0,
                spec=float(((pred == 0) & (y == 0)).sum()/max((y == 0).sum(), 1)), f1=f1_score(y, pred))

def fit_mlp(X, y, steps, lr, seed, **kw): return Net(X.shape[1], seed=seed).fit(X, y, steps=steps, lr=lr, seed=seed, **kw)

GRID = [(lr, st) for lr in (0.1, 0.5, 1.0) for st in (30, 100, 300)]
def select_mlp(X, y, seed):
    """inner 3-fold CV on the training fold only (log-loss). Never touches the outer test fold."""
    skf = StratifiedKFold(3, shuffle=True, random_state=seed); best, bl = None, 9e9
    for lr, st in GRID:
        ls = []
        for a, b in skf.split(X, y):
            net = fit_mlp(X[a], y[a], st, lr, seed)
            ls.append(log_loss(y[b], np.clip(net.predict_proba(X[b]), 1e-6, 1-1e-6)))
        if np.mean(ls) < bl: bl, best = np.mean(ls), (lr, st)
    return best

def run(n_repeats=5, dp_seeds=1, log=None, only_first=None):
    df, y = load_raw()
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=0)
    out = []
    import os
    if os.path.exists("results/exp1_partial.json") and not only_first: out = json.load(open("results/exp1_partial.json"))
    for si, (tr, te) in enumerate(rskf.split(df, y)):
        if only_first and si >= only_first: break
        if si < len(out): continue
        t0 = time.time()
        prep = Prep().fit(df.iloc[tr]); Xtr = prep.transform(df.iloc[tr]); ch_tr = prep.raw_credit
        Xte = prep.transform(df.iloc[te]); ch_te = prep.raw_credit; ytr, yte = y[tr], y[te]; n = len(ytr)
        rec = {"split": si, "n_train": n, "n_test": len(yte), "models": {}}
        M = rec["models"]
        M["majority"] = metrics(yte, np.full(len(yte), float(ytr.mean() >= .5)))
        M["credit_history_rule"] = metrics(yte, ch_te.astype(float))
        lr_ = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr); M["logreg_L2"] = metrics(yte, lr_.predict_proba(Xte)[:, 1])
        rf = RandomForestClassifier(300, max_depth=6, random_state=si).fit(Xtr, ytr); M["random_forest"] = metrics(yte, rf.predict_proba(Xte)[:, 1])
        # non-private MLP: paper settings (lr .5, 30 steps) and nested-tuned settings
        M["mlp_paper_cfg"] = metrics(yte, fit_mlp(Xtr, ytr, 30, 0.5, si).predict_proba(Xte))
        lr_s, st_s = select_mlp(Xtr, ytr, si); rec["mlp_selected_cfg"] = [lr_s, st_s]
        M["mlp_tuned"] = metrics(yte, fit_mlp(Xtr, ytr, st_s, lr_s, si).predict_proba(Xte))
        # DP: epsilon-first
        for name in ("FB30", "FB100", "PO300"):
            steps, q = cfg(name, n)
            for eps in EPS_GRID:
                sg = sigma_for(name, n, eps); ps = []
                for s in range(dp_seeds):
                    net = fit_mlp(Xtr, ytr, steps, LR, si*100+s, clip=CLIP, sigma=sg, q=q); ps.append(net.predict_proba(Xte))
                M[f"dp_{name}_eps{eps}"] = metrics(yte, np.mean(ps, 0) if dp_seeds == 1 else ps[0])
                M[f"dp_{name}_eps{eps}"]["sigma"] = sg
        # clip-norm sensitivity (PO300)
        for C in (0.25, 0.5, 3.0):
            for eps in (1, 4):
                sg = sigma_for("PO300", n, eps); steps, q = cfg("PO300", n)
                net = fit_mlp(Xtr, ytr, steps, LR, si*100, clip=C, sigma=sg, q=q)
                M[f"dpC{C}_PO300_eps{eps}"] = metrics(yte, net.predict_proba(Xte))
        out.append(rec)
        json.dump(out, open("results/exp1_partial.json", "w"))
        if log: print(f"split {si+1} done in {time.time()-t0:.1f}s", flush=True)
    return out

if __name__ == "__main__":
    nrep = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    first = int(sys.argv[2]) if len(sys.argv) > 2 else None
    res = run(nrep, log=True, only_first=first)
    json.dump(res, open("results/exp1.json" if not first else "results/exp1_pilot.json", "w"))
