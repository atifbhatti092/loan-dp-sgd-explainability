"""Experiment 2: DP-MLP with hyper-parameters (C, lr) tuned by INNER 3-fold CV on the training fold only.
Hyper-parameter tuning consumes privacy that is NOT included in the reported epsilon (stated limitation)."""
import sys, json, time, os, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("absl").setLevel(logging.ERROR)
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".")
from data import load_raw, Prep
from exp1 import metrics, fit_mlp, sigma_for, cfg, EPS_GRID
GRID = [(C, lr) for C in (0.1, 0.25, 0.5) for lr in (0.5, 2.0)]
CFGS, EPS = ("FB100", "PO300"), [0.5, 1, 2, 4, 8]
OUT = "results/exp2_partial.json"

def tune(X, y, name, eps, n, seed):
    steps, q = cfg(name, n); sg = sigma_for(name, n, eps)
    skf = StratifiedKFold(3, shuffle=True, random_state=seed); sc = []
    for C, lr in GRID:
        a_ = []
        for a, b in skf.split(X, y):
            p = fit_mlp(X[a], y[a], steps, lr, seed, clip=C, sigma=sg, q=q).predict_proba(X[b]); a_.append(roc_auc_score(y[b], p))
        sc.append(np.mean(a_))
    return GRID[int(np.argmax(sc))]

def main(budget=240):
    t0 = time.time(); df, y = load_raw()
    out = json.load(open(OUT)) if os.path.exists(OUT) else []
    for si, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(df, y)):
        if si < len(out): continue
        if time.time() - t0 > budget: print("budget reached; rerun to continue"); return False
        prep = Prep().fit(df.iloc[tr]); Xtr = prep.transform(df.iloc[tr]); Xte = prep.transform(df.iloc[te]); ytr, yte = y[tr], y[te]; n = len(ytr)
        rec = {"split": si, "models": {}, "chosen": {}}
        for name in CFGS:
            steps, q = cfg(name, n)
            for eps in EPS:
                C, lr = tune(Xtr, ytr, name, eps, n, si)
                net = fit_mlp(Xtr, ytr, steps, lr, si*100, clip=C, sigma=sigma_for(name, n, eps), q=q)
                m = metrics(yte, net.predict_proba(Xte)); rec["models"][f"dpT_{name}_eps{eps}"] = m; rec["chosen"][f"{name}_{eps}"] = [C, lr]
        out.append(rec); json.dump(out, open(OUT, "w")); print(f"split {si+1}/25 done ({time.time()-t0:.0f}s)", flush=True)
    return True
if __name__ == "__main__": main(float(sys.argv[1]) if len(sys.argv) > 1 else 240)
