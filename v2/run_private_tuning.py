"""End-to-end private tuning benchmark: epsilon covers the training runs AND the hyper-parameter selection.
Same outer splits, preprocessing and DP-SGD settings as run_cv.py, so the results are directly comparable with the
'training-run-only' numbers of the paper (results/cv_<dataset>.json).
usage: python3 run_private_tuning.py <dataset> <budget_seconds> [accountant: auto|rdp|pld]   (resumable; rerun until DONE)
m = number of candidate (C, lr) settings: 1 = no tuning (a-priori default), 2 = lr only, 6 = full grid of the paper."""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load, metrics, DELTA, ROOT
from dpnet import Net
from private_tuning import train_and_select, GRIDS

EPS = [0.5, 1, 2, 4, 8]; MS = [1, 2, 6]
CFG = {"loan": dict(repeats=5, B=64, T=300, fb=True), "german": dict(repeats=3, B=64, T=300, fb=False), "taiwan": dict(repeats=2, B=256, T=800, fb=False)}

def one_split(ds, cfg, si, tr, te, accountant):
    dtr, dte = ds.df.iloc[tr], ds.df.iloc[te]; ytr, yte = ds.y[tr], ds.y[te]
    prep = ds.make_prep().fit(dtr); Xtr, Xte = prep.transform(dtr), prep.transform(dte); n = len(ytr)
    variants = [("PO", cfg["T"], cfg["B"]/n)] + ([("FB100", 100, None)] if cfg["fb"] else [])
    rec = {"split": si, "n_train": n, "n_test": len(yte), "models": {}, "info": {}}
    for tag, steps, q in variants:
        for eps in EPS:
            for m in MS:
                net, info = train_and_select(Net, Xtr, ytr, m, eps, steps, q, seed=si, delta=DELTA, accountant=accountant)
                key = f"{tag}_eps{eps}_m{m}"
                rec["models"][key] = metrics(yte, net.predict_proba(Xte))
                rec["info"][key] = {k: info[k] for k in ("chosen", "sigma_run", "sigma_sel", "eps_check", "accountant")}
    return rec

def main(name, budget, accountant="auto"):
    t0 = time.time(); ds = load(name); cfg = CFG[name]; path = os.path.join(ROOT, "results", f"ptune_{name}.json")
    out = json.load(open(path)) if os.path.exists(path) else []
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=cfg["repeats"], random_state=0).split(ds.df, ds.split_y))
    for si, (tr, te) in enumerate(splits):
        if si < len(out): continue
        if time.time()-t0 > budget: print(f"PAUSED at {si}/{len(splits)}"); return
        out.append(one_split(ds, cfg, si, tr, te, accountant)); json.dump(out, open(path, "w"))
        print(f"{name} split {si+1}/{len(splits)} ({time.time()-t0:.0f}s)", flush=True)
    print("DONE", name, len(out))
if __name__ == "__main__": main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 240, sys.argv[3] if len(sys.argv) > 3 else "auto")
