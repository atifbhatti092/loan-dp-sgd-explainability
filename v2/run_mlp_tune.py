"""Wider hyper-parameter search for the NON-private MLP (width, depth, lr, steps, weight decay), inner 3-fold CV on the training fold only.
Answers: is the MLP's weak showing against logistic regression just under-tuning?   usage: python3 run_mlp_tune.py <dataset> <n_splits>"""
import sys, os, json, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import log_loss
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load, metrics, ROOT
from dpnet import Net
from run_cv import CFG
GRID = list(itertools.product([(16, 8), (64,), (32, 32)], [0.1, 0.5, 1.0], [100, 300, 1000], [0.0, 1e-2]))
def main(name, nsplits):
    ds = load(name); cfg = CFG[name]; out = []; path = os.path.join(ROOT, "results", f"mlptune_{name}.json")
    for si, (tr, te) in enumerate(list(RepeatedStratifiedKFold(n_splits=5, n_repeats=cfg["repeats"], random_state=0).split(ds.df, ds.split_y))[:nsplits]):
        prep = ds.make_prep().fit(ds.df.iloc[tr]); Xtr, Xte = prep.transform(ds.df.iloc[tr]), prep.transform(ds.df.iloc[te]); ytr, yte = ds.y[tr], ds.y[te]
        skf = StratifiedKFold(3, shuffle=True, random_state=si); best, bl = None, 9e9
        for hid, lr, st, wd in GRID:
            ls = [log_loss(ytr[b], np.clip(Net(Xtr.shape[1], hidden=hid, seed=si).fit(Xtr[a], ytr[a], steps=st, lr=lr, wd=wd, seed=si).predict_proba(Xtr[b]), 1e-6, 1-1e-6)) for a, b in skf.split(Xtr, ytr)]
            if np.mean(ls) < bl: bl, best = np.mean(ls), (hid, lr, st, wd)
        hid, lr, st, wd = best; m = metrics(yte, Net(Xtr.shape[1], hidden=hid, seed=si).fit(Xtr, ytr, steps=st, lr=lr, wd=wd, seed=si).predict_proba(Xte))
        out.append({"split": si, "best": [list(hid), lr, st, wd], "metrics": m}); json.dump(out, open(path, "w")); print(name, "mlptune split", si+1, flush=True)
if __name__ == "__main__": main(sys.argv[1], int(sys.argv[2]))
