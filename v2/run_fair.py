"""Group-level behaviour of non-private and DP models, with and without protected attributes (comment 7).
usage: python3 run_fair.py <dataset> <n_splits>"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load, sigma_for, ROOT
from dpnet import Net
from run_cv import CFG
FAIR_EPS = [1, 4]
def main(name, nsplits):
    ds = load(name); cfg = CFG[name]; cv = json.load(open(os.path.join(ROOT, "results", f"cv_{name}.json"))); out = []
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=cfg["repeats"], random_state=0).split(ds.df, ds.split_y))[:nsplits]
    for si, (tr, te) in enumerate(splits):
        ch = cv[si]["chosen"]; dtr, dte = ds.df.iloc[tr], ds.df.iloc[te]; ytr, yte = ds.y[tr], ds.y[te]; n = len(ytr); q = cfg["B"]/n
        G = ds.groups(dte); rec = {"split": si, "groups": {}, "models": {}}
        for variant, drop in (("with_protected", False), ("without_protected", True)):
            prep = ds.make_prep(drop_protected=drop).fit(dtr); Xtr, Xte = prep.transform(dtr), prep.transform(dte)
            lr, st = ch["mlp_np"]; preds = {"nonprivate": Net(Xtr.shape[1], seed=si).fit(Xtr, ytr, steps=st, lr=lr, seed=si, q=None if cfg["np_q"] is None else cfg["np_q"]/n).predict_proba(Xte)}
            for eps in FAIR_EPS:
                C, lr2 = ch[f"mlp_PO_{eps}"]; preds[f"dp_eps{eps}"] = Net(Xtr.shape[1], seed=si).fit(Xtr, ytr, steps=cfg["T"], lr=lr2, clip=C, sigma=sigma_for(cfg["T"], q, eps), q=q, seed=si*100).predict_proba(Xte)
            for mname, p in preds.items():
                pred = (p >= 0.5).astype(int); cell = {}
                for gname, gv in G.items():
                    for lvl in np.unique(gv):
                        m = gv == lvl; y_, p_ = yte[m].astype(int), pred[m]
                        cell[f"{gname}={lvl}"] = [int(((p_ == 1) & (y_ == 1)).sum()), int(((p_ == 1) & (y_ == 0)).sum()), int(((p_ == 0) & (y_ == 1)).sum()), int(((p_ == 0) & (y_ == 0)).sum())]  # tp fp fn tn
                rec["models"][f"{variant}|{mname}"] = cell
        out.append(rec); print(name, "fair split", si+1, "/", len(splits), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", f"fair_{name}.json"), "w"))
if __name__ == "__main__": main(sys.argv[1], int(sys.argv[2]))
