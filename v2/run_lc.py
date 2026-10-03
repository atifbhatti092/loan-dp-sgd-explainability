"""Learning curve on the 30,000-row Taiwan data: utility at fixed epsilon vs training-set size (comment 23).
Fixed 6,000-row stratified test set; training sets are stratified subsamples of the remaining 24,000 rows.
usage: python3 run_lc.py <budget_s>"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from interpret.privacy import DPExplainableBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load, metrics, sigma_for, DELTA, ROOT
from dpnet import Net
SIZES, REPS, EPS = [500, 1000, 2000, 5000, 10000, 24000], {24000: 3}, [1, 4]
GRID = [(C, lr) for C in (0.1, 0.5) for lr in (0.5, 2.0)]
def bq(n): B = int(min(256, max(32, n//8))); return B, B/n

def tune(X, y, steps, q, sg, hidden, seed):
    skf = StratifiedKFold(2, shuffle=True, random_state=seed); sc = []
    for C, lr in GRID:
        a_ = []
        for a, b in skf.split(X, y):
            p = Net(X.shape[1], hidden=hidden, seed=seed).fit(X[a], y[a], steps=steps, lr=lr, clip=C, sigma=sg, q=min(1.0, q*len(X)/len(a)), seed=seed).predict_proba(X[b]); a_.append(roc_auc_score(y[b], p))
        sc.append(np.mean(a_))
    return GRID[int(np.argmax(sc))]

def main(budget):
    t0 = time.time(); ds = load("taiwan"); path = os.path.join(ROOT, "results", "lc_taiwan.json"); out = json.load(open(path)) if os.path.exists(path) else []
    done = {(r["n"], r["rep"]) for r in out}
    pool, test = train_test_split(np.arange(len(ds.y)), test_size=6000, stratify=ds.y, random_state=0); dte = ds.df.iloc[test]; yte = ds.y[test]
    for n in SIZES:
        for rep in range(REPS.get(n, 5)):
            if (n, rep) in done: continue
            if time.time()-t0 > budget: print(f"PAUSED at n={n} rep={rep}"); return
            sub = pool if n == len(pool) else train_test_split(pool, train_size=n, stratify=ds.y[pool], random_state=rep)[0]
            dtr, ytr = ds.df.iloc[sub], ds.y[sub]; prep = ds.make_prep().fit(dtr); Xtr, Xte = prep.transform(dtr), prep.transform(dte)
            rec = {"n": n, "rep": rep, "models": {}}; M = rec["models"]; B, q = bq(n); T = 500
            M["logreg_nonprivate"] = metrics(yte, LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
            M["mlp_nonprivate"] = metrics(yte, Net(Xtr.shape[1], seed=rep).fit(Xtr, ytr, steps=1000, lr=0.5, q=min(1.0, 256/n), seed=rep).predict_proba(Xte))
            for eps in EPS:
                sg = sigma_for(T, q, eps)
                for tag, hid in (("mlp", (16, 8)), ("lr", ())):
                    C, lr = tune(Xtr, ytr, T, q, sg, hid, rep); rec.setdefault("chosen", {})[f"{tag}_{eps}"] = [C, lr]
                    M[f"dp_{tag}_eps{eps}"] = metrics(yte, Net(Xtr.shape[1], hidden=hid, seed=rep).fit(Xtr, ytr, steps=T, lr=lr, clip=C, sigma=sg, q=q, seed=rep+50).predict_proba(Xte)); M[f"dp_{tag}_eps{eps}"]["sigma"] = sg
                bounds = {i: (-4.0, 4.0) for i in range(Xtr.shape[1])}
                m = DPExplainableBoostingClassifier(epsilon=eps, delta=DELTA, privacy_bounds=bounds, random_state=rep).fit(np.clip(Xtr, -4, 4), ytr.astype(int))
                M[f"dp_ebm_eps{eps}"] = metrics(yte, m.predict_proba(np.clip(Xte, -4, 4))[:, 1])
            out.append(rec); json.dump(out, open(path, "w")); print(f"n={n} rep={rep} done ({time.time()-t0:.0f}s)", flush=True)
    print("DONE")
if __name__ == "__main__": main(float(sys.argv[1]))
